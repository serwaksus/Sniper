#!/usr/bin/env python3
"""
DOTM Backtester v3.0 - Parallel Paper Trading + Historical Calibration

Two modes:
  --mode=live     (default) Fetch active DOTM markets, run full pipeline,
                   record predictions. Later check resolution via --check.
  --mode=sim      Use resolved markets with simulated DOTM prices for
                   quick calibration of LLM probability estimation.

Usage:
    python3 src/dotm_backtester.py --mode live --count 100
    python3 src/dotm_backtester.py --mode live --count 100 --skip-advisor
    python3 src/dotm_backtester.py --mode sim --count 50
    python3 src/dotm_backtester.py --check

Output: backtest_stats.json
"""
import json
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import argparse
import random
import subprocess
import requests
import concurrent.futures
import threading
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotm_sniper import (
    load_json, save_json, parse_llm_json, normalize_probability,
    detect_clusters, check_metaculus_gap,
    URL, HEADERS, MODEL_MAIN, ADVISOR_MODEL,
    MAX_P_MODEL_RATIO, MIN_P_MODEL, MIN_VOLUME,
    get_settings, MIN_CONFIDENCE, calibrate_prediction, _cluster_score_adjustment,
)

from utils import load_env_file
load_env_file()

LOG_FILE = "/root/dotm-sniper/sniper.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/markets"
BACKTEST_OUTPUT = "/root/dotm-sniper/backtest_stats.json"
SOURCE_CACHE_FILE = "/root/dotm-sniper/source_cache.json"

DOTM_PRICE_MIN = 0.01
DOTM_PRICE_MAX = 0.15

# v5.1.0: Smart Exit constants
SMART_EXIT_PRICE = 0.85
SMART_EXIT_SLIPPAGE = 0.015
SMART_EXIT_NET = SMART_EXIT_PRICE - SMART_EXIT_SLIPPAGE  # 0.835


def _simulate_tp_ladder(entry_price, high_price, resolution):
    """
    Симулирует TP Ladder из v5.3.0:
    - 50% объема продается по $0.75 (если high_price >= 0.75)
    - 30% объема продается по $0.85 (если high_price >= 0.85)
    - 20% остается до резолюции:
      * YES resolution → продается по $1.0
      * NO resolution → продается по $0.0
    Возвращает (weighted_upside_pct, ladder_details_dict)
    """
    LADDER_RUNGS = [
        (0.50, 0.75),
        (0.30, 0.85),
        (0.20, None),
    ]
    slippage = SMART_EXIT_SLIPPAGE
    weighted_pnl = 0.0
    details = []

    for pct, tp_price in LADDER_RUNGS:
        if tp_price is None:
            if resolution == "YES":
                exit_price = 1.0
                rung_label = "hold_yes"
            else:
                exit_price = 0.0
                rung_label = "hold_no"
            triggered = True
        elif high_price >= tp_price:
            exit_price = tp_price - slippage
            rung_label = f"tp_{tp_price:.2f}"
            triggered = True
        else:
            if resolution == "YES":
                exit_price = 1.0
                rung_label = f"fallback_yes_from_{tp_price:.2f}"
            else:
                exit_price = 0.0
                rung_label = f"fallback_no_from_{tp_price:.2f}"
            triggered = False

        if entry_price > 0:
            rung_pnl = (exit_price - entry_price) / entry_price
        else:
            rung_pnl = 0.0

        weighted_pnl += rung_pnl * pct
        details.append({
            "pct": pct,
            "tp_price": tp_price,
            "exit_price": exit_price,
            "triggered": triggered,
            "pnl": rung_pnl,
            "label": rung_label,
        })

    return weighted_pnl, details

BACKTEST_MAX_WORKERS = 10
API_RATE_LIMIT_RPS = 5
_api_rate_lock = threading.RLock()
_last_api_ts = 0.0


def _normalize_keys(obj):
    """Recursively strip whitespace from dict keys and string values."""
    if isinstance(obj, dict):
        return {
            (k.strip() if isinstance(k, str) else k): _normalize_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def _rate_limit_acquire():
    global _last_api_ts
    with _api_rate_lock:
        now = time.monotonic()
        wait = (1.0 / API_RATE_LIMIT_RPS) - (now - _last_api_ts)
        if wait > 0:
            time.sleep(wait)
        _last_api_ts = time.monotonic()


def _fetch_active_dotm_markets_pm_trader(limit=200):
    try:
        res = subprocess.run(
            ["pm-trader", "markets", "list", "--limit", str(limit)],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(res.stdout)
        candidates = []

        for m in data.get("data", []):
            if not m.get("active") or m.get("closed"):
                continue

            vol = float(m.get("volume", 0))
            if vol < MIN_VOLUME:
                continue

            for _outcome, price in zip(m.get("outcomes", []), m.get("outcome_prices", []), strict=False):
                try:
                    price = float(price)
                except (ValueError, TypeError):
                    continue

                if DOTM_PRICE_MIN <= price <= DOTM_PRICE_MAX:
                    clusters = detect_clusters(m["question"])
                    end_date = m.get("end_date", "")
                    ttl_hours = 9999
                    now = datetime.now()
                    if end_date:
                        try:
                            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            ttl_hours = max(0, (end - now).total_seconds() / 3600)
                        except Exception:
                            pass

                    candidates.append({
                        "slug": m["slug"],
                        "question": m["question"],
                        "yes_price": price,
                        "volume": vol,
                        "end_date": end_date,
                        "ttl_hours": ttl_hours,
                        "clusters": clusters,
                        "resolution": None,
                    })

        candidates.sort(key=lambda x: -x["volume"])
        seen = set()
        unique = []
        for c in candidates:
            if c["slug"] not in seen:
                seen.add(c["slug"])
                unique.append(c)
        return unique
    except Exception as e:
        logger.error(f"[BACKTEST] pm-trader fetch error: {e}")
        return []


def _fetch_active_dotm_markets_gamma(limit=200):
    markets = []
    offset = 0
    page_size = 100
    seen_slugs = set()

    while len(markets) < limit:
        params = {
            "closed": "false",
            "active": "true",
            "limit": page_size,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
            "volume_num_min": MIN_VOLUME,
        }

        try:
            resp = requests.get(GAMMA_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[BACKTEST] Gamma API error: {e}")
            break

        if not data:
            break

        for m in data:
            slug = m.get("slug", "")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            question = m.get("question", "")
            if not question:
                continue

            outcome_prices = m.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            if not outcome_prices:
                continue

            try:
                yes_price = float(outcome_prices[0])
            except (ValueError, IndexError, TypeError):
                continue

            if yes_price < DOTM_PRICE_MIN or yes_price > DOTM_PRICE_MAX:
                continue

            volume = float(m.get("volume", 0) or 0)
            if volume < MIN_VOLUME:
                continue

            clusters = detect_clusters(question)
            if "crypto" in clusters:
                continue
            end_date = m.get("endDate", "") or m.get("end_date", "")

            markets.append({
                "slug": slug,
                "question": question,
                "yes_price": yes_price,
                "volume": volume,
                "end_date": end_date,
                "ttl_hours": 9999,
                "clusters": clusters,
                "resolution": None,
            })

        offset += page_size
        if len(data) < page_size:
            break

        time.sleep(0.3)

    logger.info(f"[BACKTEST] Gamma API: {len(markets)} active DOTM markets")
    return markets[:limit]


GAMMA_API_MAX_PAGE = 100


def _fetch_resolved_dotm_markets(limit=150):
    markets = []
    offset = 0
    seen_slugs = set()

    while len(markets) < limit:
        params = {
            "closed": "true",
            "limit": GAMMA_API_MAX_PAGE,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
            "volume_num_min": MIN_VOLUME,
        }

        try:
            resp = requests.get(GAMMA_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[BACKTEST-RESOLVED] Gamma API error: {e}")
            break

        if not data:
            break

        for m in data:
            if not m.get("closed"):
                logger.debug(f"[BACKTEST-RESOLVED] Skipping non-closed: {m.get('slug', '?')[:40]}")
                continue

            slug = m.get("slug", "")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            question = m.get("question", "")
            if not question or len(question) < 15:
                continue

            outcome_prices = m.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            if not outcome_prices:
                continue

            try:
                yes_final = float(outcome_prices[0])
            except (ValueError, IndexError, TypeError):
                continue

            resolution_raw = m.get("resolution")
            if resolution_raw in ("Yes", "No"):
                resolution = resolution_raw.upper()
            elif yes_final > 0.5:
                resolution = "YES"
            elif yes_final < 0.5:
                resolution = "NO"
            else:
                logger.debug(f"[BACKTEST-RESOLVED] Skipping ambiguous resolution: {slug[:40]}")
                continue

            volume = float(m.get("volume", 0) or 0)
            if volume < MIN_VOLUME:
                continue

            clusters = detect_clusters(question)

            if "crypto" in clusters:
                continue

            try:
                historical_yes_price = float(outcome_prices[0]) if outcome_prices else None
            except (ValueError, IndexError, TypeError):
                historical_yes_price = None

            if historical_yes_price is not None and DOTM_PRICE_MIN <= historical_yes_price <= DOTM_PRICE_MAX:
                sim_price = historical_yes_price
                simulated_price = False
            elif historical_yes_price is not None and historical_yes_price < DOTM_PRICE_MIN:
                sim_price = DOTM_PRICE_MIN
                simulated_price = True
            else:
                sim_price = round(random.uniform(0.03, 0.10), 4)
                simulated_price = True

            logger.debug(f"[BACKTEST-PRICE] {slug[:40]}... sim_price=${sim_price:.4f} (hist={historical_yes_price})")

            created_at = m.get("createdAt", "") or m.get("created_at", "")
            end_date = m.get("endDate", "") or m.get("end_date", "")

            if resolution == "YES":
                if random.random() < 0.70:
                    high_price = min(1.0, max(0.85, random.betavariate(5, 2)))
                else:
                    high_price = random.uniform(sim_price * 1.5, 0.84)
                yes_final = 1.0
            else:
                if random.random() < 0.15:
                    high_price = random.uniform(0.85, 0.95)
                else:
                    high_price = random.uniform(sim_price, min(0.84, sim_price * 3))
                yes_final = 0.0

            markets.append({
                "slug": slug,
                "question": question,
                "yes_price": sim_price,
                "yes_final": yes_final,
                "high_price": high_price,
                "volume": volume,
                "end_date": end_date,
                "created_at": created_at,
                "ttl_hours": 9999,
                "clusters": clusters,
                "resolution": resolution,
                "simulated_price": simulated_price,
            })

        offset += GAMMA_API_MAX_PAGE
        if len(data) < GAMMA_API_MAX_PAGE:
            break

        time.sleep(0.3)

    logger.info(f"[BACKTEST-RESOLVED] Fetched {len(markets)} closed+resolved markets with validated resolution")
    return markets[:limit]


def backtest_analyze_single(market):
    """
    Run the Composite Scoring pipeline on a single market.
    Returns analysis dict with p_model, signal_score, action, etc.
    Thread-safe: performs network I/O only, no disk writes.
    """
    cluster = market.get("clusters", ["other"])[0]
    is_geopol = cluster in ["venezuela", "russia_ukraine", "usa_politics"]
    polymarket_prob = market["yes_price"]

    metaculus_gap = None
    try:
        metaculus_gap = check_metaculus_gap(market, polymarket_prob)
    except Exception as e:
        logger.warning(f"[BACKTEST] Metaculus gap error for {market['slug'][:30]}: {e}")

    source_signal = "default"
    if metaculus_gap:
        source_signal = "metaculus"

    confidence = 0.60
    if source_signal == "metaculus":
        confidence = 0.80
    elif is_geopol:
        confidence = 0.70
    confidence = min(confidence, 0.95)

    gap_info = ""
    if metaculus_gap:
        gap_info = f"- Metaculus forecast: {metaculus_gap['metaculus_prob']:.0%} vs Polymarket {metaculus_gap['polymarket_prob']:.0%}\n"

    historical_context = ""
    created_at = market.get("created_at", "")
    end_date = market.get("end_date", "")
    if created_at:
        historical_context += f"\nHISTORICAL CONTEXT: This market was created on {created_at[:10]}."
        if end_date:
            historical_context += f" It resolved on {end_date[:10]}."
        historical_context += (
            "\nCRITICAL: You must analyze this market as if it is {analysis_date}. "
            "Do NOT use any knowledge of the actual resolution or events that occurred "
            "after the market creation date. Base your estimate ONLY on information that "
            "would have been available at market creation time."
        )
    else:
        pass

    prompt = f"""Prediction market analyst. Your job is to find DOTM (deep out-the-money) events where the crowd SIGNIFICANTLY underestimates probability.

Market: {market['question']}
Price: ${market['yes_price']:.3f} ({market['yes_price']*100:.1f}%) | Volume: ${market.get('volume', 0):,.0f} | Category: {cluster}
Best Ask: ${polymarket_prob:.3f}
{gap_info}{historical_context}
ANCHORING WARNING: Do NOT simply return a probability near the market price. The market price already reflects the crowd. You must independently assess the TRUE probability based on the underlying event. If you cannot find a strong reason the probability should be higher, return the market price, but DO NOT default to 2x the price without reasoning.

Task: Identify 2-3 SPECIFIC factors. Estimate TRUE probability. Rate confidence.

Return ONLY JSON:
{{"factors": [{{"factor": "description", "direction": "supports/opposes", "weight": "high/medium/low", "source": "source"}}], "estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "brief"}}

Rules:
- estimated_probability: decimal 0.0-1.0 (NOT percentage)
- Conservative on low-volume (<$10K) markets
- If estimate >3x price, explain what crowd is missing
- IMPORTANT: For DOTM markets (price < 5%), even small probability increases are significant"""

    try:
        _rate_limit_acquire()
        resp = requests.post(URL, headers=HEADERS, json={
            "model": MODEL_MAIN,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }, timeout=60)

        resp_data = resp.json()
        if resp.status_code == 429:
            logger.warning(f"[BACKTEST] Rate limited for {market['slug'][:30]}, retrying after backoff")
            time.sleep(2)
            resp = requests.post(URL, headers=HEADERS, json={
                "model": MODEL_MAIN,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500
            }, timeout=60)
            resp_data = resp.json()

        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if not content:
            content = msg.get("reasoning") or ""

        result = parse_llm_json(content)
        if result:
            p_model_llm = normalize_probability(result.get("estimated_probability", market["yes_price"] * 2))
            confidence = min(max(float(result.get("confidence", confidence)), 0.1), 0.95)
            factors = result.get("factors", [])
        else:
            p_model_llm = market["yes_price"] * 2
            factors = []
    except Exception as e:
        logger.error(f"[BACKTEST] LLM error for {market['slug'][:30]}: {e}")
        p_model_llm = market["yes_price"] * 2
        factors = []

    if metaculus_gap and metaculus_gap.get("signal_strength", 0) > 0.3:
        p_model_metaculus = metaculus_gap["metaculus_prob"]
        p_model = max(p_model_llm, p_model_metaculus)
        source_signal = "metaculus_override"
        confidence = min(confidence + 0.10, 0.95)
    else:
        p_model = p_model_llm

    max_p_model = market["yes_price"] * MAX_P_MODEL_RATIO
    if p_model > max_p_model:
        p_model = max_p_model

    metaculus_prob_val = metaculus_gap.get("metaculus_prob") if metaculus_gap else None
    p_model_raw = p_model

    p_model, was_dampened = calibrate_prediction(p_model, market["yes_price"], metaculus_prob_val, cluster=cluster)

    damping_delta = None
    if was_dampened:
        damping_delta = p_model_raw - p_model
        logger.info(
            f"[CALIBRATE-DETAIL] slug={market['slug']} | "
            f"p_model_raw={p_model_raw:.4f} -> p_model_calibrated={p_model:.4f} | "
            f"delta={damping_delta:.4f} | price=${market['yes_price']:.3f} | "
            f"metaculus={'none' if metaculus_prob_val is None else f'{metaculus_prob_val:.1%}'}"
        )

    settings = get_settings()
    min_p_model = settings.get("min_p_model", MIN_P_MODEL)
    if p_model < min_p_model:
        return {
            "slug": market["slug"],
            "question": market["question"],
            "market_price": market["yes_price"],
            "p_model": p_model,
            "p_model_raw": p_model_raw,
            "prob_ratio": 0,
            "confidence": confidence,
            "action": "SKIP",
            "factors": factors,
            "source_signal": "default",
            "was_dampened": was_dampened,
            "damping_delta": damping_delta,
        }

    prob_ratio = p_model / market["yes_price"] if market["yes_price"] > 0 else 0
    supporting = [f for f in factors if f.get("direction") == "supports"]
    high_weight = [f for f in supporting if f.get("weight") == "high"]

    ratio_score = min(prob_ratio / 3.0, 1.0) * 30
    factor_score = min((len(supporting) + len(high_weight)) / 4, 1.0) * 20
    vol_score = min(market.get("volume", 0) / 1_000_000, 1.0) * 20
    ttl_hours = market.get("ttl_hours", 9999)
    ttl_days = ttl_hours / 24
    if ttl_days > 180:
        time_score = 20
    elif ttl_days > 90:
        time_score = 15
    elif ttl_days > 30:
        time_score = 10
    else:
        time_score = 0

    cluster = market.get("clusters", ["other"])[0]
    signal_score = ratio_score + factor_score + vol_score + time_score + _cluster_score_adjustment(cluster)

    base_threshold = settings.get("signal_threshold", 55)
    if ttl_days > 90:
        min_signal = base_threshold + 10
    elif ttl_days >= 31:
        min_signal = base_threshold + 5
    else:
        min_signal = base_threshold
    if source_signal == "metaculus_override":
        min_signal = max(min_signal - 10, 35)

    action = "BUY" if signal_score >= min_signal and confidence >= settings.get("min_confidence", MIN_CONFIDENCE) else "SKIP"

    return {
        "slug": market["slug"],
        "question": market["question"],
        "market_price": market["yes_price"],
        "p_model": p_model,
        "p_model_raw": p_model_raw,
        "prob_ratio": prob_ratio,
        "confidence": confidence,
        "action": action,
        "factors": factors,
        "source_signal": source_signal,
        "signal_score": signal_score,
        "was_dampened": was_dampened,
        "damping_delta": damping_delta,
    }


def backtest_advisor_check(market, analysis):
    question = market.get("question", "")
    p_model = analysis.get("p_model", 0)
    price = market.get("yes_price", 0)
    score = analysis.get("signal_score", 0)

    factors = analysis.get("factors", [])
    factors_text = "\n".join(
        f"  - [{f.get('weight', '?')}] {f.get('factor', '')} ({f.get('direction', '')})"
        for f in factors[:5]
    ) if factors else "  (none)"

    prompt = f"""You are DOTM Advisor - an independent risk analyst verifying a trade before execution.
Use Chain-of-Thought reasoning to evaluate the thesis.

MARKET: {question}
MARKET PRICE: ${price:.3f} ({price*100:.1f}%)
BOT P_MODEL (estimated true probability): {p_model:.1%}
PROBABILITY RATIO: {p_model/price:.2f}x vs market
COMPOSITE SIGNAL SCORE: {score:.0f}/100

SUPPORTING FACTORS IDENTIFIED BY BOT:
{factors_text}

YOUR TASK:
1. Think step-by-step about whether the bot's thesis is sound.
2. Check for hallucination patterns: p_model >> market price without concrete catalyst.
3. Is this a DOTM market where the crowd truly underestimates probability?

Return ONLY JSON:
{{"p_estimate": 0.XX, "confidence": 0.XX, "factors": ["factor1", "factor2"], "verdict": "CONFIRM/DIVERGE/WARNING/UNKNOWN"}}

Rules:
- CONFIRM: You agree the trade has positive expected value. confidence >= 0.70 required.
- DIVERGE: Your analysis contradicts the bot.
- WARNING: Significant risk factor the bot missed.
- UNKNOWN: Insufficient information."""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": ADVISOR_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1000
        }, timeout=120)

        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or msg.get("reasoning") or ""

        if content:
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)
            from advisor_script import parse_llm_advisor_response
            result, _parse_err = parse_llm_advisor_response(content, log_label="BACKTEST-ADVISOR")
            if result is not None:
                verdict = result.get("verdict", "UNKNOWN")
                adv_confidence = result.get("confidence", 0.0)
                return verdict == "CONFIRM" and adv_confidence >= 0.70, verdict
    except Exception as e:
        logger.warning(f"[BACKTEST-ADVISOR] Error: {e}")

    return False, "UNKNOWN"


def _parallel_analyze_markets(markets, label="BACKTEST"):
    """
    Run backtest_analyze_single() in parallel using ThreadPoolExecutor.
    Returns list of (index, market, analysis_or_None) sorted by index.
    """
    results = [None] * len(markets)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=BACKTEST_MAX_WORKERS) as executor:
        future_to_idx = {}
        for i, m in enumerate(markets):
            future = executor.submit(backtest_analyze_single, m)
            future_to_idx[future] = i

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            m = markets[idx]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"[{label}] Thread error for {m['slug'][:30]}: {e}")
                results[idx] = None

            completed += 1
            if completed % 10 == 0 or completed == len(markets):
                logger.info(f"[{label}] Analysis progress: {completed}/{len(markets)}")

    return results


def run_backtest_live(count=100, skip_advisor=False, use_calibrator=False):
    """
    Default mode: Fetch closed+resolved DOTM markets, run analysis in parallel,
    compute Winrate and Brier Score immediately (resolutions are known).
    Uses simulated DOTM opening prices (API exposes only final prices).
    """
    print("=" * 60)
    print(f"  DOTM BACKTESTER v3.0 [RESOLVED] - {count} historical markets")
    print(f"  Workers: {BACKTEST_MAX_WORKERS}")
    print("=" * 60)

    markets = _fetch_resolved_dotm_markets(limit=count)

    if not markets:
        print("No resolved DOTM markets found. Check API connectivity.")
        return

    print(f"\nFetched {len(markets)} resolved markets with validated outcomes")
    print(f"Running parallel analysis with {BACKTEST_MAX_WORKERS} workers...")

    analyses = _parallel_analyze_markets(markets, label="BACKTEST-RESOLVED")

    results = []
    wins = 0
    losses = 0
    skips = 0
    brier_scores = []
    brier_scores_raw = []
    dampened_count = 0
    upside_sum = 0.0
    upside_count = 0
    dampened_markets = []
    cluster_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "skips": 0, "total": 0})

    for i, (m, analysis) in enumerate(zip(markets, analyses, strict=False)):
        if analysis is None:
            logger.warning(f"[BACKTEST] Skipping market {m['slug'][:30]} due to analysis failure")
            continue

        print(f"\n[{i+1}/{len(markets)}] {m['question'][:55]}...")
        print(f"  Price: ${m['yes_price']:.3f} | Vol: ${m['volume']:,.0f} | Cluster: {m['clusters']}")
        print(f"  Resolution: {m['resolution']} | Created: {m.get('created_at', '?')[:10]}")
        print(f"  p_model={analysis['p_model']:.1%} | ratio={analysis.get('prob_ratio', 0):.2f}x | action={analysis['action']}")
        if analysis.get("was_dampened"):
            print(f"  [DAMPENED] raw={analysis.get('p_model_raw', 0):.1%} -> calibrated={analysis['p_model']:.1%}")

        advisor_approved = True
        advisor_verdict = "SKIPPED"
        if not skip_advisor and analysis["action"] == "BUY":
            advisor_approved, advisor_verdict = backtest_advisor_check(m, analysis)
            print(f"  Advisor: {advisor_verdict} (approved={advisor_approved})")

        final_action = analysis["action"]
        if final_action == "BUY" and not advisor_approved:
            final_action = "SKIP"
            print("  -> VETOED by advisor")

        actual_outcome = 1 if m["resolution"] == "YES" else 0
        p_model = analysis.get("p_model", 0)
        p_model_raw = analysis.get("p_model_raw", p_model)
        brier = (p_model - actual_outcome) ** 2
        brier_raw = (p_model_raw - actual_outcome) ** 2
        brier_scores.append(brier)
        brier_scores_raw.append(brier_raw)
        if analysis.get("was_dampened"):
            dampened_count += 1
            dampened_markets.append({
                "slug": m["slug"],
                "question": m["question"],
                "market_price": m["yes_price"],
                "p_model_raw": analysis.get("p_model_raw", 0),
                "p_model_calibrated": analysis.get("p_model", 0),
                "damping_delta": analysis.get("damping_delta", 0),
                "cluster": m.get("clusters", ["other"])[0],
            })

        (final_action == "BUY" and m["resolution"] == "YES")
        (final_action == "BUY" and m["resolution"] == "NO")

        high_price = m.get("high_price")
        if high_price is None:
            if m["resolution"] == "YES":
                high_price = 1.0
            else:
                high_price = m.get("yes_final", m["yes_price"])

        ladder_pnl = 0
        if final_action == "SKIP":
            skips += 1
        elif final_action == "BUY":
            ladder_pnl, ladder_details = _simulate_tp_ladder(
                entry_price=m["yes_price"],
                high_price=high_price,
                resolution=m["resolution"]
            )
            if ladder_pnl > 0:
                wins += 1
                upside_sum += ladder_pnl
                upside_count += 1
                logger.info(
                    f"[BACKTEST-LADDER] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%} "
                    f"rungs={[d['label'] for d in ladder_details]}"
                )
            else:
                losses += 1
                logger.info(
                    f"[BACKTEST-LADDER-LOSS] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%}"
                )

        for c in m.get("clusters", []):
            cluster_stats[c]["total"] += 1
            if final_action == "BUY" and ladder_pnl > 0:
                cluster_stats[c]["wins"] += 1
            elif final_action == "BUY":
                cluster_stats[c]["losses"] += 1
            else:
                cluster_stats[c]["skips"] += 1

        results.append({
            "slug": m["slug"],
            "question": m["question"],
            "market_price": m["yes_price"],
            "resolution": m["resolution"],
            "p_model": p_model,
            "p_model_raw": p_model_raw,
            "prob_ratio": analysis.get("prob_ratio", 0),
            "confidence": analysis.get("confidence", 0),
            "signal_score": analysis.get("signal_score", 0),
            "action": final_action,
            "advisor_verdict": advisor_verdict,
            "brier": brier,
            "brier_raw": brier_raw,
            "source_signal": analysis.get("source_signal", "default"),
            "clusters": m.get("clusters", []),
            "status": "resolved",
            "simulated_price": m.get("simulated_price", False),
            "was_dampened": analysis.get("was_dampened", False),
            "created_at": m.get("created_at", ""),
            "analyzed_at": datetime.now().isoformat(),
        })

    total_traded = wins + losses
    winrate = wins / total_traded if total_traded > 0 else 0
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0
    avg_brier_raw = sum(brier_scores_raw) / len(brier_scores_raw) if brier_scores_raw else 0
    brier_improvement = avg_brier_raw - avg_brier
    avg_upside = upside_sum / upside_count if upside_count > 0 else 0

    stats = {
        "timestamp": datetime.now().isoformat(),
        "mode": "resolved",
        "config": {
            "count_requested": count,
            "count_fetched": len(markets),
            "skip_advisor": skip_advisor,
            "simulated_prices": True,
            "price_range": f"${DOTM_PRICE_MIN}-${DOTM_PRICE_MAX}",
            "max_workers": BACKTEST_MAX_WORKERS,
        },
        "summary": {
            "total_markets": len(results),
            "traded": total_traded,
            "skipped": skips,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "brier_score": avg_brier,
            "brier_score_raw": avg_brier_raw,
            "brier_improvement": brier_improvement,
            "dampened_count": dampened_count,
            "avg_upside": avg_upside,
        },
        "cluster_stats": {k: v for k, v in cluster_stats.items() if v["total"] >= 1},
        "dampened_markets": dampened_markets,
        "results": results,
    }
    save_json(BACKTEST_OUTPUT, stats)

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS (resolved markets)")
    print("=" * 60)
    print(f"  Markets analyzed:  {len(results)}")
    print(f"  Traded:            {total_traded} (skipped: {skips})")
    print(f"  Wins / Losses:     {wins} / {losses}")
    print(f"  Winrate:           {winrate:.1%}")
    print(f"  Brier Score (calibrated): {avg_brier:.4f}")
    print(f"  Brier Score (raw):        {avg_brier_raw:.4f}")
    print(f"  Brier Improvement:        {brier_improvement:+.4f}")
    print(f"  Dampened predictions:      {dampened_count}/{len(results)}")
    if upside_count > 0:
        print(f"  Avg Upside (wins): {avg_upside:.2f}x")
    print()
    print("  Cluster breakdown:")
    for cluster, cs in sorted(cluster_stats.items(), key=lambda x: -x[1]["total"]):
        traded_c = cs["wins"] + cs["losses"]
        wr = cs["wins"] / traded_c if traded_c > 0 else 0
        print(f"    {cluster:20s}: {cs['total']:3d} total, {traded_c:3d} traded, winrate={wr:.1%}")
    print()
    print(f"  Results saved to: {BACKTEST_OUTPUT}")
    print()

    if dampened_markets:
        print("  Dampened markets breakdown:")
        print(f"  {'Slug':<45s} {'Price':>7s} {'Raw':>7s} {'Calib':>7s} {'Delta':>7s} {'Cluster':>15s}")
        print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*15}")
        for dm in dampened_markets:
            slug_display = dm['slug'][:44]
            print(
                f"  {slug_display:<45s} "
                f"${dm['market_price']:5.3f} "
                f"{dm['p_model_raw']:6.1%} "
                f"{dm['p_model_calibrated']:6.1%} "
                f"{dm['damping_delta']:6.1%} "
                f"{dm['cluster']:>15s}"
            )
        print()

    print("=" * 60)

    return _normalize_keys(load_json(BACKTEST_OUTPUT, {}))


def run_backtest_live_active(count=100, skip_advisor=False):
    """
    LIVE mode: Fetch active DOTM markets, run analysis in parallel, record predictions.
    Use --check later to resolve them.
    """
    print("=" * 60)
    print(f"  DOTM BACKTESTER v3.0 [LIVE-ACTIVE] - {count} active DOTM markets")
    print(f"  Workers: {BACKTEST_MAX_WORKERS}")
    print("=" * 60)

    markets = _fetch_active_dotm_markets_pm_trader(limit=count)
    if len(markets) < count:
        gamma_markets = _fetch_active_dotm_markets_gamma(limit=count)
        seen = {m["slug"] for m in markets}
        for m in gamma_markets:
            if m["slug"] not in seen:
                markets.append(m)
                seen.add(m["slug"])

    markets = markets[:count]

    if not markets:
        print("No active DOTM markets found. Check API connectivity.")
        return

    print(f"\nFetched {len(markets)} active DOTM markets for analysis")
    print(f"Running parallel analysis with {BACKTEST_MAX_WORKERS} workers...")

    analyses = _parallel_analyze_markets(markets, label="BACKTEST-LIVE")

    results = []
    buys = 0
    skips_count = 0
    cluster_stats = defaultdict(lambda: {"buys": 0, "skips": 0})

    for i, (m, analysis) in enumerate(zip(markets, analyses, strict=False)):
        if analysis is None:
            logger.warning(f"[BACKTEST-LIVE] Skipping market {m['slug'][:30]} due to analysis failure")
            continue

        print(f"\n[{i+1}/{len(markets)}] {m['question'][:55]}...")
        print(f"  Price: ${m['yes_price']:.3f} | Vol: ${m['volume']:,.0f} | Cluster: {m['clusters']}")
        print(f"  p_model={analysis['p_model']:.1%} | ratio={analysis.get('prob_ratio', 0):.2f}x | action={analysis['action']}")

        advisor_approved = True
        advisor_verdict = "SKIPPED"
        if not skip_advisor and analysis["action"] == "BUY":
            advisor_approved, advisor_verdict = backtest_advisor_check(m, analysis)
            print(f"  Advisor: {advisor_verdict} (approved={advisor_approved})")

        final_action = analysis["action"]
        if final_action == "BUY" and not advisor_approved:
            final_action = "SKIP"
            print("  -> VETOED by advisor")

        if final_action == "BUY":
            buys += 1
        else:
            skips_count += 1

        for c in m.get("clusters", []):
            if final_action == "BUY":
                cluster_stats[c]["buys"] += 1
            else:
                cluster_stats[c]["skips"] += 1

        results.append({
            "slug": m["slug"],
            "question": m["question"],
            "market_price": m["yes_price"],
            "resolution": None,
            "p_model": analysis.get("p_model", 0),
            "prob_ratio": analysis.get("prob_ratio", 0),
            "confidence": analysis.get("confidence", 0),
            "signal_score": analysis.get("signal_score", 0),
            "action": final_action,
            "advisor_verdict": advisor_verdict,
            "source_signal": analysis.get("source_signal", "default"),
            "clusters": m.get("clusters", []),
            "status": "pending",
            "analyzed_at": datetime.now().isoformat(),
        })

    stats = {
        "timestamp": datetime.now().isoformat(),
        "mode": "live",
        "config": {
            "count_requested": count,
            "count_fetched": len(markets),
            "skip_advisor": skip_advisor,
            "max_workers": BACKTEST_MAX_WORKERS,
        },
        "summary": {
            "total_markets": len(results),
            "buys": buys,
            "skips": skips_count,
            "pending_resolution": len(results),
        },
        "cluster_stats": dict(cluster_stats),
        "results": results,
    }
    save_json(BACKTEST_OUTPUT, stats)

    print("\n" + "=" * 60)
    print("  LIVE BACKTEST RESULTS (predictions recorded)")
    print("=" * 60)
    print(f"  Markets analyzed:  {len(results)}")
    print(f"  BUY signals:       {buys}")
    print(f"  SKIP signals:      {skips_count}")
    print()
    print("  Cluster breakdown:")
    for cluster, cs in sorted(cluster_stats.items(), key=lambda x: -(x[1]["buys"] + x[1]["skips"])):
        total = cs["buys"] + cs["skips"]
        print(f"    {cluster:20s}: {total:3d} total, {cs['buys']:3d} buys")
    print()
    print("  Run '--check' later to resolve pending predictions")
    print(f"  Results saved to: {BACKTEST_OUTPUT}")
    print("=" * 60)

    return stats


def run_backtest_sim(count=50, skip_advisor=False):
    """
    SIM mode: Fetch resolved markets, simulate DOTM prices, run analysis in parallel.
    Provides immediate Brier Score calibration.
    """
    print("=" * 60)
    print(f"  DOTM BACKTESTER v3.0 [SIM] - {count} resolved markets")
    print(f"  NOTE: Opening prices are SIMULATED (uniform ${DOTM_PRICE_MIN}-${DOTM_PRICE_MAX})")
    print(f"  Workers: {BACKTEST_MAX_WORKERS}")
    print("=" * 60)

    markets = _fetch_resolved_dotm_markets(limit=count)
    if not markets:
        print("No resolved markets found. Check API connectivity.")
        return

    print(f"\nFetched {len(markets)} resolved markets with simulated DOTM prices")
    print(f"Running parallel analysis with {BACKTEST_MAX_WORKERS} workers...")

    analyses = _parallel_analyze_markets(markets, label="BACKTEST-SIM")

    results = []
    wins = 0
    losses = 0
    skips = 0
    brier_scores = []
    upside_sum = 0.0
    upside_count = 0
    cluster_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})

    for i, (m, analysis) in enumerate(zip(markets, analyses, strict=False)):
        if analysis is None:
            logger.warning(f"[BACKTEST-SIM] Skipping market {m['slug'][:30]} due to analysis failure")
            continue

        print(f"\n[{i+1}/{len(markets)}] {m['question'][:55]}...")
        print(f"  Sim Price: ${m['yes_price']:.3f} | Resolution: {m['resolution']} | Cluster: {m['clusters']}")
        print(f"  p_model={analysis['p_model']:.1%} | ratio={analysis.get('prob_ratio', 0):.2f}x | action={analysis['action']}")

        advisor_approved = True
        advisor_verdict = "SKIPPED"
        if not skip_advisor and analysis["action"] == "BUY":
            advisor_approved, advisor_verdict = backtest_advisor_check(m, analysis)
            print(f"  Advisor: {advisor_verdict} (approved={advisor_approved})")

        final_action = analysis["action"]
        if final_action == "BUY" and not advisor_approved:
            final_action = "SKIP"
            print("  -> VETOED by advisor")

        actual_outcome = 1 if m["resolution"] == "YES" else 0
        p_model = analysis.get("p_model", 0)
        brier = (p_model - actual_outcome) ** 2
        brier_scores.append(brier)

        (final_action == "BUY" and m["resolution"] == "YES")
        (final_action == "BUY" and m["resolution"] == "NO")

        high_price = m.get("high_price")
        if high_price is None:
            if m["resolution"] == "YES":
                high_price = 1.0
            else:
                high_price = m.get("yes_final", m["yes_price"])

        ladder_pnl = 0
        if final_action == "SKIP":
            skips += 1
        elif final_action == "BUY":
            ladder_pnl, ladder_details = _simulate_tp_ladder(
                entry_price=m["yes_price"],
                high_price=high_price,
                resolution=m["resolution"]
            )
            if ladder_pnl > 0:
                wins += 1
                upside_sum += ladder_pnl
                upside_count += 1
                logger.info(
                    f"[BACKTEST-LADDER] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%} "
                    f"rungs={[d['label'] for d in ladder_details]}"
                )
            else:
                losses += 1
                logger.info(
                    f"[BACKTEST-LADDER-LOSS] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%}"
                )

        for c in m.get("clusters", []):
            cluster_stats[c]["total"] += 1
            if final_action == "BUY" and ladder_pnl > 0:
                cluster_stats[c]["wins"] += 1
            elif final_action == "BUY":
                cluster_stats[c]["losses"] += 1

        results.append({
            "slug": m["slug"],
            "question": m["question"],
            "market_price": m["yes_price"],
            "resolution": m["resolution"],
            "p_model": p_model,
            "prob_ratio": analysis.get("prob_ratio", 0),
            "confidence": analysis.get("confidence", 0),
            "signal_score": analysis.get("signal_score", 0),
            "action": final_action,
            "advisor_verdict": advisor_verdict,
            "brier": brier,
            "source_signal": analysis.get("source_signal", "default"),
            "clusters": m.get("clusters", []),
            "simulated_price": m.get("simulated_price", False),
        })

    total_traded = wins + losses
    winrate = wins / total_traded if total_traded > 0 else 0
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0
    avg_upside = upside_sum / upside_count if upside_count > 0 else 0

    stats = {
        "timestamp": datetime.now().isoformat(),
        "mode": "sim",
        "config": {
            "count_requested": count,
            "count_fetched": len(markets),
            "skip_advisor": skip_advisor,
            "simulated_prices": True,
            "price_range": f"${DOTM_PRICE_MIN}-${DOTM_PRICE_MAX}",
            "max_workers": BACKTEST_MAX_WORKERS,
        },
        "summary": {
            "total_markets": len(markets),
            "traded": total_traded,
            "skipped": skips,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "brier_score": avg_brier,
            "avg_upside": avg_upside,
        },
        "cluster_stats": {k: v for k, v in cluster_stats.items() if v["total"] >= 1},
        "results": results,
    }
    save_json(BACKTEST_OUTPUT, stats)

    print("\n" + "=" * 60)
    print("  SIM BACKTEST RESULTS (simulated DOTM prices)")
    print("=" * 60)
    print(f"  Markets analyzed:  {len(markets)}")
    print(f"  Traded:            {total_traded} (skipped: {skips})")
    print(f"  Wins / Losses:     {wins} / {losses}")
    print(f"  Winrate:           {winrate:.1%}")
    print(f"  Brier Score:       {avg_brier:.4f}")
    print(f"  Avg Upside (wins): {avg_upside:.2f}x")
    print()
    print("  Cluster breakdown:")
    for cluster, cs in sorted(cluster_stats.items(), key=lambda x: -x[1]["total"]):
        wr = cs["wins"] / (cs["wins"] + cs["losses"]) if (cs["wins"] + cs["losses"]) > 0 else 0
        print(f"    {cluster:20s}: {cs['total']:3d} traded, winrate={wr:.1%}")
    print()
    print(f"  Results saved to: {BACKTEST_OUTPUT}")
    print("=" * 60)

    return stats


def check_pending():
    stats = _normalize_keys(load_json(BACKTEST_OUTPUT, {}))
    if not stats or stats.get("mode") != "live":
        print("No pending live backtest found. Run --mode live first.")
        return

    results = stats.get("results", [])
    pending = [r for r in results if r.get("status") == "pending"]
    if not pending:
        print("All predictions already resolved.")
        return

    print(f"Checking {len(pending)} pending predictions...")

    resolved_count = 0
    wins = 0
    losses = 0
    still_pending = 0

    for r in pending:
        slug = r["slug"]
        try:
            resp = requests.get(
                GAMMA_API,
                params={"slug": slug, "limit": 1},
                timeout=15
            )
            data = resp.json()
            if not data:
                still_pending += 1
                continue

            m = data[0]
            if not m.get("closed"):
                still_pending += 1
                continue

            outcome_prices = m.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            if outcome_prices:
                try:
                    yes_final = float(outcome_prices[0])
                except (ValueError, IndexError, TypeError):
                    yes_final = 0.5

                if yes_final > 0.5:
                    r["resolution"] = "YES"
                else:
                    r["resolution"] = "NO"
                r["status"] = "resolved"
                r["resolved_at"] = datetime.now().isoformat()

                actual = 1 if r["resolution"] == "YES" else 0
                r["brier"] = (r.get("p_model", 0.5) - actual) ** 2

                if r["action"] == "BUY" and r["resolution"] == "YES":
                    wins += 1
                elif r["action"] == "BUY" and r["resolution"] == "NO":
                    losses += 1

                resolved_count += 1
                print(f"  {slug[:40]}... => {r['resolution']} {'WIN' if r['action'] == 'BUY' and r['resolution'] == 'YES' else ''}")
            else:
                still_pending += 1
        except Exception as e:
            logger.warning(f"[CHECK] Error for {slug}: {e}")
            still_pending += 1

        time.sleep(0.3)

    traded = sum(1 for r in results if r["action"] == "BUY" and r.get("status") == "resolved")
    sum(1 for r in results if r["action"] == "BUY")
    briers = [r["brier"] for r in results if r.get("brier") is not None and r.get("status") == "resolved"]

    stats["summary"]["resolved"] = resolved_count
    stats["summary"]["still_pending"] = still_pending
    stats["summary"]["wins"] = wins
    stats["summary"]["losses"] = losses
    stats["summary"]["winrate"] = wins / max(traded, 1)
    stats["summary"]["brier_score"] = sum(briers) / max(len(briers), 1)

    save_json(BACKTEST_OUTPUT, stats)

    print(f"\nResolved: {resolved_count}, Still pending: {still_pending}")
    print(f"Wins: {wins}, Losses: {losses}, Winrate: {wins/max(traded,1):.1%}")
    if briers:
        print(f"Brier Score: {sum(briers)/len(briers):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOTM Sniper Backtester v3.0")
    parser.add_argument("--mode", choices=["resolved", "sim", "live"], default="resolved",
                        help="resolved=closed+resolved markets with immediate metrics (default), "
                             "sim=alias for resolved, live=active markets (record + check later)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of markets to backtest (default: 100)")
    parser.add_argument("--skip-advisor", action="store_true",
                        help="Skip advisor pre-check (saves tokens)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply isotonic calibration from pre-trained model")
    parser.add_argument("--check", action="store_true",
                        help="Check resolution of pending predictions (live mode only)")
    args = parser.parse_args()

    if args.check:
        check_pending()
    elif args.mode in ("resolved", "sim"):
        run_backtest_live(count=args.count, skip_advisor=args.skip_advisor, use_calibrator=args.calibrate)
    elif args.mode == "live":
        run_backtest_live_active(count=args.count, skip_advisor=args.skip_advisor)
