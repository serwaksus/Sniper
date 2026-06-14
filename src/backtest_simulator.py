#!/usr/bin/env python3
"""
DOTM Backtester v3.0 - Core Simulation Module

Provides: market fetching, LLM analysis, parallel execution, TP ladder simulation.
Main entry points are in dotm_backtester.py.
"""
from __future__ import annotations
from typing import Any
import json
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import random
import subprocess
import requests
import concurrent.futures
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotm_sniper import (
    parse_llm_json, normalize_probability,
    detect_clusters,
    URL, HEADERS, MODEL_MAIN, ADVISOR_MODEL,
    MAX_P_MODEL_RATIO, MIN_P_MODEL, MIN_VOLUME,
    get_settings, MIN_CONFIDENCE, calibrate_prediction, _cluster_score_adjustment,
)

from manifold import check_manifold_gap
from utils import load_env_file
from config import SNIPER_LOG
load_env_file()

LOG_FILE = SNIPER_LOG
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

DOTM_PRICE_MIN = 0.01
DOTM_PRICE_MAX = 0.15

# v5.1.0: Smart Exit constants
SMART_EXIT_PRICE = 0.85
SMART_EXIT_SLIPPAGE = 0.015
SMART_EXIT_NET = SMART_EXIT_PRICE - SMART_EXIT_SLIPPAGE  # 0.835


def _simulate_tp_ladder(entry_price: float, high_price: float, resolution: str) -> tuple[float, list[dict]]:
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


def _normalize_keys(obj: Any) -> Any:
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


def _rate_limit_acquire() -> None:
    global _last_api_ts
    with _api_rate_lock:
        now = time.monotonic()
        wait = (1.0 / API_RATE_LIMIT_RPS) - (now - _last_api_ts)
        if wait > 0:
            time.sleep(wait)
        _last_api_ts = time.monotonic()


def _fetch_active_dotm_markets_pm_trader(limit: int = 200) -> list[dict]:
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
                        except Exception as e:
                            logger.debug(f"[dotm_backtester] {type(e).__name__}: {e}")
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


def _fetch_active_dotm_markets_gamma(limit: int = 200) -> list[dict]:
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


def _fetch_resolved_dotm_markets(limit: int = 150) -> list[dict]:
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


def backtest_analyze_single(market: dict) -> dict:
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
        metaculus_gap = check_manifold_gap(market, polymarket_prob)
        if not metaculus_gap or not metaculus_gap.get("found"):
            from metaforecast import check_metaforecast_gap
            metaculus_gap = check_metaforecast_gap(market, polymarket_prob)
    except Exception as e:
        logger.warning(f"[BACKTEST] External forecast gap error for {market['slug'][:30]}: {e}")

    source_signal = "default"
    if metaculus_gap and metaculus_gap.get("found"):
        source_signal = "metaculus"

    confidence = 0.60
    if source_signal == "metaculus":
        confidence = 0.80
    elif is_geopol:
        confidence = 0.70
    confidence = min(confidence, 0.95)

    gap_info = ""
    if metaculus_gap and metaculus_gap.get("found"):
        src = metaculus_gap.get("source", "external")
        prob = metaculus_gap.get("probability", metaculus_gap.get("metaculus_prob", 0))
        gap_info = f"- External forecast ({src}): {prob:.0%} vs Polymarket {metaculus_gap['polymarket_prob']:.0%}\n"

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

    if metaculus_gap and metaculus_gap.get("found") and metaculus_gap.get("signal_strength", 0) > 0.3:
        p_model_metaculus = metaculus_gap.get("probability", metaculus_gap.get("metaculus_prob", 0))
        p_model = max(p_model_llm, p_model_metaculus)
        source_signal = "metaculus_override"
        confidence = min(confidence + 0.10, 0.95)
    else:
        p_model = p_model_llm

    max_p_model = market["yes_price"] * MAX_P_MODEL_RATIO
    if p_model > max_p_model:
        p_model = max_p_model

    metaculus_prob_val = metaculus_gap.get("probability", metaculus_gap.get("metaculus_prob")) if metaculus_gap and metaculus_gap.get("found") else None
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


def backtest_advisor_check(market: dict, analysis: dict) -> tuple[bool, str]:
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


def _parallel_analyze_markets(markets: list[dict], label: str = "BACKTEST") -> list[dict | None]:
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

