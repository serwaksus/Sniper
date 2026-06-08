#!/usr/bin/env python3
"""
DOTM Sniper v5.3.0 - Adaptive Signal Thresholds
Based on the mathematical edge of Deep Out-The-Money trading

v5.3.0 Changelog:
- Per-horizon signal thresholds (short/medium/long) from bot_settings
- Lowered PRICE_DELTA_THRESHOLD $0.005 -> $0.002 for DOTM sensitivity
- Added [SIGNAL-BATCH] logging for batch analysis visibility
- Fixed Renan Santos missing from hypothesis_db
"""
import subprocess
import json
import requests
import time
import re
import os
import sys
import logging
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotm_report import TelegramReporter

from news_scanner import check_market_news, extract_keywords
from utils import load_json, save_json, sanitize_for_prompt, check_and_write_pid, cleanup_pid_file
from equity_tracker import log_equity_snapshot, log_trade
from calibration_tracker import log_calibration_entry, detect_model_drift
from correlation_matrix import check_correlation_limit

PID_FILE = "/root/dotm-sniper/sniper.pid"

from utils import load_env_file
load_env_file()

_tr_instance = None

def _tr():
    global _tr_instance
    if _tr_instance is None:
        try:
            _tr_instance = TelegramReporter()
        except Exception:
            pass
    return _tr_instance

LOG_FILE = "/root/dotm-sniper/sniper.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

MODEL_LIGHT = "deepseek-chat"
MIN_P_MODEL = 0.03
MIN_CONFIDENCE = 0.65
MAX_POS_PCT = 0.10
FRACTIONAL_KELLY_MULTIPLIER = 0.25
BASE_POS_PCT = 0.02
OTHER_BOOST_POS_PCT = 0.035
MAX_CLUSTER_PCT = 0.30
MAX_POSITIONS = 5
BURN_IN_TRADES = 50
TAKE_PROFIT = 2.00

# v5.1.0: Smart Exit - automatic TP limit orders at $0.85
SMART_EXIT_PRICE = 0.85
SMART_EXIT_SLIPPAGE = 0.015  # $0.015 slippage penalty for backtesting

HYPOTHESIS_DB = "/root/dotm-sniper/hypothesis_db.json"
POSITIONS_FILE = "/root/dotm-sniper/positions.json"
SETTINGS_FILE = "/root/dotm-sniper/bot_settings.json"
CACHE_FILE = "/root/dotm-sniper/source_cache.json"
PRICE_TRACKING_FILE = "/root/dotm-sniper/price_tracking.json"
BACKTEST_STATS_FILE = "/root/dotm-sniper/backtest_stats.json"

DAILY_STATS_FILE = "/root/dotm-sniper/daily_stats.json"

PRICE_DELTA_THRESHOLD = 0.002
CACHE_TTL_SECONDS = 21600

MIN_TRADES_FOR_WEIGHT_ADJUSTMENT = 20
BAYESIAN_PRIOR_STRENGTH = 10
BACKTEST_COOLDOWN_SECONDS = 24 * 3600

MIN_BID_LIQUIDITY = 5.0


def update_daily_stats(balance, portfolio, trades_this_cycle):
    today = datetime.now().strftime("%Y-%m-%d")
    stats = load_json(DAILY_STATS_FILE, {"date": today, "trades": 0, "pnl": 0, "started": False})
    if stats.get("date") != today:
        stats = {"date": today, "trades": 0, "pnl": 0, "started": False}
    stats["started"] = True
    stats["trades"] = stats.get("trades", 0) + trades_this_cycle
    starting = get_settings().get("starting_balance", 500.0)
    stats["pnl"] = balance.get("total_value", 0) - starting
    save_json(DAILY_STATS_FILE, stats)

def load_cache():
    cache = load_json(CACHE_FILE, {"metaculus": {}, "news": {}, "last_update": None})
    now = datetime.now()
    for section in ("metaculus", "news"):
        entries = cache.get(section, {})
        stale = [k for k, v in entries.items()
                 if isinstance(v, dict) and v.get("timestamp")
                 and (now - datetime.fromisoformat(v["timestamp"])).total_seconds() > 86400]
        for k in stale:
            del entries[k]
    return cache

def parse_llm_json(response_text):
    start = response_text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(response_text)):
        c = response_text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(response_text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                break
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

def save_cache(cache):
    cache["last_update"] = datetime.now().isoformat()
    save_json(CACHE_FILE, cache)

def extract_keywords(question):
    stop_words = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for", "this", "that", "is", "are", "was", "were"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
    return [w for w in words if w not in stop_words][:10]



def get_settings():
    s = load_json(SETTINGS_FILE, {
        "min_confidence": MIN_CONFIDENCE,
        "position_size_pct": MAX_POS_PCT,
        "calibration_brier": None,
        "total_resolved": 0,
        "signal_threshold": 55,
        "min_p_model": MIN_P_MODEL
    })
    return s

def save_settings(s):
    s["__version"] = s.get("__version", 0) + 1
    save_json(SETTINGS_FILE, s)

def load_hypothesis_db():
    db = load_json(HYPOTHESIS_DB, {"hypotheses": [], "resolved": []})
    dirty = False
    active = [h for h in db.get("hypotheses", []) if not h.get("resolved")]
    if len(active) != len(db.get("hypotheses", [])):
        db["hypotheses"] = active
        dirty = True
    deduped = []
    seen = set()
    for h in db.get("resolved", []):
        if h["slug"] not in seen:
            deduped.append(h)
            seen.add(h["slug"])
    if len(deduped) != len(db.get("resolved", [])):
        db["resolved"] = deduped
        dirty = True
    if dirty:
        save_hypothesis_db(db)
    return db

def save_hypothesis_db(db):
    MAX_RESOLVED = 1000
    if len(db.get("resolved", [])) > MAX_RESOLVED:
        db["resolved"] = db["resolved"][-MAX_RESOLVED:]
    save_json(HYPOTHESIS_DB, db)

def detect_clusters(question):
    question_lower = question.lower()
    found = set()
    for cluster, keywords in CLUSTER_KEYWORDS.items():
        for kw in keywords:
            if ' ' in kw:
                if kw in question_lower:
                    found.add(cluster)
                    break
            else:
                if re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|[^a-z])', question_lower):
                    found.add(cluster)
                    break
    return list(found) if found else ["other"]

def calculate_brier_score(db):
    resolved = [h for h in db.get("resolved", []) if h.get("outcome") in ("YES", "NO")]
    if len(resolved) < BURN_IN_TRADES:        return None

    brier_scores = []
    wins = 0
    losses = 0
    for h in resolved[-BURN_IN_TRADES:]:
        p = h.get("p_model", 0.5)
        o = 1 if h.get("outcome") == "YES" else 0
        brier_scores.append((p - o) ** 2)
        if o == 1:
            wins += 1
        else:
            losses += 1

    brier = sum(brier_scores) / len(brier_scores)
    winrate = wins / (wins + losses) if (wins + losses) > 0 else 0
    logger.info(f"Stats: Brier={brier:.3f}, Winrate={winrate:.1%} ({wins}W/{losses}L) [from {len(resolved)} resolved]")

    settings = get_settings()
    old_brier = settings.get("calibration_brier")

    if old_brier is not None and brier > 0:
        if brier > 0.08 and settings.get("signal_threshold", 55) < 65:
            settings["signal_threshold"] = settings.get("signal_threshold", 55) + 2
            save_settings(settings)
            logger.info(f"[CALIBRATE] Brier {brier:.3f} > 0.08, raising signal_threshold to {settings['signal_threshold']}")
        elif brier < 0.03 and winrate > 0.1 and settings.get("signal_threshold", 55) > 40:
            settings["signal_threshold"] = settings.get("signal_threshold", 55) - 2
            save_settings(settings)
            logger.info(f"[CALIBRATE] Brier {brier:.3f} < 0.03, winrate {winrate:.0%}, lowering signal_threshold to {settings['signal_threshold']}")

    if winrate == 0 and len(resolved) >= 10 and settings.get("signal_threshold", 55) < 80:
        settings["signal_threshold"] = min(80, settings.get("signal_threshold", 55) + 5)
        save_settings(settings)
        logger.warning(f"[CALIBRATE] 0% winrate ({len(resolved)} resolved), RAISING signal_threshold to {settings['signal_threshold']} (defensive mode)")
    elif winrate < 0.30 and len(resolved) >= 20 and settings.get("signal_threshold", 55) < 75:
        settings["signal_threshold"] = min(75, settings.get("signal_threshold", 55) + 3)
        save_settings(settings)
        logger.info(f"[CALIBRATE] Low winrate ({winrate:.0%}, {len(resolved)} resolved), raising signal_threshold to {settings['signal_threshold']}")

    settings["calibration_brier"] = brier
    save_settings(settings)

    if len(resolved) >= 50:
        recent_resolved = [h for h in resolved if h.get("resolved_at") and (datetime.now() - datetime.fromisoformat(h["resolved_at"])).days <= 90]
        if len(recent_resolved) >= 20:
            from calibration import get_calibrator
            calibrator = get_calibrator()
            calibrator.fit(recent_resolved)
            calibrator.save()
            logger.info(f"[CALIBRATION] Trained isotonic model on {len(recent_resolved)} recent markets (<=90 days)")
        else:
            logger.info(f"[CALIBRATION] Only {len(recent_resolved)} recent resolved, need >=20, skipping retrain")

    return brier

def learn_from_results(db):
    resolved = db.get("resolved", [])
    if len(resolved) < 10:
        return {}

    settings = get_settings()
    factor_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    cluster_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    source_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    signal_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

    for h in resolved[-50:]:
        outcome = h.get("outcome")
        if outcome not in ("YES", "NO"):
            continue
        is_win = outcome == "YES"

        for factor in h.get("factors", []):
            key = f"{factor.get('direction')}:{factor.get('weight')}"
            if is_win:
                factor_stats[key]["wins"] += 1
            else:
                factor_stats[key]["losses"] += 1

        for cluster in h.get("clusters", []):
            if is_win:
                cluster_stats[cluster]["wins"] += 1
            else:
                cluster_stats[cluster]["losses"] += 1

        source_signal = h.get("source_signal", "default")
        if is_win:
            source_stats[source_signal]["wins"] += 1
        else:
            source_stats[source_signal]["losses"] += 1

    cluster_weights = {}
    for cluster, stats in cluster_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= MIN_TRADES_FOR_WEIGHT_ADJUSTMENT:
            winrate = stats["wins"] / total
            base_weight = {
                "venezuela": 0.30,
                "russia_ukraine": 0.25,
                "usa_politics": 0.20,
                "fed_fomc": 0.25,
                "ai_tech": 0.10,
                "sports_nba": 0.15,
                "sports_ufc": 0.15,
            }.get(cluster, 0.15)
            posterior_weight = (
                (base_weight * BAYESIAN_PRIOR_STRENGTH + winrate * total)
                / (BAYESIAN_PRIOR_STRENGTH + total)
            )
            posterior_weight = max(0.05, min(0.50, posterior_weight))
            cluster_weights[cluster] = posterior_weight
            logger.info(
                f"[LEARN] Cluster {cluster}: winrate={winrate:.1%} ({total} trades), "
                f"base={base_weight:.2f} → posterior={posterior_weight:.3f}"
            )
        else:
            logger.debug(
                f"[LEARN] Cluster {cluster}: insufficient data ({total}/{MIN_TRADES_FOR_WEIGHT_ADJUSTMENT} trades), "
                f"keeping base weight"
            )

    metaculus_bonus = 0.4
    geopol_bonus = 0.3
    sports_bonus = 0.2

    for source, stats in source_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= 3:
            winrate = stats["wins"] / total
            if source == "metaculus":
                metaculus_bonus = 0.4 * (1 + (winrate - 0.5))
                logger.info(f"Source {source}: winrate={winrate:.1%}, adjusted bonus={metaculus_bonus:.3f}")
            elif source == "geopol":
                geopol_bonus = 0.3 * (1 + (winrate - 0.5))
                logger.info(f"Source {source}: winrate={winrate:.1%}, adjusted bonus={geopol_bonus:.3f}")
            elif source == "sports":
                sports_bonus = 0.2 * (1 + (winrate - 0.5))
                logger.info(f"Source {source}: winrate={winrate:.1%}, adjusted bonus={sports_bonus:.3f}")

    settings["cluster_weights"] = cluster_weights
    settings["source_bonus_metaculus"] = metaculus_bonus
    settings["source_bonus_geopol"] = geopol_bonus
    settings["source_bonus_sports"] = sports_bonus
    save_settings(settings)

    return {
        "cluster_weights": cluster_weights,
        "source_stats": dict(source_stats)
    }


def backtest_recent(n=20):
    """
    Backtest recent resolved hypotheses to evaluate strategy performance.
    Returns simulation results and recommendations.
    """
    db = load_hypothesis_db()
    resolved = db.get("resolved", [])

    if len(resolved) < 5:
        return {"error": f"Only {len(resolved)} resolved, need at least 5", "recommendation": "skip"}

    recent = [h for h in resolved[-n:] if h.get("outcome") in ("YES", "NO")]
    if len(recent) < 5:
        return {"error": f"Only {len(recent)} with YES/NO outcome", "recommendation": "skip"}

    wins = 0
    total_pnl = 0
    brier_sum = 0

    for h in recent:
        p_model = h.get("p_model", 0.5)
        market_price = h.get("market_price", 0.5)
        outcome = 1 if h.get("outcome") == "YES" else 0
        is_win = outcome == 1

        if is_win:
            wins += 1
            pnl = (1 - market_price) / market_price
        else:
            actual_pnl = h.get("sold_pnl_pct") or h.get("pnl_at_exit")
            if actual_pnl is not None and actual_pnl != 0:
                pnl = actual_pnl
            else:
                pnl = -1

        total_pnl += pnl
        brier_sum += (p_model - outcome) ** 2

    winrate = wins / len(recent)
    avg_brier = brier_sum / len(recent)
    avg_pnl = total_pnl / len(recent)

    current_signal_threshold = get_settings().get("signal_threshold", 55)
    current_min_p = get_settings().get("min_p_model", MIN_P_MODEL)

    recommendations = []

    if winrate < 0.40:
        recommendations.append({
            "issue": "winrate_too_low",
            "current": winrate,
            "suggestion": "Raise MIN_PROB_RATIO or MIN_P_MODEL to be more selective"
        })

    if avg_brier > 0.20:
        recommendations.append({
            "issue": "poor_calibration",
            "current": avg_brier,
            "suggestion": "Improve p_model estimation or use market price as stronger prior"
        })

    if avg_pnl < 0:
        recommendations.append({
            "issue": "negative_avg_pnl",
            "current": avg_pnl,
            "suggestion": "Reduce position sizes or increase threshold"
        })

    cluster_wins = defaultdict(lambda: {"wins": 0, "total": 0})
    for h in recent:
        for c in h.get("clusters", []):
            cluster_wins[c]["total"] += 1
            if h.get("outcome") == "YES":
                cluster_wins[c]["wins"] += 1

    cluster_performance = {}
    for c, stats in cluster_wins.items():
        if stats["total"] >= 3:
            cluster_performance[c] = stats["wins"] / stats["total"]

    result = {
        "n_analyzed": len(recent),
        "winrate": winrate,
        "avg_brier": avg_brier,
        "avg_pnl": avg_pnl,
        "cluster_performance": cluster_performance,
        "recommendations": recommendations,
        "recommendation": "use_current" if not recommendations else "adjust_thresholds"
    }

    logger.info(f"[BACKTEST] n={len(recent)}, winrate={winrate:.1%}, brier={avg_brier:.3f}, pnl={avg_pnl:.2f}")
    for r in recommendations:
        logger.info(f"[BACKTEST] REC: {r['issue']} -> {r['suggestion']}")

    return result


def resolve_hypothesis_immediately(slug, current_price, entry_price):
    _cancel_all_tp_orders(slug)
    db = load_hypothesis_db()
    for h in db["hypotheses"]:
        if h["slug"] == slug and not h.get("resolved"):
            h["resolved"] = True
            h["resolved_at"] = datetime.now().isoformat()
            h["exit_price"] = current_price
            h["pnl_at_exit"] = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            h["exit_type"] = "manual"
            h["outcome"] = "SOLD"
            h["sold_pnl_pct"] = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            db["resolved"].append(h)

            positions = load_json(POSITIONS_FILE, {})
            if slug in positions:
                del positions[slug]
                save_json(POSITIONS_FILE, positions)

                try:
                    from bayesian_updater import cleanup_slug
                    cleanup_slug(slug)
                except Exception:
                    pass

            save_hypothesis_db(db)

            settings = get_settings()
            settings["total_resolved"] = settings.get("total_resolved", 0) + 1
            save_settings(settings)
            break

def repair_positions_file():
    """Fix inconsistent data in positions.json (e.g. high_price < entry_price)."""
    positions = load_json(POSITIONS_FILE, {})
    dirty = False
    for slug, p in positions.items():
        entry = p.get("entry_price", 0)
        high = p.get("high_price", 0)
        if entry > 0 and high < entry:
            p["high_price"] = entry
            logger.info(f"[REPAIR] {slug[:40]}... high_price {high:.4f} < entry_price {entry:.4f}, fixed")
            dirty = True
    if dirty:
        save_json(POSITIONS_FILE, positions)

def resolve_hypotheses():
    db = load_hypothesis_db()
    portfolio = get_portfolio()
    portfolio_slugs = {p["market_slug"] for p in portfolio}

    all_hypotheses = db.get("hypotheses", [])
    unresolved = [h for h in all_hypotheses if not h.get("resolved") and h["slug"] not in portfolio_slugs]

    if not unresolved:
        return

    market_map = {}
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "200"],
                           capture_output=True, text=True, timeout=20, start_new_session=True)
        for m in json.loads(res.stdout).get("data", []):
            market_map[m["slug"]] = m
    except Exception:
        pass

    new_resolved = 0
    for h in unresolved:
        slug = h["slug"]
        market_data = market_map.get(slug)

        if not market_data:
            h["resolved"] = True
            h["resolved_at"] = datetime.now().isoformat()
            h["outcome"] = "UNKNOWN"
            h["resolution_note"] = "market_not_found_in_api"
            db["resolved"].append(h)
            new_resolved += 1
            continue

        if not market_data.get("closed"):
            continue

        h["resolved"] = True
        h["resolved_at"] = datetime.now().isoformat()

        outcome = "UNKNOWN"
        if market_data.get("resolution") in ("YES", "NO"):
            outcome = market_data.get("resolution")
        elif market_data.get("outcome_prices"):
            yes_price = market_data.get("outcome_prices", [0.5])[0]
            outcome = "YES" if yes_price > 0.5 else "NO"

        h["outcome"] = outcome

        db["resolved"].append(h)
        new_resolved += 1

    save_hypothesis_db(db)

    if new_resolved > 0:
        settings = get_settings()
        settings["total_resolved"] = len([h for h in all_hypotheses if h.get("resolved")])
        save_settings(settings)

        if len(db.get("resolved", [])) >= BURN_IN_TRADES:
            calculate_brier_score(db)
            learn_from_results(db)

        for h in db.get("resolved", []):
            if h.get("outcome") in ("YES", "NO") and h.get("p_model") is not None:
                try:
                    log_calibration_entry(
                        slug=h["slug"],
                        question=h.get("question", ""),
                        p_model=h["p_model"],
                        p_calibrated=0,
                        market_price=h.get("market_price", 0),
                        actual_outcome=h["outcome"],
                        cluster=h.get("clusters", ["other"])[0],
                        entry_price=h.get("market_price", 0),
                        exit_price=h.get("exit_price", 0),
                        pnl_pct=h.get("pnl_at_exit", 0),
                    )
                except Exception:
                    pass

        try:
            drift_alert = detect_model_drift()
            if drift_alert:
                logger.warning(f"[CALIBRATION] {drift_alert}")
        except Exception:
            pass


def _load_price_tracking():
    tracking = load_json(PRICE_TRACKING_FILE, {})
    now = datetime.now()
    stale = [k for k, v in tracking.items()
             if isinstance(v, dict) and v.get("last_checked")
             and (now - datetime.fromisoformat(v["last_checked"])).total_seconds() > 86400]
    if stale:
        for k in stale:
            del tracking[k]
        _save_price_tracking(tracking)
    return tracking


def _save_price_tracking(tracking):
    save_json(PRICE_TRACKING_FILE, tracking)


def _check_price_delta(slug, current_price):
    """
    TAZ-3: Delta-scanning. Returns (should_analyze: bool, cached_p_model).
    If price changed < $0.005 since last check, reuse cached p_model.
    """
    tracking = _load_price_tracking()
    entry = tracking.get(slug)
    if entry:
        last_price = entry.get("last_price", 0)
        cached_p_model = entry.get("p_model")
        actual_threshold = max(PRICE_DELTA_THRESHOLD, current_price * 0.10)
        if cached_p_model is not None and abs(current_price - last_price) < actual_threshold:
            logger.info(
                f"[DELTA-SKIP] {slug[:40]}... price delta "
                f"${abs(current_price - last_price):.4f} < ${actual_threshold:.4f}, reusing p_model={cached_p_model:.1%}"
            )
            return False, cached_p_model
    return True, None


def _update_price_tracking(slug, current_price, p_model):
    tracking = _load_price_tracking()
    tracking[slug] = {
        "last_price": current_price,
        "p_model": p_model,
        "last_checked": datetime.now().isoformat(),
    }
    _save_price_tracking(tracking)


def _check_news_cache_freshness(cluster_key):
    """
    TAZ-3: Check source_cache.json freshness for a news cluster.
    Returns True if cache is fresh (< 6 hours), blocking new HTTP requests.
    """
    cache = load_json(CACHE_FILE, {"metaculus": {}, "news": {}, "last_update": None})
    news_section = cache.get("news", {})
    entry = news_section.get(cluster_key)
    if isinstance(entry, dict) and entry.get("timestamp"):
        try:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            age_seconds = (datetime.now() - cached_time).total_seconds()
            if age_seconds < CACHE_TTL_SECONDS:
                logger.info(f"[CACHE-FRESH] news cluster '{cluster_key}' age={age_seconds/3600:.1f}h < 6h, using cache")
                return True
        except (ValueError, TypeError):
            pass
    return False


def execute_trade(market, estimated_size, factors, analysis, balance):
    """Execute trade with advisor pre-check. Returns True if successful."""
    approved, verdict, adv_conf, adv_reason = advisor_pre_check(market, analysis, estimated_size, balance)
    if not approved:
        logger.info(f"[TRADE-BLOCKED] {market['slug']}: {adv_reason}")
        return False

    max_slippage = max(0.30, market["price"] * 2)
    current_ask = get_best_ask(market["slug"])
    if current_ask is not None and current_ask > market["price"] * (1 + max_slippage):
        logger.warning(f"[SNIPER] Slippage guard: ask={current_ask:.4f} > {max_slippage:.0%} above price={market['price']:.4f}, aborting")
        return False

    if not buy(market, estimated_size):
        print(f"   ❌ Buy failed for {market['slug']}")
        return False

    time.sleep(2)
    fill_data = get_actual_fill_price(market["slug"])
    if fill_data:
        log_slippage(market["slug"], market["price"], fill_data)

    shares = round(float(fill_data.get("shares", 0))) if fill_data and fill_data.get("shares", 0) > 0 else round(estimated_size / market["price"]) if market["price"] > 0 else 0

    positions = load_json(POSITIONS_FILE, {})
    if market["slug"] not in positions:
        positions[market["slug"]] = {
            "entry_price": fill_data.get("price", market["price"]) if fill_data else market["price"],
            "high_price": fill_data.get("price", market["price"]) if fill_data else market["price"],
            "trailing_on": False,
            "stop_loss": market["price"] * 0.7,
            "last_checked": datetime.now().isoformat(),
            "metaculus_prob": None,
            "market_question": market["question"],
            "outcome": market.get("outcome", "yes"),
            "clusters": market.get("clusters", ["other"]),
            "shares": shares,
        }
        save_json(POSITIONS_FILE, positions)

    if shares > 0:
        ladder_results = _place_tp_ladder(market["slug"], market["outcome"], shares)
        for price, shares_placed, ok, method in ladder_results:
            if ok:
                print(f"   🎯  TP rung placed @${price:.2f} ({shares_placed} shares)")
            else:
                print(f"   ⚠️  TP rung @{price:.2f} failed")
        if not ladder_results:
            print("   ⚠️  TP ladder placement failed, will rely on trailing_stop_check()")
    else:
        logger.warning(f"[SMART-EXIT] Zero shares for {market['slug']}, skipping TP")

    db = load_hypothesis_db()
    db["hypotheses"].append({
        "slug": market["slug"],
        "question": market["question"],
        "market_price": market["price"],
        "p_model": analysis["p_model"],
        "prob_ratio": analysis["prob_ratio"],
        "confidence": analysis["confidence"],
        "factors": factors,
        "clusters": market["clusters"],
        "size_pct": estimated_size / balance,
        "created_at": datetime.now().isoformat(),
        "resolved": False,
        "tp_limit_placed": True,
        "tp_limit_price": SMART_EXIT_PRICE,
    })
    save_hypothesis_db(db)

    try:
        log_trade(
            event_type="BUY",
            slug=market["slug"],
            question=market["question"],
            entry_price=market["price"],
            shares=shares,
            invested=estimated_size,
            reason=analysis.get("reasoning", "")[:100],
        )
    except Exception:
        pass

    if _tr():
        meta_prob = analysis.get("p_model")
        _tr().alert_new_position(
            market_slug=market["slug"],
            question=market["question"],
            entry_price=market["price"],
            amount=estimated_size,
            metaculus_prob=meta_prob,
            factors=factors,
            reasoning=analysis.get("reasoning", "")
        )

    return True


def _update_status_file():
    try:
        import shutil
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15, start_new_session=True)
        balance_data = json.loads(res.stdout).get("data", {})
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15, start_new_session=True)
        portfolio_data = json.loads(res.stdout).get("data", [])
        portfolio_data = [p for p in portfolio_data if float(p.get("shares", 0)) > 0.001]
        status = {"balance": balance_data, "portfolio": portfolio_data, "updated_at": datetime.now().isoformat()}
        save_json("/root/dotm-sniper/current_status.json", status)
        for dest in [
            "/root/.openclaw/workspace/dotm_status.json",
            "/root/.openclaw/agents/market_analyst/dotm_status.json",
            "/root/.openclaw/workspace/memory/portfolio-current.json",
        ]:
            try:
                shutil.copy("/root/dotm-sniper/current_status.json", dest)
            except Exception:
                pass
    except Exception:
        pass


def main():
    _main_inner()

def _main_inner():
    print("="*60)
    print("  DOTM SNIPER v5.3.0 - Batch Processing + Delta Scan + Advisor")
    print("="*60)

    repair_positions_file()

    settings = get_settings()
    last_backtest = settings.get("last_backtest_timestamp", 0)
    import time as _time
    now_ts = _time.time()
    if now_ts - last_backtest >= BACKTEST_COOLDOWN_SECONDS:
        bt = backtest_recent(n=20)
        if "error" not in bt:
            print(f"🧪 Backtest: winrate={bt['winrate']:.1%}, brier={bt['avg_brier']:.3f}, pnl={bt['avg_pnl']:.2f}")
            if bt.get("recommendations"):
                for r in bt["recommendations"]:
                    print(f"   ⚠️  {r['issue']}: {r['suggestion']}")
        settings["last_backtest_timestamp"] = now_ts
        save_settings(settings)
    else:
        hours_ago = (now_ts - last_backtest) / 3600
        logger.info(f"[BACKTEST-COOLDOWN] Skipping, last run {hours_ago:.1f}h ago (cooldown={BACKTEST_COOLDOWN_SECONDS/3600:.0f}h)")
    tier = get_tier_params(500)
    max_positions = settings.get("MAX_CONCURRENT_TRADES", tier["max_positions"])
    print(f"⚙️ Tier={tier['tier']}, thresholds: signal={settings.get('signal_threshold', 55)}, min_p_model={settings.get('min_p_model', 0.05):.0%}, confidence={settings['min_confidence']:.2f}, max_pos={max_positions}")

    resolve_hypotheses()
    trailing_stop_check()

    balance_data = get_balance()
    if balance_data is None:
        print("⚠️ Could not fetch balance, skipping cycle")
        return
    balance = balance_data.get("cash", 0)
    if balance <= 0:
        print("⚠️ Balance reported as $0, skipping cycle")
        return
    total_balance = balance_data.get("total_value", balance)

    tier = get_tier_params(total_balance)
    max_positions = settings.get("MAX_CONCURRENT_TRADES", tier["max_positions"])
    print(f"⚙️ Tier={tier['tier']} (balance=${total_balance:.2f}), max_pos={max_positions}, kelly={tier['kelly_mult']}")
    print(f"💰 Balance: ${balance:.2f} (total: ${total_balance:.2f})")

    portfolio = get_portfolio()
    print(f"📊 Open positions: {len(portfolio)}")

    _update_status_file()

    if len(portfolio) >= max_positions:
        print(f"⚠️ Max positions ({max_positions}) reached")
        return

    markets = fetch_markets()
    db = load_hypothesis_db()
    existing_slugs = {h["slug"] for h in db.get("hypotheses", []) if not h.get("resolved")}
    position_slugs = {p.get("market_slug", "") for p in portfolio}
    already_active = existing_slugs | position_slugs
    gamma_candidates = fetch_gamma_dotm_candidates(existing_slugs | position_slugs)
    seen = {m["slug"] for m in markets}
    for gc in gamma_candidates:
        if gc["slug"] not in seen:
            markets.append(gc)
            seen.add(gc["slug"])
    if not markets:
        print("No markets found")
        return

    print(f"📈 Candidates: {len(markets)} (pm-trader + {len(gamma_candidates)} gamma)")

    candidates_bought = 0
    available_balance = total_balance

    current_positions_for_clusters = [
        {"clusters": h.get("clusters", []), "size_pct": h.get("size_pct", 0)}
        for h in db.get("hypotheses", []) if not h.get("resolved")
    ]

    market_analyses = {}

    candidates_to_analyze = []
    for m in markets:
        if len(portfolio) + candidates_bought >= max_positions:
            break
        if available_balance < 5:
            break
        can_pass, _ = check_cluster_limits(m["clusters"], current_positions_for_clusters, portfolio_value=total_balance)
        if not can_pass:
            continue
        if m["slug"] in already_active:
            continue
        should_analyze, cached_p = _check_price_delta(m["slug"], m["price"])
        if not should_analyze and cached_p is not None:
            min_p_model = get_settings().get("min_p_model", MIN_P_MODEL)
            if cached_p >= min_p_model:
                logger.info(f"[DELTA-PROMOTE] {m['slug'][:40]}... cached p={cached_p:.1%}>={min_p_model:.0%}, promoting to scoring")
            else:
                logger.info(f"[DELTA-SKIP] {m['slug'][:40]}... cached p={cached_p:.1%}<{min_p_model:.0%}, skip")
                continue
        candidates_to_analyze.append(m)

    if candidates_to_analyze:
        candidates_to_analyze, pre_filtered = pre_filter_before_batching(candidates_to_analyze)
        if pre_filtered:
            print(f"   🔎 Pre-filtered {len(pre_filtered)} low-volume 'other' markets")
        print(f"\n📊 Batch-analyzing {len(candidates_to_analyze)} candidates (batch_size={BATCH_SIZE})...")
        for batch_start in range(0, len(candidates_to_analyze), BATCH_SIZE):
            batch = candidates_to_analyze[batch_start:batch_start + BATCH_SIZE]
            print(f"\n--- Batch {batch_start // BATCH_SIZE + 1} ({len(batch)} markets) ---")
            batch_results = batch_analyze_markets(batch)
            for m, analysis in zip(batch, batch_results):
                _update_price_tracking(m["slug"], m["price"], analysis["p_model"])
                market_analyses[m["slug"]] = (m, analysis)

    for m in markets:
        if len(portfolio) + candidates_bought >= max_positions:
            break

        if available_balance < 5:
            break

        can_pass, reason = check_cluster_limits(m["clusters"], current_positions_for_clusters, portfolio_value=total_balance)
        if not can_pass:
            continue

        if m["slug"] in already_active:
            continue

        if m["slug"] in market_analyses:
            _, analysis = market_analyses[m["slug"]]
        else:
            should_analyze, cached_p = _check_price_delta(m["slug"], m["price"])
            if not should_analyze and cached_p is not None:
                min_p_model = get_settings().get("min_p_model", MIN_P_MODEL)
                if cached_p < min_p_model:
                    continue
            analysis = full_market_analysis(m)
            _update_price_tracking(m["slug"], m["price"], analysis["p_model"])

        print(f"\n🔍 {m['question'][:55]}...")
        print(f"   Price: ${m['price']:.3f} | TTL: {m['ttl_hours']:.0f}h | Vol: ${m['volume']:,.0f}")
        print(f"   📈 P_model: {analysis['p_model']:.1%} | Ratio: {analysis.get('prob_ratio', 0):.2f}x | Conf: {analysis['confidence']:.2f}")

        if analysis["action"] == "SKIP":
            print("   ⏭️ Below threshold")
            continue

        factors = analysis.get("factors", [])

        estimated_size = position_size(
            analysis["p_model"],
            m["price"],
            available_balance,
            confidence=analysis["confidence"],
            best_ask=analysis.get("best_ask"),
            cluster=m.get("clusters", ["other"])[0]
        )

        if estimated_size <= 0:
            print("   ⏭️ Kelly edge negative, skipping")
            continue

        corr_ok, corr_reason = check_correlation_limit(
            m["clusters"][0] if m["clusters"] else "other",
            load_json(POSITIONS_FILE, {}),
            balance,
            new_investment=estimated_size,
        )
        if not corr_ok:
            logger.info(f"[CORR-SKIP] {m['slug'][:40]}... {corr_reason}")
            continue

        can_size, size_reason = check_category_limits(
            new_market=m,
            new_order_value=estimated_size,
            total_balance=total_balance,
            portfolio=portfolio
        )
        if not can_size:
            print(f"   ⏭️ Category limit: {size_reason}")
            continue

        print(f"   💵 Position size: ${estimated_size} ({estimated_size/available_balance:.1%} of balance)")

        market_for_news = {
            "question": m["question"],
            "slug": m["slug"],
            "price": m["price"],
            "metaculus_prob": analysis.get("p_model")
        }

        cluster_key = m.get("clusters", ["other"])[0]
        cache_fresh = _check_news_cache_freshness(cluster_key)
        if not cache_fresh:
            news_passed, news_reason = check_market_news(market_for_news)
            if not news_passed:
                print(f"   🚨 Trade blocked by news: {news_reason}")
                logger.info(f"[NEWS-BLOCK] {m['slug']}: {news_reason}")
                continue
        else:
            print(f"   📰 News cache fresh for cluster '{cluster_key}', skipping news check")

        if execute_trade(m, estimated_size, factors, analysis, total_balance):
            candidates_bought += 1
            available_balance -= estimated_size
            cluster = m.get("clusters", ["other"])[0]
            current_positions_for_clusters.append({
                "clusters": [cluster],
                "size_pct": estimated_size / total_balance
            })

    print(f"\n✅ Bought: {candidates_bought} | Available: ${available_balance:.2f}")

    update_daily_stats(balance_data, portfolio, candidates_bought)

    try:
        log_equity_snapshot()
    except Exception:
        pass

    db = load_hypothesis_db()
    resolved = db.get("resolved", [])
    if len(resolved) >= BURN_IN_TRADES:
        recent = resolved[-BURN_IN_TRADES:]
        wins = sum(1 for h in recent if h.get("outcome") == "YES")
        logger.info(f"Cycle complete: bought={candidates_bought}, recent_winrate={wins/len(recent):.1%}")

    try:
        from health_monitor import run_health_check
        run_health_check()
    except Exception as e:
        logger.debug(f"[HEALTH] Check failed: {e}")


# Re-export from extracted modules (at bottom to avoid circular imports)
from order_manager import (get_order_book, get_best_ask, get_balance, get_portfolio,
                           buy, _place_limit_sell, _place_tp_limit_order_single,
                           _place_tp_ladder, _get_open_tp_orders, _cancel_all_tp_orders,
                           get_actual_fill_price, log_slippage,
                           MAX_SPREAD_PCT, LIMIT_SPREAD_THRESHOLD, LIMIT_PRICE_BUFFER,
                           LIMIT_MAX_ATTEMPTS, SLIPPAGE_LOG_FILE)
from position_manager import (get_tier_params, position_size, check_cluster_limits,
                              get_category_exposure, check_category_limits,
                              MAX_EXPOSURE_PER_CATEGORY, CLUSTER_KEYWORDS)
from sell_executor import (_execute_sell, _check_sell_safety, _log_price_for_atr,
                           _calculate_atr, _get_atr_stop, _get_atr_trailing_stop,
                           trailing_stop_check,
                           TRAILING_ACTIVATION, TRAILING_STOP, CONVERGENCE_TAKE_PROFIT,
                           MIN_POSITION_CHECK_INTERVAL_HOURS, ATR_STOP_MULTIPLIER,
                           ATR_TRAILING_MULTIPLIER, ATR_LOOKBACK_DAYS, PRICE_HISTORY_FILE)
from signal_pipeline import (normalize_probability, calibrate_prediction,
                             _cluster_score_adjustment, fetch_markets,
                             fetch_gamma_dotm_candidates, pre_filter_before_batching,
                             full_market_analysis, batch_analyze_markets,
                             _parse_batch_response, _build_batch_results,
                             advisor_pre_check, metaculus_search, metaculus_get_question,
                             get_metaculus_forecast, get_time_decay_threshold,
                             check_metaculus_gap, _generate_search_queries,
                             _calculate_metaculus_match, parse_resolve_date, dates_match,
                             MIN_PROB_RATIO, MAX_P_MODEL_RATIO, MIN_VOLUME,
                             MIN_TTL_HOURS, MAX_PRICE, ALLOWED_CLUSTERS, BANNED_CLUSTERS,
                             BATCH_SIZE, CALIBRATION_DAMPING_FACTOR,
                             CALIBRATION_DOTM_THRESHOLD, CALIBRATION_AGGRESSIVE_PMODEL,
                             CALIBRATION_METACULUS_LOW, CLUSTER_SCORE_ADJUSTMENTS,
                             PRE_FILTER_OTHER_MIN_VOLUME, ADVISOR_MODEL,
                             ADVISOR_MIN_CONFIDENCE, DISPERSION_PENALTY_THRESHOLD,
                             METACULUS_GAP_THRESHOLD)

if __name__ == "__main__":
    single_run = len(sys.argv) > 1 and sys.argv[1] == "--once"

    if not check_and_write_pid(PID_FILE):
        sys.exit(1)
    try:
        if single_run:
            print("DOTM SNIPER v5.3.0 running single iteration...")
            _main_inner()
        else:
            print("DOTM SNIPER v5.3.0 starting...")
            while True:
                try:
                    _main_inner()
                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()
                print("Sleeping 30 min...")
                time.sleep(1800)
    finally:
        cleanup_pid_file(PID_FILE)
