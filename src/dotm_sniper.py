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
import time
import re
import os
import sys
import logging
import signal
from logging.handlers import RotatingFileHandler
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotm_report import TelegramReporter

import positions_db
from news_scanner import check_market_news
from utils import load_json, save_json, check_and_write_pid, cleanup_pid_file
from equity_tracker import log_equity_snapshot
from correlation_matrix import check_correlation_limit
from schema import (
    HYP_CLUSTERS,
    HYP_DB_HYPOTHESES,
    HYP_DB_RESOLVED,
    HYP_OUTCOME,
    HYP_RESOLVED,
    HYP_SIZE_PCT,
    HYP_SLUG,
    POS_ENTRY_PRICE, POS_HIGH_PRICE,
    SETTINGS_CALIBRATION_BRIER,
    SETTINGS_LAST_BACKTEST, SETTINGS_MAX_CONCURRENT,
    SETTINGS_MIN_CONFIDENCE, SETTINGS_MIN_P_MODEL, SETTINGS_POSITION_SIZE_PCT,
    SETTINGS_SIGNAL_THRESHOLD, SETTINGS_STARTING_BALANCE, SETTINGS_TOTAL_RESOLVED,
    SETTINGS_VERSION, TRACKING_LAST_CHECK, TRACKING_LAST_PRICE, TRACKING_P_MODEL,
)

PID_FILE = "/root/dotm-sniper/sniper.pid"

from utils import load_env_file, validate_env_vars  # noqa: E402
import contextlib  # noqa: E402
load_env_file()
validate_env_vars(["DEEPSEEK_API_KEY", "TG_BOT_TOKEN", "TG_CHAT_ID"])

_tr_instance = None

def _tr():
    global _tr_instance
    if _tr_instance is None:
        with contextlib.suppress(Exception):
            _tr_instance = TelegramReporter()
    return _tr_instance

LOG_FILE = "/root/dotm-sniper/logs/sniper.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
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

_shutdown_requested = False

def _handle_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("[MAIN] Shutdown signal received, finishing current cycle...")

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

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
    starting = get_settings().get(SETTINGS_STARTING_BALANCE, 500.0)
    stats["pnl"] = balance.get("total_value", 0) - starting
    save_json(DAILY_STATS_FILE, stats)

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





def get_settings():
    s = load_json(SETTINGS_FILE, {
        SETTINGS_MIN_CONFIDENCE: MIN_CONFIDENCE,
        SETTINGS_POSITION_SIZE_PCT: MAX_POS_PCT,
        SETTINGS_CALIBRATION_BRIER: None,
        SETTINGS_TOTAL_RESOLVED: 0,
        SETTINGS_SIGNAL_THRESHOLD: 55,
        SETTINGS_MIN_P_MODEL: MIN_P_MODEL
    })
    return s

def save_settings(s):
    s[SETTINGS_VERSION] = s.get(SETTINGS_VERSION, 0) + 1
    save_json(SETTINGS_FILE, s)

def load_hypothesis_db():
    db = load_json(HYPOTHESIS_DB, {HYP_DB_HYPOTHESES: [], HYP_DB_RESOLVED: []})
    dirty = False
    active = [h for h in db.get(HYP_DB_HYPOTHESES, []) if not h.get(HYP_RESOLVED)]
    if len(active) != len(db.get(HYP_DB_HYPOTHESES, [])):
        db[HYP_DB_HYPOTHESES] = active
        dirty = True
    deduped = []
    seen = set()
    for h in db.get(HYP_DB_RESOLVED, []):
        if h[HYP_SLUG] not in seen:
            deduped.append(h)
            seen.add(h[HYP_SLUG])
    if len(deduped) != len(db.get(HYP_DB_RESOLVED, [])):
        db[HYP_DB_RESOLVED] = deduped
        dirty = True
    if dirty:
        save_hypothesis_db(db)
    return db

def save_hypothesis_db(db):
    MAX_RESOLVED = 1000
    if len(db.get(HYP_DB_RESOLVED, [])) > MAX_RESOLVED:
        db[HYP_DB_RESOLVED] = db[HYP_DB_RESOLVED][-MAX_RESOLVED:]
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
    from resolution import calculate_brier_score as _impl
    return _impl(db)

def learn_from_results(db):
    from resolution import learn_from_results as _impl
    return _impl(db)

def backtest_recent(n=20):
    from resolution import backtest_recent as _impl
    return _impl(n)

def resolve_hypothesis_immediately(slug, current_price, entry_price):
    from resolution import resolve_hypothesis_immediately as _impl
    return _impl(slug, current_price, entry_price)

def repair_positions_file():
    """Fix inconsistent data in positions.json (e.g. high_price < entry_price)."""
    positions = positions_db.load_all()
    dirty = False
    for slug, p in positions.items():
        entry = p.get(POS_ENTRY_PRICE, 0)
        high = p.get(POS_HIGH_PRICE, 0)
        if entry > 0 and high < entry:
            p[POS_HIGH_PRICE] = entry
            logger.info(f"[REPAIR] {slug[:40]}... high_price {high:.4f} < entry_price {entry:.4f}, fixed")
            dirty = True
    if dirty:
        positions_db.save_all(positions)

def resolve_hypotheses():
    from resolution import resolve_hypotheses as _impl
    return _impl()


def _load_price_tracking():
    tracking = load_json(PRICE_TRACKING_FILE, {})
    now = datetime.now()
    stale = [k for k, v in tracking.items()
              if isinstance(v, dict) and v.get(TRACKING_LAST_CHECK)
             and (now - datetime.fromisoformat(v[TRACKING_LAST_CHECK])).total_seconds() > 86400]
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
        last_price = entry.get(TRACKING_LAST_PRICE, 0)
        cached_p_model = entry.get(TRACKING_P_MODEL)
        if not isinstance(cached_p_model, (int, float)):
            return True, None
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
        TRACKING_LAST_PRICE: current_price,
        TRACKING_P_MODEL: p_model,
        TRACKING_LAST_CHECK: datetime.now().isoformat(),
    }
    _save_price_tracking(tracking)


def _check_news_cache_freshness(cluster_key, slug=None):
    from news_handler import _check_news_cache_freshness as _impl
    return _impl(cluster_key, slug)


def execute_trade(market, estimated_size, factors, analysis, balance):
    from trade_executor import execute_trade as _impl
    return _impl(market, estimated_size, factors, analysis, balance)


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
            with contextlib.suppress(Exception):
                shutil.copy("/root/dotm-sniper/current_status.json", dest)
    except Exception as e:
        logger.warning(f"[status_save] {type(e).__name__}: {e}")


def main():
    _main_inner()

def _main_inner():
    print("="*60)
    print("  DOTM SNIPER v5.3.0 - Batch Processing + Delta Scan + Advisor")
    print("="*60)

    repair_positions_file()

    settings = get_settings()
    last_backtest = settings.get(SETTINGS_LAST_BACKTEST, 0)
    import time as _time
    now_ts = _time.time()
    if now_ts - last_backtest >= BACKTEST_COOLDOWN_SECONDS:
        bt = backtest_recent(n=20)
        if "error" not in bt:
            print(f"🧪 Backtest: winrate={bt['winrate']:.1%}, brier={bt['avg_brier']:.3f}, pnl={bt['avg_pnl']:.2f}")
            if bt.get("recommendations"):
                for r in bt["recommendations"]:
                    print(f"   ⚠️  {r['issue']}: {r['suggestion']}")
        settings[SETTINGS_LAST_BACKTEST] = now_ts
        save_settings(settings)
    else:
        hours_ago = (now_ts - last_backtest) / 3600
        logger.info(f"[BACKTEST-COOLDOWN] Skipping, last run {hours_ago:.1f}h ago (cooldown={BACKTEST_COOLDOWN_SECONDS/3600:.0f}h)")
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
    max_positions = settings.get(SETTINGS_MAX_CONCURRENT, tier["max_positions"])
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
    existing_slugs = {h[HYP_SLUG] for h in db.get(HYP_DB_HYPOTHESES, []) if not h.get(HYP_RESOLVED)}
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
    available_balance = balance

    current_positions_for_clusters = [
        {HYP_CLUSTERS: h.get(HYP_CLUSTERS, []), HYP_SIZE_PCT: h.get(HYP_SIZE_PCT, 0)}
        for h in db.get(HYP_DB_HYPOTHESES, []) if not h.get(HYP_RESOLVED)
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
            min_p_model = get_settings().get(SETTINGS_MIN_P_MODEL, MIN_P_MODEL)
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
            for m, analysis in zip(batch, batch_results, strict=False):
                _update_price_tracking(m["slug"], m["price"], analysis["p_model"])
                market_analyses[m["slug"]] = (m, analysis)

    for m in markets:
        if len(portfolio) + candidates_bought >= max_positions:
            break

        if available_balance < 5:
            break

        can_pass, _reason = check_cluster_limits(m["clusters"], current_positions_for_clusters, portfolio_value=total_balance)
        if not can_pass:
            continue

        if m["slug"] in already_active:
            continue

        if m["slug"] in market_analyses:
            _, analysis = market_analyses[m["slug"]]
        else:
            should_analyze, cached_p = _check_price_delta(m["slug"], m["price"])
            if not should_analyze and cached_p is not None:
                min_p_model = get_settings().get(SETTINGS_MIN_P_MODEL, MIN_P_MODEL)
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
            positions_db.load_all(),
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
        cache_fresh = _check_news_cache_freshness(cluster_key, slug=m.get("slug"))
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
                HYP_CLUSTERS: [cluster],
                HYP_SIZE_PCT: estimated_size / total_balance
            })

    print(f"\n✅ Bought: {candidates_bought} | Available: ${available_balance:.2f}")

    update_daily_stats(balance_data, portfolio, candidates_bought)

    with contextlib.suppress(Exception):
        log_equity_snapshot()

    db = load_hypothesis_db()
    resolved = db.get(HYP_DB_RESOLVED, [])
    if len(resolved) >= BURN_IN_TRADES:
        recent = resolved[-BURN_IN_TRADES:]
        wins = sum(1 for h in recent if h.get(HYP_OUTCOME) == "YES")
        logger.info(f"Cycle complete: bought={candidates_bought}, recent_winrate={wins/len(recent):.1%}")

    try:
        from health_monitor import run_health_check
        run_health_check()
    except Exception as e:
        logger.debug(f"[HEALTH] Check failed: {e}")


# Re-export from extracted modules (at bottom to avoid circular imports)
from order_manager import (get_balance, get_portfolio)  # noqa: E402
from position_manager import (get_tier_params, position_size, check_cluster_limits,  # noqa: E402
                               check_category_limits,
                               CLUSTER_KEYWORDS)
from sell_executor import (trailing_stop_check, TRAILING_STOP,  # noqa: E402, F401
                           ATR_STOP_MULTIPLIER, ATR_TRAILING_MULTIPLIER,
                           TRAILING_ACTIVATION, CONVERGENCE_TAKE_PROFIT)
from signal_pipeline import (fetch_markets,  # noqa: E402, F401
                             fetch_gamma_dotm_candidates, pre_filter_before_batching,
                             full_market_analysis, batch_analyze_markets,
                             advisor_pre_check, BATCH_SIZE,
                             normalize_probability, calibrate_prediction)

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
            while not _shutdown_requested:
                try:
                    _main_inner()
                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()
                if _shutdown_requested:
                    break
                print("Sleeping 30 min...")
                for _ in range(180):
                    if _shutdown_requested:
                        break
                    time.sleep(10)
            logger.info("[MAIN] Graceful shutdown complete")
    finally:
        cleanup_pid_file(PID_FILE)
