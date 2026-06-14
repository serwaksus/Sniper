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
from __future__ import annotations
from typing import Any
import subprocess
import json
import time
import os
import sys
import logging
import signal
from logging.handlers import RotatingFileHandler
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_formatter import StructuredFormatter
from dotm_report import TelegramReporter

from config import (
    PID_FILE, SNIPER_LOG as LOG_FILE, PRICE_TRACKING_FILE,
    DAILY_STATS_FILE, CURRENT_STATUS_FILE,
    MIN_P_MODEL, MIN_CONFIDENCE, BURN_IN_TRADES,
)

import positions_db
import hypotheses_db
from news_scanner import check_market_news
from utils import load_json, save_json, check_and_write_pid, cleanup_pid_file, load_env_file, validate_env_vars
from equity_tracker import log_equity_snapshot
from correlation_matrix import check_correlation_limit
from db import load_settings as _db_load_settings, save_settings as _db_save_settings
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
import contextlib
load_env_file()
validate_env_vars(["DEEPSEEK_API_KEY", "TG_BOT_TOKEN", "TG_CHAT_ID"])

_tr_instance = None

def _tr() -> Any:
    global _tr_instance
    if _tr_instance is None:
        with contextlib.suppress(Exception):
            _tr_instance = TelegramReporter()
    return _tr_instance

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

_handler_file = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3)
_handler_stream = logging.StreamHandler()
if os.environ.get("LOG_FORMAT") == "json":
    _formatter: logging.Formatter = StructuredFormatter(json_mode=True)
else:
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_handler_file.setFormatter(_formatter)
_handler_stream.setFormatter(_formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler_file, _handler_stream],
    force=True
)
logger = logging.getLogger(__name__)

_shutdown_requested = False

def _handle_shutdown(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("[MAIN] Shutdown signal received, finishing current cycle...")

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

MAX_POS_PCT = 0.10

PRICE_DELTA_THRESHOLD = 0.002

MIN_TRADES_FOR_WEIGHT_ADJUSTMENT = 20
BACKTEST_COOLDOWN_SECONDS = 24 * 3600


def update_daily_stats(balance: dict, portfolio: list[dict], trades_this_cycle: int) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    stats = load_json(DAILY_STATS_FILE, {"date": today, "trades": 0, "pnl": 0, "started": False})
    if stats.get("date") != today:
        stats = {"date": today, "trades": 0, "pnl": 0, "started": False}
    stats["started"] = True
    stats["trades"] = stats.get("trades", 0) + trades_this_cycle
    starting = get_settings().get(SETTINGS_STARTING_BALANCE, 500.0)
    stats["pnl"] = balance.get("total_value", 0) - starting
    save_json(DAILY_STATS_FILE, stats)

def parse_llm_json(response_text: str) -> dict | None:
    from utils import parse_llm_json as _parse
    return _parse(response_text)





def validate_settings(s: dict) -> dict:
    errors = []
    if s.get("min_p_model", 0) <= 0:
        errors.append("min_p_model must be > 0")
    if s.get("max_concurrent_trades", 1) <= 0:
        errors.append("max_concurrent_trades must be >= 1")
    if not (0 < s.get("signal_threshold", 55) <= 100):
        errors.append("signal_threshold must be in (0, 100]")
    if not (0 <= s.get("min_confidence", 0.65) <= 1):
        errors.append("min_confidence must be in [0, 1]")
    if s.get("position_size_pct", 0.03) <= 0:
        errors.append("position_size_pct must be > 0")
    if errors:
        logger.error(f"[SETTINGS-INVALID] {errors}")
        raise ValueError(f"Invalid settings: {errors}")
    return s


def _default_settings() -> dict:
    return {
        SETTINGS_MIN_CONFIDENCE: MIN_CONFIDENCE,
        SETTINGS_POSITION_SIZE_PCT: MAX_POS_PCT,
        SETTINGS_CALIBRATION_BRIER: None,
        SETTINGS_TOTAL_RESOLVED: 0,
        SETTINGS_SIGNAL_THRESHOLD: 55,
        SETTINGS_MIN_P_MODEL: MIN_P_MODEL,
    }

def get_settings() -> dict:
    s = _db_load_settings()
    if not s:
        s = _default_settings()
        _db_save_settings(s)
    return s

def save_settings(s: dict) -> None:
    s[SETTINGS_VERSION] = s.get(SETTINGS_VERSION, 0) + 1
    _db_save_settings(s)

def load_hypothesis_db() -> dict:
    return hypotheses_db.load_all()

def save_hypothesis_db(db: dict) -> None:
    hypotheses_db.save_all(db)

def detect_clusters(question: str) -> list[str]:
    from position_manager import detect_clusters as _detect
    return _detect(question)

def calculate_brier_score(db: dict) -> float | None:
    from resolution import calculate_brier_score as _impl
    return _impl(db)

def learn_from_results(db: dict) -> dict:
    from resolution import learn_from_results as _impl
    return _impl(db)

def backtest_recent(n: int = 20) -> dict:
    from resolution import backtest_recent as _impl
    return _impl(n)

def resolve_hypothesis_immediately(slug: str, current_price: float, entry_price: float) -> None:
    from resolution import resolve_hypothesis_immediately as _impl
    return _impl(slug, current_price, entry_price)

def repair_positions_file() -> None:
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

def resolve_hypotheses() -> None:
    from resolution import resolve_hypotheses as _impl
    return _impl()


def _load_price_tracking() -> dict:
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


def _save_price_tracking(tracking: dict) -> None:
    save_json(PRICE_TRACKING_FILE, tracking)


def _check_price_delta(slug: str, current_price: float) -> tuple[bool, float | None]:
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


def _update_price_tracking(slug: str, current_price: float, p_model: float) -> None:
    tracking = _load_price_tracking()
    tracking[slug] = {
        TRACKING_LAST_PRICE: current_price,
        TRACKING_P_MODEL: p_model,
        TRACKING_LAST_CHECK: datetime.now().isoformat(),
    }
    _save_price_tracking(tracking)


def _check_news_cache_freshness(cluster_key: str, slug: str | None = None) -> bool:
    from news_handler import _check_news_cache_freshness as _impl
    return _impl(cluster_key, slug)


def execute_trade(market: dict, estimated_size: float, factors: list, analysis: dict, balance: float) -> bool:
    from trade_executor import execute_trade as _impl
    return _impl(market, estimated_size, factors, analysis, balance)


def _update_status_file() -> None:
    try:
        import shutil
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15, start_new_session=True)
        balance_data = json.loads(res.stdout).get("data", {})
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15, start_new_session=True)
        portfolio_data = json.loads(res.stdout).get("data", [])
        portfolio_data = [p for p in portfolio_data if float(p.get("shares", 0)) > 0.001]
        status = {"balance": balance_data, "portfolio": portfolio_data, "updated_at": datetime.now().isoformat()}
        save_json(CURRENT_STATUS_FILE, status)
        for dest in [
            "/root/.openclaw/workspace/dotm_status.json",
            "/root/.openclaw/agents/market_analyst/dotm_status.json",
            "/root/.openclaw/workspace/memory/portfolio-current.json",
        ]:
            with contextlib.suppress(Exception):
                shutil.copy(CURRENT_STATUS_FILE, dest)
    except Exception as e:
        logger.warning(f"[status_save] {type(e).__name__}: {e}")


def main() -> None:
    _main_inner()

def cleanup_stale_orders() -> None:
    try:
        res = subprocess.run(["pm-trader", "orders", "list"], capture_output=True, text=True, timeout=20, start_new_session=True)
        orders = json.loads(res.stdout).get("data", []) if res.stdout else []
        if not orders:
            return
        all_pos = positions_db.load_all()
        active_slugs = set(all_pos.keys())
        to_cancel = [o for o in orders if o.get("slug") not in active_slugs and o.get("market_slug") not in active_slugs]
        for o in to_cancel:
            order_id = o.get("id") or o.get("order_id")
            slug = o.get("slug") or o.get("market_slug", "?")
            if order_id:
                subprocess.run(["pm-trader", "orders", "cancel", str(order_id)], capture_output=True, text=True, timeout=10, start_new_session=True)
                logger.info(f"[STARTUP-CLEANUP] Cancelled stale order {order_id} for {slug[:40]}...")
        if to_cancel:
            logger.info(f"[STARTUP-CLEANUP] Cancelled {len(to_cancel)} stale orders")
    except Exception as e:
        logger.warning(f"[STARTUP-CLEANUP] {type(e).__name__}: {e}")


def _main_inner() -> None:
    print("="*60)
    print("  DOTM SNIPER v5.3.0 - Batch Processing + Delta Scan + Advisor")
    print("="*60)

    repair_positions_file()
    cleanup_stale_orders()

    with contextlib.suppress(Exception):
        from smart_money import init_smart_money
        init_smart_money()

    settings = get_settings()
    validate_settings(settings)
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
    if portfolio is None:
        portfolio = []
    print(f"📊 Open positions: {len(portfolio)}")

    _update_status_file()

    if len(portfolio) >= max_positions:
        print(f"⚠️ Max positions ({max_positions}) reached")
        return

    markets = fetch_markets()
    db = load_hypothesis_db()
    existing_slugs = {h[HYP_SLUG] for h in db.get(HYP_DB_HYPOTHESES, [])}
    position_slugs = {p.get("market_slug", "") for p in portfolio}
    already_active = existing_slugs | position_slugs
    gamma_candidates = fetch_gamma_dotm_candidates(existing_slugs | position_slugs)
    seen: set[str] = {m["slug"] for m in markets}
    for gc in gamma_candidates:
        if gc["slug"] not in seen:
            markets.append(gc)
            seen.add(gc["slug"])
    if not markets:
        print("No markets found")
        return

    print(f"📈 Candidates: {len(markets)} (pm-trader + {len(gamma_candidates)} gamma)")

    from market_graph import build_graph_if_stale
    build_graph_if_stale([{"slug": m.get("slug", ""), "question": m.get("question", ""),
                           "price": m.get("price", 0), "clusters": m.get("clusters", [])}
                          for m in markets])

    from cascade_detector import record_prices, detect_and_find
    record_prices(markets)
    cascade_opps = detect_and_find(markets)
    if cascade_opps:
        logger.info(f"[CASCADE] {len(cascade_opps)} laggard opportunities detected")

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
            cluster=m.get("clusters", ["other"])[0],
            bid_liquidity=m.get("bid_liquidity"),
        )

        if estimated_size <= 0:
            print("   ⏭️ Kelly edge negative, skipping")
            continue

        estimated_size = conviction_adjusted_size(
            estimated_size,
            analysis.get("signal_score", 0),
            analysis.get("min_signal", 55),
        )

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

        from market_graph import check_correlation as _graph_check_corr
        _corr = _graph_check_corr(m["slug"], list(position_slugs))
        if _corr.get("correlated"):
            logger.warning(f"[GRAPH] High correlation: {_corr['warnings']}")
            estimated_size = max(5, estimated_size // 2)

        if execute_trade(m, estimated_size, factors, analysis, total_balance):
            candidates_bought += 1
            available_balance -= estimated_size
            cluster = m.get("clusters", ["other"])[0]
            current_positions_for_clusters.append({
                HYP_CLUSTERS: [cluster],
                HYP_SIZE_PCT: estimated_size / total_balance
            })

        time.sleep(0.5)

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
from order_manager import (get_balance, get_portfolio)
from position_manager import (get_tier_params, position_size, check_cluster_limits,
                                check_category_limits, conviction_adjusted_size,
                                CLUSTER_KEYWORDS)  # noqa: F401
from sell_executor import (trailing_stop_check, TRAILING_STOP,  # noqa: F401
                           ATR_STOP_MULTIPLIER, ATR_TRAILING_MULTIPLIER,
                           TRAILING_ACTIVATION, CONVERGENCE_TAKE_PROFIT)
from signal_pipeline import (fetch_markets,  # noqa: F401
                             fetch_gamma_dotm_candidates, pre_filter_before_batching,
                             full_market_analysis, batch_analyze_markets,
                             advisor_pre_check, BATCH_SIZE,
                             normalize_probability, calibrate_prediction,
                             URL, HEADERS, MODEL_MAIN, ADVISOR_MODEL,
                             MAX_P_MODEL_RATIO, MIN_VOLUME,
                             _cluster_score_adjustment,
                             )

if __name__ == "__main__":
    single_run = len(sys.argv) > 1 and sys.argv[1] == "--once"

    if "--train-ml" in sys.argv:
        from ml_predictor import train_if_ready
        result = train_if_ready()
        if result:
            print(f"Model trained: {result}")
        else:
            print("Not enough data for training")
        sys.exit(0)

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
        with contextlib.suppress(Exception):
            from db import checkpoint_wal
            checkpoint_wal()
        cleanup_pid_file(PID_FILE)
