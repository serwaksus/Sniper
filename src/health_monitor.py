#!/usr/bin/env python3
"""
Health Monitor — Main orchestrator for 25 health checks.
Sends Telegram alert ONLY if issues found. Runs once per sniper cycle.
"""
from __future__ import annotations
import json
import os
import sys
import subprocess
import logging
import shutil
from datetime import datetime

import hypotheses_db
from config import (
    PROJECT_ROOT, PRICE_TRACKING_FILE,
    CALIBRATION_MODEL_FILE,
)
from schema import (
    EQUITY_CASH, EQUITY_NUM_POSITIONS, EQUITY_POSITIONS_VALUE,
    EQUITY_SNAPSHOTS, EQUITY_TOTAL, EQUITY_UNREALIZED_PNL,
    HEALTH_LAST_CYCLE_START,
)
from health_checks import (
    _load_state, _save_state, _should_alert, _mark_alerted,
    _read_recent_log, _send_telegram,
    _check_no_trades, _check_equity_drawdown, _check_order_health,
    _check_api_health, _check_cycle_timing, _check_error_spike,
    _check_llm_usage, _check_disk_space, _check_hypothesis_db,
    _check_winrate, _check_calibration_overfit, _check_cache,
    _check_telegram, _check_crash_frequency, _check_json_integrity,
    _check_cron_health, _check_llm_error_rate, _check_screen_sessions,
    _check_disk_inodes, _check_pm_trader_health, _check_api_keys,
    _check_memory, _check_log_size, _check_sqlite_integrity,
    _check_trade_activity,
    EQUITY_FILE,
)

logger = logging.getLogger(__name__)


def _summarize_trading_activity(lines: list[str]) -> str:
    signals = sum(1 for line in lines if "=> BUY" in line)
    blocked = sum(1 for line in lines if "TRADE-BLOCKED" in line)
    executed = sum(1 for line in lines if "Bought:" in line and "Bought: 0" not in line)
    diverge_overrides = sum(1 for line in lines if "diverge_" in line and "override" in line)
    return f"signals={signals} blocked={blocked} executed={executed} diverge_overrides={diverge_overrides}"


def _summarize_no_trades(lines: list[str]) -> str:
    return _summarize_trading_activity(lines)


def _summarize_equity(state: dict) -> str:
    try:
        with open(EQUITY_FILE) as _f:
            data = json.load(_f)
        snaps = data.get(EQUITY_SNAPSHOTS, [])
        if snaps:
            eq = snaps[-1].get(EQUITY_TOTAL, 0)
            cash = snaps[-1].get(EQUITY_CASH, 0)
            pos = snaps[-1].get(EQUITY_POSITIONS_VALUE, 0)
            pnl = snaps[-1].get(EQUITY_UNREALIZED_PNL, 0)
            n_pos = snaps[-1].get(EQUITY_NUM_POSITIONS, 0)
            return f"equity=${eq:.2f} cash=${cash:.2f} pos=${pos:.2f} pnl={pnl:+.2f} positions={n_pos}"
    except Exception as e:
        logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
        pass
    return "equity=data_unavailable"


def _summarize_orders() -> str:
    try:
        res = subprocess.run(
            ["pm-trader", "orders", "list"],
            capture_output=True, text=True, timeout=15, start_new_session=True
        )
        data = json.loads(res.stdout) if res.stdout else {}
        orders = data.get("data", []) if isinstance(data.get("data"), list) else []
        pending = sum(1 for o in orders if o.get("status") == "pending")
        return f"pending_orders={pending}"
    except Exception as e:
        logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
        return "orders=api_error"


def _summarize_cycle(state: dict) -> str:
    last_cycle = state.get(HEALTH_LAST_CYCLE_START, "never")
    avg_time = state.get("avg_cycle_time", 0)
    return f"last={last_cycle} avg_cycle={avg_time:.0f}s"


def _summarize_errors(state: dict) -> str:
    errs = state.get("errors_last_hour", 0)
    return f"errors_1h={errs}"


def _summarize_llm(state: dict) -> str:
    cost = state.get("llm_cost_today", 0)
    calls = state.get("llm_calls_today", 0)
    return f"calls={calls} cost=${cost:.2f}"


def _summarize_disk() -> str:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        pct = usage.used / usage.total
        return f"used={pct:.0%} free={usage.free // (1024**3)}GB"
    except Exception as e:
        logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
        return "disk=unknown"


def _summarize_hypotheses() -> str:
    try:
        db = hypotheses_db.load_all()
        hyps = db.get("hypotheses", [])
        open_h = sum(1 for h in hyps if not h.get("resolved"))
        resolved = sum(1 for h in hyps if h.get("resolved"))
        return f"open={open_h} resolved={resolved}"
    except Exception as e:
        logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
        return "hypotheses=data_error"


def _summarize_winrate() -> str:
    try:
        db = hypotheses_db.load_all()
        resolved = [h for h in db.get("hypotheses", []) if h.get("resolved")]
        wins = sum(1 for h in resolved if h.get("pnl_pct", 0) > 0)
        total = len(resolved)
        wr = wins / total if total else 0
        return f"resolved={total} wins={wins} winrate={wr:.0%}"
    except Exception as e:
        logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
        return "winrate=data_error"


def _summarize_calib() -> str:
    try:
        with open(CALIBRATION_MODEL_FILE) as _f:
            model = json.load(_f)
        clusters = []
        for c, d in model.items():
            y = d.get("y_thresholds_", [])
            x = d.get("X_thresholds_", [])
            if y and x:
                clusters.append(f"{c}:p>{x[-1]:.0%}->{max(y):.0%}")
        return f"clusters=[{', '.join(clusters)}]"
    except Exception as e:
        logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
        return "calib=no_model"


def _summarize_cache() -> str:
    try:
        with open(PRICE_TRACKING_FILE) as _f:
            tracking = json.load(_f)
        total = len(tracking)
        high = sum(1 for v in tracking.values() if v.get("p_model", 0) >= 0.85)
        return f"tracked={total} high_p={high}"
    except Exception as e:
        logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
        return "cache=error"


def run_health_check() -> list[tuple[str, str]] | None:
    state = _load_state()
    lines = _read_recent_log(hours=24)
    alerts = []

    checks = [
        lambda: _check_no_trades(lines, state),
        lambda: _check_equity_drawdown(state),
        lambda: _check_order_health(state),
        lambda: _check_api_health(lines, state),
        lambda: _check_cycle_timing(state),
        lambda: _check_error_spike(state),
        lambda: _check_llm_usage(state),
        lambda: _check_disk_space(state),
        lambda: _check_hypothesis_db(state),
        lambda: _check_winrate(state),
        lambda: _check_calibration_overfit(state),
        lambda: _check_cache(state),
        lambda: _check_telegram(state),
        lambda: _check_crash_frequency(state),
        lambda: _check_json_integrity(state),
        lambda: _check_cron_health(state),
        lambda: _check_llm_error_rate(state),
        lambda: _check_screen_sessions(state),
        lambda: _check_disk_inodes(state),
        lambda: _check_pm_trader_health(state),
        lambda: _check_api_keys(state),
        lambda: _check_memory(state),
        lambda: _check_log_size(state),
        lambda: _check_sqlite_integrity(state),
        lambda: _check_trade_activity(state),
    ]

    summaries = [
        lambda: _summarize_no_trades(lines),
        lambda: _summarize_equity(state),
        lambda: _summarize_orders(),
        lambda: f"api={sum(1 for line in lines if '[GAMMA]' in line)}_cycles",
        lambda: _summarize_cycle(state),
        lambda: _summarize_errors(state),
        lambda: _summarize_llm(state),
        lambda: _summarize_disk(),
        lambda: _summarize_hypotheses(),
        lambda: _summarize_winrate(),
        lambda: _summarize_calib(),
        lambda: _summarize_cache(),
        lambda: "telegram=patched_dns",
        lambda: "tracebacks=scanned_5_logs",
        lambda: "files=5_critical_json",
        lambda: "cron=3_jobs",
        lambda: "llm_errors=6h_window",
        lambda: "sessions=sniper+hermes",
        lambda: "inodes=df_check",
        lambda: "pm_trader=balance_10s_timeout",
        lambda: "api_keys=deepseek_ping",
        lambda: "memory=ps_aux_rss",
        lambda: "log_size=7_files_50mb",
        lambda: "sqlite=integrity_check",
        lambda: "trade_activity=cycle_counter",
    ]

    check_names = [
        "no_trades", "equity_drawdown", "order_health", "api_health",
        "cycle_timing", "error_spike", "llm_usage", "disk_space",
        "hypothesis_db", "winrate", "calib_overfit", "cache",
        "telegram", "crash_freq", "json_integrity", "cron_health",
        "llm_errors", "screen_sessions", "disk_inodes", "pm_trader",
        "api_keys", "memory", "log_size", "sqlite_integrity", "trade_activity",
    ]

    issue_count = 0

    for i, check in enumerate(checks):
        name = check_names[i] if i < len(check_names) else f"check_{i}"
        summary = ""
        try:
            summary = summaries[i]() if i < len(summaries) else ""
        except Exception as e:
            logger.debug(f"[health_monitor] {type(e).__name__}: {e}")
            summary = "summary_error"

        try:
            result = check()
        except Exception as e:
            logger.warning(f"[HEALTH-CHECK] {name}: CRASH ({summary}) - {e}")
            continue
        if result is None:
            logger.info(f"[HEALTH-CHECK] {name}: OK | {summary}")
            continue
        alert_key, message = result
        issue_count += 1
        logger.info(f"[HEALTH-CHECK] {name}: ISSUE [{alert_key}] | {summary}")
        if _should_alert(state, alert_key):
            alerts.append((alert_key, message))
            _mark_alerted(state, alert_key)

    _save_state(state)

    if issue_count == 0:
        logger.info("[HEALTH] All 25 checks passed")
        return None

    for alert_key, message in alerts:
        logger.info(f"[HEALTH] ALERT [{alert_key}]: {message[:100]}")
    logger.info(f"[HEALTH] {issue_count} issues found, {len(alerts)} new alerts")
    return alerts


def run_hourly_report() -> list[str]:
    """Run all 25 checks, send ONE aggregated Telegram message with all issues.
    Called by cron every hour. Ignores cooldown — always reports full status."""
    state = _load_state()
    lines = _read_recent_log(hours=24)

    checks = [
        lambda: _check_no_trades(lines, state),
        lambda: _check_equity_drawdown(state),
        lambda: _check_order_health(state),
        lambda: _check_api_health(lines, state),
        lambda: _check_cycle_timing(state),
        lambda: _check_error_spike(state),
        lambda: _check_llm_usage(state),
        lambda: _check_disk_space(state),
        lambda: _check_hypothesis_db(state),
        lambda: _check_winrate(state),
        lambda: _check_calibration_overfit(state),
        lambda: _check_cache(state),
        lambda: _check_telegram(state),
        lambda: _check_crash_frequency(state),
        lambda: _check_json_integrity(state),
        lambda: _check_cron_health(state),
        lambda: _check_llm_error_rate(state),
        lambda: _check_screen_sessions(state),
        lambda: _check_disk_inodes(state),
        lambda: _check_pm_trader_health(state),
        lambda: _check_api_keys(state),
        lambda: _check_memory(state),
        lambda: _check_log_size(state),
        lambda: _check_sqlite_integrity(state),
        lambda: _check_trade_activity(state),
    ]

    check_names = [
        "no_trades", "equity_drawdown", "order_health", "api_health",
        "cycle_timing", "error_spike", "llm_usage", "disk_space",
        "hypothesis_db", "winrate", "calib_overfit", "cache",
        "telegram", "crash_freq", "json_integrity", "cron_health",
        "llm_errors", "screen_sessions", "disk_inodes", "pm_trader",
        "api_keys", "memory", "log_size", "sqlite_integrity", "trade_activity",
    ]

    issues = []
    ok_count = 0

    for i, check in enumerate(checks):
        name = check_names[i] if i < len(check_names) else f"check_{i}"
        try:
            result = check()
        except Exception as e:
            logger.warning(f"[HOURLY] {name}: CRASH - {e}")
            issues.append(f"\u2757 {name}: check crashed")
            continue
        if result is None:
            ok_count += 1
            continue
        alert_key, message = result
        clean = message.replace("<b>", "").replace("</b>", "")
        first_line = clean.split("\n")[0]
        issues.append(first_line)
        logger.info(f"[HOURLY] {name}: ISSUE [{alert_key}]")

    ts = datetime.now().strftime("%m/%d %H:%M")
    if not issues:
        logger.info(f"[HOURLY] ({ts}): All 25 checks OK")
        return []

    msg = f"\U0001f52c DOTM Hourly ({ts}): {len(issues)} issues / {ok_count} OK\n\n"
    msg += "\n".join(f"\u2022 {i}" for i in issues)

    _send_telegram(msg)
    logger.info(f"[HOURLY] Sent {len(issues)} issues to Telegram")
    _save_state(state)
    return issues


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import load_env_file
    load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--hourly" in sys.argv:
        results_h = run_hourly_report()
        print(f"{len(results_h)} issues" if results_h else "All healthy")
    else:
        results_c = run_health_check()
        print(f"{len(results_c) or 0} issues" if results_c else "All healthy")
