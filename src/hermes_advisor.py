#!/usr/bin/env python3
"""
Hermes Advisor v5.4.0 - Async Position Risk Manager with Self-Improvement
Runs parallel to dotm_sniper.py, handles reconciliation and emergency exits.
Alert throttling: Telegram only on trigger_exit or status change.
Anti-Fossil Filter: news limited to last 30 days, max 5 results.
Self-improvement: tracks predictions, generates skills, adapts to outcomes.
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
import threading
from logging.handlers import RotatingFileHandler
from log_formatter import StructuredFormatter
import positions_db

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotm_report import TelegramReporter
from utils import load_env_file, check_and_write_pid, cleanup_pid_file, validate_env_vars
from config import (
    HERMES_PID_FILE, HERMES_LOG,
)
from schema import (
    POS_ENTRY_PRICE, POS_HIGH_PRICE, POS_IN_EMERGENCY_EXIT,
    POS_MARKET_QUESTION,
    POS_PARTIAL_FILLS, POS_PARTIAL_PROCEEDS,
    POS_SHARES, POS_SHARES_AT_TP_OPEN,
)

load_env_file()
validate_env_vars(["DEEPSEEK_API_KEY", "TG_BOT_TOKEN", "TG_CHAT_ID"])

from hermes_risk import (  # noqa: F401 — re-exported for backward compatibility
    _load_alert_state,
    _save_alert_state,
    _should_send_telegram,
    _update_and_check_status,
    fetch_news_for_market,
    _prune_stale_cache,
    evaluate_emergency_exit,
    _execute_emergency_exit,
    _log_emergency_exit,
    _merge_save_positions,
    NOTIFY_SEVERITIES,
    STATUS_HOLD_SECONDS,
    STATUS_HOLD_COUNT,
    NOTIFICATION_COOLDOWN_SECONDS,
    TP_LIMIT_PRICE,
    TP_LADDER_PRICES,
    MAX_EMERGENCY_RETRIES,
    DEEPSEEK_API_KEY,
    DEEPSEEK_URL,
    HEADERS,
    _last_alert_status,
    _last_notified_at,
    _status_hold_counts,
    _alert_state_lock,
)
from hermes_resolution import _check_resolved_markets, _resolve_predictions_loop  # noqa: F401

os.makedirs(os.path.dirname(HERMES_LOG), exist_ok=True)

class UnbufferedRotatingFileHandler(RotatingFileHandler):
    def emit(self, record: Any) -> None:
        super().emit(record)
        self.flush()

_handler_file = UnbufferedRotatingFileHandler(HERMES_LOG, maxBytes=10*1024*1024, backupCount=3)
_handler_stream = logging.StreamHandler()
_formatter: logging.Formatter = StructuredFormatter(json_mode=True) if os.environ.get("LOG_FORMAT") == "json" else logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_handler_file.setFormatter(_formatter)
_handler_stream.setFormatter(_formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler_file, _handler_stream]
)
logger = logging.getLogger(__name__)

_shutdown_requested = False

def _handle_shutdown(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("[HERMES] Shutdown signal received, finishing current cycle...")

def _register_signal_handlers() -> None:
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_shutdown)
        signal.signal(signal.SIGINT, _handle_shutdown)

_register_signal_handlers()

RECONCILE_INTERVAL_SECONDS = 900
NEWS_CHECK_INTERVAL_SECONDS = 600
TELEGRAM_REPORTER = TelegramReporter()

def get_settings() -> dict:
    from db import load_settings
    return load_settings() or {}

from order_manager import get_portfolio

def get_open_orders() -> list[dict]:
    try:
        res = subprocess.run(["pm-trader", "orders", "list"],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        data = json.loads(res.stdout) if res.stdout else {}
        orders = data.get("data", []) if isinstance(data.get("data"), list) else []
        result = []
        for o in orders:
            if o.get("status") != "pending":
                continue
            result.append({
                "id": o.get("id"),
                "slug": o.get("market_slug", ""),
                "outcome": o.get("outcome", "yes"),
                "side": o.get("side", ""),
                "price": float(o.get("limit_price", 0)),
                "shares": float(o.get("amount", 0)),
            })
        return result
    except Exception as e:
        logger.error(f"[HERMES] Failed to get open orders: {e}")
        return []

def cancel_order(slug: str, outcome: str = "yes") -> bool:
    try:
        orders = get_open_orders()
        matching = [o for o in orders if o.get("slug") == slug and o.get("side") == "sell"]
        if not matching:
            return False
        order_id = matching[0].get("id")
        if not order_id:
            return False
        res = subprocess.run(["pm-trader", "orders", "cancel", str(order_id)],
                           capture_output=True, text=True, timeout=20, start_new_session=True)
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            logger.info(f"[HERMES] Canceled order {order_id} for {slug[:40]}...")
            return True
        logger.warning(f"[HERMES] Cancel failed for {slug}: {result}")
        return False
    except Exception as e:
        logger.error(f"[HERMES] Cancel exception for {slug}: {e}")
        return False

def market_sell(slug: str, outcome: str = "yes", shares: float | None = None) -> bool:
    try:
        if shares is None:
            portfolio = get_portfolio()
            if portfolio is None:
                logger.error(f"[HERMES] Cannot get portfolio for market_sell {slug}, using last known shares")
                return False
            pos = next((p for p in portfolio if p.get("market_slug") == slug), None)
            if pos:
                shares = pos.get("shares", 0)

        if not shares or shares <= 0:
            logger.warning(f"[HERMES] No shares to sell for {slug}")
            return False

        res = subprocess.run(["pm-trader", "sell", slug, outcome, str(shares)],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[HERMES] Market sell failed for {slug}: rc={res.returncode}")
            return False
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            logger.info(f"[HERMES] Market sell executed for {slug}: {shares} shares")
            return True
        logger.warning(f"[HERMES] Market sell failed for {slug}: {result}")
        return False
    except Exception as e:
        logger.error(f"[HERMES] Market sell exception for {slug}: {e}")
        return False

def reconcile_positions() -> None:
    logger.info("[HERMES] Starting position reconciliation...")

    portfolio = get_portfolio()
    if portfolio is None:
        logger.error("[HERMES] Portfolio API failed, skipping reconciliation to avoid data loss")
        return
    open_orders = get_open_orders()

    positions = positions_db.load_all()
    if not positions:
        logger.info("[HERMES] No positions to reconcile")
        return

    if len(portfolio) == 0 and len(positions) > 0:
        logger.warning(f"[HERMES] Empty portfolio but {len(positions)} tracked positions — API may be down, skipping reconciliation")
        return

    portfolio_slugs = {p.get("market_slug", "") for p in portfolio if p.get("market_slug")}

    deleted_slugs = set()
    updated_positions = {}

    for slug, pos_data in list(positions.items()):
        pos_modified = False

        if slug not in portfolio_slugs:
            if pos_data.get(POS_IN_EMERGENCY_EXIT):
                if not any(o.get("slug") == slug for o in open_orders):
                    logger.info(f"[HERMES] Emergency-exit position {slug[:40]}... no open orders, checking portfolio again")
                    fresh_portfolio = get_portfolio()
                    if fresh_portfolio is not None and slug not in {p["market_slug"] for p in fresh_portfolio}:
                        deleted_slugs.add(slug)
                        _notify_position_closed(slug, pos_data)
                continue

            logger.info(f"[HERMES] Position {slug[:40]}... not in portfolio, checking if fully closed")

            tp_order = next((o for o in open_orders if o.get("slug") == slug and o.get("side") == "sell" and any(abs(o.get("price", 0) - p) < 0.01 for p in TP_LADDER_PRICES)), None)

            if not tp_order:
                logger.info(f"[HERMES] No open TP for {slug[:40]}..., marking as closed")
                deleted_slugs.add(slug)
                _notify_position_closed(slug, pos_data)
            continue

        pos = next((p for p in portfolio if p.get("market_slug") == slug), None)
        if not pos:
            continue

        current_shares = pos.get(POS_SHARES, 0)
        recorded_shares = pos_data.get(POS_SHARES, 0)

        if current_shares != recorded_shares:
            logger.info(f"[HERMES] Share mismatch for {slug[:40]}...: recorded={recorded_shares}, actual={current_shares}")
            pos_data[POS_SHARES] = current_shares
            pos_modified = True

        entry_price = pos.get("avg_entry_price", 0)
        if entry_price > 0 and pos_data.get(POS_ENTRY_PRICE, 0) != entry_price:
            pos_data[POS_ENTRY_PRICE] = entry_price
            pos_modified = True

        tp_order = next((o for o in open_orders if o.get("slug") == slug and o.get("side") == "sell" and any(abs(o.get("price", 0) - p) < 0.01 for p in TP_LADDER_PRICES)), None)

        if tp_order:
            order_shares = tp_order.get("shares", 0)
            recorded_shares_at_tp = pos_data.get(POS_SHARES_AT_TP_OPEN, order_shares)
            current_shares_at_check = current_shares

            if current_shares_at_check < recorded_shares_at_tp and order_shares > 0:
                filled = recorded_shares_at_tp - current_shares_at_check
                total = order_shares

                if filled > 0 and filled < total:
                    logger.warning(f"[HERMES] PARTIAL FILL for {slug[:40]}...: {filled}/{total} (inferred from shares)")

                    fill_price = tp_order.get("price", TP_LIMIT_PRICE)
                    sold_value = filled * fill_price
                    pos_data[POS_SHARES] = current_shares
                    pos_data[POS_PARTIAL_FILLS] = pos_data.get(POS_PARTIAL_FILLS, 0) + filled
                    pos_data[POS_PARTIAL_PROCEEDS] = pos_data.get(POS_PARTIAL_PROCEEDS, 0) + sold_value

                    logger.info(f"[HERMES] Updated shares to {current_shares} (portfolio reflects partial fill), proceeds ${sold_value:.2f}")
                    pos_modified = True

                    _notify_partial_fill(slug, pos_data, filled, fill_price)

        if pos_modified:
            updated_positions[slug] = pos_data

    if deleted_slugs or updated_positions:
        _merge_save_positions(deleted_slugs=deleted_slugs, updated_positions=updated_positions)
        logger.info(f"[HERMES] Positions updated: {len(deleted_slugs)} deleted, {len(updated_positions)} updated")


def _notify_position_closed(slug: str, pos_data: dict) -> None:
    try:
        entry = pos_data.get(POS_ENTRY_PRICE, 0)
        high = pos_data.get(POS_HIGH_PRICE, entry)
        if entry > 0 and high > 0:
            computed_pnl_pct = (high - entry) / entry * 100
            computed_pnl_abs = (high - entry) * pos_data.get(POS_SHARES, 0)
        else:
            computed_pnl_pct = 0
            computed_pnl_abs = 0
        if TELEGRAM_REPORTER:
            TELEGRAM_REPORTER.alert_convergence(
                market_slug=slug,
                question=pos_data.get(POS_MARKET_QUESTION, "Unknown"),
                pnl_pct=computed_pnl_pct,
                pnl_abs=computed_pnl_abs,
                convergence_ratio=0
            )
    except Exception as e:
        logger.warning(f"[HERMES] Position closed notification failed: {e}")

def _notify_partial_fill(slug: str, pos_data: dict, filled: float, fill_price: float | None = None) -> None:
    try:
        if TELEGRAM_REPORTER:
            fp = fill_price or TP_LIMIT_PRICE
            TELEGRAM_REPORTER.alert_take_profit(
                market_slug=slug,
                question=pos_data.get("question", "Unknown"),
                pnl_pct=((fp - pos_data.get(POS_ENTRY_PRICE, 0)) / pos_data.get(POS_ENTRY_PRICE, 1)) * 100 if pos_data.get(POS_ENTRY_PRICE, 0) > 0 else 0,
                pnl_abs=filled * (fp - pos_data.get(POS_ENTRY_PRICE, 0))
            )
    except Exception as e:
        logger.warning(f"[HERMES] Partial fill notification failed: {e}")

def run_reconciliation_loop() -> None:
    while True:
        try:
            reconcile_positions()
        except Exception as e:
            logger.error(f"[HERMES] Reconciliation loop error: {e}")

        time.sleep(RECONCILE_INTERVAL_SECONDS)

def run_emergency_evaluation_loop() -> None:
    while True:
        try:
            evaluate_emergency_exit()
        except Exception as e:
            logger.error(f"[HERMES] Emergency evaluation error: {e}")

        time.sleep(NEWS_CHECK_INTERVAL_SECONDS)


def main() -> None:
    logger.info("="*60)
    logger.info("  HERMES ADVISOR v5.4.0 - Starting (with self-improvement)")
    logger.info("="*60)

    if not check_and_write_pid(HERMES_PID_FILE):
        logger.error("[HERMES] Another instance is already running, exiting")
        return

    import atexit
    atexit.register(cleanup_pid_file, HERMES_PID_FILE)

    reconcile_thread = threading.Thread(target=run_reconciliation_loop, daemon=True)
    emergency_thread = threading.Thread(target=run_emergency_evaluation_loop, daemon=True)
    resolution_thread = threading.Thread(target=_resolve_predictions_loop, daemon=True)

    reconcile_thread.start()
    emergency_thread.start()
    resolution_thread.start()

    logger.info("[HERMES] All 3 loops started (reconcile, emergency, resolution+skills)")

    _restart_timestamps = []

    try:
        while not _shutdown_requested:
            time.sleep(60)

            restarted_any = False
            if not reconcile_thread.is_alive():
                logger.error("[HERMES] Reconciliation thread died, restarting")
                reconcile_thread = threading.Thread(target=run_reconciliation_loop, daemon=True)
                reconcile_thread.start()
                restarted_any = True
            if not emergency_thread.is_alive():
                logger.error("[HERMES] Emergency thread died, restarting")
                emergency_thread = threading.Thread(target=run_emergency_evaluation_loop, daemon=True)
                emergency_thread.start()
                restarted_any = True
            if not resolution_thread.is_alive():
                logger.error("[HERMES] Resolution thread died, restarting")
                resolution_thread = threading.Thread(target=_resolve_predictions_loop, daemon=True)
                resolution_thread.start()
                restarted_any = True

            if restarted_any:
                _restart_timestamps.append(time.time())
                cutoff = time.time() - 600
                _restart_timestamps = [t for t in _restart_timestamps if t > cutoff]
                if len(_restart_timestamps) > 5:
                    logger.critical(
                        f"[HERMES] Thread restart storm: {len(_restart_timestamps)} restarts in 10 minutes!"
                    )

            positions = positions_db.load_all()
            active_count = len([p for p in positions.values() if not p.get(POS_IN_EMERGENCY_EXIT)])

            logger.debug(f"[HERMES] Heartbeat: {active_count} active positions")

        logger.info("[HERMES] Graceful shutdown complete")
    except KeyboardInterrupt:
        logger.info("[HERMES] Shutting down...")
    except Exception as e:
        logger.error(f"[HERMES] Fatal error: {e}")

if __name__ == "__main__":
    main()
