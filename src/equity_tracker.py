#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json, check_and_write_pid, cleanup_pid_file
from config import EQUITY_CURVE_FILE, TRADES_JOURNAL_FILE, EQUITY_TRACKER_LOG as LOG_FILE
from schema import (
    EQUITY_CASH, EQUITY_NUM_POSITIONS, EQUITY_POSITIONS, EQUITY_POSITIONS_VALUE,
    EQUITY_SNAPSHOTS, EQUITY_TIMESTAMP, EQUITY_TOTAL, EQUITY_UNREALIZED_PNL,
)

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EQUITY_FILE = EQUITY_CURVE_FILE

from utils import load_env_file
load_env_file()


from order_manager import get_balance, get_portfolio


def log_equity_snapshot() -> dict | None:
    balance = get_balance()
    if not balance:
        logger.error("[EQUITY] Failed to fetch balance")
        return None

    portfolio = get_portfolio()

    cash = float(balance.get("cash", 0))
    positions_value = sum(float(p.get("current_value", 0)) for p in portfolio)
    total_equity = (balance.get("total_value") or (cash + positions_value)) if isinstance(balance, dict) else cash + positions_value
    unrealized_pnl = sum(float(p.get("unrealized_pnl", 0)) for p in portfolio)

    snapshot = {
        EQUITY_TIMESTAMP: datetime.now().isoformat(),
        EQUITY_CASH: round(cash, 2),
        EQUITY_POSITIONS_VALUE: round(positions_value, 2),
        EQUITY_TOTAL: round(total_equity, 2),
        EQUITY_UNREALIZED_PNL: round(unrealized_pnl, 2),
        EQUITY_NUM_POSITIONS: len(portfolio),
        EQUITY_POSITIONS: [
            {
                "slug": p.get("market_slug", ""),
                "question": p.get("market_question", "")[:50],
                "entry": float(p.get("avg_entry_price", 0)),
                "current": float(p.get("live_price", 0)),
                "value": round(float(p.get("current_value", 0)), 2),
                "pnl_pct": round(float(p.get("percent_pnl", 0)), 1),
            }
            for p in portfolio
        ]
    }

    curve = load_json(EQUITY_FILE, {EQUITY_SNAPSHOTS: []})
    if not isinstance(curve, dict):
        curve = {EQUITY_SNAPSHOTS: []}
    curve[EQUITY_SNAPSHOTS].append(snapshot)

    if len(curve[EQUITY_SNAPSHOTS]) > 1440:
        curve[EQUITY_SNAPSHOTS] = curve[EQUITY_SNAPSHOTS][-1440:]

    save_json(EQUITY_FILE, curve)

    logger.info(
        f"[EQUITY] equity=${total_equity:.2f} cash=${cash:.2f} "
        f"pos=${positions_value:.2f} unrealized={unrealized_pnl:+.2f} "
        f"positions={len(portfolio)}"
    )
    return snapshot


def log_trade(event_type: str, slug: str, question: str,
              entry_price: float = 0, exit_price: float = 0,
              shares: float = 0, invested: float = 0,
              pnl_pct: float = 0, pnl_abs: float = 0,
              reason: str = "", extra: dict | None = None) -> None:
    journal = load_json(TRADES_JOURNAL_FILE, {"trades": []})
    if not isinstance(journal, dict):
        journal = {"trades": []}

    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "slug": slug,
        "question": question[:80],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": round(shares, 4),
        "invested": round(invested, 2),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_abs": round(pnl_abs, 2),
        "reason": reason,
    }
    if extra:
        entry.update(extra)

    journal["trades"].append(entry)
    save_json(TRADES_JOURNAL_FILE, journal)
    logger.info(f"[JOURNAL] {event_type}: {slug[:40]} pnl={pnl_pct:+.1f}% reason={reason}")


def get_daily_summary() -> dict[str, Any]:
    curve = load_json(EQUITY_FILE, {EQUITY_SNAPSHOTS: []})
    if not isinstance(curve, dict):
        curve = {EQUITY_SNAPSHOTS: []}
    snapshots = curve.get(EQUITY_SNAPSHOTS, [])
    if not snapshots:
        return {}

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    today_snaps = []
    yesterday_snaps = []
    for s in snapshots:
        try:
            ts = datetime.fromisoformat(s[EQUITY_TIMESTAMP])
            if ts >= today_start:
                today_snaps.append(s)
            elif ts >= yesterday_start:
                yesterday_snaps.append(s)
        except (ValueError, KeyError):
            continue

    if not today_snaps:
        return {}

    latest = today_snaps[-1]
    first_today = today_snaps[0] if today_snaps else latest
    last_yesterday = yesterday_snaps[-1] if yesterday_snaps else None

    equity_now = latest[EQUITY_TOTAL]
    equity_start = first_today[EQUITY_TOTAL]
    daily_change = equity_now - equity_start
    daily_change_pct = (daily_change / equity_start * 100) if equity_start else 0

    if last_yesterday:
        overnight_change = equity_now - last_yesterday[EQUITY_TOTAL]
    else:
        overnight_change = 0

    journal = load_json(TRADES_JOURNAL_FILE, {"trades": []})
    if not isinstance(journal, dict):
        journal = {"trades": []}
    today_trades = []
    for t in journal.get("trades", []):
        try:
            ts = datetime.fromisoformat(t["timestamp"])
            if ts >= today_start:
                today_trades.append(t)
        except (ValueError, KeyError):
            continue

    buys = [t for t in today_trades if t.get("event") == "BUY"]
    sells = [t for t in today_trades if t.get("event") in ("SELL", "STOP_LOSS", "TAKE_PROFIT", "CONVERGENCE")]
    wins = [t for t in sells if t.get("pnl_pct", 0) > 0]
    losses = [t for t in sells if t.get("pnl_pct", 0) < 0]

    equity_history = [(s[EQUITY_TIMESTAMP], s[EQUITY_TOTAL]) for s in today_snaps]

    return {
        "equity_now": equity_now,
        "equity_start": equity_start,
        "daily_change": round(daily_change, 2),
        "daily_change_pct": round(daily_change_pct, 1),
        "overnight_change": round(overnight_change, 2),
        "cash": latest[EQUITY_CASH],
        "positions_value": latest[EQUITY_POSITIONS_VALUE],
        "num_positions": latest[EQUITY_NUM_POSITIONS],
        "positions": latest.get(EQUITY_POSITIONS, []),
        "buys_today": len(buys),
        "sells_today": len(sells),
        "wins_today": len(wins),
        "losses_today": len(losses),
        "trades_today": today_trades,
        "equity_history": equity_history,
    }


def main() -> None:
    import html

    snapshot = log_equity_snapshot()
    if not snapshot:
        return

    force = "--force" in sys.argv
    now = datetime.now()
    hour = now.hour

    if not force and hour != 21:
        logger.info(f"[DAILY] Not report hour ({hour}), skipping")
        return

    summary = get_daily_summary()
    if not summary:
        logger.info("[DAILY] No summary data")
        return

    msg = "📊 <b>Daily Summary</b>\n"
    msg += f"🕐 {now.strftime('%Y-%m-%d %H:%M')}\n\n"
    msg += f"💰 <b>Equity: ${summary['equity_now']:.2f}</b>\n"
    msg += f"   Daily: {'+' if summary['daily_change'] >= 0 else ''}"
    msg += f"${summary['daily_change']:.2f} ({summary['daily_change_pct']:+.1f}%)\n"
    msg += f"   Cash: ${summary['cash']:.2f} | Positions: ${summary['positions_value']:.2f}\n\n"

    msg += f"📈 Trades: {summary['buys_today']} buys, {summary['sells_today']} sells "
    msg += f"(✅{summary['wins_today']} ❌{summary['losses_today']})\n\n"

    positions = summary.get("positions", [])
    if positions:
        msg += "📍 Positions:\n"
        for p in sorted(positions, key=lambda x: x.get("pnl_pct", 0), reverse=True):
            emoji = "🟢" if p["pnl_pct"] > 0 else "🔴" if p["pnl_pct"] < -10 else "🟡"
            msg += f"{emoji} {html.escape(p['question'])} {p['pnl_pct']:+.1f}%\n"

    equity_hist = summary.get("equity_history", [])
    if len(equity_hist) >= 2:
        min_eq = min(e[1] for e in equity_hist)
        max_eq = max(e[1] for e in equity_hist)
        msg += f"\n📉 Range: ${min_eq:.2f} - ${max_eq:.2f}"

    from tg_sender import send_telegram
    send_telegram(msg)
    logger.info("[DAILY] Telegram summary send attempted")


if __name__ == "__main__":
    EQUITY_PID_FILE = "/tmp/equity_tracker.pid"
    if not check_and_write_pid(EQUITY_PID_FILE):
        print("Another equity_tracker instance running, exiting")
        sys.exit(1)
    import atexit
    atexit.register(lambda: cleanup_pid_file(EQUITY_PID_FILE))
    main()
