#!/usr/bin/env python3
import subprocess
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json

EQUITY_FILE = "/root/dotm-sniper/equity_curve.json"
TRADES_JOURNAL_FILE = "/root/dotm-sniper/trades_journal.json"

LOG_FILE = "/root/dotm-sniper/equity_tracker.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _load_env():
    env_path = "/root/dotm-sniper/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, val = line.partition('=')
                    os.environ.setdefault(key.strip(), val.strip())

_load_env()


def get_balance():
    try:
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True,
                           timeout=15, start_new_session=True)
        if res.returncode != 0:
            return None
        return json.loads(res.stdout).get("data", {})
    except Exception:
        return None


def get_portfolio():
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True,
                           timeout=15, start_new_session=True)
        if res.returncode != 0:
            return []
        data = json.loads(res.stdout).get("data", [])
        return [p for p in data if float(p.get("shares", 0)) > 0.001]
    except Exception:
        return []


def log_equity_snapshot():
    balance = get_balance()
    if not balance:
        logger.error("[EQUITY] Failed to fetch balance")
        return None

    portfolio = get_portfolio()

    cash = float(balance.get("cash", 0))
    positions_value = sum(float(p.get("current_value", 0)) for p in portfolio)
    total_equity = cash + positions_value
    unrealized_pnl = sum(float(p.get("unrealized_pnl", 0)) for p in portfolio)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_equity": round(total_equity, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "num_positions": len(portfolio),
        "positions": [
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

    curve = load_json(EQUITY_FILE, {"snapshots": []})
    if not isinstance(curve, dict):
        curve = {"snapshots": []}
    curve["snapshots"].append(snapshot)

    if len(curve["snapshots"]) > 2880:
        curve["snapshots"] = curve["snapshots"][-2880:]

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
              reason: str = "", extra: Optional[Dict] = None):
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


def get_daily_summary() -> Dict[str, Any]:
    curve = load_json(EQUITY_FILE, {"snapshots": []})
    if not isinstance(curve, dict):
        curve = {"snapshots": []}
    snapshots = curve.get("snapshots", [])
    if not snapshots:
        return {}

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    today_snaps = []
    yesterday_snaps = []
    for s in snapshots:
        try:
            ts = datetime.fromisoformat(s["timestamp"])
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

    equity_now = latest["total_equity"]
    equity_start = first_today["total_equity"]
    daily_change = equity_now - equity_start
    daily_change_pct = (daily_change / equity_start * 100) if equity_start else 0

    if last_yesterday:
        overnight_change = equity_now - last_yesterday["total_equity"]
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

    equity_history = [(s["timestamp"], s["total_equity"]) for s in today_snaps]

    return {
        "equity_now": equity_now,
        "equity_start": equity_start,
        "daily_change": round(daily_change, 2),
        "daily_change_pct": round(daily_change_pct, 1),
        "overnight_change": round(overnight_change, 2),
        "cash": latest["cash"],
        "positions_value": latest["positions_value"],
        "num_positions": latest["num_positions"],
        "positions": latest.get("positions", []),
        "buys_today": len(buys),
        "sells_today": len(sells),
        "wins_today": len(wins),
        "losses_today": len(losses),
        "trades_today": today_trades,
        "equity_history": equity_history,
    }


def main():
    import html
    import requests

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

    token = os.environ.get("TG_BOT_TOKEN", "")
    chat_id = os.environ.get("TG_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("[DAILY] Telegram not configured")
        return

    msg = f"📊 <b>Daily Summary</b>\n"
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

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=20
        )
        logger.info("[DAILY] Telegram summary sent")
    except Exception as e:
        logger.error(f"[DAILY] Telegram error: {e}")


if __name__ == "__main__":
    main()
