#!/usr/bin/env python3
import subprocess
import json
import os
import sys
import logging
import html
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any

LOG_FILE = "/root/dotm-sniper/report.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from utils import load_env_file
load_env_file()


class TelegramReporter:
    def __init__(self):
        self.token = os.environ.get("TG_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TG_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning("Telegram reporter DISABLED: missing TG_BOT_TOKEN or TG_CHAT_ID")

    def _send(self, message: str) -> bool:
        if not self.enabled:
            logger.warning("Telegram send skipped: reporter not enabled")
            return False
        message = message[:4096]
        for attempt in range(3):
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True
                    },
                    timeout=20
                )
                if response.ok:
                    logger.info("Telegram message sent successfully")
                    return True
                else:
                    logger.error(f"Telegram send failed: {response.status_code} {response.text[:200]}")
                    return False
            except Exception as e:
                logger.warning(f"Telegram send attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    import time; time.sleep(2)
        logger.error("Telegram send failed after 3 attempts")
        return False

    def alert_new_position(self, market_slug: str, question: str, entry_price: float,
                          amount: float, metaculus_prob: Optional[float] = None,
                          factors: Optional[List[Dict]] = None, reasoning: Optional[str] = None):
        msg = f"🚨 <b>New Position</b>\n\n"
        msg += f"📌 {html.escape(str(question[:55]))}...\n\n"
        msg += f"💰 Entry: <b>${entry_price:.3f}</b>\n"
        msg += f"💵 Size: ${amount:.2f}\n"
        if metaculus_prob is not None:
            msg += f"📊 Hermes estimate: {metaculus_prob:.0%}\n"

        if factors:
            supporting = [f for f in factors if f.get('direction') == 'supports']
            if supporting:
                msg += f"\n📋 <b>Why this trade:</b>\n"
                for f in supporting[:3]:
                    source = f.get('source', 'analysis')
                    weight = f.get('weight', 'medium')
                    factor_text = f.get('factor', '')[:80]
                    emoji = "🔺" if weight == 'high' else "▫️"
                    msg += f"{emoji} {html.escape(str(factor_text))}\n"
                    if source and source != 'analysis':
                        msg += f"   └ {html.escape(str(source))}\n"

        if reasoning:
            msg += f"\n💡 <i>{html.escape(str(reasoning[:120]))}</i>\n"

        self._send(msg)

    def alert_take_profit(self, market_slug: str, question: str, pnl_pct: float, pnl_abs: float):
        msg = f"✅ <b>Take Profit</b>\n\n"
        msg += f"📌 {html.escape(str(question[:50]))}...\n\n"
        msg += f"📈 P&L: <b>+{pnl_pct:.1f}%</b> (${pnl_abs:.2f})\n"
        self._send(msg)

    def alert_stop_loss(self, market_slug: str, question: str, pnl_pct: float, pnl_abs: float):
        msg = f"❌ <b>Stop Loss</b>\n\n"
        msg += f"📌 {html.escape(str(question[:50]))}...\n\n"
        msg += f"📉 P&L: <b>{pnl_pct:.1f}%</b> (${pnl_abs:.2f})\n"
        self._send(msg)

    def alert_convergence(self, market_slug: str, question: str, pnl_pct: float,
                          pnl_abs: float, convergence_ratio: float):
        msg = f"🎯 <b>Gap Convergence</b> (edge captured)\n\n"
        msg += f"📌 {html.escape(str(question[:50]))}...\n\n"
        msg += f"📈 P&L: <b>+{pnl_pct:.1f}%</b> (${pnl_abs:.2f})\n"
        msg += f"📊 Convergence: {convergence_ratio:.0%}\n"
        self._send(msg)

    def alert_news_blocked(self, market_slug: str, question: str, reason: str):
        msg = f"🚨 <b>Trade Blocked by News API</b>\n\n"
        msg += f"📌 {html.escape(str(question[:50]))}...\n\n"
        msg += f"⚠️ Reason: {html.escape(str(reason))}\n"
        self._send(msg)

    def send_daily_report(self, balance_data: Dict[str, Any], portfolio: List[Dict[str, Any]],
                         history_summary: Dict[str, Any], top_markets: List[Dict[str, Any]]):
        import pytz
        msk_tz = pytz.timezone("Europe/Moscow")
        now_msk = datetime.now(msk_tz)
        cash = balance_data.get("cash", 0) if balance_data else 0
        total_value = balance_data.get("total_value", 0) if balance_data else 0
        pnl = balance_data.get("pnl", 0) if balance_data else 0
        starting = balance_data.get("starting_balance", 500) if balance_data else 500

        msg = f"📊 <b>DOTM Sniper Report</b>\n"
        msg += f"🕐 {now_msk.strftime('%H:%M')} МСК\n\n"
        msg += f"💰 Balance: <b>${total_value:.2f}</b>\n"
        msg += f"   Cash: ${cash:.2f}\n"
        msg += f"   P&L: {'+' if pnl >= 0 else ''}{pnl:.2f} ({pnl/max(starting, 1)*100:+.1f}%)\n\n"

        summary = history_summary
        msg += f"📈 History: {summary.get('total_trades', 0)} trades "
        msg += f"(✅{summary.get('wins', 0)} ❌{summary.get('losses', 0)}) "
        msg += f"P&L: ${summary.get('total_pnl', 0):.2f}\n\n"

        if portfolio:
            msg += f"📈 Active Positions ({len(portfolio)}):\n"
            for pos in sorted(portfolio, key=lambda x: x.get("percent_pnl", 0), reverse=True):
                pnl_pct = pos.get("percent_pnl", 0)
                emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < -10 else "🟡"
                msg += f"{emoji} {html.escape(str(pos.get('market_question', 'N/A')[:40]))}...\n"
                msg += f"   {pnl_pct:+.1f}% (${pos.get('current_value', 0):.2f})\n"
        else:
            msg += "📈 No open positions\n"

        if top_markets:
            msg += "\n🔥 Top Markets:\n"
            for m in top_markets[:3]:
                vol = float(m.get("volume", 0))
                msg += f"   ${vol:,.0f} - {html.escape(str(m.get('question', '')[:35]))}...\n"

        return self._send(msg)


def get_balance():
    try:
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15)
        data = json.loads(res.stdout).get("data", {})
        logger.info(f"Balance fetched: ${data.get('total_value', 0):.2f}")
        return data
    except Exception as e:
        logger.error(f"Failed to get balance: {e}")
        return None


def get_portfolio():
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15)
        data = json.loads(res.stdout).get("data", [])
        data = [p for p in data if float(p.get("shares", 0)) > 0.001]
        logger.info(f"Portfolio fetched: {len(data)} positions (dust filtered)")
        return data
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        return []


def get_markets():
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "5"],
                            capture_output=True, text=True, timeout=20)
        data = json.loads(res.stdout)
        return sorted(data.get("data", []), key=lambda x: -float(x.get("volume", 0)))[:5]
    except Exception as e:
        logger.error(f"Failed to get markets: {e}")
        return []


def load_history():
    HISTORY_FILE = "/root/dotm-sniper/trades_history.json"
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"trades": [], "summary": {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0}}


def main():
    force = "--force" in sys.argv
    import pytz
    msk_tz = pytz.timezone("Europe/Moscow")
    now_msk = datetime.now(msk_tz)
    hour_msk = now_msk.hour

    logger.info(f"Report started (MSK={hour_msk}, force={force})")

    if not force and (hour_msk < 9 or hour_msk >= 22):
        logger.info(f"Outside schedule (MSK={hour_msk}), skipping (use --force to override)")
        return

    balance = get_balance()
    if not balance:
        logger.error("No balance data, aborting report")
        return

    portfolio = get_portfolio()
    history = load_history()
    top_markets = get_markets()

    reporter = TelegramReporter()
    sent = reporter.send_daily_report(balance, portfolio, history.get("summary", {}), top_markets)
    logger.info(f"Report Telegram send result: {sent}")

    status_file = "/root/dotm-sniper/current_status.json"
    try:
        with open(status_file, 'w') as f:
            json.dump({"balance": balance, "portfolio": portfolio, "updated_at": datetime.now().isoformat()}, f, indent=2, default=str)
        import shutil
        for dest in [
            "/root/.openclaw/workspace/dotm_status.json",
            "/root/.openclaw/agents/market_analyst/dotm_status.json",
            "/root/.openclaw/workspace/memory/portfolio-current.json",
        ]:
            try:
                shutil.copy(status_file, dest)
            except Exception as e:
                logger.warning(f"Could not copy status to {dest}: {e}")
    except Exception as e:
        logger.error(f"Status file error: {e}")

    logger.info("Report complete")


if __name__ == "__main__":
    main()