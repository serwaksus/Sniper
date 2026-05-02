#!/usr/bin/env python3
import subprocess, json, time
from datetime import datetime

def get_balance():
    try:
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15)
        return json.loads(res.stdout).get("data", {})
    except:
        return {}

def get_portfolio():
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15)
        return json.loads(res.stdout).get("data", [])
    except:
        return []

def get_markets():
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "5"],
                            capture_output=True, text=True, timeout=20)
        data = json.loads(res.stdout)
        return sorted(data.get("data", []), key=lambda x: -float(x.get("volume", 0)))[:5]
    except:
        return []

def send_telegram(message):
    import requests
    token = "8593160940:AAFETgh0-SsnHcPAJh-_aSM8tXvJHh52lqo"
    chat_id = "730132245"
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                     json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                     timeout=10)
    except:
        pass

def load_history():
    import os
    HISTORY_FILE = "/root/dotm-sniper/trades_history.json"
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"trades": [], "summary": {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0}}

def main():
    now = datetime.now()
    hour_msk = (now.hour + 3) % 24
    if hour_msk < 9 or hour_msk > 22:
        print(f"Outside schedule (MSK={hour_msk}), skipping")
        return

    balance = get_balance()
    portfolio = get_portfolio()
    history = load_history()

    cash = balance.get("cash", 0)
    total_value = balance.get("total_value", 0)
    pnl = balance.get("pnl", 0)
    starting = balance.get("starting_balance", 500)
    summary = history.get("summary", {})

    msg = f"📊 <b>DOTM Sniper Report</b>\n"
    msg += f"🕐 {now.strftime('%H:%M')} МСК\n\n"
    msg += f"💰 Баланс: <b>${total_value:.2f}</b>\n"
    msg += f"   Наличные: ${cash:.2f}\n"
    msg += f"   P&L: {'+' if pnl >= 0 else ''}{pnl:.2f} ({pnl/starting*100:+.1f}%)\n\n"

    msg += f"📈 История: {summary.get('total_trades', 0)} сделок "
    msg += f"(✅{summary.get('wins', 0)} ❌{summary.get('losses', 0)}) "
    msg += f"P&L: ${summary.get('total_pnl', 0):.2f}\n\n"

    if portfolio:
        msg += f"📈 Открытые позиции ({len(portfolio)}):\n"
        for pos in sorted(portfolio, key=lambda x: x.get("percent_pnl", 0), reverse=True)[:5]:
            pnl_pct = pos.get("percent_pnl", 0)
            emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < -10 else "🟡"
            msg += f"{emoji} {pos.get('market_question', 'N/A')[:45]}...\n"
            msg += f"   {pnl_pct:+.1f}% (${pos.get('current_value', 0):.2f})\n"
    else:
        msg += "📈 Открытых позиций нет\n"

    if history.get("trades"):
        msg += "\n📜 Последние закрытые:\n"
        for trade in sorted(history["trades"], key=lambda x: x["closed_at"], reverse=True)[:3]:
            emoji = "✅" if trade["pnl"] > 0 else "❌"
            msg += f"{emoji} {trade['question'][:35]}... {trade['pnl_pct']:+.0f}%\n"

    top_markets = get_markets()
    if top_markets:
        msg += "\n🔥 Топ рынки:\n"
        for m in top_markets[:3]:
            vol = float(m.get("volume", 0))
            msg += f"   ${vol:,.0f} - {m.get('question', '')[:35]}...\n"

    print(msg)
    send_telegram(msg)

    status_file = "/root/dotm-sniper/current_status.json"
    try:
        with open(status_file, 'w') as f:
            json.dump({"balance": balance, "portfolio": portfolio, "updated_at": now.isoformat()}, f, indent=2, default=str)
        import shutil
        shutil.copy(status_file, "/root/.openclaw/workspace/dotm_status.json")
        shutil.copy(status_file, "/root/.openclaw/agents/market_analyst/dotm_status.json")
        shutil.copy(status_file, "/root/.openclaw/workspace/memory/portfolio-current.json")
    except:
        pass