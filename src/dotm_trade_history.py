#!/usr/bin/env python3
import subprocess, json, os
from datetime import datetime

HISTORY_FILE = "/root/dotm-sniper/trades_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"trades": [], "summary": {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0}}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_trade_history():
    try:
        res = subprocess.run(["pm-trader", "history", "--limit", "50"],
                           capture_output=True, text=True, timeout=15)
        data = json.loads(res.stdout)
        return data.get("data", [])
    except:
        return []

def get_current_portfolio():
    try:
        res = subprocess.run(["pm-trader", "portfolio"],
                           capture_output=True, text=True, timeout=15)
        return json.loads(res.stdout).get("data", [])
    except:
        return []

def get_balance():
    try:
        res = subprocess.run(["pm-trader", "balance"],
                           capture_output=True, text=True, timeout=15)
        return json.loads(res.stdout).get("data", {})
    except:
        return {}

def check_closed_positions():
    history = load_history()
    current_portfolio = get_current_portfolio()
    current_slugs = {pos.get("market_slug") for pos in current_portfolio}

    trade_history = get_trade_history()
    closed_trades = []

    for trade in trade_history:
        slug = trade.get("market_slug")
        if slug and slug not in current_slugs:
            if not any(t.get("slug") == slug for t in history["trades"]):
                closed_trades.append({
                    "slug": slug,
                    "question": trade.get("market_question", ""),
                    "outcome": trade.get("outcome", ""),
                    "pnl": trade.get("unrealized_pnl", 0),
                    "pnl_pct": trade.get("percent_pnl", 0),
                    "closed_at": datetime.now().isoformat()
                })

    if closed_trades:
        for trade in closed_trades:
            history["trades"].append(trade)
            history["summary"]["total_trades"] += 1
            if trade["pnl"] > 0:
                history["summary"]["wins"] += 1
            else:
                history["summary"]["losses"] += 1
            history["summary"]["total_pnl"] += trade["pnl"]
        save_history(history)
        print(f"Logged {len(closed_trades)} closed trades")

    return history

def show_summary():
    history = load_history()
    summary = history["summary"]
    balance = get_balance()

    print("\n" + "="*50)
    print("📊 DOTM SNIPER - TRADING HISTORY")
    print("="*50)
    print(f"💰 Total P&L: ${summary['total_pnl']:.2f}")
    print(f"📈 Total Trades: {summary['total_trades']}")
    print(f"✅ Wins: {summary['wins']} | ❌ Losses: {summary['losses']}")
    if summary['total_trades'] > 0:
        win_rate = summary['wins'] / summary['total_trades'] * 100
        print(f"📊 Win Rate: {win_rate:.1f}%")

    print("\n--- Recent Closed Trades ---")
    for trade in sorted(history["trades"], key=lambda x: x["closed_at"], reverse=True)[:10]:
        emoji = "✅" if trade["pnl"] > 0 else "❌"
        print(f"{emoji} {trade['question'][:45]}...")
        print(f"   P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:+.1f}%)")

    return history

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        show_summary()
    else:
        check_closed_positions()