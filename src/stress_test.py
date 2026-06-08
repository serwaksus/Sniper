#!/usr/bin/env python3
"""
Monte Carlo stress test and competitive edge monitor for DOTM Sniper.
Simulates worst-case scenarios, estimates risk of ruin, and tracks edge degradation.
"""
import os
import sys
import random
import logging
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import HYP_DB_RESOLVED

TRADES_JOURNAL = "/root/dotm-sniper/trades_journal.json"
EQUITY_FILE = "/root/dotm-sniper/equity_curve.json"
HYPOTHESIS_DB = "/root/dotm-sniper/hypothesis_db.json"
CALIBRATION_LOG = "/root/dotm-sniper/calibration_log.json"

logger = logging.getLogger(__name__)


def _get_live_stats() -> dict:
    import hypotheses_db
    db = hypotheses_db.load_all()
    if not isinstance(db, dict):
        db = {HYP_DB_RESOLVED: []}
    resolved = [r for r in db.get(HYP_DB_RESOLVED, []) if r.get("exit_type") == "manual"]
    if not resolved:
        return {"trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "avg_pnl": 0}

    wins = [r for r in resolved if r.get("pnl_at_exit", 0) > 0]
    losses = [r for r in resolved if r.get("pnl_at_exit", 0) < 0]
    win_rate = len(wins) / len(resolved) if resolved else 0
    avg_win = sum(r["pnl_at_exit"] for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r["pnl_at_exit"] for r in losses) / len(losses) if losses else 0
    avg_pnl = sum(r["pnl_at_exit"] for r in resolved) / len(resolved) if resolved else 0

    return {
        "trades": len(resolved),
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "avg_pnl": round(avg_pnl, 3),
        "wins": len(wins),
        "losses": len(losses),
    }


def monte_carlo_simulation(
    start_balance: float = 500,
    monthly_deposit: float = 500,
    months: int = 36,
    trades_per_month: int = 10,
    win_rate: float = 0.289,
    avg_win_pct: float = 2.5,
    avg_loss_pct: float = -0.30,
    position_size_pct: float = 0.02,
    fee_per_trade: float = 0.02,
    n_simulations: int = 10000,
    stop_if_ruined: bool = True,
) -> dict:
    ruin_threshold = 0.10 * start_balance
    results = {
        "final_balances": [],
        "max_drawdowns": [],
        "min_balances": [],
        "ruined": 0,
        "profitable": 0,
        "months_to_target": [],
    }

    for _ in range(n_simulations):
        balance = start_balance
        peak = balance
        max_dd = 0
        min_balance = balance
        ruined = False
        hit_target_month = None

        for month in range(1, months + 1):
            if stop_if_ruined and balance < ruin_threshold:
                ruined = True
                break

            balance += monthly_deposit

            tier_mult = 0.25
            if balance >= 50000:
                tier_mult = 0.40
            elif balance >= 10000:
                tier_mult = 0.35
            elif balance >= 2000:
                tier_mult = 0.30

            for _ in range(trades_per_month):
                if balance < 10:
                    ruined = True
                    break

                size = balance * position_size_pct * tier_mult * 4
                size = max(5, min(size, balance * 0.10))

                if random.random() < win_rate:
                    pnl = size * avg_win_pct
                else:
                    pnl = size * avg_loss_pct

                fee = size * fee_per_trade
                net_pnl = pnl - fee
                balance += net_pnl

            if ruined:
                break

            peak = max(peak, balance)
            dd = (peak - balance) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            min_balance = min(min_balance, balance)

            if hit_target_month is None and balance >= 71400:
                hit_target_month = month

        results["final_balances"].append(balance)
        results["max_drawdowns"].append(max_dd)
        results["min_balances"].append(min_balance)
        if ruined:
            results["ruined"] += 1
        if balance > start_balance + monthly_deposit * months:
            results["profitable"] += 1
        if hit_target_month:
            results["months_to_target"].append(hit_target_month)

    finals = sorted(results["final_balances"])
    draws = sorted(results["max_drawdowns"])

    return {
        "params": {
            "start_balance": start_balance,
            "monthly_deposit": monthly_deposit,
            "months": months,
            "trades_per_month": trades_per_month,
            "win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "position_size_pct": position_size_pct,
            "fee_per_trade_pct": fee_per_trade,
            "n_simulations": n_simulations,
        },
        "n_simulations": n_simulations,
        "p5_final": round(finals[int(0.05 * n_simulations)], 0),
        "p25_final": round(finals[int(0.25 * n_simulations)], 0),
        "p50_final": round(finals[int(0.50 * n_simulations)], 0),
        "p75_final": round(finals[int(0.75 * n_simulations)], 0),
        "p95_final": round(finals[int(0.95 * n_simulations)], 0),
        "avg_final": round(sum(finals) / len(finals), 0),
        "max_drawdown_p50": round(draws[int(0.50 * n_simulations)], 3),
        "max_drawdown_p95": round(draws[int(0.95 * n_simulations)], 3),
        "ruin_probability": round(results["ruined"] / n_simulations, 4),
        "profitable_pct": round(results["profitable"] / n_simulations, 3),
        "months_to_71k_p50": round(sorted(results["months_to_target"])[int(0.50 * len(results["months_to_target"]))], 1) if results["months_to_target"] else None,
        "months_to_71k_p25": round(sorted(results["months_to_target"])[int(0.25 * len(results["months_to_target"]))], 1) if results["months_to_target"] else None,
        "pct_reaching_71k": round(len(results["months_to_target"]) / n_simulations, 3),
    }


def check_edge_degradation(quarterly_min_trades: int = 10) -> dict:
    import hypotheses_db
    db = hypotheses_db.load_all()
    if not isinstance(db, dict):
        db = {HYP_DB_RESOLVED: []}
    resolved = [r for r in db.get(HYP_DB_RESOLVED, []) if r.get("exit_type") == "manual"]

    if len(resolved) < quarterly_min_trades:
        return {
            "status": "INSUFFICIENT_DATA",
            "trades": len(resolved),
            "message": f"Need {quarterly_min_trades} trades, have {len(resolved)}",
        }

    wins = [r for r in resolved if r.get("pnl_at_exit", 0) > 0]
    current_hit_rate = len(wins) / len(resolved)

    backtest_hit_rate = 0.289
    threshold_warning = 0.20
    threshold_critical = 0.15

    status = "GREEN"
    if current_hit_rate < threshold_critical:
        status = "CRITICAL"
    elif current_hit_rate < threshold_warning:
        status = "WARNING"

    avg_pnl = sum(r.get("pnl_at_exit", 0) for r in resolved) / len(resolved)

    by_quarter = defaultdict(list)
    for r in resolved:
        try:
            dt = datetime.fromisoformat(r.get("resolved_at", r.get("created_at", "")))
            qkey = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            by_quarter[qkey].append(r)
        except (ValueError, TypeError):
            pass

    quarterly_stats = {}
    for qkey, trades in sorted(by_quarter.items()):
        qwins = sum(1 for t in trades if t.get("pnl_at_exit", 0) > 0)
        quarterly_stats[qkey] = {
            "trades": len(trades),
            "hit_rate": round(qwins / len(trades), 3),
            "avg_pnl": round(sum(t.get("pnl_at_exit", 0) for t in trades) / len(trades), 3),
        }

    edge_decay = current_hit_rate < backtest_hit_rate * 0.7

    return {
        "status": status,
        "total_trades": len(resolved),
        "current_hit_rate": round(current_hit_rate, 3),
        "backtest_hit_rate": backtest_hit_rate,
        "avg_pnl_per_trade": round(avg_pnl, 3),
        "edge_decay_detected": edge_decay,
        "quarterly_breakdown": quarterly_stats,
        "recommendation": (
            "CONTINUE" if status == "GREEN" else
            "REDUCE_POSITION_SIZE" if status == "WARNING" else
            "PAUSE_AND_RECALIBRATE"
        ),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--monte-carlo", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sims", type=int, default=5000)
    parser.add_argument("--months", type=int, default=36)
    args = parser.parse_args()

    if args.monte_carlo or args.all:
        print("\n=== MONTE CARLO STRESS TEST ===\n")

        stats = _get_live_stats()
        print(f"Live stats: {stats['trades']} trades, win_rate={stats['win_rate']}, avg_pnl={stats['avg_pnl']:+.1%}")

        scenarios = [
            ("Conservative (7%/mo)", 0.289, 2.5, -0.30, 0.02),
            ("Moderate (12%/mo)", 0.35, 2.5, -0.30, 0.02),
            ("Pessimistic (high fees)", 0.289, 2.0, -0.35, 0.05),
            ("Worst case (20% win)", 0.20, 1.5, -0.40, 0.05),
        ]

        for name, wr, aw, al, fee in scenarios:
            result = monte_carlo_simulation(
                win_rate=wr, avg_win_pct=aw, avg_loss_pct=al,
                fee_per_trade=fee, n_simulations=args.sims, months=args.months,
            )
            print(f"\n--- {name} ---")
            print(f"  Final balance: p5=${result['p5_final']:,.0f}  p50=${result['p50_final']:,.0f}  p95=${result['p95_final']:,.0f}")
            print(f"  Max drawdown:  p50={result['max_drawdown_p50']:.1%}  p95={result['max_drawdown_p95']:.1%}")
            print(f"  Risk of ruin:  {result['ruin_probability']:.1%}")
            print(f"  Reach $71k:    {result['pct_reaching_71k']:.0%} of simulations"
                  + (f" (median {result['months_to_71k_p50']:.0f}mo)" if result['months_to_71k_p50'] else ""))

    if args.edge or args.all:
        print("\n=== COMPETITIVE EDGE MONITOR ===\n")
        report = check_edge_degradation()
        print(f"Status: {report['status']}")
        print(f"Hit rate: {report.get('current_hit_rate', 'N/A')} (backtest: {report.get('backtest_hit_rate', 'N/A')})")
        avg_pnl = report.get('avg_pnl_per_trade', None)
        print(f"Avg PnL per trade: {avg_pnl:+.1%}" if avg_pnl is not None else "Avg PnL per trade: N/A")
        print(f"Edge decay: {report.get('edge_decay_detected', 'N/A')}")
        print(f"Recommendation: {report.get('recommendation', 'N/A')}")
        if report.get("quarterly_breakdown"):
            print("\nQuarterly breakdown:")
            for q, s in report["quarterly_breakdown"].items():
                print(f"  {q}: {s['trades']} trades, hit_rate={s['hit_rate']:.1%}, avg_pnl={s['avg_pnl']:+.1%}")


if __name__ == "__main__":
    main()
