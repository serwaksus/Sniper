#!/usr/bin/env python3
"""
Backtest v2 Runner — realistic event-driven backtest with no look-ahead bias.

Usage:
  python3 run_backtest.py                  # default run
  python3 run_backtest.py --balance 1000   # custom starting balance
  python3 run_backtest.py --no-advisor     # skip advisor veto simulation
  python3 run_backtest.py --walk-forward   # walk-forward validation
  python3 run_backtest.py --refresh        # force refresh market data
  python3 run_backtest.py --compare        # compare Conservative vs AggressiveMicro
"""
import sys
import os
import json
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from backtest_v2.engine import run_backtest
from backtest_v2.walk_forward import run_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

PROFILE_CONSERVATIVE = {
    "name": "Conservative",
    "kelly_fraction": 0.25,
    "base_pct": 0.03,
    "max_pct": 0.10,
    "max_positions": 12,
    "max_cluster_pct": 0.40,
    "signal_threshold": 35,
    "min_confidence": 0.50,
    "min_prob_ratio": 1.8,
    "advisor_veto_rate": 0.20,
    "news_block_rate": 0.05,
}

PROFILE_AGGRESSIVE = {
    "name": "AggressiveMicro",
    "kelly_fraction": 0.50,
    "base_pct": 0.05,
    "max_pct": 0.15,
    "max_positions": 6,
    "max_cluster_pct": 0.45,
    "signal_threshold": 45,
    "min_confidence": 0.60,
    "min_prob_ratio": 2.5,
    "advisor_veto_rate": 0.10,
    "news_block_rate": 0.03,
}

PROFILE_AGGRESSIVE_LOOSE = {
    "name": "AggressiveMicro-Loose",
    "kelly_fraction": 0.50,
    "base_pct": 0.05,
    "max_pct": 0.15,
    "max_positions": 6,
    "max_cluster_pct": 0.45,
    "signal_threshold": 35,
    "min_confidence": 0.50,
    "min_prob_ratio": 1.8,
    "advisor_veto_rate": 0.10,
    "news_block_rate": 0.03,
}


def print_results(summary: dict):
    """Print backtest results in readable format."""
    print("\n" + "=" * 60)
    print("  BACKTEST v2 RESULTS — Realistic Simulation")
    print("=" * 60)
    
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return
    
    if "folds" in summary:
        print(f"  Method:           Walk-Forward ({summary['folds']} folds)")
        print(f"  Cumulative PnL:   ${summary['cumulative_pnl']:.2f}")
        print(f"  Total Trades:     {summary['cumulative_trades']}")
        print(f"  Win Rate:         {summary['cumulative_win_rate']:.1%}")
        print()
        for fr in summary.get("fold_results", []):
            print(f"  Fold {fr['fold']:2d} ({fr['test_month']}): "
                  f"trades={fr.get('total_trades',0):3d} "
                  f"WR={fr.get('win_rate',0):.1%} "
                  f"PnL=${fr.get('total_pnl',0):+.2f} "
                  f"DD={fr.get('max_drawdown',0):.1%}")
        return
    
    print(f"  Starting Balance: ${summary.get('starting_balance', 0):.2f}")
    print(f"  Final Equity:     ${summary.get('final_equity', 0):.2f}")
    print(f"  Total PnL:        ${summary.get('total_pnl', 0):+.2f} ({summary.get('total_pnl_pct', 0):+.1%})")
    print(f"  Total Trades:     {summary.get('total_trades', 0)}")
    print(f"  Wins / Losses:    {summary.get('wins', 0)} / {summary.get('losses', 0)}")
    print(f"  Win Rate:         {summary.get('win_rate', 0):.1%}")
    print(f"  Avg Win:          +{summary.get('avg_win_pct', 0):.1%}")
    print(f"  Avg Loss:         {summary.get('avg_loss_pct', 0):.1%}")
    print(f"  Max Drawdown:     {summary.get('max_drawdown', 0):.1%}")
    print(f"  Sharpe Ratio:     {summary.get('sharpe_ratio', 0):.2f}")
    print(f"  Rejected Trades:  {summary.get('rejected_trades', 0)}")
    
    if summary.get("exit_reasons"):
        print(f"\n  Exit Reasons:")
        for reason, count in sorted(summary["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    if summary.get("rejected_reasons"):
        print(f"\n  Rejection Reasons:")
        for reason, count in sorted(summary["rejected_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    config = summary.get("config", {})
    print(f"\n  Config: advisor={config.get('use_advisor')}, news={config.get('use_news')}, "
          f"metaculus={config.get('use_metaculus')}, seed={config.get('seed')}")
    print("=" * 60)


def print_comparison_3(res_c, res_a, res_l):
    c1, c2, c3, c4 = 22, 15, 17, 17

    def fmt(val, f):
        if val is None:
            return "N/A"
        if f == "pct":
            return f"{val:.1%}"
        if f == "pct2":
            return f"{val:+.1%}"
        if f == "dollar":
            return f"${val:,.0f}"
        if f == "f2":
            return f"{val:.2f}"
        return str(val)

    def row(label, vc, va, vl, f="s"):
        print(f"\u2551 {label:<{c1}} \u2502 {fmt(vc,f):>{c2}} \u2502 {fmt(va,f):>{c3}} \u2502 {fmt(vl,f):>{c4}} \u2551")

    def divider():
        h = "\u2500"
        print(f"\u2551 {h*c1}\u253c{h*(c2+2)}\u253c{h*(c3+2)}\u253c{h*(c4+1)}\u2551")

    w = c1 + c2 + c3 + c4 + 14
    print()
    print("\u2554" + "\u2550" * w + "\u2557")
    title = "BACKTEST: 3-PROFILE COMPARISON"
    print(f"\u2551{title:^{w}}\u2551")
    print("\u2560" + "\u2550" * w + "\u2563")
    nc = res_c.get("profile_name", "Conservative")
    na = res_a.get("profile_name", "AggressiveMicro")
    nl = res_l.get("profile_name", "Aggro-Loose")
    row("Metric", nc, na, nl)
    divider()
    row("Trades", res_c.get("total_trades",0), res_a.get("total_trades",0), res_l.get("total_trades",0))
    row("Win rate", res_c.get("win_rate",0), res_a.get("win_rate",0), res_l.get("win_rate",0), "pct")
    row("Avg win", res_c.get("avg_win_pct",0), res_a.get("avg_win_pct",0), res_l.get("avg_win_pct",0), "pct2")
    row("Avg loss", res_c.get("avg_loss_pct",0), res_a.get("avg_loss_pct",0), res_l.get("avg_loss_pct",0), "pct2")
    row("Total P&L", res_c.get("total_pnl",0), res_a.get("total_pnl",0), res_l.get("total_pnl",0), "dollar")
    row("Final equity", res_c.get("final_equity",0), res_a.get("final_equity",0), res_l.get("final_equity",0), "dollar")
    row("Max drawdown", res_c.get("max_drawdown",0), res_a.get("max_drawdown",0), res_l.get("max_drawdown",0), "pct")
    row("Sharpe ratio", res_c.get("sharpe_ratio",0), res_a.get("sharpe_ratio",0), res_l.get("sharpe_ratio",0), "f2")
    row("Rejected", res_c.get("rejected_trades",0), res_a.get("rejected_trades",0), res_l.get("rejected_trades",0))
    print("\u255a" + "\u2550" * w + "\u255d")

    for label, res in [(nc, res_c), (na, res_a), (nl, res_l)]:
        print(f"\n  Top Exit Reasons ({label}):")
        for reason, count in sorted(res.get("exit_reasons",{}).items(), key=lambda x: -x[1])[:5]:
            print(f"    {reason}: {count}")
        print(f"  Top Rejections ({label}):")
        grouped = {}
        for k, v in res.get("rejected_reasons",{}).items():
            key = k.split(":")[0] if ":" in k else k
            grouped[key] = grouped.get(key, 0) + v
        for reason, count in sorted(grouped.items(), key=lambda x: -x[1])[:4]:
            print(f"    {reason}: {count}")


def print_comparison(res_conservative: dict, res_aggressive: dict):
    """Print side-by-side comparison of two backtest results."""
    name_a = res_conservative.get("profile_name", "Conservative")
    name_b = res_aggressive.get("profile_name", "AggressiveMicro")

    col1 = 22
    col2 = 15
    col3 = 18

    def fmt_pct(val):
        return f"{val:+.1%}" if val < 0 else f"{val:.1%}"

    def fmt_dollar(val):
        return f"${val:,.0f}"

    def row(label, va, vb, fmt="s"):
        if fmt == "pct":
            sa = fmt_pct(va) if va is not None else "N/A"
            sb = fmt_pct(vb) if vb is not None else "N/A"
        elif fmt == "dollar":
            sa = fmt_dollar(va) if va is not None else "N/A"
            sb = fmt_dollar(vb) if vb is not None else "N/A"
        elif fmt == "pct_plain":
            sa = f"{va:.1%}" if va is not None else "N/A"
            sb = f"{vb:.1%}" if vb is not None else "N/A"
        elif fmt == "f2":
            sa = f"{va:.2f}" if va is not None else "N/A"
            sb = f"{vb:.2f}" if vb is not None else "N/A"
        else:
            sa = str(va) if va is not None else "N/A"
            sb = str(vb) if vb is not None else "N/A"
        print(f"\u2551 {label:<{col1}} \u2502 {sa:>{col2}} \u2502 {sb:>{col3}} \u2551")

    def divider():
        h = "\u2500"
        print(f"\u2551 {h * col1}\u253c{h * (col2 + 2)}\u253c{h * (col3 + 1)}\u2551")

    w = col1 + col2 + col3 + 10

    print()
    print("\u2554" + "\u2550" * w + "\u2557")
    title = f"BACKTEST COMPARISON: {name_a.upper()} vs {name_b.upper()}"
    print(f"\u2551{title:^{w}}\u2551")
    print("\u2560" + "\u2550" * w + "\u2563")

    row("Metric", name_a, name_b)
    divider()
    row("Trades", res_conservative.get("total_trades", 0), res_aggressive.get("total_trades", 0))
    row("Win rate", res_conservative.get("win_rate", 0), res_aggressive.get("win_rate", 0), fmt="pct_plain")
    row("Avg win", res_conservative.get("avg_win_pct", 0), res_aggressive.get("avg_win_pct", 0), fmt="pct")
    row("Avg loss", res_conservative.get("avg_loss_pct", 0), res_aggressive.get("avg_loss_pct", 0), fmt="pct")
    row("Total P&L", res_conservative.get("total_pnl", 0), res_aggressive.get("total_pnl", 0), fmt="dollar")
    row("Final equity", res_conservative.get("final_equity", 0), res_aggressive.get("final_equity", 0), fmt="dollar")
    row("Max drawdown", res_conservative.get("max_drawdown", 0), res_aggressive.get("max_drawdown", 0), fmt="pct_plain")
    row("Sharpe ratio", res_conservative.get("sharpe_ratio", 0), res_aggressive.get("sharpe_ratio", 0), fmt="f2")
    row("Rejected trades", res_conservative.get("rejected_trades", 0), res_aggressive.get("rejected_trades", 0))
    print("\u255a" + "\u2550" * w + "\u255d")

    for label, res in [(name_a, res_conservative), (name_b, res_aggressive)]:
        print(f"\n  Exit Reasons ({label}):")
        if res.get("exit_reasons"):
            for reason, count in sorted(res["exit_reasons"].items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
        else:
            print("    (none)")

        print(f"\n  Top Rejection Reasons ({label}):")
        if res.get("rejected_reasons"):
            for reason, count in sorted(res["rejected_reasons"].items(), key=lambda x: -x[1])[:5]:
                print(f"    {reason}: {count}")
        else:
            print("    (none)")

    print()


def main():
    parser = argparse.ArgumentParser(description="DOTM Sniper Backtest v2")
    parser.add_argument("--balance", type=float, default=500.0, help="Starting balance")
    parser.add_argument("--max-markets", type=int, default=300, help="Max markets to fetch")
    parser.add_argument("--no-advisor", action="store_true", help="Skip advisor veto")
    parser.add_argument("--no-news", action="store_true", help="Skip news block")
    parser.add_argument("--no-metaculus", action="store_true", help="Skip metaculus")
    parser.add_argument("--walk-forward", action="store_true", help="Walk-forward validation")
    parser.add_argument("--refresh", action="store_true", help="Force refresh market data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default="", help="Save results to file")
    parser.add_argument("--compare", action="store_true", help="Compare Conservative vs AggressiveMicro profiles")
    args = parser.parse_args()
    
    print(f"DOTM Sniper Backtest v2")
    print(f"  Balance: ${args.balance}")
    print(f"  Max markets: {args.max_markets}")
    print(f"  Advisor: {'OFF' if args.no_advisor else 'ON'}")
    print(f"  News: {'OFF' if args.no_news else 'ON'}")
    print(f"  Metaculus: {'OFF' if args.no_metaculus else 'ON'}")
    if args.compare:
        print(f"  Mode: COMPARE (3 profiles)")
    print()
    
    if args.compare:
        results_conservative = run_backtest(
            starting_balance=args.balance,
            max_markets=args.max_markets,
            use_advisor=not args.no_advisor,
            use_news=not args.no_news,
            use_metaculus=not args.no_metaculus,
            force_refresh=args.refresh,
            seed=args.seed,
            profile=PROFILE_CONSERVATIVE,
        )

        markets_used = results_conservative.get("_markets")
        results_aggressive = run_backtest(
            starting_balance=args.balance,
            max_markets=args.max_markets,
            use_advisor=not args.no_advisor,
            use_news=not args.no_news,
            use_metaculus=not args.no_metaculus,
            force_refresh=args.refresh,
            seed=args.seed,
            markets=markets_used,
            profile=PROFILE_AGGRESSIVE,
        )

        results_loose = run_backtest(
            starting_balance=args.balance,
            max_markets=args.max_markets,
            use_advisor=not args.no_advisor,
            use_news=not args.no_news,
            use_metaculus=not args.no_metaculus,
            force_refresh=args.refresh,
            seed=args.seed,
            markets=markets_used,
            profile=PROFILE_AGGRESSIVE_LOOSE,
        )

        print_comparison_3(results_conservative, results_aggressive, results_loose)

        if args.save:
            combined = {
                "conservative": results_conservative,
                "aggressive": results_aggressive,
                "aggressive_loose": results_loose,
            }
            with open(args.save, 'w') as f:
                json.dump(combined, f, indent=2, default=str)
            print(f"\nResults saved to {args.save}")
    elif args.walk_forward:
        results = run_walk_forward(
            starting_balance=args.balance,
            max_markets=args.max_markets,
            force_refresh=args.refresh,
        )
        print_results(results)
    else:
        results = run_backtest(
            starting_balance=args.balance,
            max_markets=args.max_markets,
            use_advisor=not args.no_advisor,
            use_news=not args.no_news,
            use_metaculus=not args.no_metaculus,
            force_refresh=args.refresh,
            seed=args.seed,
        )
        print_results(results)
    
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
