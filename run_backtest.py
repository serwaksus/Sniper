#!/usr/bin/env python3
"""
Backtest v2 Runner — realistic event-driven backtest with no look-ahead bias.

Usage:
  python3 run_backtest.py                  # default run
  python3 run_backtest.py --balance 1000   # custom starting balance
  python3 run_backtest.py --no-advisor     # skip advisor veto simulation
  python3 run_backtest.py --walk-forward   # walk-forward validation
  python3 run_backtest.py --refresh        # force refresh market data
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
    args = parser.parse_args()
    
    print(f"DOTM Sniper Backtest v2")
    print(f"  Balance: ${args.balance}")
    print(f"  Max markets: {args.max_markets}")
    print(f"  Advisor: {'OFF' if args.no_advisor else 'ON'}")
    print(f"  News: {'OFF' if args.no_news else 'ON'}")
    print(f"  Metaculus: {'OFF' if args.no_metaculus else 'ON'}")
    print()
    
    if args.walk_forward:
        results = run_walk_forward(
            starting_balance=args.balance,
            max_markets=args.max_markets,
            force_refresh=args.refresh,
        )
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
