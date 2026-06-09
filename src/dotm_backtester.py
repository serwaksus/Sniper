#!/usr/bin/env python3
"""
DOTM Backtester v3.0 - Parallel Paper Trading + Historical Calibration

Two modes:
  --mode=live     (default) Fetch active DOTM markets, run full pipeline,
                   record predictions. Later check resolution via --check.
  --mode=sim      Use resolved markets with simulated DOTM prices for
                   quick calibration of LLM probability estimation.

Usage:
    python3 src/dotm_backtester.py --mode live --count 100
    python3 src/dotm_backtester.py --mode live --count 100 --skip-advisor
    python3 src/dotm_backtester.py --mode sim --count 50
    python3 src/dotm_backtester.py --check

Output: backtest_stats.json
"""
from __future__ import annotations
import json
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotm_sniper import load_json, save_json

from backtest_simulator import (
    _simulate_tp_ladder,
    _normalize_keys,
    _fetch_active_dotm_markets_pm_trader,
    _fetch_active_dotm_markets_gamma,
    _fetch_resolved_dotm_markets,
    backtest_analyze_single,  # noqa: F401 — re-exported for backward compat
    backtest_advisor_check,
    _parallel_analyze_markets,
    DOTM_PRICE_MIN, DOTM_PRICE_MAX,
    BACKTEST_MAX_WORKERS,
    GAMMA_API,
)
from backtest_stats import (
    process_resolved_results,
    process_sim_results,
    process_live_results,
    print_resolved_report,
    print_sim_report,
    print_live_report,
)
from utils import load_env_file
from config import SNIPER_LOG, BACKTEST_STATS_FILE as BACKTEST_OUTPUT

load_env_file()

LOG_FILE = SNIPER_LOG
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


def run_backtest_live(count: int = 100, skip_advisor: bool = False, use_calibrator: bool = False) -> dict | None:
    """Resolved mode: Fetch closed+resolved DOTM markets, run analysis, compute metrics."""
    print("=" * 60)
    print(f"  DOTM BACKTESTER v3.0 [RESOLVED] - {count} historical markets")
    print(f"  Workers: {BACKTEST_MAX_WORKERS}")
    print("=" * 60)

    markets = _fetch_resolved_dotm_markets(limit=count)

    if not markets:
        print("No resolved DOTM markets found. Check API connectivity.")
        return

    print(f"\nFetched {len(markets)} resolved markets with validated outcomes")
    print(f"Running parallel analysis with {BACKTEST_MAX_WORKERS} workers...")

    analyses = _parallel_analyze_markets(markets, label="BACKTEST-RESOLVED")

    results, summary, cluster_stats, dampened_markets = process_resolved_results(
        markets, analyses, skip_advisor, backtest_advisor_check, _simulate_tp_ladder,
    )

    config = {
        "count_requested": count,
        "count_fetched": len(markets),
        "skip_advisor": skip_advisor,
        "simulated_prices": True,
        "price_range": f"${DOTM_PRICE_MIN}-${DOTM_PRICE_MAX}",
        "max_workers": BACKTEST_MAX_WORKERS,
    }

    stats = {
        "timestamp": datetime.now().isoformat(),
        "mode": "resolved",
        "config": config,
        "summary": summary,
        "cluster_stats": {k: v for k, v in cluster_stats.items() if v.get("total", 0) >= 1},
        "dampened_markets": dampened_markets,
        "results": results,
    }
    save_json(BACKTEST_OUTPUT, stats)

    print_resolved_report(summary, cluster_stats, dampened_markets, config, BACKTEST_OUTPUT)

    return _normalize_keys(load_json(BACKTEST_OUTPUT, {}))


def run_backtest_live_active(count: int = 100, skip_advisor: bool = False) -> dict | None:
    """LIVE mode: Fetch active DOTM markets, run analysis, record predictions."""
    print("=" * 60)
    print(f"  DOTM BACKTESTER v3.0 [LIVE-ACTIVE] - {count} active DOTM markets")
    print(f"  Workers: {BACKTEST_MAX_WORKERS}")
    print("=" * 60)

    markets = _fetch_active_dotm_markets_pm_trader(limit=count)
    if len(markets) < count:
        gamma_markets = _fetch_active_dotm_markets_gamma(limit=count)
        seen = {m["slug"] for m in markets}
        for m in gamma_markets:
            if m["slug"] not in seen:
                markets.append(m)
                seen.add(m["slug"])

    markets = markets[:count]

    if not markets:
        print("No active DOTM markets found. Check API connectivity.")
        return

    print(f"\nFetched {len(markets)} active DOTM markets for analysis")
    print(f"Running parallel analysis with {BACKTEST_MAX_WORKERS} workers...")

    analyses = _parallel_analyze_markets(markets, label="BACKTEST-LIVE")

    results, summary, cluster_stats = process_live_results(
        markets, analyses, skip_advisor, backtest_advisor_check,
    )

    config = {
        "count_requested": count,
        "count_fetched": len(markets),
        "skip_advisor": skip_advisor,
        "max_workers": BACKTEST_MAX_WORKERS,
    }

    stats = {
        "timestamp": datetime.now().isoformat(),
        "mode": "live",
        "config": config,
        "summary": summary,
        "cluster_stats": dict(cluster_stats),
        "results": results,
    }
    save_json(BACKTEST_OUTPUT, stats)

    print_live_report(summary, cluster_stats, config, BACKTEST_OUTPUT)

    return stats


def run_backtest_sim(count: int = 50, skip_advisor: bool = False) -> dict | None:
    """SIM mode: Fetch resolved markets, simulate DOTM prices, run analysis."""
    print("=" * 60)
    print(f"  DOTM BACKTESTER v3.0 [SIM] - {count} resolved markets")
    print(f"  NOTE: Opening prices are SIMULATED (uniform ${DOTM_PRICE_MIN}-${DOTM_PRICE_MAX})")
    print(f"  Workers: {BACKTEST_MAX_WORKERS}")
    print("=" * 60)

    markets = _fetch_resolved_dotm_markets(limit=count)
    if not markets:
        print("No resolved markets found. Check API connectivity.")
        return

    print(f"\nFetched {len(markets)} resolved markets with simulated DOTM prices")
    print(f"Running parallel analysis with {BACKTEST_MAX_WORKERS} workers...")

    analyses = _parallel_analyze_markets(markets, label="BACKTEST-SIM")

    results, summary, cluster_stats = process_sim_results(
        markets, analyses, skip_advisor, backtest_advisor_check, _simulate_tp_ladder,
    )

    config = {
        "count_requested": count,
        "count_fetched": len(markets),
        "skip_advisor": skip_advisor,
        "simulated_prices": True,
        "price_range": f"${DOTM_PRICE_MIN}-${DOTM_PRICE_MAX}",
        "max_workers": BACKTEST_MAX_WORKERS,
    }

    stats = {
        "timestamp": datetime.now().isoformat(),
        "mode": "sim",
        "config": config,
        "summary": summary,
        "cluster_stats": {k: v for k, v in cluster_stats.items() if v.get("total", 0) >= 1},
        "results": results,
    }
    save_json(BACKTEST_OUTPUT, stats)

    print_sim_report(summary, cluster_stats, config, BACKTEST_OUTPUT)

    return stats


def check_pending() -> None:
    """Check resolution of pending live predictions."""
    import requests

    stats = _normalize_keys(load_json(BACKTEST_OUTPUT, {}))
    if not stats or stats.get("mode") != "live":
        print("No pending live backtest found. Run --mode live first.")
        return

    results = stats.get("results", [])
    pending = [r for r in results if r.get("status") == "pending"]
    if not pending:
        print("All predictions already resolved.")
        return

    print(f"Checking {len(pending)} pending predictions...")

    resolved_count = 0
    wins = 0
    losses = 0
    still_pending = 0

    for r in pending:
        slug = r["slug"]
        try:
            resp = requests.get(
                GAMMA_API,
                params={"slug": slug, "limit": 1},
                timeout=15
            )
            data = resp.json()
            if not data:
                still_pending += 1
                continue

            m = data[0]
            if not m.get("closed"):
                still_pending += 1
                continue

            outcome_prices = m.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            if outcome_prices:
                try:
                    yes_final = float(outcome_prices[0])
                except (ValueError, IndexError, TypeError):
                    yes_final = 0.5

                if yes_final > 0.5:
                    r["resolution"] = "YES"
                else:
                    r["resolution"] = "NO"
                r["status"] = "resolved"
                r["resolved_at"] = datetime.now().isoformat()

                actual = 1 if r["resolution"] == "YES" else 0
                r["brier"] = (r.get("p_model", 0.5) - actual) ** 2

                if r["action"] == "BUY" and r["resolution"] == "YES":
                    wins += 1
                elif r["action"] == "BUY" and r["resolution"] == "NO":
                    losses += 1

                resolved_count += 1
                print(f"  {slug[:40]}... => {r['resolution']} {'WIN' if r['action'] == 'BUY' and r['resolution'] == 'YES' else ''}")
            else:
                still_pending += 1
        except Exception as e:
            logger.warning(f"[CHECK] Error for {slug}: {e}")
            still_pending += 1

        time.sleep(0.3)

    traded = sum(1 for r in results if r["action"] == "BUY" and r.get("status") == "resolved")
    briers = [r["brier"] for r in results if r.get("brier") is not None and r.get("status") == "resolved"]

    stats["summary"]["resolved"] = resolved_count
    stats["summary"]["still_pending"] = still_pending
    stats["summary"]["wins"] = wins
    stats["summary"]["losses"] = losses
    stats["summary"]["winrate"] = wins / max(traded, 1)
    stats["summary"]["brier_score"] = sum(briers) / max(len(briers), 1)

    save_json(BACKTEST_OUTPUT, stats)

    print(f"\nResolved: {resolved_count}, Still pending: {still_pending}")
    print(f"Wins: {wins}, Losses: {losses}, Winrate: {wins/max(traded,1):.1%}")
    if briers:
        print(f"Brier Score: {sum(briers)/len(briers):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOTM Sniper Backtester v3.0")
    parser.add_argument("--mode", choices=["resolved", "sim", "live"], default="resolved",
                        help="resolved=closed+resolved markets with immediate metrics (default), "
                             "sim=alias for resolved, live=active markets (record + check later)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of markets to backtest (default: 100)")
    parser.add_argument("--skip-advisor", action="store_true",
                        help="Skip advisor pre-check (saves tokens)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply isotonic calibration from pre-trained model")
    parser.add_argument("--check", action="store_true",
                        help="Check resolution of pending predictions (live mode only)")
    args = parser.parse_args()

    if args.check:
        check_pending()
    elif args.mode in ("resolved", "sim"):
        run_backtest_live(count=args.count, skip_advisor=args.skip_advisor, use_calibrator=args.calibrate)
    elif args.mode == "live":
        run_backtest_live_active(count=args.count, skip_advisor=args.skip_advisor)
