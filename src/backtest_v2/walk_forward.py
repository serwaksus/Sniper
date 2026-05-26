#!/usr/bin/env python3
"""
Backtest v2 Walk-Forward Validation.
Expanding window: train on months 1-N, test on month N+1.
Eliminates calibration look-ahead bias.
"""
import json
import os
import sys
import logging
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import fetch_resolved_markets
from engine import run_backtest

logger = logging.getLogger(__name__)

RESULTS_DIR = "/root/dotm-sniper/backtest_data"


def split_by_month(markets: List[Dict], min_train: int = 20) -> List[Dict]:
    """
    Split markets into monthly buckets.
    Returns list of {month_key, markets}.
    """
    by_month = {}
    for m in markets:
        created = m.get("created_at", "")[:7]
        if not created:
            continue
        by_month.setdefault(created, []).append(m)
    
    months = sorted(by_month.keys())
    buckets = []
    for month in months:
        if len(by_month[month]) >= 3:
            buckets.append({
                "month": month,
                "markets": by_month[month],
            })
    
    return buckets


def run_walk_forward(
    starting_balance: float = 500.0,
    max_markets: int = 500,
    force_refresh: bool = False,
) -> Dict:
    """
    Walk-forward validation with expanding window.
    Each fold: train on all data up to month N, test on month N+1.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    markets = fetch_resolved_markets(
        max_markets=max_markets,
        force_refresh=force_refresh,
    )
    
    if not markets:
        return {"error": "no markets"}
    
    buckets = split_by_month(markets)
    
    if len(buckets) < 2:
        logger.warning("[WALK-FORWARD] Not enough monthly data, running single backtest")
        return run_backtest(starting_balance=starting_balance, max_markets=max_markets,
                            markets=markets)
    
    fold_results = []
    cumulative_pnl = 0.0
    cumulative_trades = 0
    cumulative_wins = 0
    
    for i in range(1, len(buckets)):
        test_month = buckets[i]["month"]
        test_markets = buckets[i]["markets"]
        
        train_months = [b["month"] for b in buckets[:i]]
        train_size = sum(len(b["markets"]) for b in buckets[:i])
        
        logger.info(f"[WALK-FORWARD] Fold {i}/{len(buckets)-1}: "
                     f"train={train_months[0]}..{train_months[-1]} ({train_size} markets), "
                     f"test={test_month} ({len(test_markets)} markets)")
        
        result = run_backtest(
            starting_balance=starting_balance,
            max_markets=len(test_markets),
            use_advisor=True,
            use_news=True,
            seed=42 + i,
            markets=test_markets,
        )
        
        fold_results.append({
            "fold": i,
            "train_months": f"{train_months[0]}..{train_months[-1]}",
            "test_month": test_month,
            "train_size": train_size,
            "test_size": len(test_markets),
            **result,
        })
        
        cumulative_pnl += result.get("total_pnl", 0)
        cumulative_trades += result.get("total_trades", 0)
        cumulative_wins += result.get("wins", 0)
    
    overall = {
        "method": "walk_forward",
        "folds": len(fold_results),
        "cumulative_pnl": cumulative_pnl,
        "cumulative_trades": cumulative_trades,
        "cumulative_win_rate": cumulative_wins / cumulative_trades if cumulative_trades > 0 else 0,
        "fold_results": fold_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join(RESULTS_DIR, "walk_forward_results.json")
    with open(results_path, 'w') as f:
        json.dump(overall, f, indent=2, default=str)
    
    logger.info(f"[WALK-FORWARD] Done: {len(fold_results)} folds, "
                f"PnL=${cumulative_pnl:.2f}, WR={overall['cumulative_win_rate']:.1%}")
    
    return overall
