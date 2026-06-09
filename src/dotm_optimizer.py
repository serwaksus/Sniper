#!/usr/bin/env python3
"""
DOTM Optimizer v5.1.0 - Walk-Forward Grid Search + Monte Carlo Validation

Three-phase pipeline:
  1. Chronological walk-forward split (50/50) on 1000 resolved markets
  2. Parallel grid search on In-Sample to find optimal thresholds
  3. Out-of-Sample validation + Monte Carlo simulation (10k runs)

v5.1.0 Smart Exit integration:
  - Take-Profit at $0.85 with $0.015 slippage ($0.835 net)
  - Unconditional TP limit orders on all DOTM entries

Stress-test hardening:
  - Slippage & fee penalty: entry price += $0.015 (thin order book simulation)
  - TP exit: $0.85 - $0.015 = $0.835 net
  - 50/50 split for statistical significance on OOS (target 40-50 trades)

Data source: backtest_stats.json (local, no external API calls).

Usage:
    python3 src/dotm_optimizer.py
"""
import os
import sys
import random
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotm_sniper import load_json, save_json
from config import BACKTEST_STATS_FILE as BACKTEST_FILE, OPTIMIZER_OUTPUT_FILE as OPTIMIZER_OUTPUT

COMPOSITE_THRESHOLDS = [65, 70, 75, 80]
VOLUME_THRESHOLDS = [100000, 150000, 200000, 250000]

VOLUME_TO_CONFIDENCE = {
    100000: 0.60,
    150000: 0.65,
    200000: 0.70,
    250000: 0.75,
}

N_MONTE_CARLO = 10000
N_MC_TRADES = 100
POSITION_PCT = 0.035

SLIPPAGE_PENALTY = 0.015

GRID_MAX_WORKERS = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_grid_lock = threading.RLock()
_progress_counter = {"done": 0, "total": 0}


def load_markets():
    data = load_json(BACKTEST_FILE, {})
    results = data.get("results", [])
    if not results:
        logger.error("backtest_stats.json contains no results. Run dotm_backtester.py first.")
        sys.exit(1)
    results.sort(key=lambda x: x.get("created_at", ""))
    return results


def walk_forward_split(markets):
    n = len(markets)
    split = n // 2
    is_markets = markets[:split]
    oos_markets = markets[split:]
    logger.info(
        "Walk-forward split: %d In-Sample | %d Out-of-Sample (sorted by created_at)",
        len(is_markets), len(oos_markets),
    )
    return is_markets, oos_markets


def _evaluate_single(markets, composite_thresh, volume_thresh):
    min_confidence = VOLUME_TO_CONFIDENCE.get(volume_thresh, 0.60)

    wins = 0
    losses = 0
    pnl_sum = 0.0
    upside_sum = 0.0
    trade_outcomes = []

    for m in markets:
        sig = m.get("signal_score")
        if sig is None:
            continue
        if sig < composite_thresh:
            continue
        conf = m.get("confidence", 0)
        if conf < min_confidence:
            continue

        raw_price = m["market_price"]
        entry_price = raw_price + SLIPPAGE_PENALTY
        if entry_price >= 1.0:
            continue

        # v5.1.0: Smart Exit TP check
        high_price = m.get("high_price")
        if high_price is None:
            if m["resolution"] == "YES":
                high_price = 1.0
            else:
                high_price = m.get("yes_final", raw_price)

        tp_hit = high_price >= 0.85 if high_price is not None else False

        if tp_hit:
            wins += 1
            # Net exit price: 0.85 - 0.015 slippage = 0.835
            net_exit = 0.835
            upside = (net_exit - entry_price) / entry_price if entry_price > 0 else 0.0
            upside_sum += upside
            pnl_sum += upside
            trade_outcomes.append(upside)
        elif m["resolution"] == "YES":
            wins += 1
            upside = (1.0 - entry_price) / entry_price if entry_price > 0 else 0.0
            upside_sum += upside
            pnl_sum += upside
            trade_outcomes.append(upside)
        else:
            losses += 1
            pnl_sum -= 1.0
            trade_outcomes.append(-1.0)

    total = wins + losses
    if total == 0:
        return {
            "composite_threshold": composite_thresh,
            "volume_threshold": volume_thresh,
            "min_confidence": min_confidence,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "avg_ev": 0.0,
            "avg_upside": 0.0,
            "score": -999.0,
        }, []

    winrate = wins / total
    avg_ev = pnl_sum / total
    avg_upside = upside_sum / wins if wins > 0 else 0.0
    score = avg_ev

    return {
        "composite_threshold": composite_thresh,
        "volume_threshold": volume_thresh,
        "min_confidence": min_confidence,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "avg_ev": avg_ev,
        "avg_upside": avg_upside,
        "score": score,
    }, trade_outcomes


def _grid_worker(markets, composite_thresh, volume_thresh):
    result, outcomes = _evaluate_single(markets, composite_thresh, volume_thresh)
    with _grid_lock:
        _progress_counter["done"] += 1
        done = _progress_counter["done"]
        total = _progress_counter["total"]
        logger.info(
            "Grid [%d/%d] score>=%d vol>=%d => trades=%d wr=%.1f%% ev=%.2f",
            done, total, composite_thresh, volume_thresh,
            result["trades"], result["winrate"] * 100, result["avg_ev"],
        )
    return result, outcomes


def grid_search(markets):
    combos = [
        (ct, vt)
        for ct in COMPOSITE_THRESHOLDS
        for vt in VOLUME_THRESHOLDS
    ]
    _progress_counter["done"] = 0
    _progress_counter["total"] = len(combos)

    results = []
    outcomes_map = {}

    with ThreadPoolExecutor(max_workers=GRID_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_grid_worker, markets, ct, vt): (ct, vt)
            for ct, vt in combos
        }
        for future in as_completed(futures):
            result, outcomes = future.result()
            results.append(result)
            ct, vt = futures[future]
            outcomes_map[(ct, vt)] = outcomes

    results.sort(key=lambda x: (-x["score"], -x["winrate"]))
    best = results[0]
    best_key = (best["composite_threshold"], best["volume_threshold"])
    best_outcomes = outcomes_map.get(best_key, [])

    return results, best, best_outcomes


def validate_oos(oos_markets, best_params):
    result, outcomes = _evaluate_single(
        oos_markets,
        best_params["composite_threshold"],
        best_params["volume_threshold"],
    )
    logger.info(
        "OOS Validation: trades=%d winrate=%.1f%% avg_upside=%.2fx",
        result["trades"], result["winrate"] * 100, result["avg_upside"],
    )
    return result, outcomes


def monte_carlo_simulation(outcomes, n_sim=N_MONTE_CARLO, n_trades=N_MC_TRADES, pos_pct=POSITION_PCT):
    if not outcomes:
        return {
            "ruin_probability": 0.0,
            "max_drawdown_worst": 0.0,
            "max_drawdown_avg": 0.0,
            "avg_final_capital": 0.0,
            "median_final_capital": 0.0,
            "p5_final_capital": 0.0,
            "p95_final_capital": 0.0,
        }

    rng = random.Random(42)
    max_drawdowns = []
    final_capitals = []
    ruined = 0

    for _ in range(n_sim):
        capital = 1.0
        peak = 1.0
        max_dd = 0.0

        for _ in range(n_trades):
            o = rng.choice(outcomes)
            capital *= (1.0 + pos_pct * o)
            capital = max(capital, 0.0)
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if capital <= 0.01:
                ruined += 1
                break

        max_drawdowns.append(max_dd)
        final_capitals.append(capital)

    sorted_finals = sorted(final_capitals)
    n = len(sorted_finals)

    return {
        "ruin_probability": ruined / n_sim,
        "max_drawdown_worst": max(max_drawdowns),
        "max_drawdown_avg": sum(max_drawdowns) / len(max_drawdowns),
        "avg_final_capital": sum(final_capitals) / len(final_capitals),
        "median_final_capital": sorted_finals[n // 2],
        "p5_final_capital": sorted_finals[int(n * 0.05)],
        "p95_final_capital": sorted_finals[int(n * 0.95)],
    }


def print_report(best_is, is_grid, oos_result, mc_result, n_is, n_oos):
    W = 72

    print("\n" + "=" * W)
    print("  DOTM OPTIMIZER v5.1.0 - WALK-FORWARD VALIDATION REPORT")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * W)

    print(f"\n{'─' * W}")
    print("  PHASE 1: CHRONOLOGICAL WALK-FORWARD SPLIT")
    print(f"{'─' * W}")
    print(f"  Total markets (sorted by created_at): {n_is + n_oos}")
    print(f"  In-Sample  (train): {n_is:>4d} markets  (first 50%)")
    print(f"  Out-of-Sample (test): {n_oos:>4d} markets  (last 50%)")
    print(f"  Slippage penalty:    ${SLIPPAGE_PENALTY:.3f} per contract (entry degradation)")

    print(f"\n{'─' * W}")
    print("  PHASE 2: GRID SEARCH ON IN-SAMPLE (parallel)")
    print(f"{'─' * W}")
    print(
        f"  {'Score>=':>8s}  {'Conf>=':>7s}  {'Trades':>6s}  "
        f"{'W':>3s}  {'L':>3s}  {'Winrate':>7s}  {'Avg EV':>7s}  {'Rank':>4s}"
    )
    print(f"  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*3}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*4}")

    for rank, r in enumerate(is_grid, 1):
        marker = " <<<" if r == best_is else ""
        print(
            f"  {r['composite_threshold']:>8d}  {r['min_confidence']:>7.2f}  "
            f"{r['trades']:>6d}  {r['wins']:>3d}  {r['losses']:>3d}  "
            f"{r['winrate']*100:>5.1f}%  {r['avg_ev']:>+7.2f}  {rank:>4d}{marker}"
        )

    print("\n  BEST IN-SAMPLE COMBINATION:")
    print(f"    Signal Score Threshold >= {best_is['composite_threshold']}")
    print(f"    Confidence Threshold  >= {best_is['min_confidence']:.2f}")
    print(f"    (mapped from volume filter ${best_is['volume_threshold']:,})")
    print(f"    Trades executed:  {best_is['trades']}")
    print(f"    Wins / Losses:    {best_is['wins']} / {best_is['losses']}")
    print(f"    Winrate:          {best_is['winrate']:.1%}")
    print(f"    Avg EV per trade: {best_is['avg_ev']:+.2f}  (after slippage ${SLIPPAGE_PENALTY:.3f})")
    print(f"    Avg Upside (win): {best_is['avg_upside']:.2f}x  (after slippage)")

    print(f"\n{'─' * W}")
    print("  PHASE 3: OUT-OF-SAMPLE VALIDATION (pure exam)")
    print(f"{'─' * W}")
    beat_60 = oos_result["winrate"] >= 0.60
    verdict_60 = "PASS" if beat_60 else "FAIL"
    print(f"    OOS Trades:        {oos_result['trades']}")
    print(f"    OOS Wins / Losses: {oos_result['wins']} / {oos_result['losses']}")
    print(f"    OOS Winrate:       {oos_result['winrate']:.1%}")
    print(f"    OOS Avg EV:        {oos_result['avg_ev']:+.2f}  (after slippage)")
    print(f"    OOS Avg Upside:    {oos_result['avg_upside']:.2f}x  (after slippage)")
    print(f"    60% Winrate Test:  {verdict_60}")
    if oos_result["trades"] > 0:
        degradation = best_is["winrate"] - oos_result["winrate"]
        print(f"    WR Degradation:    {degradation:+.1%} (IS -> OOS)")

    print(f"\n{'─' * W}")
    print("  PHASE 4: MONTE CARLO SIMULATION")
    print(f"  ({N_MONTE_CARLO:,} runs x {N_MC_TRADES} trades, position={POSITION_PCT:.1%})")
    print(f"{'─' * W}")
    safe = mc_result["ruin_probability"] < 0.05 and mc_result["max_drawdown_worst"] < 0.50
    print(f"    Ruin Probability:       {mc_result['ruin_probability']:.2%}")
    print(f"    Worst-Case Max DD:      {mc_result['max_drawdown_worst']:.1%}")
    print(f"    Average Max DD:         {mc_result['max_drawdown_avg']:.1%}")
    print(f"    Avg Final Capital:      {mc_result['avg_final_capital']:.2f}x")
    print(f"    Median Final Capital:   {mc_result['median_final_capital']:.2f}x")
    print(f"    P5  Final Capital:      {mc_result['p5_final_capital']:.2f}x")
    print(f"    P95 Final Capital:      {mc_result['p95_final_capital']:.2f}x")
    print(f"    3.5% Position Safe?     {'YES' if safe else 'NO'}")
    print("      (ruin<5% AND worst_dd<50%)")

    print(f"\n{'═' * W}")
    if oos_result["trades"] == 0:
        print("  VERDICT: INSUFFICIENT DATA - 0 OOS trades with current parameters")
    elif beat_60 and safe:
        print("  VERDICT: STRATEGY VIABLE")
        print(f"    OOS Winrate {oos_result['winrate']:.1%} >= 60% | "
              f"Ruin Risk {mc_result['ruin_probability']:.2%} | "
              f"Max DD {mc_result['max_drawdown_worst']:.1%}")
    elif beat_60 and not safe:
        print("  VERDICT: PROFITABLE BUT RISKY")
        print(f"    Winrate OK ({oos_result['winrate']:.1%}) but risk too high "
              f"(ruin={mc_result['ruin_probability']:.2%}, dd={mc_result['max_drawdown_worst']:.1%})")
        print(f"    Recommendation: reduce position size below {POSITION_PCT:.1%}")
    else:
        print("  VERDICT: NEEDS CALIBRATION")
        print(f"    OOS Winrate {oos_result['winrate']:.1%} < 60% - model overfits IS data")
        print("    Recommendation: raise composite threshold or add filters")
    print(f"{'═' * W}\n")


def main():
    print("=" * 72)
    print("  DOTM OPTIMIZER v5.1.0")
    print("  Walk-Forward Grid Search + Monte Carlo Validation")
    print(f"  Slippage penalty: ${SLIPPAGE_PENALTY:.3f} | TP: $0.85 (net $0.835) | Split: 50/50")
    print("=" * 72)

    markets = load_markets()
    print(f"\n  Loaded {len(markets)} resolved markets from {BACKTEST_FILE}")

    is_markets, oos_markets = walk_forward_split(markets)
    n_is = len(is_markets)
    n_oos = len(oos_markets)

    if n_is == 0 or n_oos == 0:
        print("  ERROR: Empty split. Need at least 10 markets.")
        return

    print(f"\n  Running grid search on {n_is} In-Sample markets...")
    print(f"  Grid: {len(COMPOSITE_THRESHOLDS)} x {len(VOLUME_THRESHOLDS)} = "
          f"{len(COMPOSITE_THRESHOLDS) * len(VOLUME_THRESHOLDS)} combinations")
    print(f"  Workers: {GRID_MAX_WORKERS}\n")

    is_grid, best_is, _is_best_outcomes = grid_search(is_markets)

    print(f"\n  Best IS: score>={best_is['composite_threshold']}, "
          f"conf>={best_is['min_confidence']:.2f} => "
          f"{best_is['trades']} trades, WR={best_is['winrate']:.1%}, "
          f"EV={best_is['avg_ev']:+.2f}")

    print(f"\n  Validating on {n_oos} Out-of-Sample markets...")
    oos_result, oos_outcomes = validate_oos(oos_markets, best_is)

    print(f"  Running Monte Carlo ({N_MONTE_CARLO:,} simulations x {N_MC_TRADES} trades)...")
    mc_result = monte_carlo_simulation(oos_outcomes)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_markets": len(markets),
            "n_in_sample": n_is,
            "n_out_of_sample": n_oos,
            "composite_thresholds": COMPOSITE_THRESHOLDS,
            "volume_thresholds": VOLUME_THRESHOLDS,
            "slippage_penalty": SLIPPAGE_PENALTY,
            "monte_carlo_simulations": N_MONTE_CARLO,
            "monte_carlo_trades": N_MC_TRADES,
            "position_pct": POSITION_PCT,
        },
        "best_in_sample": best_is,
        "grid_results": is_grid,
        "oos_validation": oos_result,
        "monte_carlo": mc_result,
    }
    save_json(OPTIMIZER_OUTPUT, output)

    print_report(best_is, is_grid, oos_result, mc_result, n_is, n_oos)

    print(f"  Full results saved to: {OPTIMIZER_OUTPUT}")


if __name__ == "__main__":
    main()
