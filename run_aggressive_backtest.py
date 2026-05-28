#!/usr/bin/env python3
"""
Aggressive Backtest Runner — parametric strategy comparison.
Runs 4 strategy presets (Baseline, Aggressive, High-Roller, YOLO)
on the same dataset with full walk-forward validation.
"""
import sys
import os
import json
import logging
import random
import time
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "backtest_v2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import fetch_resolved_markets, generate_price_series, generate_order_book
from execution import simulate_buy, simulate_sell, simulate_tp_ladder, MIN_ORDER_USD, FEE_PCT
from portfolio import Position, CONVERGENCE_TP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PRESETS = {
    "baseline": {
        "label": "Baseline (current)",
        "kelly_range": (0.25, 0.40),
        "max_pos_pct_range": (0.03, 0.05),
        "max_pos_cap_range": (0.10, 0.15),
        "signal_threshold": 35,
        "min_confidence": 0.50,
        "min_prob_ratio": 1.8,
        "dotm_max_price": 0.15,
        "hard_stop_loss": -0.80,
        "trailing_activation": 0.30,
        "trailing_stop": 0.15,
        "take_profit": 1.50,
        "advisor_veto_rate": 0.20,
        "news_block_rate": 0.05,
        "max_cluster_pct": 0.40,
        "max_positions_range": (12, 30),
        "tp_ladder": [(0.50, 0.75), (0.30, 0.85)],
    },
    "aggressive": {
        "label": "Aggressive",
        "kelly_range": (0.40, 0.60),
        "max_pos_pct_range": (0.05, 0.08),
        "max_pos_cap_range": (0.15, 0.20),
        "signal_threshold": 25,
        "min_confidence": 0.45,
        "min_prob_ratio": 1.5,
        "dotm_max_price": 0.15,
        "hard_stop_loss": -0.80,
        "trailing_activation": 0.30,
        "trailing_stop": 0.10,
        "take_profit": 2.50,
        "advisor_veto_rate": 0.10,
        "news_block_rate": 0.03,
        "max_cluster_pct": 0.50,
        "max_positions_range": (15, 40),
        "tp_ladder": [(0.40, 0.75), (0.30, 0.85), (0.20, 0.95)],
    },
    "high_roller": {
        "label": "High-Roller",
        "kelly_range": (0.60, 0.80),
        "max_pos_pct_range": (0.08, 0.12),
        "max_pos_cap_range": (0.20, 0.25),
        "signal_threshold": 20,
        "min_confidence": 0.40,
        "min_prob_ratio": 1.3,
        "dotm_max_price": 0.20,
        "hard_stop_loss": -0.60,
        "trailing_activation": 0.25,
        "trailing_stop": 0.12,
        "take_profit": 3.00,
        "advisor_veto_rate": 0.00,
        "news_block_rate": 0.00,
        "max_cluster_pct": 0.50,
        "max_positions_range": (20, 50),
        "tp_ladder": [(0.30, 0.75), (0.30, 0.85), (0.25, 0.95)],
    },
    "yolo": {
        "label": "YOLO",
        "kelly_range": (0.80, 1.00),
        "max_pos_pct_range": (0.12, 0.18),
        "max_pos_cap_range": (0.25, 0.30),
        "signal_threshold": 0,
        "min_confidence": 0.30,
        "min_prob_ratio": 1.0,
        "dotm_max_price": 0.25,
        "hard_stop_loss": -0.50,
        "trailing_activation": 0.20,
        "trailing_stop": 0.15,
        "take_profit": 0,
        "advisor_veto_rate": 0.00,
        "news_block_rate": 0.00,
        "max_cluster_pct": 0.60,
        "max_positions_range": (30, 60),
        "tp_ladder": [(0.30, 0.75), (0.30, 0.85), (0.30, 0.95)],
    },
}


def get_tier_aggressive(balance: float, preset: Dict) -> Dict:
    kelly_lo, kelly_hi = preset["kelly_range"]
    pct_lo, pct_hi = preset["max_pos_pct_range"]
    cap_lo, cap_hi = preset["max_pos_cap_range"]
    pos_lo, pos_hi = preset["max_positions_range"]

    if balance < 2000:
        return {"kelly": kelly_lo, "base_pct": pct_lo, "other_pct": pct_lo * 1.2,
                "max_pct": cap_lo, "max_positions": pos_lo, "tier": "micro"}
    elif balance < 10000:
        frac = (balance - 2000) / 8000
        return {"kelly": kelly_lo + (kelly_hi - kelly_lo) * frac,
                "base_pct": pct_lo + (pct_hi - pct_lo) * frac,
                "other_pct": pct_lo * 1.2 + (pct_hi * 1.2 - pct_lo * 1.2) * frac,
                "max_pct": cap_lo + (cap_hi - cap_lo) * frac,
                "max_positions": int(pos_lo + (pos_hi - pos_lo) * frac),
                "tier": "growth"}
    elif balance < 50000:
        return {"kelly": kelly_hi, "base_pct": pct_hi, "other_pct": pct_hi * 1.2,
                "max_pct": cap_hi, "max_positions": pos_hi, "tier": "established"}
    else:
        return {"kelly": kelly_hi, "base_pct": pct_hi, "other_pct": pct_hi * 1.2,
                "max_pct": cap_hi, "max_positions": pos_hi, "tier": "scale"}


def estimate_signal(market: Dict, preset: Dict) -> Dict:
    price = market["entry_price"]
    vol = market.get("volume", 1000)
    liq = market.get("liquidity", 100)
    ttl_days = market.get("ttl_days", 30)

    base_ratio = 2.5 if price < 0.03 else (2.0 if price < 0.07 else 1.5)
    vol_bonus = min(vol / 50000, 1.0) * 0.3
    liq_bonus = min(liq / 500, 1.0) * 0.2
    ttl_bonus = 0.3 if ttl_days > 90 else (0.2 if ttl_days > 30 else 0.0)

    p_model = price * (base_ratio + vol_bonus + liq_bonus + ttl_bonus)
    p_model = min(p_model, price * 10, 0.95)

    prob_ratio = p_model / price if price > 0 else 0

    ratio_score = min(prob_ratio / 3.0, 1.0) * 25
    vol_score = min(vol / 100_000, 1.0) * 15
    ttl_score = 8 if ttl_days > 30 else (4 if ttl_days > 7 else 0)
    price_score = 15 if price < 0.03 else (8 if price < 0.07 else 0)

    signal_score = ratio_score + vol_score + ttl_score + price_score
    confidence = 0.55 + min(vol / 5000, 0.15) + min(liq / 200, 0.10)
    if price < 0.03:
        confidence += 0.05
    confidence = min(confidence, 0.90)

    threshold = preset["signal_threshold"]
    min_conf = preset["min_confidence"]
    min_ratio = preset["min_prob_ratio"]
    max_price = preset["dotm_max_price"]

    action = "BUY"
    if signal_score < threshold:
        action = "SKIP"
    if confidence < min_conf:
        action = "SKIP"
    if prob_ratio < min_ratio:
        action = "SKIP"
    if price > max_price:
        action = "SKIP"

    return {"p_model": p_model, "prob_ratio": prob_ratio,
            "signal_score": signal_score, "confidence": confidence, "action": action}


def run_preset_backtest(preset: Dict, markets: List[Dict], starting_balance: float = 500.0,
                         seed: int = 42) -> Dict:
    random.seed(seed)

    balance = starting_balance
    positions: Dict[str, Dict] = {}
    trades: List[Dict] = []
    rejected: List[Dict] = []
    equity_curve: List[Dict] = []
    cluster_exposure: Dict[str, float] = {}

    analyzed = 0
    for market in markets:
        slug = market["slug"]
        entry_price = market["entry_price"]
        liquidity = market.get("liquidity", 100)
        cluster = market.get("category", "other")
        if cluster == "other":
            cluster = f"other_{hash(market.get('question', '')) % 8}"

        if balance < 5:
            break

        signal = estimate_signal(market, preset)
        analyzed += 1

        if signal["action"] != "BUY":
            continue

        if preset["advisor_veto_rate"] > 0:
            veto_prob = preset["advisor_veto_rate"]
            if signal["confidence"] < 0.70:
                veto_prob = min(veto_prob * 2.5, 0.6)
            if signal["prob_ratio"] < 2.5:
                veto_prob = min(veto_prob * 2.0, 0.6)
            if random.random() < veto_prob:
                rejected.append({"slug": slug, "reason": "advisor_veto"})
                continue

        if preset["news_block_rate"] > 0 and random.random() < preset["news_block_rate"]:
            rejected.append({"slug": slug, "reason": "news_block"})
            continue

        tier = get_tier_aggressive(balance, preset)
        eff_price = entry_price
        if eff_price <= 0.001:
            continue

        b = (1 - eff_price) / eff_price
        p = signal["p_model"]
        q = 1 - p
        kelly_full = (b * p - q) / b
        if kelly_full <= 0:
            continue

        kelly_dollar = kelly_full * tier["kelly"]
        cap = tier["other_pct"] if cluster.startswith("other") else tier["base_pct"]
        size_pct = min(kelly_dollar, cap)
        amount = round(balance * size_pct)
        amount = min(amount, round(balance * tier["max_pct"]))

        if amount < 5:
            continue

        total_equity = balance + sum(p_d["cost"] for p_d in positions.values())
        cluster_exp = cluster_exposure.get(cluster, 0) + amount
        if total_equity > 0 and cluster_exp / total_equity > preset["max_cluster_pct"]:
            rejected.append({"slug": slug, "reason": "cluster_limit"})
            continue

        if len(positions) >= tier["max_positions"]:
            rejected.append({"slug": slug, "reason": "max_positions"})
            continue

        if amount > balance:
            continue

        buy_result = simulate_buy(entry_price, amount, liquidity)
        if not buy_result["filled"]:
            rejected.append({"slug": slug, "reason": f"buy_failed:{buy_result['reason']}"})
            continue

        fee = buy_result.get("fee", buy_result["cost"] * FEE_PCT)
        total_cost = buy_result["cost"] + fee
        balance -= total_cost

        positions[slug] = {
            "entry_price": buy_result["effective_price"],
            "shares": buy_result["shares"],
            "cost": total_cost,
            "liquidity": liquidity,
            "cluster": cluster,
            "p_model": signal["p_model"],
            "high_price": buy_result["effective_price"],
            "trailing_on": False,
            "stop_loss": 0.0,
            "trailing_confirmed": False,
            "tp_filled": False,
            "shares_after_tp": buy_result["shares"],
        }
        cluster_exposure[cluster] = cluster_exposure.get(cluster, 0) + total_cost

    price_cache = {}
    for market in markets:
        price_cache[market["slug"]] = generate_price_series(market, num_steps=60)

    for step_idx in range(31):
        current_prices = {}
        for slug, pos in positions.items():
            series = price_cache.get(slug, [pos["entry_price"]])
            idx = min(step_idx, len(series) - 1)
            current_prices[slug] = series[idx]

        for slug, pos in list(positions.items()):
            price = current_prices.get(slug, 0)
            if price <= 0:
                continue

            pos["high_price"] = max(pos["high_price"], price)

            if pos["high_price"] > pos["entry_price"] * (1 + preset["trailing_activation"]):
                pos["trailing_on"] = True
                pos["stop_loss"] = pos["high_price"] * (1 - preset["trailing_stop"])

            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
            sold = False

            metaculus_prob = pos["p_model"] * 1.5
            if price > 0 and metaculus_prob > 0:
                convergence = price / metaculus_prob
                if convergence >= CONVERGENCE_TP:
                    sell_r = simulate_sell(price, pos["shares_after_tp"], pos["entry_price"], pos["liquidity"])
                    if sell_r["filled"]:
                        fee = sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                        net = sell_r["proceeds"] - fee
                        balance += net
                        trades.append({"slug": slug, "pnl_pct": (net - pos["cost"]) / pos["cost"],
                                       "pnl_abs": net - pos["cost"], "reason": "convergence"})
                        del positions[slug]
                        sold = True

            if not sold and pos["trailing_on"] and price <= pos["stop_loss"]:
                sell_r = simulate_sell(price, pos["shares_after_tp"], pos["entry_price"], pos["liquidity"])
                if sell_r["filled"]:
                    fee = sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                    net = sell_r["proceeds"] - fee
                    balance += net
                    trades.append({"slug": slug, "pnl_pct": (net - pos["cost"]) / pos["cost"],
                                   "pnl_abs": net - pos["cost"], "reason": "trailing_stop"})
                    del positions[slug]
                    sold = True

            if not sold and pnl_pct <= preset["hard_stop_loss"]:
                sell_r = simulate_sell(price, pos["shares_after_tp"], pos["entry_price"], pos["liquidity"], force_market=True)
                if sell_r["filled"]:
                    fee = sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                    net = sell_r["proceeds"] - fee
                    balance += net
                    trades.append({"slug": slug, "pnl_pct": (net - pos["cost"]) / pos["cost"],
                                   "pnl_abs": net - pos["cost"], "reason": "hard_stop"})
                    del positions[slug]
                    sold = True

            if not sold and preset["take_profit"] > 0 and pnl_pct >= preset["take_profit"]:
                sell_r = simulate_sell(price, pos["shares_after_tp"], pos["entry_price"], pos["liquidity"])
                if sell_r["filled"]:
                    fee = sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                    net = sell_r["proceeds"] - fee
                    balance += net
                    trades.append({"slug": slug, "pnl_pct": (net - pos["cost"]) / pos["cost"],
                                   "pnl_abs": net - pos["cost"], "reason": "take_profit"})
                    del positions[slug]
                    sold = True

            if not sold and not pos["tp_filled"] and price >= 0.75:
                tp_r = simulate_tp_ladder(pos["shares"], pos["entry_price"], price,
                                           pos["liquidity"], ladder=preset["tp_ladder"])
                if tp_r["rungs_filled"] > 0:
                    pos["tp_filled"] = True
                    pos["shares_after_tp"] = tp_r["shares_held_to_expiry"]
                    balance += tp_r["total_proceeds"]

        eq = balance + sum(
            p["shares_after_tp"] * current_prices.get(s, 0) for s, p in positions.items()
        )
        peak = max((e["equity"] for e in equity_curve), default=eq)
        dd = (eq - peak) / peak if peak > 0 else 0
        equity_curve.append({"step": step_idx, "equity": eq, "drawdown": dd,
                              "balance": balance, "positions": len(positions)})

    for slug, pos in list(positions.items()):
        series = price_cache.get(slug, [pos["entry_price"]])
        final_price = series[-1]
        sell_r = simulate_sell(final_price, pos["shares_after_tp"], pos["entry_price"], pos["liquidity"], force_market=True)
        proceeds = sell_r["proceeds"] if sell_r["filled"] else pos["shares_after_tp"] * final_price * 0.5
        fee = sell_r.get("fee", proceeds * FEE_PCT) if sell_r["filled"] else proceeds * FEE_PCT
        net = proceeds - fee
        balance += net
        trades.append({"slug": slug, "pnl_pct": (net - pos["cost"]) / pos["cost"],
                       "pnl_abs": net - pos["cost"], "reason": "resolution"})

    if not trades:
        return {"total_trades": 0, "label": preset["label"]}

    wins = [t for t in trades if t["pnl_abs"] > 0]
    losses = [t for t in trades if t["pnl_abs"] <= 0]
    total_pnl = sum(t["pnl_abs"] for t in trades)
    max_dd = min((e["drawdown"] for e in equity_curve), default=0)

    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]["equity"]
        curr = equity_curve[i]["equity"]
        if prev > 0:
            returns.append(curr / prev - 1)

    sharpe = 0
    if returns:
        avg_ret = sum(returns) / len(returns)
        var = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std = var ** 0.5
        if std > 0:
            sharpe = avg_ret / std * (252 ** 0.5)

    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    ruin_count = sum(1 for e in equity_curve if e["equity"] < starting_balance * 0.1)

    final_eq = equity_curve[-1]["equity"] if equity_curve else starting_balance

    return {
        "label": preset["label"],
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "avg_win_pct": sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0,
        "avg_loss_pct": sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / starting_balance,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "final_equity": final_eq,
        "exit_reasons": reasons,
        "rejected": len(rejected),
        "ruin_steps": ruin_count,
        "analyzed": analyzed,
    }


def run_walk_forward(preset: Dict, markets: List[Dict], n_folds: int = 5,
                     starting_balance: float = 500.0) -> Dict:
    n = len(markets)
    fold_size = n // n_folds
    fold_results = []

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min(test_start + fold_size, n)
        test_markets = markets[test_start:test_end]

        result = run_preset_backtest(preset, test_markets, starting_balance=starting_balance, seed=42 + i)
        if result["total_trades"] > 0:
            fold_results.append(result)
            starting_balance = max(result["final_equity"], 50)

    if not fold_results:
        return {"label": preset["label"], "folds": 0, "total_trades": 0}

    total_trades = sum(f["total_trades"] for f in fold_results)
    total_wins = sum(f["wins"] for f in fold_results)
    total_pnl = sum(f["total_pnl"] for f in fold_results)
    avg_wr = total_wins / total_trades if total_trades > 0 else 0
    worst_dd = min(f["max_drawdown"] for f in fold_results)
    avg_sharpe = sum(f["sharpe_ratio"] for f in fold_results) / len(fold_results)
    final_eq = fold_results[-1]["final_equity"]

    all_reasons = {}
    for f in fold_results:
        for r, c in f.get("exit_reasons", {}).items():
            all_reasons[r] = all_reasons.get(r, 0) + c

    return {
        "label": preset["label"],
        "folds": len(fold_results),
        "total_trades": total_trades,
        "win_rate": avg_wr,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / 500.0,
        "max_drawdown": worst_dd,
        "sharpe_ratio": avg_sharpe,
        "final_equity": final_eq,
        "exit_reasons": all_reasons,
        "fold_details": [{"fold": i + 1, "trades": f["total_trades"], "wr": f["win_rate"],
                          "pnl": f["total_pnl"], "dd": f["max_drawdown"],
                          "final_eq": f["final_equity"]}
                         for i, f in enumerate(fold_results)],
    }


def main():
    logger.info("[AGGRESSIVE-BT] Loading markets...")
    markets = fetch_resolved_markets(max_markets=2000, force_refresh=False)
    logger.info(f"[AGGRESSIVE-BT] Loaded {len(markets)} markets")

    random.seed(42)
    random.shuffle(markets)

    results = {}
    for key, preset in PRESETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"[AGGRESSIVE-BT] Running: {preset['label']}")
        t0 = time.time()

        full = run_preset_backtest(preset, markets, starting_balance=500.0, seed=42)
        wf = run_walk_forward(preset, markets, n_folds=5, starting_balance=500.0)

        elapsed = time.time() - t0
        results[key] = {"full": full, "walk_forward": wf, "elapsed_sec": round(elapsed, 1)}

        logger.info(f"  Full:    {full['total_trades']} trades, WR={full['win_rate']:.1%}, "
                     f"PnL={full['total_pnl_pct']:.0%}, DD={full['max_drawdown']:.1%}, "
                     f"Sharpe={full['sharpe_ratio']:.2f}, Final=${full['final_equity']:,.0f}")
        logger.info(f"  WF:      {wf['total_trades']} trades, WR={wf['win_rate']:.1%}, "
                     f"PnL={wf['total_pnl_pct']:.0%}, DD={wf['max_drawdown']:.1%}, "
                     f"Final=${wf['final_equity']:,.0f}")
        logger.info(f"  Time:    {elapsed:.1f}s")

    print("\n" + "=" * 100)
    print(f"{'Strategy':<20} {'Trades':>7} {'WR':>7} {'PnL%':>8} {'Final$':>10} {'MaxDD':>8} {'Sharpe':>7} {'WF Final$':>10} {'WF DD':>8}")
    print("-" * 100)
    for key in ["baseline", "aggressive", "high_roller", "yolo"]:
        f = results[key]["full"]
        wf = results[key]["walk_forward"]
        print(f"{f['label']:<20} {f['total_trades']:>7} {f['win_rate']:>6.1%} "
              f"{f['total_pnl_pct']:>+7.0%} {f['final_equity']:>10,.0f} "
              f"{f['max_drawdown']:>7.1%} {f['sharpe_ratio']:>7.2f} "
              f"{wf['final_equity']:>10,.0f} {wf['max_drawdown']:>7.1%}")

    print("\n" + "=" * 100)
    print("EXIT REASONS BREAKDOWN:")
    print("-" * 100)
    for key in ["baseline", "aggressive", "high_roller", "yolo"]:
        f = results[key]["full"]
        reasons = " | ".join(f"{k}={v}" for k, v in sorted(f.get("exit_reasons", {}).items()))
        print(f"  {f['label']:<20} {reasons}")

    print("\n" + "=" * 100)
    print("WALK-FORWARD FOLD DETAILS:")
    print("-" * 100)
    for key in ["baseline", "aggressive", "high_roller", "yolo"]:
        wf = results[key]["walk_forward"]
        print(f"\n  {wf['label']}:")
        for fd in wf.get("fold_details", []):
            print(f"    Fold {fd['fold']}: {fd['trades']} trades, WR={fd['wr']:.1%}, "
                  f"PnL=${fd['pnl']:,.0f}, DD={fd['dd']:.1%}, Final=${fd['final_eq']:,.0f}")

    out_path = "/root/dotm-sniper/backtest_aggressive_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n[AGGRESSIVE-BT] Results saved to {out_path}")


if __name__ == "__main__":
    main()
