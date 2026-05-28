#!/usr/bin/env python3
"""
Standalone backtest: Conservative vs AggressiveMicro profiles
with Monte Carlo simulation. No imports from backtest_v2 needed.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

FEE_PCT = 0.02
MIN_ORDER_USD = 5.0
DOTM_MAX_PRICE = 0.30
MIN_VOLUME = 25000

HOURLY_VOL_BY_PRICE = {
    "lt03": 0.031,
    "lt07": 0.018,
    "lt15": 0.057,
}

CORRELATED_GROUPS = {
    "trump_admin": ["usa_politics", "russia_ukraine", "geopolitics", "venezuela"],
    "us_economic": ["fed_fomc", "usa_politics"],
    "sports": ["sports_nba", "sports_ufc", "sports"],
    "tech_ai": ["ai_tech", "tech"],
}

CATEGORY_ADJ = {
    "venezuela": 0.08,
    "russia_ukraine": 0.06,
    "geopolitics": 0.05,
    "usa_politics": 0.04,
    "fed_fomc": 0.04,
    "sports_ufc": 0.02,
    "other": 0.03,
    "sports_nba": -0.02,
    "crypto": -0.05,
}

PROFILE_A = {
    "name": "Conservative",
    "base_pct": 0.02,
    "kelly_fraction": 0.25,
    "max_pct": 0.10,
    "max_positions": 12,
    "max_cluster_pct": 0.30,
    "min_signal_score": 55,
    "min_confidence": 0.65,
    "min_prob_ratio": 3.0,
}

PROFILE_B = {
    "name": "AggressiveMicro",
    "base_pct": 0.04,
    "kelly_fraction": 0.50,
    "max_pct": 0.15,
    "max_positions": 6,
    "max_cluster_pct": 0.45,
    "min_signal_score": 65,
    "min_confidence": 0.70,
    "min_prob_ratio": 3.5,
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_data")
DATA_FILES = [
    "markets_v3_m500_2024-06-01_2026-06-01.json",
    "markets_v3_m100_2024-06-01_2026-06-01.json",
    "markets_v3_m50_2024-06-01_2026-06-01.json",
]


def load_markets() -> list:
    for fname in DATA_FILES:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = json.load(f)
            dotm = [
                m for m in raw
                if m.get("entry_price", 1.0) <= DOTM_MAX_PRICE
                and m.get("volume", 0) >= MIN_VOLUME
            ]
            dotm.sort(key=lambda m: m.get("created_at", ""))
            print(f"[DATA] Loaded {len(raw)} from {fname}, {len(dotm)} DOTM (price<={DOTM_MAX_PRICE}, vol>={MIN_VOLUME})")
            return dotm
    print("[DATA] No cached market files found. Exiting.")
    sys.exit(1)


def estimate_signal(market: dict) -> dict:
    entry_price = market["entry_price"]
    category = market.get("category", "other")
    volume = market.get("volume", 1000)
    ttl_days = market.get("ttl_days", 30)

    base_p = 0.05 + (0.30 - entry_price) * 0.5
    cat_adj = CATEGORY_ADJ.get(category, 0)
    vol_bonus = min(volume / 500000, 1.0) * 0.03
    ttl_bonus = min(ttl_days / 180, 1.0) * 0.04
    p_model = max(0.03, min(0.95, base_p + cat_adj + vol_bonus + ttl_bonus))

    prob_ratio = p_model / entry_price if entry_price > 0 else 0

    confidence = min(
        0.95,
        0.55
        + (p_model - entry_price) * 0.5
        + min(volume / 500000, 1.0) * 0.10,
    )

    ratio_score = min(prob_ratio / 3.0, 1.0) * 30
    supporting = max(1, int(p_model * 10))
    high_weight = max(0, supporting - 3)
    factor_score = min((supporting + high_weight) / 4, 1.0) * 20
    vol_score = min(volume / 500000, 1.0) * 20
    if ttl_days > 180:
        ttl_score = 20
    elif ttl_days > 90:
        ttl_score = 15
    elif ttl_days > 30:
        ttl_score = 12
    elif ttl_days > 14:
        ttl_score = 8
    elif ttl_days > 2:
        ttl_score = 5
    else:
        ttl_score = 0
    cluster_adj = {"other": 15, "sports_nba": -15}.get(category, 0)
    signal_score = ratio_score + factor_score + vol_score + ttl_score + cluster_adj

    return {
        "p_model": p_model,
        "prob_ratio": prob_ratio,
        "confidence": confidence,
        "signal_score": signal_score,
    }


def get_cluster_group(category: str) -> str:
    for group_name, members in CORRELATED_GROUPS.items():
        if category in members:
            return group_name
    return category


def generate_price_path(entry_price: float, resolution: str, ttl_days: int,
                        volume: float, num_steps: int, rng: np.random.Generator) -> list:
    if entry_price < 0.03:
        hourly_vol = HOURLY_VOL_BY_PRICE["lt03"]
    elif entry_price < 0.07:
        hourly_vol = HOURLY_VOL_BY_PRICE["lt07"]
    else:
        hourly_vol = HOURLY_VOL_BY_PRICE["lt15"]

    if volume < 2000:
        hourly_vol *= 1.3
    elif volume > 50000:
        hourly_vol *= 0.7

    steps_per_day = max(1, num_steps // max(ttl_days, 1))
    daily_vol = hourly_vol * (steps_per_day ** 0.5)

    resolution_price = 1.0 if resolution == "YES" else 0.0
    prices = [entry_price]
    dt = 1.0 / num_steps

    for step in range(1, num_steps + 1):
        progress = step / num_steps
        remaining = max(0.01, 1.0 - progress)
        mr_strength = 0.02 / remaining

        if resolution_price > 0.5:
            drift = mr_strength * max(0, resolution_price - prices[-1]) * 0.1
        else:
            drift = mr_strength * min(0, resolution_price - prices[-1]) * 0.1

        shock = rng.normal(0, daily_vol * dt ** 0.5)
        if rng.random() < 0.05:
            shock += rng.normal(0, daily_vol * 0.3)

        new_price = prices[-1] * (1 + drift + shock)
        new_price = max(0.001, min(0.99, new_price))
        prices.append(new_price)

    convergence_steps = min(3, num_steps)
    for i in range(max(0, num_steps - convergence_steps + 1), num_steps + 1):
        w = (i - (num_steps - convergence_steps + 1)) / convergence_steps
        prices[i] = prices[i] * (1 - w) + resolution_price * w

    return prices


def generate_order_book(entry_price: float, liquidity: float, rng: np.random.Generator) -> dict:
    if entry_price < 0.03:
        spread_pct = rng.uniform(0.30, 0.50)
        ask_base = max(5, liquidity * 0.01) * rng.uniform(0.5, 2.0)
        bid_base = ask_base * 0.4
    elif entry_price < 0.07:
        spread_pct = rng.uniform(0.15, 0.30)
        ask_base = max(10, liquidity * 0.02) * rng.uniform(0.5, 2.0)
        bid_base = ask_base * 0.5
    else:
        spread_pct = rng.uniform(0.08, 0.20)
        ask_base = max(20, liquidity * 0.03) * rng.uniform(0.5, 2.0)
        bid_base = ask_base * 0.6

    asks = []
    bids = []
    ask_start = entry_price * (1 + spread_pct * 0.5)
    bid_start = entry_price * (1 - spread_pct * 0.5)

    for i in range(5):
        ask_price = round(min(0.99, ask_start * (1 + 0.20 * i)), 4)
        ask_size = round(ask_base * (1.3 ** i) * rng.uniform(0.5, 1.5), 1)
        asks.append({"price": ask_price, "size": ask_size})

        bid_price = round(max(0.001, bid_start * (1 - 0.20 * i)), 4)
        if bid_price > 0:
            bid_size = round(bid_base * (1.3 ** i) * rng.uniform(0.5, 1.5), 1)
            bids.append({"price": bid_price, "size": bid_size})

    asks.sort(key=lambda x: x["price"])
    bids.sort(key=lambda x: -x["price"])
    return {"asks": asks, "bids": bids}


def walk_the_book(asks: list, amount_usd: float) -> tuple:
    remaining = amount_usd
    total_cost = 0.0
    total_shares = 0.0
    for level in asks:
        price = level["price"]
        size = level["size"]
        if price <= 0:
            continue
        level_cost = price * size
        fill_cost = min(remaining, level_cost)
        fill_shares = fill_cost / price
        total_cost += fill_cost
        total_shares += fill_shares
        remaining -= fill_cost
        if remaining <= 0.01:
            break
    if total_cost < MIN_ORDER_USD:
        return 0, 0, 0
    effective_price = total_cost / total_shares if total_shares > 0 else 0
    return effective_price, total_shares, total_cost


def walk_the_book_sell(bids: list, shares: float) -> tuple:
    remaining = shares
    total_proceeds = 0.0
    total_filled = 0.0
    for level in bids:
        price = level["price"]
        size = level["size"]
        if price <= 0:
            continue
        fill_shares = min(remaining, size)
        total_proceeds += fill_shares * price
        total_filled += fill_shares
        remaining -= fill_shares
        if remaining <= 0.01:
            break
    effective_price = total_proceeds / total_filled if total_filled > 0 else 0
    return effective_price, total_filled, total_proceeds


def simulate_buy(entry_price: float, amount_usd: float, liquidity: float,
                 rng: np.random.Generator) -> dict:
    if entry_price <= 0.001 or amount_usd < MIN_ORDER_USD:
        return {"filled": False, "reason": "price_too_low_or_amount_too_small"}

    book = generate_order_book(entry_price, liquidity, rng)
    asks = book["asks"]
    bids = book["bids"]

    if not asks:
        return {"filled": False, "reason": "no_asks"}

    best_ask = asks[0]["price"]
    best_bid = bids[0]["price"] if bids else 0
    if best_bid > 0 and best_ask > 0:
        spread = (best_ask - best_bid) / best_ask
        if spread > 0.50:
            return {"filled": False, "reason": f"spread={spread:.1%}"}

    eff_price, shares, cost = walk_the_book(asks, amount_usd)
    if eff_price == 0:
        return {"filled": False, "reason": "cannot_fill_min"}

    slippage = (eff_price - entry_price) / entry_price if entry_price > 0 else 0
    if slippage > 0.50:
        return {"filled": False, "reason": f"slippage={slippage:.1%}"}

    fee = round(cost * FEE_PCT, 4)
    return {
        "filled": True,
        "effective_price": eff_price,
        "shares": shares,
        "cost": cost,
        "fee": fee,
        "slippage_pct": slippage,
    }


def simulate_sell(current_price: float, shares: float, entry_price: float,
                  liquidity: float, rng: np.random.Generator,
                  force_market: bool = False) -> dict:
    if shares <= 0:
        return {"filled": False, "proceeds": 0, "reason": "no_shares"}

    book = generate_order_book(current_price, liquidity, rng)
    bids = book["bids"]

    if not bids or bids[0]["price"] <= 0:
        if force_market:
            proceeds = shares * current_price * 0.5
            fee = proceeds * FEE_PCT
            return {"filled": True, "proceeds": proceeds, "fee": fee,
                    "effective_price": current_price * 0.5}
        return {"filled": False, "proceeds": 0, "reason": "no_bids"}

    if not force_market:
        best_bid = bids[0]["price"]
        asks = book["asks"]
        best_ask = asks[0]["price"] if asks else 0
        if best_ask > 0:
            spread = (best_ask - best_bid) / best_ask
            if spread > 0.50:
                return {"filled": False, "proceeds": 0, "reason": "spread_too_wide"}
        if entry_price > 0 and best_bid < entry_price * 0.70:
            return {"filled": False, "proceeds": 0, "reason": "bid_below_entry"}

    eff_price, filled, proceeds = walk_the_book_sell(bids, shares)
    if filled <= 0:
        if force_market:
            proceeds = shares * current_price * 0.5
            fee = proceeds * FEE_PCT
            return {"filled": True, "proceeds": proceeds, "fee": fee,
                    "effective_price": current_price * 0.5}
        return {"filled": False, "proceeds": 0, "reason": "no_fill"}

    fee = round(proceeds * FEE_PCT, 4)
    return {"filled": True, "proceeds": proceeds, "fee": fee,
            "effective_price": eff_price, "shares_filled": filled}


def simulate_profile(markets: list, profile: dict, starting_balance: float = 1500.0,
                     seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    balance = starting_balance
    positions = {}
    trades = []
    equity_curve = []
    cluster_exposure = {}
    rejection_reasons = {}
    signals_passed = 0

    for market in markets:
        slug = market["slug"]
        entry_price = market["entry_price"]
        liquidity = market.get("liquidity", 100)
        category = market.get("category", "other")
        volume = market.get("volume", 1000)
        ttl_days = market.get("ttl_days", 30)
        resolution = market.get("resolution", "NO")

        if balance < 5:
            break

        signal = estimate_signal(market)

        if signal["signal_score"] < profile["min_signal_score"]:
            rejection_reasons["low_signal"] = rejection_reasons.get("low_signal", 0) + 1
            continue
        if signal["confidence"] < profile["min_confidence"]:
            rejection_reasons["low_confidence"] = rejection_reasons.get("low_confidence", 0) + 1
            continue
        if signal["prob_ratio"] < profile["min_prob_ratio"]:
            rejection_reasons["low_prob_ratio"] = rejection_reasons.get("low_prob_ratio", 0) + 1
            continue

        signals_passed += 1

        b = (1 - entry_price) / entry_price
        p = signal["p_model"]
        q = 1 - p
        kelly_full = (b * p - q) / b
        if kelly_full <= 0:
            rejection_reasons["negative_kelly"] = rejection_reasons.get("negative_kelly", 0) + 1
            continue

        kelly_dollars = balance * profile["base_pct"] * profile["kelly_fraction"] * signal["confidence"]
        if kelly_dollars < MIN_ORDER_USD:
            rejection_reasons["below_min_size"] = rejection_reasons.get("below_min_size", 0) + 1
            continue
        kelly_dollars = min(kelly_dollars, balance * profile["max_pct"])

        if len(positions) >= profile["max_positions"]:
            rejection_reasons["max_positions"] = rejection_reasons.get("max_positions", 0) + 1
            continue

        cluster_group = get_cluster_group(category)
        total_equity = balance + sum(p_d["cost"] for p_d in positions.values())
        new_cluster_exp = cluster_exposure.get(cluster_group, 0) + kelly_dollars
        if total_equity > 0 and new_cluster_exp / total_equity > profile["max_cluster_pct"]:
            rejection_reasons["cluster_limit"] = rejection_reasons.get("cluster_limit", 0) + 1
            continue

        buy_result = simulate_buy(entry_price, kelly_dollars, liquidity, rng)
        if not buy_result["filled"]:
            rejection_reasons["buy_failed"] = rejection_reasons.get("buy_failed", 0) + 1
            continue

        fee = buy_result["fee"]
        total_cost = buy_result["cost"] + fee
        if total_cost > balance:
            rejection_reasons["insufficient_balance"] = rejection_reasons.get("insufficient_balance", 0) + 1
            continue

        balance -= total_cost

        positions[slug] = {
            "entry_price": buy_result["effective_price"],
            "shares": buy_result["shares"],
            "cost": total_cost,
            "liquidity": liquidity,
            "cluster": cluster_group,
            "category": category,
            "p_model": signal["p_model"],
            "volume": volume,
            "ttl_days": ttl_days,
            "resolution": resolution,
            "high_price": buy_result["effective_price"],
            "trailing_on": False,
            "stop_loss": 0.0,
            "shares_held": buy_result["shares"],
            "tp_rungs_executed": [],
        }
        cluster_exposure[cluster_group] = cluster_exposure.get(cluster_group, 0) + total_cost

    for slug, pos in positions.items():
        num_steps = max(30, min(120, pos["ttl_days"] * 4))
        path = generate_price_path(
            pos["entry_price"], pos["resolution"], pos["ttl_days"],
            pos["volume"], num_steps, rng
        )
        pos["_price_path"] = path

    for step_idx in range(121):
        current_prices = {}
        for slug, pos in positions.items():
            path = pos.get("_price_path", [pos["entry_price"]])
            idx = min(step_idx, len(path) - 1)
            current_prices[slug] = path[idx]

        for slug in list(positions.keys()):
            pos = positions[slug]
            price = current_prices.get(slug, 0)
            if price <= 0:
                continue

            if price > pos["high_price"]:
                pos["high_price"] = price

            entry = pos["entry_price"]
            pnl_pct = (price - entry) / entry if entry > 0 else 0
            sold = False
            sell_shares = pos["shares_held"]

            if pnl_pct <= -0.30:
                sell_r = simulate_sell(price, sell_shares, entry, pos["liquidity"], rng, force_market=True)
                if sell_r["filled"]:
                    net = sell_r["proceeds"] - sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                    balance += net
                    pnl_abs = net - pos["cost"]
                    trades.append({
                        "slug": slug, "entry_price": entry, "exit_price": price,
                        "pnl_pct": pnl_abs / pos["cost"] if pos["cost"] > 0 else 0,
                        "pnl_abs": pnl_abs, "reason": "hard_stop",
                    })
                    cluster_exposure[pos["cluster"]] = max(0, cluster_exposure.get(pos["cluster"], 0) - pos["cost"])
                    del positions[slug]
                    sold = True

            if not sold and pos["high_price"] > entry * 1.30:
                pos["trailing_on"] = True
                pos["stop_loss"] = pos["high_price"] * 0.75

            if not sold and pos["trailing_on"] and price <= pos["stop_loss"]:
                sell_r = simulate_sell(price, sell_shares, entry, pos["liquidity"], rng, force_market=True)
                if sell_r["filled"]:
                    net = sell_r["proceeds"] - sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                    balance += net
                    pnl_abs = net - pos["cost"]
                    trades.append({
                        "slug": slug, "entry_price": entry, "exit_price": price,
                        "pnl_pct": pnl_abs / pos["cost"] if pos["cost"] > 0 else 0,
                        "pnl_abs": pnl_abs, "reason": "trailing_stop",
                    })
                    cluster_exposure[pos["cluster"]] = max(0, cluster_exposure.get(pos["cluster"], 0) - pos["cost"])
                    del positions[slug]
                    sold = True

            original_shares = pos["shares"]
            if not sold and len(pos["tp_rungs_executed"]) == 0 and price >= 0.75:
                tp_shares_50 = original_shares * 0.50
                if tp_shares_50 > 0 and tp_shares_50 <= pos["shares_held"]:
                    sell_r = simulate_sell(price, tp_shares_50, entry, pos["liquidity"], rng)
                    if sell_r["filled"]:
                        net = sell_r["proceeds"] - sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                        balance += net
                        pos["shares_held"] -= tp_shares_50
                        pos["tp_rungs_executed"].append(0.75)

            if not sold and len(pos.get("tp_rungs_executed", [])) == 1 and price >= 0.85:
                tp_shares_30 = original_shares * 0.30
                if tp_shares_30 > 0 and tp_shares_30 <= pos["shares_held"]:
                    sell_r = simulate_sell(price, tp_shares_30, entry, pos["liquidity"], rng)
                    if sell_r["filled"]:
                        net = sell_r["proceeds"] - sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                        balance += net
                        pos["shares_held"] -= tp_shares_30
                        pos["tp_rungs_executed"].append(0.85)

            if not sold and pnl_pct >= 1.50 and pos["shares_held"] > 0:
                sell_portion = pos["shares_held"] * 0.40
                sell_r = simulate_sell(price, sell_portion, entry, pos["liquidity"], rng)
                if sell_r["filled"]:
                    net = sell_r["proceeds"] - sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                    balance += net
                    pos["shares_held"] -= sell_portion

            if not sold and pnl_pct >= 3.00 and pos["shares_held"] > 0:
                sell_portion = pos["shares_held"] * 0.35
                sell_r = simulate_sell(price, sell_portion, entry, pos["liquidity"], rng)
                if sell_r["filled"]:
                    net = sell_r["proceeds"] - sell_r.get("fee", sell_r["proceeds"] * FEE_PCT)
                    balance += net
                    pos["shares_held"] -= sell_portion

        eq = balance + sum(
            p["shares_held"] * current_prices.get(s, 0) for s, p in positions.items()
        )
        peak = max((e["equity"] for e in equity_curve), default=eq)
        dd = (eq - peak) / peak if peak > 0 else 0
        equity_curve.append({
            "step": step_idx, "equity": eq, "drawdown": dd,
            "balance": balance, "open_positions": len(positions),
        })

    for slug in list(positions.keys()):
        pos = positions[slug]
        path = pos.get("_price_path", [pos["entry_price"]])
        final_price = path[-1]
        sell_shares = pos["shares_held"]
        resolution_price = 1.0 if pos["resolution"] == "YES" else 0.0

        if sell_shares > 0:
            if resolution_price >= 0.5:
                proceeds = sell_shares * min(final_price, 0.99)
            else:
                proceeds = sell_shares * max(final_price * 0.1, 0.001)
            fee = proceeds * FEE_PCT
            net = proceeds - fee
            balance += net
            pnl_abs = net - pos["cost"]
            trades.append({
                "slug": slug, "entry_price": pos["entry_price"], "exit_price": final_price,
                "pnl_pct": pnl_abs / pos["cost"] if pos["cost"] > 0 else 0,
                "pnl_abs": pnl_abs, "reason": "resolution",
            })

    if not trades:
        return {
            "total_trades": 0, "signals_passed": signals_passed,
            "final_equity": starting_balance, "total_pnl": 0,
            "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0, "equity_curve": equity_curve,
            "trades": trades, "rejection_reasons": rejection_reasons,
        }

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

    sharpe = 0.0
    if returns:
        avg_ret = sum(returns) / len(returns)
        var = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std = var ** 0.5
        if std > 0:
            sharpe = avg_ret / std * (252 ** 0.5)

    exit_reasons = {}
    for t in trades:
        exit_reasons[t["reason"]] = exit_reasons.get(t["reason"], 0) + 1

    return {
        "name": profile["name"],
        "total_trades": len(trades),
        "signals_passed": signals_passed,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "avg_win_pct": sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0,
        "avg_loss_pct": sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0,
        "total_pnl": total_pnl,
        "final_equity": equity_curve[-1]["equity"] if equity_curve else starting_balance,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "exit_reasons": exit_reasons,
        "equity_curve": [{"step": e["step"], "equity": round(e["equity"], 2),
                          "drawdown": round(e["drawdown"], 4)} for e in equity_curve],
        "trades": trades,
        "rejection_reasons": rejection_reasons,
    }


def monte_carlo_simulation(markets: list, profile: dict, n_simulations: int = 10000,
                           starting_balance: float = 1500.0) -> dict:
    final_balances = []
    max_drawdowns = []
    months_to_5k_list = []
    months_to_10k_list = []
    ruin_count = 0
    sample_paths = []
    total_trades_list = []
    sample_interval = max(1, n_simulations // 100)

    for sim_idx in range(n_simulations):
        if sim_idx % 500 == 0 and sim_idx > 0:
            pct = sim_idx / n_simulations * 100
            print(f"  MC {profile['name']}: {pct:.0f}% ({sim_idx}/{n_simulations})", end="\r")

        sim_seed = 42 + sim_idx * 7
        rng = np.random.default_rng(sim_seed)

        order = rng.permutation(len(markets)).tolist()
        shuffled = [markets[i] for i in order]

        noisy = []
        for m in shuffled:
            nm = dict(m)
            price_noise = 1.0 + rng.uniform(-0.10, 0.10)
            nm["entry_price"] = max(0.005, min(0.29, m["entry_price"] * price_noise))
            vol_noise = 1.0 + rng.uniform(-0.15, 0.15)
            nm["volume"] = max(100, m.get("volume", 1000) * vol_noise)
            noisy.append(nm)

        result = simulate_profile(noisy, profile, starting_balance=starting_balance, seed=sim_seed)

        eq = result["final_equity"]
        final_balances.append(eq)
        max_drawdowns.append(result["max_drawdown"])
        total_trades_list.append(result["total_trades"])

        if sim_idx % sample_interval == 0:
            sample_paths.append([e["equity"] for e in result["equity_curve"]])

        equity_curve = result["equity_curve"]
        total_steps = len(equity_curve)
        m5k = None
        m10k = None
        if total_steps > 0:
            for i, e in enumerate(equity_curve):
                if m5k is None and e["equity"] >= 5000:
                    m5k = i / total_steps * 24.0
                if m10k is None and e["equity"] >= 10000:
                    m10k = i / total_steps * 24.0
        months_to_5k_list.append(m5k)
        months_to_10k_list.append(m10k)

        if any(e["equity"] < 500 for e in equity_curve):
            ruin_count += 1

    print(f"  MC {profile['name']}: 100% ({n_simulations}/{n_simulations})   ")

    max_len = max((len(p) for p in sample_paths), default=0)
    fan_p5 = []
    fan_p25 = []
    fan_p50 = []
    fan_p75 = []
    fan_p95 = []
    for step in range(max_len):
        values = [p[step] for p in sample_paths if step < len(p)]
        if values:
            fan_p5.append(float(np.percentile(values, 5)))
            fan_p25.append(float(np.percentile(values, 25)))
            fan_p50.append(float(np.percentile(values, 50)))
            fan_p75.append(float(np.percentile(values, 75)))
            fan_p95.append(float(np.percentile(values, 95)))

    arr_fb = np.array(final_balances)
    arr_dd = np.array(max_drawdowns)
    m5k_arr = np.array([x for x in months_to_5k_list if x is not None])
    m10k_arr = np.array([x for x in months_to_10k_list if x is not None])

    prob_5k_12mo = sum(1 for x in months_to_5k_list if x is not None and x <= 12) / n_simulations
    prob_10k_24mo = sum(1 for x in months_to_10k_list if x is not None and x <= 24) / n_simulations

    return {
        "n_simulations": n_simulations,
        "mean_final_equity": float(np.mean(arr_fb)),
        "median_final_equity": float(np.median(arr_fb)),
        "p5_final_equity": float(np.percentile(arr_fb, 5)),
        "p25_final_equity": float(np.percentile(arr_fb, 25)),
        "p50_final_equity": float(np.percentile(arr_fb, 50)),
        "p75_final_equity": float(np.percentile(arr_fb, 75)),
        "p95_final_equity": float(np.percentile(arr_fb, 95)),
        "p50_final_balance": float(np.percentile(arr_fb, 50)),
        "mean_max_drawdown": float(np.mean(arr_dd)),
        "median_max_drawdown": float(np.median(arr_dd)),
        "p5_max_drawdown": float(np.percentile(arr_dd, 5)),
        "p25_max_drawdown": float(np.percentile(arr_dd, 25)),
        "p50_max_drawdown": float(np.percentile(arr_dd, 50)),
        "p75_max_drawdown": float(np.percentile(arr_dd, 75)),
        "p95_max_drawdown": float(np.percentile(arr_dd, 95)),
        "p50_months_to_5k": float(np.median(m5k_arr)) if len(m5k_arr) > 0 else None,
        "p50_months_to_10k": float(np.median(m10k_arr)) if len(m10k_arr) > 0 else None,
        "probability_of_ruin": ruin_count / n_simulations,
        "probability_of_5k_12mo": prob_5k_12mo,
        "probability_of_10k_24mo": prob_10k_24mo,
        "ruin_probability": ruin_count / n_simulations,
        "mean_trades": float(np.mean(total_trades_list)) if total_trades_list else 0,
        "fan_chart": {
            "p5": fan_p5,
            "p25": fan_p25,
            "p50": fan_p50,
            "p75": fan_p75,
            "p95": fan_p95,
        },
        "final_balances": final_balances,
    }


def print_report(res_a: dict, res_b: dict, mc_a: dict, mc_b: dict,
                 n_markets: int, n_sims: int):
    name_a = PROFILE_A["name"]
    name_b = PROFILE_B["name"]
    col1 = 25
    col2 = 14
    col3 = 17

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
        print(f"║ {label:<{col1}} │ {sa:>{col2}} │ {sb:>{col3}} ║")

    def divider():
        inner = col1 + 1 + col2 + 3 + col3 + 1
        print(f"║ {'─' * col1}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 1)}║")

    w = col1 + col2 + col3 + 10
    print()
    print("╔" + "═" * w + "╗")
    title = f"BACKTEST COMPARISON: {name_a.upper()} vs {name_b.upper()}"
    print(f"║{title:^{w}}║")
    print("╠" + "═" * w + "╣")

    row("Metric", name_a, name_b)
    divider()
    row("Markets analyzed", n_markets, n_markets)
    row("Signals passed filter", res_a.get("signals_passed", 0), res_b.get("signals_passed", 0))
    row("Trades executed", res_a.get("total_trades", 0), res_b.get("total_trades", 0))
    row("Win rate", res_a.get("win_rate", 0), res_b.get("win_rate", 0), fmt="pct_plain")
    row("Avg win", res_a.get("avg_win_pct", 0), res_b.get("avg_win_pct", 0), fmt="pct")
    row("Avg loss", res_a.get("avg_loss_pct", 0), res_b.get("avg_loss_pct", 0), fmt="pct")
    row("Total P&L", res_a.get("total_pnl", 0), res_b.get("total_pnl", 0), fmt="dollar")
    row("Final equity", res_a.get("final_equity", 0), res_b.get("final_equity", 0), fmt="dollar")
    row("Max drawdown", res_a.get("max_drawdown", 0), res_b.get("max_drawdown", 0), fmt="pct_plain")
    row("Sharpe ratio", res_a.get("sharpe_ratio", 0), res_b.get("sharpe_ratio", 0), fmt="f2")
    divider()
    mc_label = f"MONTE CARLO ({n_sims} sims)"
    row(mc_label, "", "")
    divider()
    row("Mean final equity", mc_a.get("mean_final_equity"), mc_b.get("mean_final_equity"), fmt="dollar")
    row("Median final equity", mc_a.get("median_final_equity"), mc_b.get("median_final_equity"), fmt="dollar")
    row("5th percentile", mc_a.get("p5_final_equity"), mc_b.get("p5_final_equity"), fmt="dollar")
    row("25th percentile", mc_a.get("p25_final_equity"), mc_b.get("p25_final_equity"), fmt="dollar")
    row("50th percentile", mc_a.get("p50_final_equity"), mc_b.get("p50_final_equity"), fmt="dollar")
    row("75th percentile", mc_a.get("p75_final_equity"), mc_b.get("p75_final_equity"), fmt="dollar")
    row("95th percentile", mc_a.get("p95_final_equity"), mc_b.get("p95_final_equity"), fmt="dollar")
    row("P(ruin < $100)", mc_a.get("ruin_probability", 0), mc_b.get("ruin_probability", 0), fmt="pct_plain")
    row("P(ruin < $500)", mc_a.get("probability_of_ruin", 0), mc_b.get("probability_of_ruin", 0), fmt="pct_plain")
    row("Mean max drawdown", mc_a.get("mean_max_drawdown", 0), mc_b.get("mean_max_drawdown", 0), fmt="pct_plain")
    divider()
    m5k_a = mc_a.get("p50_months_to_5k")
    m5k_b = mc_b.get("p50_months_to_5k")
    m10k_a = mc_a.get("p50_months_to_10k")
    m10k_b = mc_b.get("p50_months_to_10k")
    row("P50 months to $5K", f"{m5k_a:.1f}" if m5k_a else "N/A", f"{m5k_b:.1f}" if m5k_b else "N/A")
    row("P50 months to $10K", f"{m10k_a:.1f}" if m10k_a else "N/A", f"{m10k_b:.1f}" if m10k_b else "N/A")
    row("P($5K within 12mo)", mc_a.get("probability_of_5k_12mo", 0), mc_b.get("probability_of_5k_12mo", 0), fmt="pct_plain")
    row("P($10K within 24mo)", mc_a.get("probability_of_10k_24mo", 0), mc_b.get("probability_of_10k_24mo", 0), fmt="pct_plain")
    print("╚" + "═" * w + "╝")
    print()


def generate_html_report(res_a, res_b, mc_a, mc_b, markets, seed, starting_balance=1500.0):
    eq_a = [e["equity"] for e in res_a.get("equity_curve", [])]
    eq_b = [e["equity"] for e in res_b.get("equity_curve", [])]
    dd_a = [e["drawdown"] for e in res_a.get("equity_curve", [])]
    dd_b = [e["drawdown"] for e in res_b.get("equity_curve", [])]
    steps_a = list(range(len(eq_a)))
    steps_b = list(range(len(eq_b)))

    peak_a = []
    cp = 0
    for e in eq_a:
        cp = max(cp, e)
        peak_a.append(cp)
    peak_b = []
    cp = 0
    for e in eq_b:
        cp = max(cp, e)
        peak_b.append(cp)

    trades_a = res_a.get("trades", [])
    trades_b = res_b.get("trades", [])

    def make_trade_markers(trades, equity):
        wx, wy, lx, ly = [], [], [], []
        if not trades or not equity:
            return wx, wy, lx, ly
        n = len(equity)
        for i, t in enumerate(trades):
            x = int(i * (n - 1) / max(len(trades) - 1, 1))
            x = min(x, n - 1)
            y = equity[x]
            if t["pnl_abs"] > 0:
                wx.append(x)
                wy.append(y)
            else:
                lx.append(x)
                ly.append(y)
        return wx, wy, lx, ly

    wa_x, wa_y, la_x, la_y = make_trade_markers(trades_a, eq_a)
    wb_x, wb_y, lb_x, lb_y = make_trade_markers(trades_b, eq_b)

    signal_map = {}
    for m in markets:
        sig = estimate_signal(m)
        signal_map[m.get("slug", "")] = sig["signal_score"]

    scores_a = []
    scores_b = []
    for m in markets:
        sig = estimate_signal(m)
        if (sig["signal_score"] >= PROFILE_A["min_signal_score"]
                and sig["confidence"] >= PROFILE_A["min_confidence"]
                and sig["prob_ratio"] >= PROFILE_A["min_prob_ratio"]):
            scores_a.append(round(sig["signal_score"], 1))
        if (sig["signal_score"] >= PROFILE_B["min_signal_score"]
                and sig["confidence"] >= PROFILE_B["min_confidence"]
                and sig["prob_ratio"] >= PROFILE_B["min_prob_ratio"]):
            scores_b.append(round(sig["signal_score"], 1))

    buckets = [(55, 60), (60, 65), (65, 70), (70, 75), (75, 200)]
    bucket_labels = ["55-60", "60-65", "65-70", "70-75", "75+"]

    def compute_bucket_winrate(trades):
        results = []
        for low, high in buckets:
            bt = [t for t in trades if t.get("slug") in signal_map
                  and low <= signal_map[t["slug"]] < high]
            wins = sum(1 for t in bt if t["pnl_abs"] > 0)
            total = len(bt)
            results.append(round(wins / total, 3) if total > 0 else 0)
        return results

    winrate_a = compute_bucket_winrate(trades_a)
    winrate_b = compute_bucket_winrate(trades_b)

    def profit_factor(trades):
        gw = sum(t["pnl_abs"] for t in trades if t["pnl_abs"] > 0)
        gl = abs(sum(t["pnl_abs"] for t in trades if t["pnl_abs"] < 0))
        return gw / gl if gl > 0 else 999.99

    pf_a = profit_factor(trades_a)
    pf_b = profit_factor(trades_b)

    def cagr_val(final, start, months=24):
        if start <= 0 or final <= 0:
            return 0.0
        return (final / start) ** (12.0 / months) - 1

    cagr_a = cagr_val(res_a.get("final_equity", starting_balance), starting_balance)
    cagr_b = cagr_val(res_b.get("final_equity", starting_balance), starting_balance)

    def analyze_drawdown(equity_curve, trades):
        max_dd_depth = 0.0
        max_dd_duration = 0
        cur_dd_dur = 0
        pk = 0.0
        for e in equity_curve:
            eq = e["equity"]
            if eq > pk:
                pk = eq
                cur_dd_dur = 0
            else:
                cur_dd_dur += 1
                dd = (eq - pk) / pk if pk > 0 else 0
                if dd < max_dd_depth:
                    max_dd_depth = dd
            if cur_dd_dur > max_dd_duration:
                max_dd_duration = cur_dd_dur
        max_cl = 0
        cur_cl = 0
        for t in trades:
            if t["pnl_abs"] <= 0:
                cur_cl += 1
                if cur_cl > max_cl:
                    max_cl = cur_cl
            else:
                cur_cl = 0
        recovery_steps = None
        pk_at_dd = 0.0
        dd_step = 0
        for i, e in enumerate(equity_curve):
            if e["equity"] > pk_at_dd:
                pk_at_dd = e["equity"]
            dd = (e["equity"] - pk_at_dd) / pk_at_dd if pk_at_dd > 0 else 0
            if dd <= max_dd_depth and dd_step == 0 and pk_at_dd > starting_balance:
                dd_step = i
        if dd_step > 0:
            for i in range(dd_step, len(equity_curve)):
                if equity_curve[i]["equity"] >= pk_at_dd:
                    recovery_steps = i - dd_step
                    break
        return max_dd_depth, max_dd_duration, max_cl, recovery_steps

    ec_a = res_a.get("equity_curve", [])
    ec_b = res_b.get("equity_curve", [])
    dd_depth_a, dd_dur_a, consec_a, recovery_a = analyze_drawdown(ec_a, trades_a)
    dd_depth_b, dd_dur_b, consec_b, recovery_b = analyze_drawdown(ec_b, trades_b)

    verdict = "KEEP Conservative \u2014 AggressiveMicro does not improve risk-adjusted returns"
    verdict_class = "keep"
    if mc_b:
        if mc_b.get("probability_of_ruin", 0) > 0.15:
            verdict = "REJECT AggressiveMicro \u2014 ruin risk too high"
            verdict_class = "reject"
        elif (mc_b.get("p50_final_balance", 0) > mc_a.get("p50_final_balance", 0) * 1.2
              and mc_b.get("mean_max_drawdown", 0) > -0.30):
            verdict = "ADOPT AggressiveMicro \u2014 significantly higher returns with acceptable risk"
            verdict_class = "adopt"
        elif mc_b.get("p50_final_balance", 0) > mc_a.get("p50_final_balance", 0):
            verdict = "CONSIDER AggressiveMicro \u2014 moderate improvement, verify with live trading"
            verdict_class = "consider"

    equity_traces = [
        {"x": steps_a, "y": peak_a, "line": {"color": "rgba(0,0,255,0)"}, "showlegend": False, "hoverinfo": "skip"},
        {"x": steps_a, "y": eq_a, "fill": "tonexty", "fillcolor": "rgba(255,0,0,0.08)",
         "line": {"color": "blue", "width": 2}, "name": "Conservative"},
        {"x": steps_b, "y": peak_b, "line": {"color": "rgba(255,0,0,0)"}, "showlegend": False, "xaxis": "x2", "hoverinfo": "skip"},
        {"x": steps_b, "y": eq_b, "fill": "tonexty", "fillcolor": "rgba(255,100,100,0.08)",
         "line": {"color": "red", "width": 2}, "name": "AggressiveMicro", "xaxis": "x2"},
        {"x": wa_x, "y": wa_y, "mode": "markers", "marker": {"color": "green", "size": 8, "symbol": "triangle-up"},
         "name": "Win (Conservative)", "showlegend": True},
        {"x": la_x, "y": la_y, "mode": "markers", "marker": {"color": "rgba(0,150,0,0.4)", "size": 7, "symbol": "x"},
         "name": "Loss (Conservative)", "showlegend": True},
        {"x": wb_x, "y": wb_y, "mode": "markers", "marker": {"color": "limegreen", "size": 8, "symbol": "triangle-up"},
         "name": "Win (AggressiveMicro)", "xaxis": "x2"},
        {"x": lb_x, "y": lb_y, "mode": "markers", "marker": {"color": "rgba(200,0,0,0.4)", "size": 7, "symbol": "x"},
         "name": "Loss (AggressiveMicro)", "xaxis": "x2"},
    ]
    equity_layout = {
        "title": "Equity Curves with Drawdown Shading",
        "xaxis": {"title": "Step", "domain": [0, 1]},
        "xaxis2": {"title": "Step", "overlaying": "x", "side": "top"},
        "yaxis": {"title": "Equity ($)"},
        "legend": {"orientation": "h"},
    }

    js_lines = []
    js_lines.append("Plotly.newPlot('equity-chart', %s, %s);" % (json.dumps(equity_traces), json.dumps(equity_layout)))

    mc_sections_html = ""
    if mc_a and mc_b and mc_a.get("fan_chart"):
        fan_a = mc_a["fan_chart"]
        fan_b = mc_b["fan_chart"]
        fan_steps_a = list(range(len(fan_a["p5"])))
        fan_steps_b = list(range(len(fan_b["p5"])))

        fan_a_traces = [
            {"x": fan_steps_a, "y": fan_a["p95"], "line": {"color": "rgba(0,0,255,0.3)"}, "name": "P95"},
            {"x": fan_steps_a, "y": fan_a["p75"], "fill": "tonexty", "fillcolor": "rgba(0,0,255,0.1)",
             "line": {"color": "rgba(0,0,255,0.3)"}, "name": "P75"},
            {"x": fan_steps_a, "y": fan_a["p50"], "fill": "tonexty", "fillcolor": "rgba(0,0,255,0.15)",
             "line": {"color": "blue", "width": 2}, "name": "P50 (Median)"},
            {"x": fan_steps_a, "y": fan_a["p25"], "fill": "tonexty", "fillcolor": "rgba(0,0,255,0.15)",
             "line": {"color": "rgba(0,0,255,0.3)"}, "name": "P25"},
            {"x": fan_steps_a, "y": fan_a["p5"], "fill": "tonexty", "fillcolor": "rgba(0,0,255,0.1)",
             "line": {"color": "rgba(0,0,255,0.3)"}, "name": "P5"},
        ]
        fan_a_layout = {"title": "Monte Carlo Fan Chart \u2014 Conservative (10,000 sims)", "yaxis": {"title": "Equity ($)"}, "xaxis": {"title": "Step"}}

        fan_b_traces = [
            {"x": fan_steps_b, "y": fan_b["p95"], "line": {"color": "rgba(255,0,0,0.3)"}, "name": "P95"},
            {"x": fan_steps_b, "y": fan_b["p75"], "fill": "tonexty", "fillcolor": "rgba(255,0,0,0.1)",
             "line": {"color": "rgba(255,0,0,0.3)"}, "name": "P75"},
            {"x": fan_steps_b, "y": fan_b["p50"], "fill": "tonexty", "fillcolor": "rgba(255,0,0,0.15)",
             "line": {"color": "red", "width": 2}, "name": "P50 (Median)"},
            {"x": fan_steps_b, "y": fan_b["p25"], "fill": "tonexty", "fillcolor": "rgba(255,0,0,0.15)",
             "line": {"color": "rgba(255,0,0,0.3)"}, "name": "P25"},
            {"x": fan_steps_b, "y": fan_b["p5"], "fill": "tonexty", "fillcolor": "rgba(255,0,0,0.1)",
             "line": {"color": "rgba(255,0,0,0.3)"}, "name": "P5"},
        ]
        fan_b_layout = {"title": "Monte Carlo Fan Chart \u2014 AggressiveMicro (10,000 sims)", "yaxis": {"title": "Equity ($)"}, "xaxis": {"title": "Step"}}

        js_lines.append("Plotly.newPlot('mc-fan-a', %s, %s);" % (json.dumps(fan_a_traces), json.dumps(fan_a_layout)))
        js_lines.append("Plotly.newPlot('mc-fan-b', %s, %s);" % (json.dumps(fan_b_traces), json.dumps(fan_b_layout)))

        fb_a = mc_a.get("final_balances", [])
        fb_b = mc_b.get("final_balances", [])
        hist_traces = [
            {"x": fb_a, "type": "histogram", "name": "Conservative", "opacity": 0.7, "marker": {"color": "blue"}},
            {"x": fb_b, "type": "histogram", "name": "AggressiveMicro", "opacity": 0.7, "marker": {"color": "red"}},
        ]
        hist_shapes = []
        if fb_a:
            hist_shapes.extend([
                {"type": "line", "x0": mc_a.get("p5_final_equity", 0), "x1": mc_a.get("p5_final_equity", 0),
                 "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "blue", "width": 2, "dash": "dash"}},
                {"type": "line", "x0": mc_a.get("p50_final_equity", 0), "x1": mc_a.get("p50_final_equity", 0),
                 "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "blue", "width": 3}},
                {"type": "line", "x0": mc_a.get("p95_final_equity", 0), "x1": mc_a.get("p95_final_equity", 0),
                 "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "blue", "width": 2, "dash": "dash"}},
            ])
        if fb_b:
            hist_shapes.extend([
                {"type": "line", "x0": mc_b.get("p5_final_equity", 0), "x1": mc_b.get("p5_final_equity", 0),
                 "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "red", "width": 2, "dash": "dash"}},
                {"type": "line", "x0": mc_b.get("p50_final_equity", 0), "x1": mc_b.get("p50_final_equity", 0),
                 "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "red", "width": 3}},
                {"type": "line", "x0": mc_b.get("p95_final_equity", 0), "x1": mc_b.get("p95_final_equity", 0),
                 "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "red", "width": 2, "dash": "dash"}},
            ])
        hist_layout = {
            "title": "Distribution of Final Balances",
            "xaxis": {"title": "Final Balance ($)"},
            "yaxis": {"title": "Count"},
            "barmode": "overlay",
            "shapes": hist_shapes,
        }
        js_lines.append("Plotly.newPlot('mc-hist', %s, %s);" % (json.dumps(hist_traces), json.dumps(hist_layout)))

        m5k_a_val = mc_a.get("p50_months_to_5k")
        m5k_b_val = mc_b.get("p50_months_to_5k")
        m10k_a_val = mc_a.get("p50_months_to_10k")
        m10k_b_val = mc_b.get("p50_months_to_10k")

        mc_sections_html = (
            '<div class="section"><h2>3. Monte Carlo Results</h2>'
            '<h3>Conservative \u2014 Fan Chart</h3>'
            '<div id="mc-fan-a" style="width:100%;height:500px;"></div>'
            '<h3>AggressiveMicro \u2014 Fan Chart</h3>'
            '<div id="mc-fan-b" style="width:100%;height:500px;"></div>'
            '<h3>Final Balance Distribution</h3>'
            '<div id="mc-hist" style="width:100%;height:500px;"></div>'
            '<h3>Risk Metrics</h3>'
            '<table>'
            '<tr><th>Metric</th><th>Conservative</th><th>AggressiveMicro</th></tr>'
            '<tr><td>P50 months to $5,000</td><td>' + (("%.1f" % m5k_a_val) if m5k_a_val else "N/A") + '</td><td>' + (("%.1f" % m5k_b_val) if m5k_b_val else "N/A") + '</td></tr>'
            '<tr><td>P50 months to $10,000</td><td>' + (("%.1f" % m10k_a_val) if m10k_a_val else "N/A") + '</td><td>' + (("%.1f" % m10k_b_val) if m10k_b_val else "N/A") + '</td></tr>'
            '<tr><td>Probability of ruin (&lt;$500)</td><td>' + ("%.1f%%" % (mc_a.get("probability_of_ruin", 0) * 100)) + '</td><td>' + ("%.1f%%" % (mc_b.get("probability_of_ruin", 0) * 100)) + '</td></tr>'
            '<tr><td>Probability of $5K within 12 months</td><td>' + ("%.1f%%" % (mc_a.get("probability_of_5k_12mo", 0) * 100)) + '</td><td>' + ("%.1f%%" % (mc_b.get("probability_of_5k_12mo", 0) * 100)) + '</td></tr>'
            '</table></div>'
        )
    else:
        mc_sections_html = (
            '<div class="section"><h2>3. Monte Carlo Results</h2>'
            '<p>Monte Carlo simulation was skipped (--no-mc flag).</p></div>'
        )

    dd_traces = [
        {"x": steps_a, "y": [d * 100 for d in dd_a], "name": "Conservative", "line": {"color": "blue"}},
        {"x": steps_b, "y": [d * 100 for d in dd_b], "name": "AggressiveMicro", "line": {"color": "red"}},
    ]
    dd_layout = {
        "title": "Drawdown Over Time",
        "yaxis": {"title": "Drawdown (%)"},
        "xaxis": {"title": "Step"},
    }
    js_lines.append("Plotly.newPlot('dd-chart', %s, %s);" % (json.dumps(dd_traces), json.dumps(dd_layout)))

    dd_text_lines = [
        "=" * 60,
        "DRAWDOWN ANALYSIS",
        "=" * 60,
        "",
        "CONSERVATIVE:",
        "  Max drawdown depth: %.1f%%" % (dd_depth_a * 100),
        "  Max drawdown duration: %d steps" % dd_dur_a,
        "  Max consecutive losses: %d" % consec_a,
        "  Recovery time: %s" % ("%d steps" % recovery_a if recovery_a else "Not recovered"),
        "",
        "AGGRESSIVEMICRO:",
        "  Max drawdown depth: %.1f%%" % (dd_depth_b * 100),
        "  Max drawdown duration: %d steps" % dd_dur_b,
        "  Max consecutive losses: %d" % consec_b,
        "  Recovery time: %s" % ("%d steps" % recovery_b if recovery_b else "Not recovered"),
        "=" * 60,
    ]
    dd_text = "\n".join(dd_text_lines)

    signal_traces = []
    if scores_a:
        signal_traces.append({"x": scores_a, "type": "histogram", "name": "Conservative",
                              "opacity": 0.7, "marker": {"color": "blue"}, "nbinsx": 25})
    if scores_b:
        signal_traces.append({"x": scores_b, "type": "histogram", "name": "AggressiveMicro",
                              "opacity": 0.7, "marker": {"color": "red"}, "nbinsx": 25})
    signal_layout = {
        "title": "Signal Score Distribution (Qualifying Trades)",
        "xaxis": {"title": "Signal Score"},
        "yaxis": {"title": "Count"},
        "barmode": "overlay",
    }
    js_lines.append("Plotly.newPlot('signal-hist', %s, %s);" % (json.dumps(signal_traces), json.dumps(signal_layout)))

    wr_traces = []
    wr_a_pct = [r * 100 for r in winrate_a]
    wr_b_pct = [r * 100 for r in winrate_b]
    wr_traces.append({"x": bucket_labels, "y": wr_a_pct, "type": "bar", "name": "Conservative",
                      "marker": {"color": "rgba(0,0,255,0.7)"}})
    wr_traces.append({"x": bucket_labels, "y": wr_b_pct, "type": "bar", "name": "AggressiveMicro",
                      "marker": {"color": "rgba(255,0,0,0.7)"}})
    wr_layout = {
        "title": "Win Rate by Signal Score Bucket",
        "xaxis": {"title": "Signal Score Range"},
        "yaxis": {"title": "Win Rate (%)"},
        "barmode": "group",
    }
    js_lines.append("Plotly.newPlot('winrate-chart', %s, %s);" % (json.dumps(wr_traces), json.dumps(wr_layout)))

    js_code = "\n".join(js_lines)

    css = (
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; "
        "margin: 20px; background: #f5f5f5; color: #333; }"
        "h1 { text-align: center; color: #2c3e50; }"
        "h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }"
        "h3 { color: #34495e; }"
        ".section { background: white; padding: 20px 30px; margin: 20px 0; border-radius: 8px; "
        "box-shadow: 0 2px 8px rgba(0,0,0,0.1); }"
        "table { border-collapse: collapse; width: 100%; margin: 10px 0; }"
        "th, td { border: 1px solid #ddd; padding: 10px 14px; text-align: center; }"
        "th { background: #3498db; color: white; font-weight: 600; }"
        "tr:nth-child(even) { background: #f8f9fa; }"
        "tr:hover { background: #e8f4fd; }"
        "pre { background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 6px; "
        "overflow-x: auto; font-size: 13px; line-height: 1.5; }"
        ".verdict { font-size: 1.4em; padding: 25px; text-align: center; font-weight: bold; "
        "border-left: 5px solid; }"
        ".verdict.reject { color: #d32f2f; border-color: #d32f2f; background: #ffebee; }"
        ".verdict.adopt { color: #2e7d32; border-color: #2e7d32; background: #e8f5e9; }"
        ".verdict.consider { color: #e65100; border-color: #e65100; background: #fff3e0; }"
        ".verdict.keep { color: #1565c0; border-color: #1565c0; background: #e3f2fd; }"
        ".plotly-chart { margin: 15px 0; }"
    )

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>DOTM Sniper Backtest Report</title>\n"
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>\n"
        "<style>\n" + css + "\n</style>\n"
        "</head>\n<body>\n"
        "<h1>DOTM Sniper Backtest Report</h1>\n"
        "<p style='text-align:center;color:#7f8c8d;'>Seed: " + str(seed) +
        " | Starting Balance: $" + "{:,.0f}".format(starting_balance) +
        " | Markets: " + str(len(markets)) + "</p>\n"

        "<div class='section'><h2>1. Summary Comparison</h2>\n"
        "<table>\n"
        "<tr><th>Metric</th><th>Conservative</th><th>AggressiveMicro</th></tr>\n"
        "<tr><td>Total trades qualified</td><td>" + str(res_a.get("signals_passed", 0)) + "</td><td>" + str(res_b.get("signals_passed", 0)) + "</td></tr>\n"
        "<tr><td>Total trades executed</td><td>" + str(res_a.get("total_trades", 0)) + "</td><td>" + str(res_b.get("total_trades", 0)) + "</td></tr>\n"
        "<tr><td>Win rate</td><td>" + ("%.1f%%" % (res_a.get("win_rate", 0) * 100)) + "</td><td>" + ("%.1f%%" % (res_b.get("win_rate", 0) * 100)) + "</td></tr>\n"
        "<tr><td>Avg win %</td><td>" + ("%+.1f%%" % (res_a.get("avg_win_pct", 0) * 100)) + "</td><td>" + ("%+.1f%%" % (res_b.get("avg_win_pct", 0) * 100)) + "</td></tr>\n"
        "<tr><td>Avg loss %</td><td>" + ("%+.1f%%" % (res_a.get("avg_loss_pct", 0) * 100)) + "</td><td>" + ("%+.1f%%" % (res_b.get("avg_loss_pct", 0) * 100)) + "</td></tr>\n"
        "<tr><td>Profit Factor</td><td>" + ("%.2f" % pf_a) + "</td><td>" + ("%.2f" % pf_b) + "</td></tr>\n"
        "<tr><td>Sharpe Ratio</td><td>" + ("%.2f" % res_a.get("sharpe_ratio", 0)) + "</td><td>" + ("%.2f" % res_b.get("sharpe_ratio", 0)) + "</td></tr>\n"
        "<tr><td>Max Drawdown</td><td>" + ("%.1f%%" % (res_a.get("max_drawdown", 0) * 100)) + "</td><td>" + ("%.1f%%" % (res_b.get("max_drawdown", 0) * 100)) + "</td></tr>\n"
        "<tr><td>Final balance</td><td>$" + ("{:,.0f}".format(res_a.get("final_equity", 0))) + "</td><td>$" + ("{:,.0f}".format(res_b.get("final_equity", 0))) + "</td></tr>\n"
        "<tr><td>CAGR %</td><td>" + ("%.1f%%" % (cagr_a * 100)) + "</td><td>" + ("%.1f%%" % (cagr_b * 100)) + "</td></tr>\n"
        "</table></div>\n"

        "<div class='section'><h2>2. Equity Curves</h2>\n"
        "<div id='equity-chart' class='plotly-chart' style='width:100%;height:600px;'></div></div>\n"

        + mc_sections_html +

        "<div class='section'><h2>4. Drawdown Analysis</h2>\n"
        "<pre>" + dd_text + "</pre>\n"
        "<div id='dd-chart' class='plotly-chart' style='width:100%;height:500px;'></div></div>\n"

        "<div class='section'><h2>5. Signal Score Distribution</h2>\n"
        "<div id='signal-hist' class='plotly-chart' style='width:100%;height:500px;'></div>\n"
        "<div id='winrate-chart' class='plotly-chart' style='width:100%;height:500px;'></div></div>\n"

        "<div class='section verdict " + verdict_class + "'>\n"
        "<h2>6. Verdict</h2>\n"
        "<p>" + verdict + "</p></div>\n"

        "<script>\n" + js_code + "\n</script>\n"
        "</body>\n</html>"
    )

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"[REPORT] HTML report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest: Conservative vs AggressiveMicro")
    parser.add_argument("--balance", type=float, default=1500.0)
    parser.add_argument("--mc-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mc", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("BACKTEST: Conservative vs AggressiveMicro")
    print(f"Balance: ${args.balance:,.0f} | MC sims: {args.mc_sims} | Seed: {args.seed}")
    print("=" * 60)

    markets = load_markets()
    n_markets = len(markets)
    print(f"[DATA] {n_markets} DOTM markets loaded\n")

    print(f"[RUN] Simulating {PROFILE_A['name']}...")
    t0 = time.time()
    res_a = simulate_profile(markets, PROFILE_A, starting_balance=args.balance, seed=args.seed)
    elapsed_a = time.time() - t0
    print(f"  Done in {elapsed_a:.1f}s | {res_a['total_trades']} trades | "
          f"P&L ${res_a['total_pnl']:,.0f} | Final ${res_a['final_equity']:,.0f}")

    print(f"[RUN] Simulating {PROFILE_B['name']}...")
    t0 = time.time()
    res_b = simulate_profile(markets, PROFILE_B, starting_balance=args.balance, seed=args.seed)
    elapsed_b = time.time() - t0
    print(f"  Done in {elapsed_b:.1f}s | {res_b['total_trades']} trades | "
          f"P&L ${res_b['total_pnl']:,.0f} | Final ${res_b['final_equity']:,.0f}")

    mc_a = {}
    mc_b = {}
    if not args.no_mc:
        print(f"\n[MC] Monte Carlo ({args.mc_sims} sims) for {PROFILE_A['name']}...")
        t0 = time.time()
        mc_a = monte_carlo_simulation(markets, PROFILE_A, n_simulations=args.mc_sims,
                                      starting_balance=args.balance)
        elapsed_mc_a = time.time() - t0
        print(f"  Done in {elapsed_mc_a:.1f}s | Mean equity ${mc_a['mean_final_equity']:,.0f} | "
              f"Ruin {mc_a['ruin_probability']:.1%}")

        print(f"[MC] Monte Carlo ({args.mc_sims} sims) for {PROFILE_B['name']}...")
        t0 = time.time()
        mc_b = monte_carlo_simulation(markets, PROFILE_B, n_simulations=args.mc_sims,
                                      starting_balance=args.balance)
        elapsed_mc_b = time.time() - t0
        print(f"  Done in {elapsed_mc_b:.1f}s | Mean equity ${mc_b['mean_final_equity']:,.0f} | "
              f"Ruin {mc_b['ruin_probability']:.1%}")
    else:
        print("\n[MC] Skipping Monte Carlo (--no-mc flag)")

    print_report(res_a, res_b, mc_a, mc_b, n_markets, args.mc_sims)

    generate_html_report(res_a, res_b, mc_a, mc_b, markets, args.seed, args.balance)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "backtest_aggressive_micro_results.json")
    output = {
        "profile_a": PROFILE_A,
        "profile_b": PROFILE_B,
        "profile_a_results": {
            "total_trades": res_a.get("total_trades", 0),
            "signals_passed": res_a.get("signals_passed", 0),
            "wins": res_a.get("wins", 0),
            "losses": res_a.get("losses", 0),
            "win_rate": res_a.get("win_rate", 0),
            "avg_win_pct": res_a.get("avg_win_pct", 0),
            "avg_loss_pct": res_a.get("avg_loss_pct", 0),
            "total_pnl": res_a.get("total_pnl", 0),
            "final_equity": res_a.get("final_equity", 0),
            "max_drawdown": res_a.get("max_drawdown", 0),
            "sharpe_ratio": res_a.get("sharpe_ratio", 0),
            "exit_reasons": res_a.get("exit_reasons", {}),
            "rejection_reasons": res_a.get("rejection_reasons", {}),
            "trades": res_a.get("trades", []),
            "equity_curve": res_a.get("equity_curve", []),
        },
        "profile_b_results": {
            "total_trades": res_b.get("total_trades", 0),
            "signals_passed": res_b.get("signals_passed", 0),
            "wins": res_b.get("wins", 0),
            "losses": res_b.get("losses", 0),
            "win_rate": res_b.get("win_rate", 0),
            "avg_win_pct": res_b.get("avg_win_pct", 0),
            "avg_loss_pct": res_b.get("avg_loss_pct", 0),
            "total_pnl": res_b.get("total_pnl", 0),
            "final_equity": res_b.get("final_equity", 0),
            "max_drawdown": res_b.get("max_drawdown", 0),
            "sharpe_ratio": res_b.get("sharpe_ratio", 0),
            "exit_reasons": res_b.get("exit_reasons", {}),
            "rejection_reasons": res_b.get("rejection_reasons", {}),
            "trades": res_b.get("trades", []),
            "equity_curve": res_b.get("equity_curve", []),
        },
        "monte_carlo_a": mc_a,
        "monte_carlo_b": mc_b,
        "comparison_summary": {
            "markets_analyzed": n_markets,
            "starting_balance": args.balance,
            "mc_sims": args.mc_sims if not args.no_mc else 0,
            "seed": args.seed,
            "winner_by_pnl": PROFILE_A["name"] if res_a.get("total_pnl", 0) >= res_b.get("total_pnl", 0) else PROFILE_B["name"],
            "winner_by_sharpe": PROFILE_A["name"] if res_a.get("sharpe_ratio", 0) >= res_b.get("sharpe_ratio", 0) else PROFILE_B["name"],
            "winner_by_final_equity": PROFILE_A["name"] if res_a.get("final_equity", 0) >= res_b.get("final_equity", 0) else PROFILE_B["name"],
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[SAVE] Results saved to {out_path}")


if __name__ == "__main__":
    main()
