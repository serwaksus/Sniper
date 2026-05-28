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


def monte_carlo_simulation(markets: list, profile: dict, n_simulations: int = 1000,
                           starting_balance: float = 1500.0) -> dict:
    final_equities = []
    max_drawdowns = []
    sharpe_ratios = []
    ruin_count = 0
    total_trades_list = []

    for sim_idx in range(n_simulations):
        if sim_idx % 100 == 0 and sim_idx > 0:
            pct = sim_idx / n_simulations * 100
            print(f"  MC {profile['name']}: {pct:.0f}% ({sim_idx}/{n_simulations})", end="\r")

        sim_seed = 42 + sim_idx * 7
        rng = np.random.default_rng(sim_seed)

        shuffled = list(markets)
        rng.shuffle(np.arange(len(shuffled)))
        order = rng.permutation(len(shuffled)).tolist()
        shuffled = [shuffled[i] for i in order]

        noisy = []
        for m in shuffled:
            nm = dict(m)
            noise_price = 1.0 + rng.uniform(-0.10, 0.10)
            nm["entry_price"] = max(0.005, min(0.29, m["entry_price"] * noise_price))
            noisy.append(nm)

        result = simulate_profile(noisy, profile, starting_balance=starting_balance, seed=sim_seed)

        eq = result["final_equity"]
        final_equities.append(eq)
        max_drawdowns.append(result["max_drawdown"])
        sharpe_ratios.append(result["sharpe_ratio"])
        total_trades_list.append(result["total_trades"])

        if eq < 100:
            ruin_count += 1

    arr_eq = np.array(final_equities)
    arr_dd = np.array(max_drawdowns)
    arr_sh = np.array(sharpe_ratios)

    return {
        "n_simulations": n_simulations,
        "mean_final_equity": float(np.mean(arr_eq)),
        "median_final_equity": float(np.median(arr_eq)),
        "p5_final_equity": float(np.percentile(arr_eq, 5)),
        "p95_final_equity": float(np.percentile(arr_eq, 95)),
        "ruin_probability": ruin_count / n_simulations,
        "mean_max_drawdown": float(np.mean(arr_dd)),
        "median_max_drawdown": float(np.median(arr_dd)),
        "mean_sharpe": float(np.mean(arr_sh)),
        "median_sharpe": float(np.median(arr_sh)),
        "mean_trades": float(np.mean(total_trades_list)),
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
    row("95th percentile", mc_a.get("p95_final_equity"), mc_b.get("p95_final_equity"), fmt="dollar")
    row("P(ruin < $100)", mc_a.get("ruin_probability", 0), mc_b.get("ruin_probability", 0), fmt="pct_plain")
    row("Mean max drawdown", mc_a.get("mean_max_drawdown", 0), mc_b.get("mean_max_drawdown", 0), fmt="pct_plain")
    print("╚" + "═" * w + "╝")
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest: Conservative vs AggressiveMicro")
    parser.add_argument("--balance", type=float, default=1500.0)
    parser.add_argument("--mc-sims", type=int, default=1000)
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
