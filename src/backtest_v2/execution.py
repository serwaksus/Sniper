#!/usr/bin/env python3
"""
Backtest v2 Execution Engine — realistic order execution simulation.
Walk-the-book for entry/exit, slippage, partial fills, $5 minimum.
"""
import logging
from data_loader import generate_order_book

logger = logging.getLogger(__name__)

MIN_ORDER_USD = 5.0
MAX_SLIPPAGE_PCT = 0.50
MAX_SPREAD_PCT = 0.50
FEE_PCT = 0.02  # 2% transaction cost per trade (spread + Polymarket fees)


def walk_the_book(asks: list[dict], amount_usd: float) -> tuple[float, float, float]:
    """
    Walk the ask side of the order book to fill a buy order.
    Returns (effective_price, total_shares, total_cost).
    If cannot fill minimum $5, returns (0, 0, 0).
    """
    remaining_usd = amount_usd
    total_cost = 0.0
    total_shares = 0.0

    for level in asks:
        price = level["price"]
        size = level["size"]
        if price <= 0:
            continue

        level_cost = price * size
        fill_cost = min(remaining_usd, level_cost)
        fill_shares = fill_cost / price

        total_cost += fill_cost
        total_shares += fill_shares
        remaining_usd -= fill_cost

        if remaining_usd <= 0.01:
            break

    if total_cost < MIN_ORDER_USD:
        return 0, 0, 0

    effective_price = total_cost / total_shares if total_shares > 0 else 0
    return effective_price, total_shares, total_cost


def walk_the_book_sell(bids: list[dict], shares: float) -> tuple[float, float, float]:
    """
    Walk the bid side of the order book to fill a sell order.
    Returns (effective_price, shares_filled, proceeds_usd).
    """
    remaining_shares = shares
    total_proceeds = 0.0
    total_shares_filled = 0.0

    for level in bids:
        price = level["price"]
        size = level["size"]
        if price <= 0:
            continue

        fill_shares = min(remaining_shares, size)
        proceeds = fill_shares * price

        total_proceeds += proceeds
        total_shares_filled += fill_shares
        remaining_shares -= fill_shares

        if remaining_shares <= 0.01:
            break

    effective_price = total_proceeds / total_shares_filled if total_shares_filled > 0 else 0
    return effective_price, total_shares_filled, total_proceeds


def simulate_buy(
    entry_price: float,
    amount_usd: float,
    liquidity: float,
    slippage_buffer: float = 0.0,
) -> dict:
    """
    Simulate a realistic buy order execution.
    Returns dict with: filled, effective_price, shares, cost, slippage_pct, reason.
    """
    result = {
        "filled": False,
        "effective_price": 0,
        "shares": 0,
        "cost": 0,
        "slippage_pct": 0,
        "reason": "",
    }

    if entry_price <= 0.001:
        result["reason"] = f"entry_price={entry_price:.6f} too low"
        return result

    if amount_usd < MIN_ORDER_USD:
        result["reason"] = f"amount=${amount_usd:.2f} < ${MIN_ORDER_USD} minimum"
        return result

    book = generate_order_book(entry_price, liquidity)
    asks = book["asks"]
    bids = book["bids"]

    if not asks:
        result["reason"] = "no asks in order book"
        return result

    best_ask = asks[0]["price"]
    best_bid = bids[0]["price"] if bids else 0

    if best_bid > 0 and best_ask > 0:
        spread = (best_ask - best_bid) / best_ask
        if spread > MAX_SPREAD_PCT:
            result["reason"] = f"spread={spread:.1%} > {MAX_SPREAD_PCT:.0%}"
            return result

    mid_price = (best_ask + best_bid) / 2 if best_bid > 0 else best_ask

    effective_price, shares, cost = walk_the_book(asks, amount_usd)

    if effective_price == 0:
        result["reason"] = "cannot fill minimum $5"
        return result

    slippage = (effective_price - entry_price) / entry_price if entry_price > 0 else 0

    if slippage > MAX_SLIPPAGE_PCT + slippage_buffer:
        result["reason"] = f"slippage={slippage:.1%} > {MAX_SLIPPAGE_PCT:.0%} guard"
        result["effective_price"] = effective_price
        result["slippage_pct"] = slippage
        return result

    result["filled"] = True
    result["effective_price"] = effective_price
    result["shares"] = shares
    result["cost"] = cost
    result["fee"] = round(cost * FEE_PCT, 4)
    result["cost_with_fee"] = round(cost * (1 + FEE_PCT), 4)
    result["slippage_pct"] = slippage
    result["reason"] = "ok"
    result["book"] = book

    return result


def simulate_sell(
    current_price: float,
    shares: float,
    entry_price: float,
    liquidity: float,
    force_market: bool = False,
) -> dict:
    """
    Simulate a realistic sell order execution.
    Returns dict with: filled, effective_price, shares_filled, proceeds, slippage_pct, reason.
    """
    result = {
        "filled": False,
        "effective_price": 0,
        "shares_filled": 0,
        "proceeds": 0,
        "slippage_pct": 0,
        "reason": "",
    }

    if shares <= 0:
        result["reason"] = "no shares to sell"
        return result

    book = generate_order_book(current_price, liquidity)
    bids = book["bids"]
    asks = book["asks"]

    if not bids:
        result["reason"] = "no bids in order book"
        return result

    best_bid = bids[0]["price"]

    if best_bid <= 0:
        result["reason"] = "best_bid <= 0"
        return result

    if not force_market:
        best_ask = asks[0]["price"] if asks else 0
        if best_ask > 0 and best_bid > 0:
            spread = (best_ask - best_bid) / best_ask
            if spread > MAX_SPREAD_PCT:
                result["reason"] = f"spread={spread:.1%} too wide for limit sell"
                return result

        if entry_price > 0 and best_bid < entry_price * 0.70:
            result["reason"] = f"bid={best_bid:.4f} >30% below entry={entry_price:.4f}"
            return result

    effective_price, shares_filled, proceeds = walk_the_book_sell(bids, shares)

    if shares_filled <= 0:
        result["reason"] = "no shares filled"
        return result

    slippage = (current_price - effective_price) / current_price if current_price > 0 else 0

    result["filled"] = True
    result["effective_price"] = effective_price
    result["shares_filled"] = shares_filled
    result["proceeds"] = proceeds
    result["fee"] = round(proceeds * FEE_PCT, 4)
    result["proceeds_after_fee"] = round(proceeds * (1 - FEE_PCT), 4)
    result["slippage_pct"] = slippage
    result["partial_fill"] = shares_filled < shares * 0.99
    result["reason"] = "ok"

    return result


def simulate_tp_ladder(
    shares: float,
    entry_price: float,
    current_price: float,
    liquidity: float,
    ladder: list[tuple[float, float]] = None,
) -> dict:
    """
    Simulate TP ladder execution.
    ladder: [(pct_of_shares, tp_price), ...]
    Returns: {total_proceeds, total_shares_sold, rungs_filled, rungs_failed}
    """
    if ladder is None:
        ladder = [(0.50, 0.75), (0.30, 0.85)]

    total_proceeds = 0.0
    total_shares_sold = 0.0
    rungs_filled = 0
    rungs_failed = 0
    allocated = 0.0

    for pct, tp_price in ladder:
        rung_shares = max(round(5.0 / tp_price), round(shares * pct), 1)
        if allocated + rung_shares > shares:
            rung_shares = shares - allocated
        if rung_shares <= 0:
            continue

        if current_price < tp_price:
            break

        book = generate_order_book(tp_price, liquidity)
        bids = book["bids"]

        if not bids or bids[0]["price"] < tp_price * 0.90:
            rungs_failed += 1
            continue

        eff_price, filled, proceeds = walk_the_book_sell(bids, rung_shares)
        if filled > 0:
            fee = proceeds * FEE_PCT
            net_proceeds = proceeds - fee
            total_proceeds += net_proceeds
            total_shares_sold += filled
            allocated += filled
            rungs_filled += 1
        else:
            rungs_failed += 1

    return {
        "total_proceeds": total_proceeds,
        "total_shares_sold": total_shares_sold,
        "rungs_filled": rungs_filled,
        "rungs_failed": rungs_failed,
        "shares_held_to_expiry": shares - total_shares_sold,
    }
