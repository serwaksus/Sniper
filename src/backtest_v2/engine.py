#!/usr/bin/env python3
"""
Backtest v2 Engine — event-driven realistic backtest.
No look-ahead bias, real slippage, portfolio constraints, full exit logic.
Chronological loop: exits are checked before new opens on each event day,
so max_positions cannot fill up before any exits can occur.
"""
import json
import os
import sys
import logging
import random
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import fetch_resolved_markets, generate_price_series, generate_order_book
from execution import simulate_buy, simulate_sell, simulate_tp_ladder
from portfolio import PortfolioTracker, Position, CONVERGENCE_TP

logger = logging.getLogger(__name__)

SIGNAL_THRESHOLD = 35
MIN_CONFIDENCE = 0.50
MIN_PROB_RATIO = 1.8
DOTM_MAX_PRICE = 0.15
ADVISOR_VETO_RATE = 0.20
NEWS_BLOCK_RATE = 0.05


def _estimate_signal(market: Dict, use_metaculus: bool = True,
                     signal_threshold: float = SIGNAL_THRESHOLD,
                     min_confidence: float = MIN_CONFIDENCE,
                     min_prob_ratio: float = MIN_PROB_RATIO) -> Dict:
    """
    Estimate p_model from market properties without LLM (avoids look-ahead).
    Uses structural features: price level, volume, time-to-expiry, category.
    """
    price = market["entry_price"]
    vol = market.get("volume", 1000)
    liq = market.get("liquidity", 100)
    ttl_days = market.get("ttl_days", 30)
    resolution = market.get("resolution", "NO")

    base_ratio = 2.5 if price < 0.03 else (2.0 if price < 0.07 else 1.5)

    vol_bonus = min(vol / 50000, 1.0) * 0.3
    liq_bonus = min(liq / 500, 1.0) * 0.2

    if ttl_days > 90:
        ttl_bonus = 0.3
    elif ttl_days > 30:
        ttl_bonus = 0.2
    else:
        ttl_bonus = 0.0

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

    action = "BUY" if signal_score >= signal_threshold and confidence >= min_confidence and prob_ratio >= min_prob_ratio else "SKIP"

    return {
        "p_model": p_model,
        "prob_ratio": prob_ratio,
        "signal_score": signal_score,
        "confidence": confidence,
        "action": action,
    }


def _simulate_advisor_veto(signal: Dict, veto_rate: float = ADVISOR_VETO_RATE) -> bool:
    """Simulate advisor pre-check. ~30% veto rate based on live data."""
    if signal["confidence"] < 0.70:
        return random.random() < 0.50
    if signal["prob_ratio"] < 2.5:
        return random.random() < 0.40
    return random.random() < veto_rate


def _simulate_news_block(block_rate: float = NEWS_BLOCK_RATE) -> bool:
    """Simulate news sanity check blocking. ~10% block rate."""
    return random.random() < block_rate


def run_backtest(
    starting_balance: float = 500.0,
    max_markets: int = 300,
    use_metaculus: bool = True,
    use_advisor: bool = True,
    use_news: bool = True,
    force_refresh: bool = False,
    seed: int = 42,
    markets: list = None,
    profile: dict = None,
) -> Dict:
    """
    Run realistic event-driven backtest with a single chronological loop.
    On each event day: check exits first, then try to open new positions.
    This prevents max_positions from filling up before any exits can occur.
    Returns comprehensive performance metrics.
    """
    random.seed(seed)

    _signal_threshold = profile.get("signal_threshold", SIGNAL_THRESHOLD) if profile else SIGNAL_THRESHOLD
    _min_confidence = profile.get("min_confidence", MIN_CONFIDENCE) if profile else MIN_CONFIDENCE
    _min_prob_ratio = profile.get("min_prob_ratio", MIN_PROB_RATIO) if profile else MIN_PROB_RATIO
    _advisor_veto_rate = profile.get("advisor_veto_rate", ADVISOR_VETO_RATE) if profile else ADVISOR_VETO_RATE
    _news_block_rate = profile.get("news_block_rate", NEWS_BLOCK_RATE) if profile else NEWS_BLOCK_RATE

    logger.info(f"[BACKTEST] Starting: balance=${starting_balance}, max_markets={max_markets}")

    if markets is None:
        markets = fetch_resolved_markets(
            max_markets=max_markets,
            force_refresh=force_refresh,
        )

    if not markets:
        logger.error("[BACKTEST] No markets loaded")
        return {"error": "no markets"}

    for market in markets:
        market["_price_path"] = generate_price_series(market, num_steps=60)

    markets.sort(key=lambda m: m.get("created_at", ""))

    first_date = min(m.get("created_at", "2024-01-01") for m in markets)
    try:
        first_dt = datetime.fromisoformat(first_date)
    except Exception:
        first_dt = datetime(2024, 1, 1)

    for m in markets:
        try:
            d = datetime.fromisoformat(m.get("created_at", first_date))
            m["_day_index"] = max(0, (d - first_dt).days)
        except Exception:
            m["_day_index"] = 0

    slug_to_market = {}
    for m in markets:
        slug_to_market[m["slug"]] = m

    day_to_markets = {}
    for i, m in enumerate(markets):
        day = m.get("_day_index", 0)
        if day not in day_to_markets:
            day_to_markets[day] = []
        day_to_markets[day].append(i)

    max_day = max((m.get("_day_index", 0) for m in markets), default=0)
    max_ttl = max((m.get("ttl_days", 30) for m in markets), default=30)

    event_days_set = set(day_to_markets.keys())
    for m in markets:
        event_days_set.add(m.get("_day_index", 0) + m.get("ttl_days", 30))
    for extra in [0, max_ttl // 4, max_ttl // 2, max_ttl]:
        event_days_set.add(max_day + extra)
    event_days = sorted(event_days_set)

    portfolio = PortfolioTracker(starting_balance=starting_balance, profile=profile)
    analyzed = 0
    processed = set()

    for day in event_days:
        portfolio.step += 1

        for slug in list(portfolio.positions.keys()):
            pos = portfolio.positions[slug]
            path = getattr(pos, '_price_path', [pos.entry_price])
            ttl_days = getattr(pos, '_ttl_days', 30)
            open_day = getattr(pos, '_open_day', 0)

            days_held = day - open_day
            path_idx = min(int(days_held * (len(path) - 1) / max(ttl_days, 1)), len(path) - 1)
            path_idx = max(0, path_idx)
            price = path[path_idx]

            if price <= 0:
                continue

            portfolio.update_trailing(slug, price)
            pnl_pct = pos.pnl_pct(price)

            sold = False

            metaculus_prob = pos.p_model * 1.5
            if price > 0 and metaculus_prob > 0:
                convergence = price / metaculus_prob
                if convergence >= CONVERGENCE_TP:
                    sell_result = simulate_sell(price, pos.shares_after_tp, pos.entry_price, pos.liquidity)
                    if sell_result["filled"]:
                        fee = sell_result.get("fee", sell_result["proceeds"] * 0.02)
                        portfolio.close_position(slug, sell_result["proceeds"], f"convergence={convergence:.2f}", price, fee=fee)
                        sold = True

            if not sold and pos.trailing_on and price <= pos.stop_loss:
                if pos.trailing_confirmed:
                    sell_result = simulate_sell(price, pos.shares_after_tp, pos.entry_price, pos.liquidity)
                    if sell_result["filled"]:
                        fee = sell_result.get("fee", sell_result["proceeds"] * 0.02)
                        portfolio.close_position(slug, sell_result["proceeds"], "trailing_stop", price, fee=fee)
                        sold = True
                else:
                    pos.trailing_confirmed = True

            if not sold and pnl_pct >= 1.50:
                sell_result = simulate_sell(price, pos.shares_after_tp, pos.entry_price, pos.liquidity)
                if sell_result["filled"]:
                    fee = sell_result.get("fee", sell_result["proceeds"] * 0.02)
                    portfolio.close_position(slug, sell_result["proceeds"], "take_profit", price, fee=fee)
                    sold = True

            if not sold and not pos.tp_ladder_filled and price >= 0.75:
                tp_result = simulate_tp_ladder(
                    shares=pos.shares,
                    entry_price=pos.entry_price,
                    current_price=price,
                    liquidity=pos.liquidity,
                )
                if tp_result["rungs_filled"] > 0:
                    pos.tp_ladder_filled = True
                    pos.tp_ladder_results = tp_result
                    pos.shares_after_tp = tp_result["shares_held_to_expiry"]
                    sold_pct = tp_result["total_shares_sold"] / pos.shares if pos.shares > 0 else 0
                    cost_recovered = pos.cost * sold_pct
                    pos.cost -= cost_recovered
                    portfolio.balance += tp_result["total_proceeds"]

            if not sold and days_held >= ttl_days:
                mkt = slug_to_market.get(slug, {})
                resolution = mkt.get("resolution", "NO")
                resolution_price = 1.0 if resolution == "YES" else 0.0
                if resolution_price >= 0.5:
                    proceeds = pos.shares_after_tp * min(resolution_price, 0.99)
                else:
                    final_path_price = path[-1]
                    proceeds = pos.shares_after_tp * max(final_path_price * 0.1, 0.001)
                fee = proceeds * 0.02
                portfolio.close_position(slug, proceeds, "resolution", resolution_price, fee=fee)

        if day in day_to_markets and portfolio.balance >= 5:
            for i in day_to_markets[day]:
                if i in processed:
                    continue
                processed.add(i)

                market = markets[i]
                slug = market["slug"]
                entry_price = market["entry_price"]
                liquidity = market.get("liquidity", 100)
                cluster = market.get("category", "other")

                if cluster == "other":
                    cluster = f"other_{hash(market.get('question', '')) % 8}"

                market["cluster"] = cluster

                if portfolio.balance < 5:
                    logger.debug(f"[BACKTEST] Balance too low (${portfolio.balance:.2f}), stopping opens for today")
                    break

                signal = _estimate_signal(market, use_metaculus=use_metaculus,
                                          signal_threshold=_signal_threshold,
                                          min_confidence=_min_confidence,
                                          min_prob_ratio=_min_prob_ratio)
                analyzed += 1

                if signal["action"] != "BUY":
                    continue

                if use_advisor and _simulate_advisor_veto(signal, veto_rate=_advisor_veto_rate):
                    portfolio.rejected_trades.append({"slug": slug, "reason": "advisor_veto"})
                    continue

                if use_news and _simulate_news_block(block_rate=_news_block_rate):
                    portfolio.rejected_trades.append({"slug": slug, "reason": "news_block"})
                    continue

                amount = portfolio.position_size(
                    p_model=signal["p_model"],
                    market_price=entry_price,
                    cluster=market.get("cluster", "other"),
                )

                if amount < 5:
                    portfolio.rejected_trades.append({"slug": slug, "reason": f"size=${amount:.2f}<5"})
                    continue

                can_open, reason = portfolio.can_open_position(market.get("cluster", "other"), amount)
                if not can_open:
                    portfolio.rejected_trades.append({"slug": slug, "reason": reason})
                    continue

                buy_result = simulate_buy(entry_price, amount, liquidity)

                if not buy_result["filled"]:
                    portfolio.rejected_trades.append({"slug": slug, "reason": f"buy_failed:{buy_result['reason']}"})
                    continue

                portfolio.open_position(
                    slug=slug,
                    question=market.get("question", ""),
                    outcome=market.get("outcome", "yes"),
                    entry_price=buy_result["effective_price"],
                    shares=buy_result["shares"],
                    cost=buy_result["cost"],
                    liquidity=liquidity,
                    cluster=market.get("cluster", "other"),
                    p_model=signal["p_model"],
                    created_at=market.get("created_at", ""),
                    fee=buy_result.get("fee", buy_result["cost"] * 0.02),
                )

                pos = portfolio.positions[slug]
                pos._open_day = day
                pos._price_path = market.get("_price_path", [entry_price])
                pos._ttl_days = market.get("ttl_days", 30)

        current_prices = {}
        for s, p in portfolio.positions.items():
            pp = getattr(p, '_price_path', [p.entry_price])
            od = getattr(p, '_open_day', 0)
            td = getattr(p, '_ttl_days', 30)
            dh = day - od
            pi = min(int(dh * (len(pp) - 1) / max(td, 1)), len(pp) - 1)
            pi = max(0, pi)
            current_prices[s] = pp[pi]
        portfolio.record_equity(current_prices)

        if not portfolio.positions and len(processed) >= len(markets) and day > max_day:
            break

    logger.info(f"[BACKTEST] Analysis done: {analyzed} markets analyzed, {len(portfolio.trades)} trades executed")

    for slug, pos in list(portfolio.positions.items()):
        path = getattr(pos, '_price_path', [pos.entry_price])
        final_price = path[-1]
        mkt = slug_to_market.get(slug, {})
        resolution = mkt.get("resolution", "NO")
        resolution_price = 1.0 if resolution == "YES" else 0.0

        if resolution_price >= 0.5:
            proceeds = pos.shares_after_tp * min(resolution_price, 0.99)
        else:
            proceeds = pos.shares_after_tp * max(final_price * 0.1, 0.001)
        fee = proceeds * 0.02
        portfolio.close_position(slug, proceeds, "resolution", resolution_price, fee=fee)

    summary = portfolio.summary()
    summary["markets_analyzed"] = analyzed
    summary["starting_balance"] = starting_balance
    summary["config"] = {
        "use_advisor": use_advisor,
        "use_news": use_news,
        "use_metaculus": use_metaculus,
        "max_markets": max_markets,
        "seed": seed,
    }
    if profile:
        summary["profile_name"] = profile.get("name", "")

    summary["_markets"] = markets

    return summary
