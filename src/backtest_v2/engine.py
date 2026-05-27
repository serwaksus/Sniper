#!/usr/bin/env python3
"""
Backtest v2 Engine — event-driven realistic backtest.
No look-ahead bias, real slippage, portfolio constraints, full exit logic.
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


def _estimate_signal(market: Dict, use_metaculus: bool = True) -> Dict:
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
    
    action = "BUY" if signal_score >= SIGNAL_THRESHOLD and confidence >= MIN_CONFIDENCE and prob_ratio >= MIN_PROB_RATIO else "SKIP"
    
    return {
        "p_model": p_model,
        "prob_ratio": prob_ratio,
        "signal_score": signal_score,
        "confidence": confidence,
        "action": action,
    }


def _simulate_advisor_veto(signal: Dict) -> bool:
    """Simulate advisor pre-check. ~30% veto rate based on live data."""
    if signal["confidence"] < 0.70:
        return random.random() < 0.50
    if signal["prob_ratio"] < 2.5:
        return random.random() < 0.40
    return random.random() < ADVISOR_VETO_RATE


def _simulate_news_block() -> bool:
    """Simulate news sanity check blocking. ~10% block rate."""
    return random.random() < NEWS_BLOCK_RATE


def run_backtest(
    starting_balance: float = 500.0,
    max_markets: int = 300,
    use_metaculus: bool = True,
    use_advisor: bool = True,
    use_news: bool = True,
    force_refresh: bool = False,
    seed: int = 42,
    markets: list = None,
) -> Dict:
    """
    Run realistic event-driven backtest.
    Returns comprehensive performance metrics.
    """
    random.seed(seed)
    
    logger.info(f"[BACKTEST] Starting: balance=${starting_balance}, max_markets={max_markets}")
    
    if markets is None:
        markets = fetch_resolved_markets(
            max_markets=max_markets,
            force_refresh=force_refresh,
        )
    
    if not markets:
        logger.error("[BACKTEST] No markets loaded")
        return {"error": "no markets"}
    
    random.shuffle(markets)
    
    portfolio = PortfolioTracker(starting_balance=starting_balance)
    
    analyzed = 0
    for market in markets:
        portfolio.step += 1
        slug = market["slug"]
        entry_price = market["entry_price"]
        liquidity = market.get("liquidity", 100)
        cluster = market.get("category", "other")
        
        # Sub-cluster within "other" to avoid concentration
        if cluster == "other":
            cluster = f"other_{hash(market.get('question', '')) % 8}"
        
        market["cluster"] = cluster
        
        if portfolio.balance < 5:
            logger.debug(f"[BACKTEST] Balance too low (${portfolio.balance:.2f}), stopping")
            break
        
        signal = _estimate_signal(market, use_metaculus=use_metaculus)
        analyzed += 1
        
        if signal["action"] != "BUY":
            continue
        
        if use_advisor and _simulate_advisor_veto(signal):
            portfolio.rejected_trades.append({"slug": slug, "reason": "advisor_veto"})
            continue
        
        if use_news and _simulate_news_block():
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
    
    logger.info(f"[BACKTEST] Analysis done: {analyzed} markets, {len(portfolio.positions)} positions opened")
    
    price_series_cache = {}
    for market in markets:
        price_series_cache[market["slug"]] = generate_price_series(market, num_steps=60)
    
    for step_idx in range(31):
        current_prices = {}
        for slug, pos in list(portfolio.positions.items()):
            series = price_series_cache.get(slug, [pos.entry_price])
            idx = min(step_idx, len(series) - 1)
            current_prices[slug] = series[idx]
        
        for slug, pos in list(portfolio.positions.items()):
            price = current_prices.get(slug, 0)
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
            
            if not sold and pnl_pct <= -0.80:
                sell_result = simulate_sell(price, pos.shares_after_tp, pos.entry_price, pos.liquidity, force_market=True)
                if sell_result["filled"]:
                    fee = sell_result.get("fee", sell_result["proceeds"] * 0.02)
                    portfolio.close_position(slug, sell_result["proceeds"], "hard_stop_loss", price, fee=fee)
                    sold = True
            
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
                    portfolio.balance += tp_result["total_proceeds"]
        
        portfolio.record_equity(current_prices)
    
    for slug, pos in list(portfolio.positions.items()):
        series = price_series_cache.get(slug, [pos.entry_price])
        final_price = series[-1]
        sell_result = simulate_sell(final_price, pos.shares_after_tp, pos.entry_price, pos.liquidity, force_market=True)
        proceeds = sell_result["proceeds"] if sell_result["filled"] else pos.shares_after_tp * final_price * 0.5
        fee = sell_result.get("fee", proceeds * 0.02) if sell_result["filled"] else proceeds * 0.02
        portfolio.close_position(slug, proceeds, "resolution", final_price, fee=fee)
    
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
    
    return summary
