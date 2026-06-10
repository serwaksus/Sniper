from __future__ import annotations
import time
import logging
import contextlib
from datetime import datetime, UTC
from typing import Any

import positions_db
import hypotheses_db
from equity_tracker import log_trade
from schema import (
    HYP_CLUSTERS, HYP_CONFIDENCE, HYP_CREATED_AT, HYP_DB_HYPOTHESES,
    HYP_FACTORS, HYP_MARKET_PRICE, HYP_P_MODEL, HYP_PROB_RATIO,
    HYP_QUESTION, HYP_RESOLVED, HYP_SIZE_PCT, HYP_SLUG,
    HYP_SOURCE_SIGNAL, HYP_TP_LIMIT_PLACED, HYP_TP_LIMIT_PRICE,
    POS_CLUSTERS, POS_ENTRY_PRICE, POS_HIGH_PRICE, POS_LAST_CHECKED,
    POS_MARKET_QUESTION, POS_METACULUS_PROB, POS_OUTCOME, POS_SHARES,
    POS_STOP_LOSS, POS_TRAILING_ON,
)
from db import record_trade

logger = logging.getLogger(__name__)

SMART_EXIT_PRICE = 0.85


def _get_sniper_deps() -> tuple[Any, Any, Any]:
    from dotm_sniper import load_hypothesis_db, save_hypothesis_db, _tr
    return load_hypothesis_db, save_hypothesis_db, _tr


def execute_trade(market: dict[str, Any], estimated_size: float, factors: list[str], analysis: dict[str, Any], balance: float) -> bool:
    """Execute trade with advisor pre-check. Returns True if successful."""
    slug = market.get("slug", "")

    from sell_executor import check_portfolio_drawdown
    if check_portfolio_drawdown():
        logger.warning(f"[TRADE] Portfolio drawdown stop active, skipping {slug[:60]}")
        return False

    existing = positions_db.get(slug)
    if existing is not None:
        logger.info(f"[TRADE] {slug[:60]} already tracked, skipping")
        return False

    hyp = hypotheses_db.get(slug)
    if hyp is not None:
        logger.info(f"[TRADE] {slug[:60]} already in hypothesis_db, skipping")
        return False

    from signal_pipeline import advisor_pre_check
    from order_manager import get_best_ask, buy, _place_tp_ladder, get_actual_fill_price, log_slippage
    load_hypothesis_db, save_hypothesis_db, _tr = _get_sniper_deps()

    if not isinstance(estimated_size, (int, float)) or estimated_size <= 0:
        return False
    signal_score = analysis.get("signal_score", 0)
    min_signal = analysis.get("min_signal", 50)
    if signal_score >= min_signal * 1.5:
        logger.info(f"[TRADE] High conviction ({signal_score:.0f} >= {min_signal * 1.5:.0f}), skipping advisor")
    else:
        approved, _verdict, _adv_conf, adv_reason = advisor_pre_check(market, analysis, estimated_size, balance)
        if not approved:
            logger.info(f"[TRADE-BLOCKED] {slug}: {adv_reason}")
            with contextlib.suppress(Exception):
                record_trade(mode="demo", slug=slug, action="skip", price=market["price"],
                             size_usd=estimated_size, p_model=analysis.get("p_model", 0),
                             confidence=analysis.get("confidence", 0),
                             signal_score=analysis.get("signal_score", 0),
                             prob_ratio=analysis.get("prob_ratio", 0),
                             reason=f"advisor_veto: {adv_reason}",
                             cluster=market.get("clusters", ["other"])[0] if market.get("clusters") else "",
                             source="sniper",
                             metadata={"advisor_verdict": _verdict, "advisor_blocked": True})
            return False

    if market["price"] < 0.10:
        max_slippage = max(0.50, market["price"] * 10)
    else:
        max_slippage = max(0.30, market["price"] * 2)
    current_ask = get_best_ask(slug)
    if current_ask is not None and current_ask > market["price"] * (1 + max_slippage):
        logger.warning(f"[SNIPER] Slippage guard: ask={current_ask:.4f} > {max_slippage:.0%} above price={market['price']:.4f}, aborting")
        with contextlib.suppress(Exception):
            record_trade(mode="demo", slug=slug, action="skip", price=market["price"],
                         size_usd=estimated_size, p_model=analysis.get("p_model", 0),
                         confidence=analysis.get("confidence", 0),
                         signal_score=analysis.get("signal_score", 0),
                         prob_ratio=analysis.get("prob_ratio", 0),
                         reason="slippage_guard",
                         cluster=market.get("clusters", ["other"])[0] if market.get("clusters") else "",
                         source="sniper",
                         metadata={"slippage_blocked": True, "ask": current_ask})
        return False

    positions_db.update(slug, {
        "status": "pending_fill",
        POS_ENTRY_PRICE: market.get("price", 0),
        POS_SHARES: 0,
        POS_OUTCOME: market.get("outcome", "yes"),
        POS_CLUSTERS: market.get("clusters", ["other"]),
        POS_MARKET_QUESTION: market.get("question", ""),
        "created_at": datetime.now(UTC).isoformat(),
    })
    logger.info(f"[TRADE] {slug[:60]} pending position recorded before buy")

    if not buy(market, estimated_size):
        print(f"   ❌ Buy failed for {slug}")
        return False

    time.sleep(10)
    fill_data = get_actual_fill_price(slug)
    if fill_data:
        log_slippage(slug, market["price"], fill_data)

    shares = round(float(fill_data.get("shares", 0))) if fill_data and fill_data.get("shares", 0) > 0 else round(estimated_size / market["price"]) if market["price"] > 0 else 0

    if shares <= 0:
        return False

    actual_price = fill_data.get("price", market["price"]) if fill_data else market["price"]

    positions_db.update(slug, {
        "status": "active",
        POS_ENTRY_PRICE: actual_price,
        POS_HIGH_PRICE: actual_price,
        POS_TRAILING_ON: False,
        POS_STOP_LOSS: round(actual_price * 0.80, 4),
        POS_LAST_CHECKED: datetime.now().isoformat(),
        POS_METACULUS_PROB: None,
        POS_MARKET_QUESTION: market.get("question", ""),
        POS_OUTCOME: market.get("outcome", "yes"),
        POS_CLUSTERS: market.get("clusters", ["other"]),
        POS_SHARES: shares,
    })

    with contextlib.suppress(Exception):
        from bayesian_updater import init_posterior
        init_posterior(slug, analysis.get("p_model", market["price"] * 2), market["price"])

    if shares > 0:
        ladder_results = _place_tp_ladder(market["slug"], market["outcome"], shares, entry_price=actual_price)
        for price, shares_placed, ok, _method in ladder_results:
            if ok:
                print(f"   🎯  TP rung placed @${price:.2f} ({shares_placed} shares)")
            else:
                print(f"   ⚠️  TP rung @{price:.2f} failed")
        if not ladder_results:
            print("   ⚠️  TP ladder placement failed, will rely on trailing_stop_check()")
    else:
        logger.warning(f"[SMART-EXIT] Zero shares for {market['slug']}, skipping TP")

    db = load_hypothesis_db()
    db[HYP_DB_HYPOTHESES].append({
        HYP_SLUG: market["slug"],
        HYP_QUESTION: market["question"],
        HYP_MARKET_PRICE: market["price"],
        HYP_P_MODEL: analysis["p_model"],
        HYP_PROB_RATIO: analysis["prob_ratio"],
        HYP_CONFIDENCE: analysis["confidence"],
        HYP_FACTORS: factors,
        HYP_CLUSTERS: market["clusters"],
        HYP_SIZE_PCT: estimated_size / balance,
        HYP_CREATED_AT: datetime.now().isoformat(),
        HYP_RESOLVED: False,
        HYP_TP_LIMIT_PLACED: True,
        HYP_TP_LIMIT_PRICE: SMART_EXIT_PRICE,
        HYP_SOURCE_SIGNAL: analysis.get("source_signal", "default"),
    })
    save_hypothesis_db(db)

    with contextlib.suppress(Exception):
        log_trade(
            event_type="BUY",
            slug=market["slug"],
            question=market["question"],
            entry_price=market["price"],
            shares=shares,
            invested=estimated_size,
            reason=analysis.get("reasoning", "")[:100],
        )

    with contextlib.suppress(Exception):
        record_trade(mode="demo", slug=market[HYP_SLUG], action="buy",
                     price=actual_price, size_usd=estimated_size, shares=shares,
                     p_model=analysis.get("p_model", 0),
                     confidence=analysis.get("confidence", 0),
                     signal_score=analysis.get("signal_score", 0),
                     prob_ratio=analysis.get("prob_ratio", 0),
                     reason=analysis.get("reasoning", "")[:200],
                     cluster=market.get("clusters", ["other"])[0] if market.get("clusters") else "",
                     source="sniper",
                     metadata={"factors": factors,
                               "source_signal": analysis.get("source_signal", "default"),
                               "advisor_verdict": "CONFIRM" if signal_score >= min_signal * 1.5 else "checked",
                               "slippage_blocked": False})

    if _tr():
        meta_prob = analysis.get("p_model")
        _tr().alert_new_position(
            market_slug=market["slug"],
            question=market["question"],
            entry_price=market["price"],
            amount=estimated_size,
            metaculus_prob=meta_prob,
            factors=factors,
            reasoning=analysis.get("reasoning", "")
        )

    return True
