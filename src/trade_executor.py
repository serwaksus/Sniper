import time
import logging
import contextlib
from datetime import datetime

import positions_db
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

logger = logging.getLogger(__name__)

SMART_EXIT_PRICE = 0.85
POSITIONS_FILE = "/root/dotm-sniper/positions.json"


def _get_sniper_deps():
    from dotm_sniper import load_hypothesis_db, save_hypothesis_db, _tr
    return load_hypothesis_db, save_hypothesis_db, _tr


def execute_trade(market, estimated_size, factors, analysis, balance):
    """Execute trade with advisor pre-check. Returns True if successful."""
    from signal_pipeline import advisor_pre_check
    from order_manager import get_best_ask, buy, _place_tp_ladder, get_actual_fill_price, log_slippage
    load_hypothesis_db, save_hypothesis_db, _tr = _get_sniper_deps()

    if not isinstance(estimated_size, (int, float)) or estimated_size <= 0:
        return False
    approved, _verdict, _adv_conf, adv_reason = advisor_pre_check(market, analysis, estimated_size, balance)
    if not approved:
        logger.info(f"[TRADE-BLOCKED] {market['slug']}: {adv_reason}")
        return False

    max_slippage = max(0.30, market["price"] * 2)
    current_ask = get_best_ask(market["slug"])
    if current_ask is not None and current_ask > market["price"] * (1 + max_slippage):
        logger.warning(f"[SNIPER] Slippage guard: ask={current_ask:.4f} > {max_slippage:.0%} above price={market['price']:.4f}, aborting")
        return False

    if not buy(market, estimated_size):
        print(f"   ❌ Buy failed for {market['slug']}")
        return False

    time.sleep(10)
    fill_data = get_actual_fill_price(market["slug"])
    if fill_data:
        log_slippage(market["slug"], market["price"], fill_data)

    shares = round(float(fill_data.get("shares", 0))) if fill_data and fill_data.get("shares", 0) > 0 else round(estimated_size / market["price"]) if market["price"] > 0 else 0

    if shares <= 0:
        return False

    positions = positions_db.load_all()
    if market["slug"] not in positions:
        positions[market["slug"]] = {
            POS_ENTRY_PRICE: fill_data.get("price", market["price"]) if fill_data else market["price"],
            POS_HIGH_PRICE: fill_data.get("price", market["price"]) if fill_data else market["price"],
            POS_TRAILING_ON: False,
            POS_STOP_LOSS: market["price"] * 0.7,
            POS_LAST_CHECKED: datetime.now().isoformat(),
            POS_METACULUS_PROB: None,
            POS_MARKET_QUESTION: market["question"],
            POS_OUTCOME: market.get("outcome", "yes"),
            POS_CLUSTERS: market.get("clusters", ["other"]),
            POS_SHARES: shares,
        }
        positions_db.save_all(positions)

    if shares > 0:
        ladder_results = _place_tp_ladder(market["slug"], market["outcome"], shares)
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
