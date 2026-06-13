from __future__ import annotations
import subprocess
import json
import os
import sys
import logging
import contextlib
from datetime import datetime, timedelta, UTC
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import positions_db
from utils import load_json, save_json
from config import PRICE_HISTORY_FILE
from schema import (
    HYP_DB_RESOLVED, HYP_SLUG,
    POS_CLUSTERS, POS_ENTRY_PRICE, POS_HIGH_PRICE, POS_LAST_CHECKED,
    POS_LIMIT_SELL_ATTEMPTS, POS_LIMIT_SELL_PRICE, POS_LIMIT_SELL_SINCE,
    POS_MARKET_QUESTION, POS_METACULUS_PROB, POS_OUTCOME, POS_SELLING_IN_PROGRESS,
    POS_SHARES, POS_STOP_LOSS, POS_STOP_TYPE, POS_TP_LADDER_FAILED,
    POS_TRAILING_CONFIRM_TIME, POS_TRAILING_CONFIRMED, POS_TRAILING_ON,
)

logger = logging.getLogger(__name__)

TRAILING_ACTIVATION = 0.30
TRAILING_STOP = 0.25
CONVERGENCE_TAKE_PROFIT = 0.60
MIN_POSITION_CHECK_INTERVAL_HOURS = 3
ATR_STOP_MULTIPLIER = 2.5
ATR_TRAILING_MULTIPLIER = 1.5
ATR_LOOKBACK_DAYS = 7

MAX_SPREAD_PCT = 0.15
LIMIT_SPREAD_THRESHOLD = 0.03
LIMIT_PRICE_BUFFER = 0.005
LIMIT_MAX_ATTEMPTS = 3

_om: Any = None
_sniper: Any = None
_pm: Any = None
_et: Any = None

def _get_om() -> Any:
    global _om
    if _om is None:
        from order_manager import (
            get_order_book,
            get_portfolio,
            _get_open_tp_orders,
            _place_limit_sell,
            _place_tp_ladder,
            _cancel_all_tp_orders,
        )
        _om = type("OM", (), {
            "get_order_book": staticmethod(get_order_book),
            "get_portfolio": staticmethod(get_portfolio),
            "_get_open_tp_orders": staticmethod(_get_open_tp_orders),
            "_place_limit_sell": staticmethod(_place_limit_sell),
            "_place_tp_ladder": staticmethod(_place_tp_ladder),
            "_cancel_all_tp_orders": staticmethod(_cancel_all_tp_orders),
        })()
    return _om

def _get_sniper() -> Any:
    global _sniper
    if _sniper is None:
        from dotm_sniper import load_hypothesis_db, resolve_hypothesis_immediately, _tr
        from signal_pipeline import get_metaculus_forecast
        _sniper = type("Sniper", (), {
            "load_hypothesis_db": staticmethod(load_hypothesis_db),
            "get_metaculus_forecast": staticmethod(get_metaculus_forecast),
            "resolve_hypothesis_immediately": staticmethod(resolve_hypothesis_immediately),
            "_tr": staticmethod(_tr),
        })()
    return _sniper

def _get_pm() -> Any:
    global _pm
    if _pm is None:
        from position_manager import check_cluster_limits
        _pm = type("PM", (), {
            "check_cluster_limits": staticmethod(check_cluster_limits),
        })()
    return _pm

def _get_et() -> Any:
    global _et
    if _et is None:
        from equity_tracker import log_trade
        _et = type("ET", (), {
            "log_trade": staticmethod(log_trade),
        })()
    return _et


def _execute_sell(slug: str, outcome: str, shares: float, current_price: float, entry_price: float, force_market: bool = False) -> tuple[bool, float | None, str]:
    """
    Smart sell: use limit order when spread is wide, market order when safe or forced.
    Returns (sold: bool, effective_price: float or None, method: str)
    """
    book = _get_om().get_order_book(slug)
    best_bid = book.get("best_bid")
    best_ask = book.get("best_ask")

    if best_bid is None or best_bid <= 0:
        logger.warning(f"[SELL] {slug[:40]}... no bids at all")
        return False, None, "no_bids"

    spread = (best_ask - best_bid) if (best_bid and best_ask) else 0

    if not force_market and spread > LIMIT_SPREAD_THRESHOLD:
        positions = positions_db.load_all()
        pos = positions.get(slug, {})
        limit_attempts = pos.get(POS_LIMIT_SELL_ATTEMPTS, 0)

        if _get_om()._get_open_tp_orders(slug):
            logger.info(f"[LIMIT-SELL] {slug[:40]}... limit already pending")
        elif limit_attempts < LIMIT_MAX_ATTEMPTS:
            limit_price = best_bid + LIMIT_PRICE_BUFFER
            logger.info(
                f"[LIMIT-SELL] {slug[:40]}... spread=${spread:.4f} > ${LIMIT_SPREAD_THRESHOLD}, "
                f"placing limit at ${limit_price:.4f} (attempt {limit_attempts + 1}/{LIMIT_MAX_ATTEMPTS})"
            )
            ok, _reason = _get_om()._place_limit_sell(slug, outcome, shares, limit_price)
            if ok:
                pos[POS_LIMIT_SELL_ATTEMPTS] = limit_attempts + 1
                pos[POS_LIMIT_SELL_PRICE] = limit_price
                pos[POS_LIMIT_SELL_SINCE] = datetime.now().isoformat()
                positions_db.update(slug, pos)
                return False, limit_price, "limit_pending"

        logger.warning(
            f"[FORCE-MARKET] {slug[:40]}... {limit_attempts} limit attempts exhausted, forcing market sell"
        )

    logger.info(f"[MARKET-SELL] {slug[:40]}... bid={best_bid:.4f} spread=${spread:.4f}")
    try:
        om = _get_om()
        om._cancel_all_tp_orders(slug)
        current_portfolio = om.get_portfolio()
        if current_portfolio is None:
            return False, best_bid, "portfolio_error"
        actual_shares = 0.0
        for p in current_portfolio:
            if p.get("slug") == slug or p.get("market_slug") == slug:
                actual_shares = float(p.get("size", p.get("shares", 0)))
                break
        if actual_shares <= 0:
            logger.warning(f"[SELL] {slug[:40]}... actual_shares=0, position already sold")
            return False, best_bid, "already_sold"

        sell_res = subprocess.run(["pm-trader", "sell", slug, outcome, str(int(actual_shares))],
                             capture_output=True, text=True, timeout=20, start_new_session=True)
        result = json.loads(sell_res.stdout) if sell_res.stdout else {}
        if result.get("ok"):
            return True, best_bid, "market"

        if not result.get("ok"):
            limit_price = max(best_bid, 0.01)
            ok, _reason = om._place_limit_sell(slug, outcome, actual_shares, limit_price)
            if ok:
                logger.info(f"[AGGRESSIVE-LIMIT] {slug[:40]}... market sell failed, placed limit at bid={limit_price:.4f}")
                return False, limit_price, "limit_pending"
            logger.warning(f"[SELL] {slug[:40]}... both market and limit failed: {result.get('error', 'unknown')}")
    except Exception as e:
        logger.warning(f"[market_sell] {type(e).__name__}: {e}")
    return False, best_bid, "market_failed"


def _log_price_for_atr(slug: str, price: float) -> None:
    try:
        history = load_json(PRICE_HISTORY_FILE, {})
        if not isinstance(history, dict):
            history = {}
        slug_data = history.get(slug, [])
        slug_data.append({"t": datetime.now().isoformat(), "p": round(price, 6)})
        if len(slug_data) > 1008:
            slug_data = slug_data[-1008:]
        history[slug] = slug_data
        save_json(PRICE_HISTORY_FILE, history)
    except Exception as e:
        logger.debug(f"[atr_log] {type(e).__name__}: {e}")


def _calculate_atr(slug: str, current_price: float) -> float:
    try:
        history = load_json(PRICE_HISTORY_FILE, {})
        if not isinstance(history, dict):
            return abs(current_price) * 0.10
        slug_data = history.get(slug, [])
        if len(slug_data) < 2:
            return abs(current_price) * 0.10

        cutoff = (datetime.now() - timedelta(days=ATR_LOOKBACK_DAYS)).isoformat()
        recent = [e for e in slug_data if e.get("t", "") >= cutoff]

        if len(recent) < 2:
            return abs(current_price) * 0.10

        true_ranges = []
        for i in range(1, len(recent)):
            h = abs(recent[i]["p"] - recent[i - 1]["p"])
            true_ranges.append(h)

        if not true_ranges:
            return abs(current_price) * 0.10

        atr = sum(true_ranges) / len(true_ranges)
        return max(atr, abs(current_price) * 0.03)
    except Exception as e:
        logger.debug(f"[sell_executor] {type(e).__name__}: {e}")
        return abs(current_price) * 0.10


def _get_atr_stop(slug: str, entry_price: float, current_price: float) -> float:
    atr = _calculate_atr(slug, current_price)
    return current_price - ATR_STOP_MULTIPLIER * atr


def _get_atr_trailing_stop(slug: str, high_price: float, current_price: float) -> float:
    atr = _calculate_atr(slug, current_price)
    return high_price - ATR_TRAILING_MULTIPLIER * atr


def _check_sell_safety(slug: str, current_price: float, shares: float) -> tuple[bool, str, float | None]:
    """
    Verify order book has sufficient liquidity before placing a market sell.
    Returns (safe: bool, reason: str, effective_price: float or None)
    """
    book = _get_om().get_order_book(slug)
    best_bid = book.get("best_bid")
    best_ask = book.get("best_ask")

    if best_bid is None or best_bid <= 0:
        logger.warning(f"[SLIPPAGE-GUARD] {slug[:40]}... no bids in order book, aborting sell")
        return False, "no_bids", None

    if best_ask and best_ask > 0:
        spread = (best_ask - best_bid) / best_ask
        if spread > MAX_SPREAD_PCT:
            logger.warning(
                f"[SLIPPAGE-GUARD] {slug[:40]}... spread={spread:.1%} > {MAX_SPREAD_PCT:.0%}, "
                f"bid={best_bid:.4f} ask={best_ask:.4f}, aborting sell"
            )
            return False, f"spread_too_wide:{spread:.1%}", best_bid

    bid_threshold = 0.40 if current_price < 0.15 else 0.70
    if best_bid < current_price * bid_threshold:
        logger.warning(
            f"[SLIPPAGE-GUARD] {slug[:40]}... best_bid={best_bid:.4f} is >{(1-bid_threshold)*100:.0f}% below mid={current_price:.4f}, "
            f"likely empty order book, aborting sell"
        )
        return False, f"bid_far_from_mid:{best_bid:.4f}_vs_{current_price:.4f}", best_bid

    logger.info(
        f"[SLIPPAGE-GUARD] {slug[:40]}... OK: bid={best_bid:.4f} ask={best_ask} mid={current_price:.4f}"
    )
    return True, "ok", best_bid

def trailing_stop_check() -> None:
    om = _get_om()
    sniper = _get_sniper()
    portfolio = om.get_portfolio()
    if portfolio is None:
        logger.error("[STOP_LOSS] Portfolio API error, skipping this cycle")
        return
    if not portfolio:
        pass

    current_slugs = {p["market_slug"] for p in portfolio}
    now = datetime.now()

    all_pos = positions_db.load_all()
    orphan_tp_slugs = [s for s in all_pos if s not in current_slugs]
    for s in orphan_tp_slugs:
        with contextlib.suppress(Exception):
            om._cancel_all_tp_orders(s)
            logger.info(f"[CLEANUP-TP] Cancelled orphan TP orders for {s[:40]}...")

    db = sniper.load_hypothesis_db()
    resolved_slugs = {h[HYP_SLUG] for h in db.get(HYP_DB_RESOLVED, [])}

    for pos in portfolio:
        slug = pos["market_slug"]
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue

        stored_pos = positions_db.get(slug)
        stored_entry = stored_pos.get(POS_ENTRY_PRICE, 0) if stored_pos else 0
        entry_price = stored_entry if stored_entry > 0 else pos.get("avg_entry_price", 0)
        outcome = pos.get("outcome", "yes")

        if slug in resolved_slugs:
            if positions_db.get(slug) is not None:
                positions_db.delete(slug)
                logger.info(f"[SKIP-RESOLVED] {slug[:40]}... already resolved in hypothesis_db, removed from positions")
                try:
                    from bayesian_updater import cleanup_slug
                    cleanup_slug(slug)
                except Exception as e:
                    logger.debug(f"[bayesian_cleanup] {type(e).__name__}: {e}")
            continue

        book = om.get_order_book(slug)
        mid_price = book.get("mid_price")
        live_price = pos.get("live_price", 0)
        current_price = mid_price if mid_price is not None else live_price

        if current_price <= 0:
            continue

        if entry_price <= 0:
            entry_price = current_price
            logger.warning(f"[EMERGENCY] {slug[:40]}... entry_price=0, using current_price={current_price:.4f}")

        _log_price_for_atr(slug, current_price)

        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        positions = positions_db.load_all()
        if slug not in positions:
            if shares <= 0:
                logger.info(f"[SKIP] {slug[:40]}... 0 shares in portfolio, not recreating")
                continue
            atr_stop = _get_atr_stop(slug, entry_price, current_price)
            positions[slug] = {
                POS_ENTRY_PRICE: entry_price,
                POS_HIGH_PRICE: max(entry_price, current_price),
                POS_TRAILING_ON: False,
                POS_STOP_LOSS: atr_stop,
                POS_STOP_TYPE: "atr",
                POS_LAST_CHECKED: now.isoformat(),
                POS_METACULUS_PROB: None,
                POS_MARKET_QUESTION: pos.get("market_question", ""),
                POS_OUTCOME: pos.get("outcome", "yes"),
                POS_CLUSTERS: pos.get("clusters", []),
                POS_SHARES: shares
            }
            positions_db.save_all(positions)

        p = positions[slug]

        if p.get(POS_SELLING_IN_PROGRESS):
            sell_since = p.get(POS_LIMIT_SELL_SINCE)
            if sell_since:
                try:
                    elapsed = (datetime.now(UTC) - datetime.fromisoformat(str(sell_since))).total_seconds()
                    if elapsed > 3600:
                        logger.warning(f"[STUCK-SELL] {slug[:60]} selling_in_progress for {int(elapsed)}s, clearing")
                        p[POS_SELLING_IN_PROGRESS] = False
                        positions_db.update(slug, {POS_SELLING_IN_PROGRESS: False})
                    else:
                        logger.info(f"[SKIP] {slug[:40]}... sell already in progress")
                        continue
                except (ValueError, TypeError):
                    pass
            else:
                logger.warning(f"[STUCK-SELL] {slug[:60]} selling_in_progress without timestamp, clearing")
                p[POS_SELLING_IN_PROGRESS] = False
                positions_db.update(slug, {POS_SELLING_IN_PROGRESS: False})

        last_checked = None
        if p.get(POS_LAST_CHECKED):
            try:
                last_checked = datetime.fromisoformat(p[POS_LAST_CHECKED])
            except Exception as e:
                logger.debug(f"[sell_executor] {type(e).__name__}: {e}")
                last_checked = None

        check_interval = MIN_POSITION_CHECK_INTERVAL_HOURS * 3600
        if last_checked and (now - last_checked).total_seconds() < check_interval:
            logger.info(f"[POLLING] {slug[:40]}... skipping, checked {(now - last_checked).total_seconds()/3600:.1f}h ago")
            continue

        p[POS_LAST_CHECKED] = now.isoformat()

        p[POS_HIGH_PRICE] = max(p.get(POS_HIGH_PRICE, current_price), current_price, entry_price)

        if p[POS_HIGH_PRICE] > entry_price * (1 + TRAILING_ACTIVATION):
            p[POS_TRAILING_ON] = True
            atr_trail = _get_atr_trailing_stop(slug, p[POS_HIGH_PRICE], current_price)
            fixed_trail = p[POS_HIGH_PRICE] * (1 - TRAILING_STOP)
            p[POS_STOP_LOSS] = max(atr_trail, fixed_trail)

        meta = sniper.get_metaculus_forecast(pos.get("market_question", ""), None)
        metaculus_prob = None
        if meta.get("found"):
            metaculus_prob = meta.get("probability")
            p[POS_METACULUS_PROB] = metaculus_prob

        positions[slug] = p
        positions_db.save_all(positions)

        if not om._get_open_tp_orders(slug) and current_price < 0.70 and not p.get(POS_TP_LADDER_FAILED):
            try:
                ladder_results = om._place_tp_ladder(slug, outcome, shares, entry_price=entry_price)
                if any(ok for _, _, ok, _ in ladder_results):
                    logger.info(f"[TP-REFRESH] Placed TP ladder for {slug[:40]}... (was missing)")
                else:
                    p[POS_TP_LADDER_FAILED] = True
            except Exception as e:
                logger.debug(f"[sell_executor] {type(e).__name__}: {e}")
                p[POS_TP_LADDER_FAILED] = True

        sold = False
        sold_reason = ""

        p[POS_SELLING_IN_PROGRESS] = True
        positions[slug] = p
        positions_db.save_all(positions)

        convergence = None
        if metaculus_prob and metaculus_prob > 0.05:
            convergence = current_price / metaculus_prob
            logger.info(f"[CONVERGENCE] {slug[:40]}... mid={current_price:.3f}, meta={metaculus_prob:.0%}, ratio={convergence:.2f}")
            if convergence >= CONVERGENCE_TAKE_PROFIT and not om._get_open_tp_orders(slug):
                sold_reason = f"convergence={convergence:.2f} >= {CONVERGENCE_TAKE_PROFIT}"
                logger.info(f"[TAKE-PROFIT] Gap convergence reached, no TP ladder: {sold_reason}")
                try:
                    sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                    if sold:
                        if eff_price is None:
                            eff_price = current_price
                        logger.info(f"SOLD take-profit convergence ({method}): {slug} pnl={pnl_pct:.2%}")
                        pnl_abs = shares * (eff_price - entry_price)
                        actual_pnl = (eff_price - entry_price) / entry_price if entry_price > 0 else pnl_pct
                        if sniper._tr():
                            sniper._tr().alert_convergence(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs, convergence)
                    else:
                        logger.warning(f"[CONVERGENCE-SELL] Failed to sell {slug[:40]}... method={method}")
                except Exception as e:
                    logger.warning(f"[CONVERGENCE-SELL] Failed for {slug}: {e}")
            elif convergence >= CONVERGENCE_TAKE_PROFIT:
                logger.info(f"[CONVERGENCE] {slug[:40]}... convergence={convergence:.2f} but TP ladder active, letting limits execute")

        if not sold and entry_price > 0 and current_price <= entry_price * 0.50:
            sold_reason = f"hard_stop_50pct: price=${current_price:.4f} <= entry*0.50=${entry_price * 0.50:.4f}"
            logger.warning(f"[HARD-STOP] -50% stop triggered: {slug[:40]}... price=${current_price:.4f} entry=${entry_price:.4f}")
            try:
                sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price, force_market=True)
                if method == "already_sold":
                    sold = True
                    positions = positions_db.load_all()
                    if slug in positions:
                        del positions[slug]
                        positions_db.save_all(positions)
                    logger.info(f"[HARD-STOP] {slug[:40]}... already sold, cleaned up from positions_db")
                elif sold:
                    eff_price = eff_price or current_price
                    actual_pnl = (eff_price - entry_price) / entry_price
                    pnl_abs = shares * (eff_price - entry_price)
                    logger.info(f"SOLD hard stop -50% ({method}): {slug} pnl={actual_pnl:.2%}")
                    if sniper._tr():
                        sniper._tr().alert_stop_loss(market_slug=slug, question=pos.get("market_question", ""), pnl_pct=actual_pnl * 100, pnl_abs=pnl_abs)
            except Exception as e:
                logger.warning(f"[hard_stop_sell] {type(e).__name__}: {e}")

        if not sold and stored_pos:
            created_at = stored_pos.get("created_at")
            ttl_hours = stored_pos.get("ttl_hours")
            if created_at and ttl_hours and ttl_hours > 0:
                try:
                    age_hours = (now - datetime.fromisoformat(str(created_at))).total_seconds() / 3600
                    age_ratio = age_hours / ttl_hours
                    if age_ratio > 0.60 and current_price < entry_price * 1.5:
                        sold_reason = f"time_decay: age={age_ratio:.0%} of TTL, price=${current_price:.4f} < entry*1.5"
                        logger.info(f"[TIME-DECAY] Selling stale position: {slug[:40]}... age={age_ratio:.0%}")
                        sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                        if sold:
                            eff_price = eff_price or current_price
                            actual_pnl = (eff_price - entry_price) / entry_price
                            pnl_abs = shares * (eff_price - entry_price)
                            logger.info(f"SOLD time-decay ({method}): {slug} pnl={actual_pnl:.2%}")
                            if sniper._tr():
                                sniper._tr().alert_stop_loss(market_slug=slug, question=pos.get("market_question", ""), pnl_pct=actual_pnl * 100, pnl_abs=pnl_abs)
                except (ValueError, TypeError) as e:
                    logger.debug(f"[time_decay] {type(e).__name__}: {e}")

        if not sold and current_price <= p.get(POS_STOP_LOSS, 0):
            sold_reason = f"atr_stop: price=${current_price:.4f} <= atr_stop"
            logger.warning(f"[STOP-LOSS] ATR stop triggered: {slug[:40]}... price=${current_price:.4f}")
            try:
                pos_data = positions.get(slug, {})
                limit_attempts = pos_data.get(POS_LIMIT_SELL_ATTEMPTS, 0)
                safety_failures = pos_data.get("_safety_failures", 0)
                force = limit_attempts >= LIMIT_MAX_ATTEMPTS or safety_failures >= 5
                if not force:
                    safe, safe_reason, _sell_price = _check_sell_safety(slug, current_price, shares)
                    if not safe:
                        pos_data["_safety_failures"] = safety_failures + 1
                        positions_db.update(slug, pos_data)
                        logger.warning(
                            f"[STOP-DELAYED] {slug[:40]}... sell unsafe ({safety_failures + 1}/5): {safe_reason}. "
                            f"mid={current_price:.4f} entry={entry_price:.4f}"
                        )
                        if sniper._tr():
                            sniper._tr().alert_stop_loss(slug, pos.get("market_question", ""), pnl_pct * 100, shares * (current_price - entry_price))
                    else:
                        sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                        if sold:
                            if eff_price is None:
                                eff_price = current_price
                            actual_pnl = (eff_price - entry_price) / entry_price if entry_price > 0 else pnl_pct
                            logger.info(f"SOLD hard stop ({method}): {slug} mid_pnl={pnl_pct:.2%} eff_pnl={actual_pnl:.2%}")
                            pnl_abs = shares * (eff_price - entry_price)
                            if sniper._tr():
                                sniper._tr().alert_stop_loss(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
                else:
                    logger.warning(f"[EMERGENCY-SELL] {slug[:40]}... forcing market after {limit_attempts} limit attempts")
                    sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price, force_market=True)
                    if sold:
                        if eff_price is None:
                            eff_price = current_price
                        actual_pnl = (eff_price - entry_price) / entry_price if entry_price > 0 else pnl_pct
                        logger.info(f"SOLD emergency ({method}): {slug} pnl={actual_pnl:.2%}")
                        pnl_abs = shares * (eff_price - entry_price)
                        if sniper._tr():
                            sniper._tr().alert_stop_loss(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
            except Exception as e:
                logger.warning(f"[stop_loss_sell] {type(e).__name__}: {e}")

        if not sold and p.get(POS_TRAILING_ON) and current_price <= p.get(POS_STOP_LOSS, 0):
            if not p.get(POS_TRAILING_CONFIRMED):
                p[POS_TRAILING_CONFIRMED] = True
                p[POS_TRAILING_CONFIRM_TIME] = now.isoformat()
                logger.info(f"[TRAILING-STOP] Confirming for {slug[:40]}... (1/2)")
                p.pop(POS_SELLING_IN_PROGRESS, None)
                positions[slug] = p
                positions_db.save_all(positions)
                continue
            else:
                confirm_time = p.get(POS_TRAILING_CONFIRM_TIME)
                if confirm_time:
                    try:
                        elapsed = (now - datetime.fromisoformat(confirm_time)).total_seconds()
                        if elapsed < 300:
                            logger.info(f"[TRAILING-STOP] Waiting confirmation for {slug[:40]}... ({elapsed:.0f}s/300s)")
                            p.pop(POS_SELLING_IN_PROGRESS, None)
                            positions[slug] = p
                            positions_db.save_all(positions)
                            continue
                    except (ValueError, TypeError):
                        pass
            sold_reason = f"trailing={current_price:.3f} <= {p.get(POS_STOP_LOSS, 0):.3f}"
            logger.info(f"[TRAILING-STOP] Triggered for {slug[:40]}...")
            try:
                sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                if sold:
                    if eff_price is None:
                        eff_price = current_price
                    logger.info(f"SOLD trailing stop ({method}): {slug}")
                    p.pop(POS_TRAILING_CONFIRMED, None)
                    p.pop(POS_TRAILING_CONFIRM_TIME, None)
                    pnl_abs = shares * (eff_price - entry_price)
                    actual_pnl = (eff_price - entry_price) / entry_price if entry_price > 0 else pnl_pct
                    if sniper._tr():
                        if actual_pnl > 0:
                            sniper._tr().alert_take_profit(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
                        else:
                            sniper._tr().alert_stop_loss(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
            except Exception as e:
                logger.warning(f"[trailing_stop_sell] {type(e).__name__}: {e}")

        if not sold and current_price >= 0.75:
            tp_orders = om._get_open_tp_orders(slug)
            if not tp_orders:
                sold_reason = f"price=${current_price:.3f} >= $0.75 (TP ladder fallback)"
                logger.info(f"[TAKE-PROFIT] {slug[:40]}... price=${current_price:.3f} (no TP ladder, selling)")
                try:
                    sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                    if sold:
                        if eff_price is None:
                            eff_price = current_price
                        logger.info(f"SOLD take-profit ({method}): {slug}")
                        pnl_abs = shares * (eff_price - entry_price)
                        if sniper._tr():
                            sniper._tr().alert_take_profit(slug, pos.get("market_question", ""), pnl_pct * 100, pnl_abs)
                except Exception as e:
                    logger.warning(f"[take_profit_sell] {type(e).__name__}: {e}")
            else:
                logger.info(f"[TAKE-PROFIT] {slug[:40]}... +{pnl_pct:.0f}% but TP ladder active, letting limit orders execute")

        if sold:
            om._cancel_all_tp_orders(slug)
            sniper.resolve_hypothesis_immediately(slug, current_price, entry_price)
            try:
                eff_price_val = eff_price if eff_price else current_price
                actual_pnl_val = (eff_price_val - entry_price) / entry_price if entry_price > 0 else 0
                _get_et().log_trade(
                    event_type="SELL",
                    slug=slug,
                    question=pos.get("market_question", ""),
                    entry_price=entry_price,
                    exit_price=eff_price_val,
                    shares=shares,
                    invested=shares * entry_price,
                    pnl_pct=actual_pnl_val * 100,
                    pnl_abs=shares * (eff_price_val - entry_price),
                    reason=sold_reason,
                )
            except Exception as e:
                logger.warning(f"[trade_log] {type(e).__name__}: {e}")
            positions_db.delete(slug)
            try:
                from bayesian_updater import cleanup_slug
                cleanup_slug(slug)
            except Exception as e:
                logger.debug(f"[bayesian_cleanup] {type(e).__name__}: {e}")
        else:
            p.pop(POS_SELLING_IN_PROGRESS, None)
            if not sold and sold_reason:
                override_ts = (now - timedelta(hours=MIN_POSITION_CHECK_INTERVAL_HOURS - 0.25)).isoformat()
                p[POS_LAST_CHECKED] = override_ts
                logger.info(f"[RETRY-SOON] {slug[:40]}... sell failed ({sold_reason}), recheck in 15min")
            positions[slug] = p
            positions_db.save_all(positions)

    current_slugs = {p["market_slug"] for p in portfolio}
    all_pos = positions_db.load_all()
    stale = []
    for s in list(all_pos.keys()):
        if s in resolved_slugs:
            stale.append(s)
            continue
        if s not in current_slugs:
            pos_data = all_pos[s]
            miss_count = pos_data.get("_miss_count", 0) + 1
            pos_data["_miss_count"] = miss_count
            if miss_count >= 3:
                stale.append(s)
            else:
                all_pos[s] = pos_data
    positions_db.save_all(all_pos)
    for s in stale:
        with contextlib.suppress(Exception):
            om._cancel_all_tp_orders(s)
        positions_db.delete(s)
        try:
            from bayesian_updater import cleanup_slug
            cleanup_slug(s)
        except Exception as e:
            logger.debug(f"[bayesian_cleanup] {type(e).__name__}: {e}")
        logger.info(f"[CLEANUP] Removed stale position: {s}")


def check_portfolio_drawdown() -> bool:
    """Check if portfolio is in >10% drawdown from high-water mark. Returns True if trading should pause."""
    try:
        from config import EQUITY_HISTORY_FILE
        from utils import load_json
        history = load_json(EQUITY_HISTORY_FILE, [])
        if not isinstance(history, list) or len(history) < 2:
            return False

        high_water = max(s.get("total_equity", 0) for s in history if isinstance(s, dict))
        current = history[-1].get("total_equity", 0) if isinstance(history[-1], dict) else 0

        if high_water <= 0:
            return False

        drawdown = (high_water - current) / high_water
        if drawdown >= 0.10:
            logger.warning(f"[DRAWDOWN-STOP] Portfolio drawdown {drawdown:.1%} >= 10%, pausing new trades")
            return True
        return False
    except Exception as e:
        logger.debug(f"[sell_executor] {type(e).__name__}: {e}")
        return False
