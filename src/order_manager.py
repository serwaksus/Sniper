from __future__ import annotations
import subprocess
import json
import os
import sys
import logging
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json
from schema import HYP_TP_LIMIT_PLACED
from config import SLIPPAGE_LOG_FILE

logger = logging.getLogger(__name__)

MAX_SPREAD_PCT = 0.40
LIMIT_SPREAD_THRESHOLD = 0.03
LIMIT_PRICE_BUFFER = 0.005
LIMIT_MAX_ATTEMPTS = 3


def get_order_book(slug: str) -> dict[str, float | None]:
    try:
        res = subprocess.run(["pm-trader", "book", slug, "--depth", "3"],
                           capture_output=True, text=True, timeout=15, start_new_session=True)
        data = json.loads(res.stdout)
        asks = data.get("data", {}).get("asks", [])
        bids = data.get("data", {}).get("bids", [])
        best_ask = float(asks[0]["price"]) if asks and float(asks[0].get("price", 0)) > 0 else None
        best_bid = float(bids[0]["price"]) if bids and float(bids[0].get("price", 0)) > 0 else None
        mid_price: float | None = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = best_ask or best_bid
        return {"best_bid": best_bid, "best_ask": best_ask, "mid_price": mid_price}
    except Exception as e:
        logger.debug(f"[order_manager] {type(e).__name__}: {e}")
        return {"best_bid": None, "best_ask": None, "mid_price": None}

def get_best_ask(slug: str) -> float | None:
    book = get_order_book(slug)
    return book.get("best_ask")

def get_balance() -> dict[str, Any] | None:
    try:
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[SNIPER] pm-trader balance failed: rc={res.returncode}")
            return None
        return json.loads(res.stdout).get("data", {})
    except Exception as e:
        logger.debug(f"[order_manager] {type(e).__name__}: {e}")
        return None

def get_portfolio() -> list[dict[str, Any]] | None:
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[SNIPER] pm-trader portfolio failed: rc={res.returncode}")
            return None
        data = json.loads(res.stdout).get("data", [])
        return [p for p in data if float(p.get("shares", 0)) > 0.001]
    except Exception as e:
        logger.debug(f"[order_manager] {type(e).__name__}: {e}")
        return None

def buy(market: dict[str, Any], amount: float) -> bool:
    try:
        book = get_order_book(market["slug"])
        best_ask = book.get("best_ask")
        best_bid = book.get("best_bid")

        if best_ask is None or best_ask <= 0:
            logger.warning(f"[SNIPER] No valid ask for {market['slug']}, aborting buy")
            return False

        market_price = market.get("price", 0)
        max_slippage = max(0.30, market_price * 2)
        max_acceptable = market_price * (1 + max_slippage)
        if market_price > 0 and best_ask > max_acceptable:
            logger.warning(
                f"[SNIPER] Slippage guard in buy(): ask={best_ask:.4f} > {max_acceptable:.4f} "
                f"({max_slippage:.0%} above price={market_price:.4f}) for {market['slug']}, aborting"
            )
            return False

        if best_bid is not None and best_bid > 0 and best_ask > 0:
            spread = best_ask - best_bid
            if spread / best_ask > MAX_SPREAD_PCT:
                logger.warning(
                    f"[SNIPER] Spread too wide in buy(): spread={spread:.4f} "
                    f"({spread/best_ask:.1%}) for {market['slug']}, aborting"
                )
                return False

        limit_price = min(best_ask * 1.15, max_acceptable)
        limit_price = max(limit_price, market_price)

        estimated_shares = int(amount / limit_price) if limit_price > 0 else 0
        if estimated_shares < 1:
            logger.warning(f"[SNIPER] Estimated shares < 1 for ${amount} @ {limit_price:.4f}, aborting")
            return False

        logger.info(
            f"[SNIPER] Placing limit buy for {market['slug'][:40]}... "
            f"${amount} @ limit={limit_price:.4f} (ask={best_ask:.4f}, max={max_acceptable:.4f})"
        )

        res = subprocess.run(
            ["pm-trader", "orders", "place", market["slug"], market["outcome"],
             "buy", str(estimated_shares), f"{limit_price:.4f}"],
            capture_output=True, text=True, timeout=30, start_new_session=True
        )
        if res.returncode != 0:
            logger.error(f"[SNIPER] Limit buy failed for {market['slug']}: rc={res.returncode} {res.stderr[:200]}")
            return False
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            print(f"  ✅ {market['question'][:45]}... ${amount} @ limit {limit_price:.4f}")
            return True
        else:
            logger.warning(f"[SNIPER] Limit buy not ok: {result}")
            print(f"  ❌ {result}")
            return False
    except Exception as e:
        logger.debug(f"[order_manager] {type(e).__name__}: {e}")
        print(f"  ❌ {e}")
        return False


def _place_limit_sell(slug: str, outcome: str, shares: float, limit_price: float) -> tuple[bool, str]:
    try:
        res = subprocess.run(
            ["pm-trader", "orders", "place", slug, outcome, "sell", str(int(shares)), f"{limit_price:.4f}"],
            capture_output=True, text=True, timeout=20, start_new_session=True
        )
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            return True, "limit_placed"
    except Exception as e:
        logger.warning(f"[limit_sell] {type(e).__name__}: {e}")
    return False, "limit_failed"


def _place_tp_limit_order_single(slug: str, outcome: str, shares: float, price: float) -> tuple[bool, str]:
    try:
        res = subprocess.run(
            ["pm-trader", "orders", "place", slug, outcome, "sell", str(int(shares)), f"{price:.4f}"],
            capture_output=True, text=True, timeout=20, start_new_session=True
        )
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            logger.info(f"[SMART-EXIT] TP limit placed for {slug[:40]}... @{price:.2f} (shares={shares})")
            return True, HYP_TP_LIMIT_PLACED
        else:
            logger.warning(f"[SMART-EXIT] TP limit failed for {slug[:40]}... response={result}")
    except Exception as e:
        logger.warning(f"[SMART-EXIT] Exception placing TP for {slug[:40]}...: {e}")
    return False, "tp_limit_failed"


def _place_tp_ladder(slug: str, outcome: str, total_shares: float, entry_price: float = 0) -> list[tuple[float, float, bool, str]]:
    """v5.4.0 TP Ladder: entry-price-based targets (2x and 3x), capped at 0.95"""
    existing = _get_open_tp_orders(slug)
    if existing:
        existing_prices = {o.get("limit_price") for o in existing}
        logger.info(f"[TP-LADDER] {slug[:40]}... {len(existing)} existing TP orders (prices={existing_prices}), skipping duplicate placement")
        return [(float(o.get("limit_price", 0)), float(o.get("amount", 0)), True, "existing") for o in existing]

    if entry_price <= 0:
        entry_price = 0.10

    rung1_price = min(entry_price * 2, 0.95)
    rung2_price = min(entry_price * 3, 0.95)

    if rung1_price >= 0.95:
        ladder = [(1.0, 0.90)]
    else:
        ladder = [(0.50, rung1_price), (0.50, rung2_price)]

    results: list[tuple[float, float, bool, str]] = []
    allocated = 0.0
    for pct, price in ladder:
        shares = round(total_shares * pct)
        shares = max(shares, 1)
        if shares * price < 5.0:
            if allocated < total_shares:
                shares = int(total_shares - allocated)
            if shares * price < 5.0:
                single_price = round(min(0.95, price * 1.5), 2)
                single_shares = max(1, round(total_shares))
                ok, m = _place_limit_sell(slug, outcome, single_shares, single_price)
                return [(single_price, float(single_shares), ok, m)] if ok else []
        ok, m = _place_tp_limit_order_single(slug, outcome, shares, price)
        results.append((price, float(shares), ok, m))
        allocated += shares
    logger.info(f"[TP-LADDER] {slug[:40]}... placed {len(results)} rungs, {total_shares - allocated} held to expiry")
    return results


def _get_open_tp_orders(slug: str) -> list[dict[str, Any]]:
    try:
        res = subprocess.run(["pm-trader", "orders", "list"], capture_output=True, text=True, timeout=30, start_new_session=True)
        data = json.loads(res.stdout) if res.stdout else {}
        orders = data.get("data", []) if isinstance(data.get("data"), list) else []
        return [o for o in orders if o.get("market_slug") == slug and o.get("side") == "sell" and o.get("status") == "pending"]
    except Exception as e:
        logger.debug(f"[order_manager] {type(e).__name__}: {e}")
        return []


def _cancel_all_tp_orders(slug: str) -> None:
    try:
        orders = _get_open_tp_orders(slug)
        for order in orders:
            order_id = order.get("id")
            if order_id:
                res = subprocess.run(["pm-trader", "orders", "cancel", str(order_id)], timeout=20, start_new_session=True)
                logger.info(f"[TP-CANCEL] Canceled sell order {order_id} for {slug[:40]}..., rc={res.returncode}")
    except Exception as e:
        logger.warning(f"[TP-CANCEL] Failed for {slug}: {e}")


def get_actual_fill_price(slug: str) -> dict[str, Any] | None:
    try:
        res = subprocess.run(
            ["pm-trader", "history", "--limit", "5"],
            capture_output=True, text=True, timeout=15, start_new_session=True
        )
        data = json.loads(res.stdout)
        for trade in data.get("data", []):
            if trade.get("market_slug") == slug and trade.get("side") == "buy":
                return {
                    "avg_price": float(trade.get("avg_price", 0)),
                    "amount_usd": float(trade.get("amount_usd", 0)),
                    "shares": float(trade.get("shares", 0)),
                    "slippage": float(trade.get("slippage", 0)),
                    "levels_filled": int(trade.get("levels_filled", 0)),
                }
    except Exception as e:
        logger.warning(f"[SLIPPAGE] Failed to get fill price for {slug}: {e}")
    return None


def log_slippage(slug: str, expected_price: float, fill_data: dict[str, Any] | None) -> None:
    if not fill_data:
        return
    actual_price = fill_data["avg_price"]
    slippage_pct = (actual_price - expected_price) / expected_price if expected_price > 0 else 0

    entry = {
        "slug": slug,
        "expected_price": expected_price,
        "actual_price": actual_price,
        "slippage_pct": round(slippage_pct, 4),
        "amount_usd": fill_data["amount_usd"],
        "shares": fill_data["shares"],
        "levels_filled": fill_data["levels_filled"],
        "timestamp": datetime.now().isoformat(),
    }

    try:
        logs = load_json(SLIPPAGE_LOG_FILE, [])
        logs.append(entry)
        logs = logs[-500:]
        os.makedirs(os.path.dirname(SLIPPAGE_LOG_FILE), exist_ok=True)
        save_json(SLIPPAGE_LOG_FILE, logs)
    except Exception as e:
        logger.warning(f"[SLIPPAGE] Failed to write log: {e}")

    if abs(slippage_pct) > 0.05:
        logger.warning(
            f"[SLIPPAGE-HIGH] {slug[:40]}... expected=${expected_price:.4f} "
            f"actual=${actual_price:.4f} slippage={slippage_pct:+.2%} "
            f"({fill_data['levels_filled']} levels, ${fill_data['amount_usd']:.2f})"
        )
