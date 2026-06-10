"""Order book depth analysis via Polymarket CLOB API."""
from __future__ import annotations

import logging
import time

import requests
from typing import Any

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"
BOOK_CACHE_TTL = 300
_book_cache: dict[str, tuple[float, dict]] = {}


def fetch_order_book(condition_token_id: str, timeout: int = 10) -> dict[str, Any] | None:
    try:
        resp = requests.get(
            f"{CLOB_BASE}/book",
            params={"token_id": condition_token_id},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug(f"[ORDERBOOK] CLOB returned {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        logger.debug(f"[ORDERBOOK] fetch failed: {e}")
        return None


def compute_imbalance(bids: list[dict], asks: list[dict]) -> float:
    bid_vol = sum(float(b.get("size", 0)) * float(b.get("price", 0)) for b in bids[:20])
    ask_vol = sum(float(a.get("size", 0)) * float(a.get("price", 0)) for a in asks[:20])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def detect_bid_wall(bids: list[dict], price: float) -> tuple[bool, float]:
    if not bids or price <= 0:
        return False, 0.0
    top_bids = sorted(bids, key=lambda b: float(b.get("price", 0)), reverse=True)[:3]
    sizes = [float(b.get("size", 0)) * float(b.get("price", 0)) for b in top_bids]
    wall_size = sum(sizes)
    has_wall = wall_size >= 5000 or any(s >= 5000 for s in sizes)
    return has_wall, wall_size


def analyze_orderbook_depth(condition_token_id: str, market_price: float) -> dict[str, Any]:
    now = time.time()
    cached = _book_cache.get(condition_token_id)
    if cached and now - cached[0] < BOOK_CACHE_TTL:
        return cached[1]

    book = fetch_order_book(condition_token_id)
    if not book:
        return _empty_result("fetch_failed")

    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids and not asks:
        return _empty_result("empty_book")

    imbalance = compute_imbalance(bids, asks)
    bid_vol = sum(float(b.get("size", 0)) * float(b.get("price", 0)) for b in bids[:20])
    ask_vol = sum(float(a.get("size", 0)) * float(a.get("price", 0)) for a in asks[:20])
    has_wall, wall_size = detect_bid_wall(bids, market_price)

    score = 0
    reason = ""
    if imbalance > 0.4:
        score = 15
        reason = f"imbalance={imbalance:.2f}"
    elif has_wall:
        score = 12
        reason = f"bid_wall=${wall_size:.0f}"

    result = {
        "imbalance": round(imbalance, 3),
        "bid_volume_usd": round(bid_vol, 2),
        "ask_volume_usd": round(ask_vol, 2),
        "has_bid_wall": has_wall,
        "wall_size_usd": round(wall_size, 2),
        "signal_score": score,
        "reason": reason,
    }

    _book_cache[condition_token_id] = (now, result)
    return result


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "imbalance": 0.0,
        "bid_volume_usd": 0.0,
        "ask_volume_usd": 0.0,
        "has_bid_wall": False,
        "wall_size_usd": 0.0,
        "signal_score": 0,
        "reason": reason,
    }
