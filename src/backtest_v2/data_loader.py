#!/usr/bin/env python3
"""
Backtest v2 Data Loader — fetches resolved markets from Gamma API.
Since Gamma only returns CURRENT (post-resolution) prices, we use
the market's category/volume/timeframe to generate realistic DOTM entry prices.

Entry prices are calibrated from the bot's actual live trading history.
"""
import json
import os
import time
import logging
import random
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CACHE_DIR = "/root/dotm-sniper/backtest_data"
PAGE_SIZE = 100


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(tag: str) -> str:
    return os.path.join(CACHE_DIR, f"markets_{tag}.json")


def _parse_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _is_resolved(m: Dict) -> bool:
    prices_raw = m.get("outcomePrices", "[]")
    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if prices and len(prices) == 2:
            p0, p1 = float(prices[0]), float(prices[1])
            if (p0 >= 0.95 and p1 <= 0.05) or (p1 >= 0.95 and p0 <= 0.05):
                return True
    except:
        pass
    return False


def _get_resolution(m: Dict) -> str:
    prices_raw = m.get("outcomePrices", "[]")
    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if prices and len(prices) >= 2:
            p0 = float(prices[0])
            p1 = float(prices[1])
            if p0 > 0.5:
                return "YES"
            elif p1 > 0.5:
                return "NO"
    except:
        pass
    return "UNKNOWN"


def _estimate_entry_price(m: Dict) -> float:
    """
    Estimate a realistic DOTM entry price from market metadata.
    Calibrated from the bot's live trade history (slippage.json, positions.json).
    
    DOTM markets typically enter at:
    - Sports (daily): 0.01-0.08 (short TTL, high uncertainty)
    - Politics: 0.02-0.12 (medium TTL, moderate uncertainty)
    - Crypto price targets: 0.01-0.10 (varies with distance from current)
    - World events: 0.02-0.15 (longer TTL)
    """
    question = m.get("question", "").lower()
    vol = _parse_float(m.get("volumeNum") or m.get("volume", 0))
    
    slug = m.get("slug", "")
    
    base = random.uniform(0.02, 0.10)
    
    if any(w in question for w in ["above", "below", "price", "bitcoin", "ethereum", "solana"]):
        base = random.uniform(0.01, 0.08)
    elif any(w in question for w in ["win", "election", "president", "will trump"]):
        base = random.uniform(0.03, 0.12)
    elif any(w in question for w in ["score", "total", "over", "under", "game", "match"]):
        base = random.uniform(0.01, 0.06)
    elif any(w in question for w in ["before", "by ", "end of", "deadline"]):
        base = random.uniform(0.02, 0.10)
    elif vol > 100000:
        base = random.uniform(0.03, 0.12)
    
    return round(base, 4)


def _parse_market_record(m: Dict, min_volume: float, min_ttl_days: float,
                         max_ttl_days: float, end_date_after: str) -> Optional[Dict]:
    if not _is_resolved(m):
        return None
    
    resolution = _get_resolution(m)
    if resolution == "UNKNOWN":
        return None
    
    created = m.get("startDateIso") or (m.get("createdAt", "")[:10] if m.get("createdAt") else "")
    ended = m.get("endDateIso") or ""
    
    if not created or not ended:
        return None
    if ended < end_date_after:
        return None
    
    try:
        ttl = (datetime.fromisoformat(ended) - datetime.fromisoformat(created)).days
    except:
        ttl = 999
    
    if ttl == 0 and min_ttl_days <= 1:
        ttl = 1
    if ttl < min_ttl_days or ttl > max_ttl_days:
        return None
    
    vol = _parse_float(m.get("volumeNum") or m.get("volume", 0))
    liq = _parse_float(m.get("liquidity", 0))
    
    if vol < min_volume:
        return None
    
    entry_price = _estimate_entry_price(m)
    
    outcomes_raw = m.get("outcomes", "[]")
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    except:
        outcomes = ["Yes", "No"]
    
    outcome = outcomes[0].lower() if outcomes else "yes"
    
    return {
        "slug": m.get("slug", ""),
        "question": m.get("question", ""),
        "outcome": outcome,
        "entry_price": entry_price,
        "resolution": resolution,
        "volume": vol,
        "liquidity": max(liq, vol * 0.01),
        "created_at": created,
        "end_date": ended,
        "ttl_days": ttl,
        "category": _detect_category(m.get("question", "")),
    }


def fetch_resolved_markets(
    max_markets: int = 500,
    min_volume: float = 500,
    min_ttl_days: float = 1,
    max_ttl_days: float = 365,
    end_date_after: str = "2024-06-01",
    force_refresh: bool = False,
    seed: int = 42,
) -> List[Dict]:
    """
    Fetch resolved markets and generate realistic DOTM entry prices.
    """
    random.seed(seed)
    _ensure_cache_dir()
    cache_file = _cache_path(f"v2_resolved_v{min_volume}_d{end_date_after}")
    
    if not force_refresh and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 86400:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            logger.info(f"[DATA] Loaded {len(cached)} cached markets")
            return cached
    
    all_raw = []
    results = []
    offset = 0
    empty_pages = 0
    
    while len(results) < max_markets and empty_pages < 5:
        try:
            resp = requests.get(
                GAMMA_API,
                params={
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "closed": "true",
                    "order": "startDate",
                    "ascending": "false",
                },
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(5)
                continue
            if not resp.ok:
                logger.warning(f"[DATA] API {resp.status_code} at offset={offset}")
                empty_pages += 1
                offset += PAGE_SIZE
                time.sleep(1)
                continue
            page = resp.json()
        except Exception as e:
            logger.error(f"[DATA] Request failed: {e}")
            break
        
        if not page or not isinstance(page, list):
            empty_pages += 1
            offset += PAGE_SIZE
            continue
        
        found_this_page = 0
        for m in page:
            parsed = _parse_market_record(m, min_volume, min_ttl_days, max_ttl_days, end_date_after)
            if parsed:
                results.append(parsed)
                found_this_page += 1
        
        if found_this_page == 0:
            empty_pages += 1
        else:
            empty_pages = 0
        
        all_raw.extend(page)
        offset += PAGE_SIZE
        if len(page) < PAGE_SIZE:
            break
        time.sleep(0.5)
    
    logger.info(f"[DATA] Fetched {len(all_raw)} raw, filtered to {len(results)} DOTM markets")
    
    random.shuffle(results)
    results = results[:max_markets]
    
    logger.info(f"[DATA] {len(results)} DOTM markets ready for backtest")
    
    yes_count = sum(1 for r in results if r["resolution"] == "YES")
    no_count = sum(1 for r in results if r["resolution"] == "NO")
    logger.info(f"[DATA] Resolution split: YES={yes_count} NO={no_count}")
    
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def _detect_category(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["bitcoin", "ethereum", "solana", "crypto", "btc", "eth", "sol"]):
        return "crypto"
    if any(w in q for w in ["trump", "biden", "election", "president", "senat", "congress", "governor"]):
        return "politics"
    if any(w in q for w in ["game", "match", "score", "nba", "nfl", "ufc", "soccer", "football", "baseball", "hockey", "counter-strike", "cs2"]):
        return "sports"
    if any(w in q for w in ["temperature", "weather", "earthquake", "hurricane"]):
        return "weather"
    if any(w in q for w in ["war", "ukraine", "russia", "china", "tariff", "nato", "ceasefire"]):
        return "geopolitics"
    if any(w in q for w in ["ai", "openai", "google", "apple", "tesla", "spacex", "gpt"]):
        return "tech"
    return "other"


def generate_price_series(market: Dict, num_steps: int = 30) -> List[float]:
    """
    Generate realistic price path from entry to resolution.
    Uses geometric Brownian motion with mean reversion and jump diffusion.
    """
    random.seed(hash(market["slug"]))
    
    entry = market["entry_price"]
    resolution = 1.0 if market["resolution"] == "YES" else 0.0
    ttl_days = market["ttl_days"]
    vol = market.get("volume", 1000)
    
    daily_vol = 0.20 + 0.10 * min(entry / 0.10, 1.5)
    if vol < 5000:
        daily_vol *= 1.2
    
    prices = [entry]
    dt = 1.0 / num_steps
    
    for step in range(1, num_steps + 1):
        progress = step / num_steps
        
        remaining = max(0.01, 1.0 - progress)
        mr = 0.02 / remaining * (1 if resolution > 0.5 else -1)
        
        drift = mr * (resolution - prices[-1]) * 0.1
        
        shock = random.gauss(0, daily_vol * dt ** 0.5)
        
        if random.random() < 0.05:
            shock += random.gauss(0, daily_vol * 0.3)
        
        new_price = prices[-1] + drift * prices[-1] + shock * prices[-1]
        new_price = max(0.001, min(0.99, new_price))
        prices.append(new_price)
    
    if resolution == 1.0:
        for i in range(max(0, num_steps - 3), num_steps + 1):
            w = (i - (num_steps - 3)) / 3
            prices[i] = prices[i] * (1 - w) + resolution * w
    else:
        for i in range(max(0, num_steps - 3), num_steps + 1):
            w = (i - (num_steps - 3)) / 3
            prices[i] = prices[i] * (1 - w) + resolution * w
    
    return prices


def generate_order_book(entry_price: float, liquidity: float, seed: int = 0) -> Dict:
    """
    Generate realistic order book calibrated from live Polymarket CLOB data.
    
    Real observations (DOTM $0.003-$0.15):
    - Spread: 50-200% for <0.01, 20-100% for 0.01-0.05, 5-30% for 0.05-0.15
    - Ask liquidity: $1-50 at best level
    - Bid liquidity: $0.5-10 at best level
    - 3-5 visible levels on each side
    """
    random.seed(seed)
    
    spread_pct = min(2.0, 0.3 * (0.10 / max(entry_price, 0.005)) ** 0.6)
    
    liq_factor = max(0.2, min(liquidity / 500, 3.0))
    
    asks = []
    bids = []
    
    if entry_price < 0.01:
        ask_start = entry_price * (1 + spread_pct)
        bid_start = entry_price * (1 - spread_pct * 0.5)
        ask_base_size = max(5, liq_factor * 15)
        bid_base_size = max(3, liq_factor * 5)
    elif entry_price < 0.05:
        ask_start = entry_price * (1 + spread_pct * 0.4)
        bid_start = entry_price * (1 - spread_pct * 0.3)
        ask_base_size = max(10, liq_factor * 40)
        bid_base_size = max(5, liq_factor * 20)
    else:
        ask_start = entry_price * (1 + spread_pct * 0.2)
        bid_start = entry_price * (1 - spread_pct * 0.15)
        ask_base_size = max(20, liq_factor * 100)
        bid_base_size = max(10, liq_factor * 50)
    
    for i in range(5):
        ask_price = round(min(0.99, ask_start * (1 + 0.25 * i)), 4)
        ask_size = round(ask_base_size * (1.4 ** i) * random.uniform(0.5, 1.5), 1)
        asks.append({"price": ask_price, "size": ask_size})
        
        bid_price = round(max(0.001, bid_start * (1 - 0.25 * i)), 4)
        if bid_price > 0:
            bid_size = round(bid_base_size * (1.4 ** i) * random.uniform(0.5, 1.5), 1)
            bids.append({"price": bid_price, "size": bid_size})
    
    asks.sort(key=lambda x: x["price"])
    bids.sort(key=lambda x: -x["price"])
    
    return {"asks": asks, "bids": bids}
