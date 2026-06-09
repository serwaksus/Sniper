"""
market_fetcher.py — Market fetching and pre-filtering.
Extracted from signal_pipeline.py.
"""
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, UTC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from utils import load_env_file
from position_manager import detect_clusters
from schema import HYP_CLUSTERS, HYP_SLUG

load_env_file()

logger = logging.getLogger(__name__)

MIN_VOLUME = 25000
MIN_TTL_HOURS = 48
MAX_PRICE = 0.40
ALLOWED_CLUSTERS = {"ai_tech", "russia_ukraine", "usa_politics", "fed_fomc", "sports_nba", "sports_ufc"}
BANNED_CLUSTERS = {"crypto"}
PRE_FILTER_OTHER_MIN_VOLUME = 100_000


def fetch_markets():
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "200"],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[MARKETS] pm-trader markets failed: rc={res.returncode}")
            return []
        data = json.loads(res.stdout)
        candidates = []
        now = datetime.now(UTC).replace(tzinfo=None)

        for m in data.get("data", []):
            if not m.get("active") or m.get("closed") or m.get("status") in ("closing", "resolved", "invalid"):
                continue

            vol = float(m.get("volume", 0))
            if vol < MIN_VOLUME:
                continue

            liq = float(m.get("liquidity", 0))
            if liq < 100:
                continue

            end_date = m.get("end_date", "")
            ttl_hours = 999
            if end_date:
                try:
                    end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00")).replace(tzinfo=None)
                    ttl_hours = max(0, (end - now).total_seconds() / 3600)
                except Exception as e:
                    logger.debug(f"[ttl_parse] {type(e).__name__}: {e}")

            if ttl_hours < MIN_TTL_HOURS:
                continue

            for outcome, price in zip(m.get("outcomes", []), m.get("outcome_prices", []), strict=False):
                try:
                    price = float(price)
                except (ValueError, TypeError):
                    continue
                if price <= 0 or price > 1.0:
                    continue
                if price <= MAX_PRICE:
                    clusters = detect_clusters(m["question"])
                    if any(c in BANNED_CLUSTERS for c in clusters):
                        continue
                    is_allowed = any(c in ALLOWED_CLUSTERS for c in clusters)
                    if any(c in ("sports_nba", "sports_ufc") for c in clusters) and (vol < 250_000 or ttl_hours < 48):
                        continue
                    is_other_high_vol = clusters == ["other"] and vol >= PRE_FILTER_OTHER_MIN_VOLUME
                    if not is_allowed and not is_other_high_vol:
                        continue
                    candidates.append({
                        "id": m["condition_id"],
                        HYP_SLUG: m[HYP_SLUG],
                        "question": m["question"],
                        "outcome": outcome,
                        "price": price,
                        "volume": vol,
                        "liquidity": float(m.get("liquidity", 0)),
                        "end_date": end_date,
                        "ttl_hours": ttl_hours,
                        HYP_CLUSTERS: clusters,
                        "oracle_type": m.get("oracle_type", "unknown")
                    })

        candidates.sort(key=lambda x: -x["volume"])
        seen_slugs = set()
        unique = []
        for c in candidates:
            if c[HYP_SLUG] not in seen_slugs:
                seen_slugs.add(c[HYP_SLUG])
                unique.append(c)
        return unique[:30]
    except Exception as e:
        logger.error(f"[MARKETS] Fetch error: {e}")
        return []


def fetch_gamma_dotm_candidates(existing_slugs: set) -> list:
    try:
        url = "https://gamma-api.polymarket.com/markets"
        all_markets = []
        for offset in (0, 100):
            params = {
                "closed": "false",
                "limit": 100,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"[GAMMA] status={resp.status_code} at offset={offset}")
                break
            batch = resp.json()
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                break
        markets = all_markets
        now = datetime.now(UTC).replace(tzinfo=None)
        candidates = []
        for m in markets:
            if m.get("active") is False:
                continue
            slug = m.get(HYP_SLUG, "")
            if slug in existing_slugs:
                continue
            question = m.get("question", "")
            if not question:
                continue
            outcome_prices = m.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except Exception as e:
                    logger.debug(f"[market_fetcher] {type(e).__name__}: {e}")
                    continue
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except Exception as e:
                    logger.debug(f"[market_fetcher] {type(e).__name__}: {e}")
                    continue
            vol = float(m.get("volume", 0))
            if vol < MIN_VOLUME:
                continue
            end_date = m.get("endDate", m.get("end_date", ""))
            ttl_hours = 999
            if end_date:
                try:
                    end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00")).replace(tzinfo=None)
                    ttl_hours = max(0, (end - now).total_seconds() / 3600)
                except Exception as e:
                    logger.debug(f"[ttl_parse] {type(e).__name__}: {e}")
            if ttl_hours < MIN_TTL_HOURS:
                continue
            for outcome, price_str in zip(outcomes, outcome_prices, strict=False):
                try:
                    price = float(price_str)
                except (ValueError, TypeError):
                    continue
                if price <= 0 or price > MAX_PRICE:
                    continue
                clusters = detect_clusters(question)
                if any(c in BANNED_CLUSTERS for c in clusters):
                    continue
                is_allowed = any(c in ALLOWED_CLUSTERS for c in clusters)
                is_other_high_vol = clusters == ["other"] and vol >= PRE_FILTER_OTHER_MIN_VOLUME
                if not is_allowed and not is_other_high_vol:
                    continue
                candidates.append({
                    "id": m.get("conditionId", m.get("condition_id", "")),
                    HYP_SLUG: slug,
                    "question": question,
                    "outcome": outcome,
                    "price": price,
                    "volume": vol,
                    "liquidity": float(m.get("liquidity", 0)),
                    "end_date": end_date,
                    "ttl_hours": ttl_hours,
                    HYP_CLUSTERS: clusters,
                    "oracle_type": m.get("oracleType", m.get("oracle_type", "unknown")),
                    "source": "gamma",
                })
        candidates.sort(key=lambda x: -x["volume"])
        seen = set()
        unique = []
        for c in candidates:
            if c[HYP_SLUG] not in seen and c[HYP_SLUG] not in existing_slugs:
                seen.add(c[HYP_SLUG])
                unique.append(c)
        logger.info(f"[GAMMA] Fetched {len(markets)} markets, {len(unique)} new DOTM candidates")
        return unique[:20]
    except Exception as e:
        logger.error(f"[GAMMA] Fetch error: {e}")
        return []


def pre_filter_before_batching(markets: list[dict]) -> list[dict]:
    kept = []
    skipped = []
    for m in markets:
        clusters = m.get(HYP_CLUSTERS, ["other"])
        if any(c in BANNED_CLUSTERS for c in clusters):
            skipped.append(m)
            continue
        is_other = clusters == ["other"] or (len(clusters) == 1 and clusters[0] == "other")
        if is_other:
            volume = m.get("volume", 0)
            if volume < PRE_FILTER_OTHER_MIN_VOLUME:
                slug = m.get(HYP_SLUG, "unknown")
                logger.info(f"[PRE-FILTER] Skipping low-volume 'other' market: {slug}")
                skipped.append(m)
                continue
        kept.append(m)
    return kept, skipped
