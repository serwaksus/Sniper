"""
metaculus.py — Metaculus integration: forecast fetching, gap detection, caching.

Two-step pipeline:
  1. Search Metaculus posts via /api/posts/?search= (working endpoint)
  2. Get probabilities via Metaforecast GraphQL (Metaculus API aggregations
     are null by design — community predictions are not exposed publicly)

The module maintains backward-compatible function signatures so that
signal_pipeline.py and signal_scorer.py continue to work unchanged.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time as _time
from datetime import datetime, UTC
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from fuzzywuzzy import fuzz

from utils import load_json, save_json, load_env_file
from config import CACHE_FILE

load_env_file()

logger = logging.getLogger(__name__)

METACULUS_API_KEY = os.environ.get("METACULUS_TOKEN", "")
METACULUS_POSTS_URL = "https://www.metaculus.com/api/posts/"
METACULUS_HEADERS = {"Authorization": f"Token {METACULUS_API_KEY}"}
METACULUS_GAP_THRESHOLD = 0.08
DATE_WINDOW_DAYS = 7
DISPERSION_PENALTY_THRESHOLD = 0.25

# Metaforecast GraphQL for probability data
_METAForecast_URL = "https://metaforecast.org/api/graphql"
_METAForecast_PROXIES = {
    "https": os.environ.get("ALL_PROXY", ""),
    "http": os.environ.get("ALL_PROXY", ""),
}
_METAForecast_HEADERS = {"Content-Type": "application/json"}

# Rate limiting (Metaculus API can return 429)
_LAST_API_CALL: float = 0.0
_MIN_API_INTERVAL = 3.0  # seconds between API calls


def _rate_limit() -> None:
    """Ensure we don't hit Metaculus API too fast."""
    global _LAST_API_CALL
    elapsed = _time.time() - _LAST_API_CALL
    if elapsed < _MIN_API_INTERVAL:
        _time.sleep(_MIN_API_INTERVAL - elapsed)
    _LAST_API_CALL = _time.time()


# ── Cache management ─────────────────────────────────────────────────────────


def load_cache() -> dict:
    cache = load_json(CACHE_FILE, {"metaculus": {}, "news": {}, "last_update": None})
    now = datetime.now()
    for section in ("metaculus", "news"):
        entries = cache.get(section, {})
        stale = [
            k
            for k, v in entries.items()
            if isinstance(v, dict)
            and v.get("timestamp")
            and (now - datetime.fromisoformat(v["timestamp"])).total_seconds() > 86400
        ]
        for k in stale:
            del entries[k]
    return cache


def save_cache(cache: dict) -> None:
    cache["last_update"] = datetime.now().isoformat()
    save_json(CACHE_FILE, cache)


# ── Probability helpers ──────────────────────────────────────────────────────


def normalize_probability(p: float | str | None) -> float:
    if p is None:
        return 0
    if isinstance(p, str):
        p = p.strip().rstrip("%").strip()
        if not p:
            return 0
    p = float(p)
    if p > 1.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))


# ── Metaculus post search (new /api/posts/ endpoint) ─────────────────────────


def metaculus_search(query: str, limit: int = 10) -> list[dict]:
    """Search Metaculus posts via /api/posts/?search=.

    This endpoint works correctly (unlike /api2/questions/?search=
    which ignores the search parameter).
    """
    _rate_limit()
    try:
        resp = requests.get(
            METACULUS_POSTS_URL,
            headers=METACULUS_HEADERS,
            params={"search": query, "limit": limit, "status": "open"},
            timeout=30,
        )
        if resp.status_code == 429:
            logger.warning("[METACULUS] Rate limited (429), backing off")
            _time.sleep(10)
            return []
        if resp.status_code != 200:
            logger.warning(f"[METACULUS] Search HTTP {resp.status_code}")
            return []
        results = resp.json().get("results", [])
        logger.info(f"[METACULUS] Search '{query[:30]}': {len(results)} results")
        return results
    except requests.exceptions.Timeout:
        logger.warning("[METACULUS] Search timeout")
        return []
    except Exception as e:
        logger.warning(f"[METACULUS] Search error: {type(e).__name__}: {e}")
        return []


def metaculus_get_question(qid: int) -> dict | None:
    """Get a single question by ID from the new /api/ endpoint."""
    _rate_limit()
    for attempt in range(3):
        try:
            resp = requests.get(
                f"https://www.metaculus.com/api/questions/{qid}/",
                headers=METACULUS_HEADERS,
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code >= 500 and attempt < 2:
                _time.sleep(2**attempt)
                continue
        except requests.exceptions.Timeout:
            if attempt < 2:
                _time.sleep(2**attempt)
                continue
            logger.warning(f"[metaculus_get_question] Timeout after {attempt + 1} attempts")
        except Exception as e:
            logger.warning(f"[metaculus_get_question] {type(e).__name__}: {e}")
            break
    return None


# ── Metaforecast bridge: get probability for Metaculus question ──────────────


def _metaforecast_get_prob(metaculus_id: int) -> dict | None:
    """Query Metaforecast GraphQL for a Metaculus question's probability.

    Metaforecast indexes Metaculus questions and provides their community
    prediction probabilities (which Metaculus API returns as null).
    """
    mf_id = f"metaculus-{metaculus_id}"
    query = (
        '{ question(id: "'
        + mf_id
        + '") { title options { name probability } '
        'qualityIndicators { numForecasts stars } } }'
    )
    try:
        r = requests.post(
            _METAForecast_URL,
            json={"query": query},
            proxies=_METAForecast_PROXIES,
            timeout=15,
            headers=_METAForecast_HEADERS,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        q = data.get("data", {}).get("question")
        if not q:
            return None

        opts = {o["name"]: o.get("probability") for o in q.get("options", [])}
        qi = q.get("qualityIndicators", {}) or {}

        # For binary questions, get "Yes" probability
        yes_prob = opts.get("Yes")
        if yes_prob is None:
            # Try first non-None option
            probs = [v for v in opts.values() if v is not None]
            yes_prob = probs[0] if probs else None

        return {
            "probability": yes_prob,
            "num_forecasts": qi.get("numForecasts") or 0,
            "stars": qi.get("stars") or 0,
            "title": q.get("title", ""),
        }
    except Exception as e:
        logger.debug(f"[METAForecast] Error for {mf_id}: {e}")
        return None


# ── Date utilities ───────────────────────────────────────────────────────────


def parse_resolve_date(date_str: str | None) -> Any:
    if not date_str:
        return None
    date_str = date_str.replace("Z", "+00:00")
    try:
        from dateutil.parser import parse

        return parse(date_str)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        pass
    return None


def dates_match(date1: str, date2: str, window_days: int = DATE_WINDOW_DAYS) -> bool:
    d1 = parse_resolve_date(date1)
    d2 = parse_resolve_date(date2)
    if d1 is None or d2 is None:
        return False
    if d1.tzinfo is not None:
        d1 = d1.astimezone(UTC).replace(tzinfo=None)
    if d2.tzinfo is not None:
        d2 = d2.astimezone(UTC).replace(tzinfo=None)
    diff = abs((d1 - d2).total_seconds() / 86400)
    return diff <= window_days


# ── Matching ─────────────────────────────────────────────────────────────────


def _generate_search_queries(question: str) -> list[str]:
    stop = {
        "will", "the", "a", "an", "be", "by", "of", "in", "on", "at",
        "to", "for", "is", "are", "was", "were", "before", "after",
        "any", "this", "that", "there",
    }
    words = [
        w
        for w in re.sub(r"[^\w\s]", " ", question.lower()).split()
        if w not in stop and len(w) >= 3
    ]
    if not words:
        return []

    queries = []
    mid = len(words) // 2
    if len(words) >= 6:
        queries.append(" ".join(words[:3]))
        queries.append(" ".join(words[mid : mid + 3]))
        queries.append(" ".join(words[-3:]))
    elif len(words) >= 3:
        queries.append(" ".join(words[:3]))
        queries.append(" ".join(words[-2:]))
    else:
        queries.append(" ".join(words))

    nums = [w for w in words if any(c.isdigit() for c in w)]
    if nums:
        queries.append(" ".join(nums[:2]))

    return queries[:5]


def _calculate_metaculus_match(pm_question: str, result: dict) -> float:
    """Calculate match score between Polymarket question and Metaculus result."""
    pm_lower = pm_question.lower()
    meta_title = (result.get("title", "") or result.get("short_title", "")).lower()

    pm_words = set(re.sub(r"[^\w\s]", " ", pm_lower).split())
    meta_words = set(re.sub(r"[^\w\s]", " ", meta_title).split())

    stop = {
        "will", "the", "a", "an", "be", "by", "of", "in", "on", "at",
        "to", "for", "is", "are", "was", "were", "before", "after",
        "any", "this", "that",
    }
    pm_clean = pm_words - stop
    meta_clean = meta_words - stop

    overlap = len(pm_clean & meta_clean)
    base_score = overlap / max(len(pm_clean), 1) if pm_clean else 0

    key_phrases = [
        "ai safety", "artificial intelligence", "anthropic", "ukraine",
        "nato", "nuclear", "china", "taiwan", "trump", "fed", "powell",
        "bitcoin", "recession", "election", "democratic", "republican",
    ]
    substring_bonus = 0.0
    for phrase in key_phrases:
        if phrase in pm_lower and phrase in meta_title:
            substring_bonus += 0.15
        elif phrase in pm_lower and phrase.split()[0] in meta_title:
            substring_bonus += 0.05

    pm_nums = set(w for w in pm_clean if any(c.isdigit() for c in w))
    meta_nums = set(w for w in meta_clean if any(c.isdigit() for c in w))
    if pm_nums and meta_nums and pm_nums & meta_nums:
        base_score += 0.1

    similarity = fuzz.partial_ratio(pm_lower, meta_title) / 100.0
    if similarity > 0.70:
        base_score += 0.15

    return min(base_score + substring_bonus, 1.0)


# ── Main API: get_metaculus_forecast ─────────────────────────────────────────


def get_metaculus_forecast(
    pm_question: str, pm_resolve_date: str | None = None
) -> dict:
    """Get Metaculus community forecast for a Polymarket question.

    Two-step pipeline:
      1. Search Metaculus posts by keyword → find matching question IDs
      2. Query Metaforecast for probability data (Metaculus API returns null)

    Returns dict with:
        - found: bool
        - probability: float | None
        - question_title: str
        - url: str
        - forecaster_count: int
    """
    cache = load_cache()
    cache_key = pm_question

    # Check cache (1h TTL)
    if cache_key in cache.get("metaculus", {}):
        cached = cache["metaculus"][cache_key]
        if cached.get("timestamp"):
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cached_time).total_seconds() < 3600:
                return cached

    # Step 1: Search Metaculus posts
    search_queries = [pm_question, *_generate_search_queries(pm_question)]

    best_match = None
    raw_best_score = 0.0
    for query in search_queries:
        results = metaculus_search(query, limit=10)
        if not results:
            continue
        for r in results:
            score = _calculate_metaculus_match(pm_question, r)
            if score > raw_best_score:
                raw_best_score = score
                best_match = r
        if best_match and raw_best_score >= 0.40:
            break

    if not best_match or raw_best_score < 0.25:
        result = {
            "found": False,
            "probability": None,
            "reason": "no_title_match",
            "best_score": raw_best_score,
            "timestamp": datetime.now().isoformat(),
        }
        cache.setdefault("metaculus", {})[cache_key] = result
        save_cache(cache)
        return result

    meta_title = best_match.get("title", "") or best_match.get("short_title", "")

    # Date matching
    q_data = best_match.get("question", {})
    if not q_data:
        q_data = best_match

    if pm_resolve_date:
        meta_resolve = q_data.get("scheduled_resolve_time") or best_match.get(
            "scheduled_resolve_time"
        )
        if meta_resolve and not dates_match(pm_resolve_date, meta_resolve):
            result = {
                "found": False,
                "probability": None,
                "reason": "date_mismatch",
                "meta_date": meta_resolve,
                "pm_date": pm_resolve_date,
                "timestamp": datetime.now().isoformat(),
            }
            cache.setdefault("metaculus", {})[cache_key] = result
            save_cache(cache)
            return result

    # Get question ID
    qid = q_data.get("id") or best_match.get("id")
    if qid is None:
        return {
            "found": False,
            "probability": None,
            "reason": "no_question_id",
            "timestamp": datetime.now().isoformat(),
        }

    fc_count = best_match.get("forecasts_count", 0) or 0

    # Step 2: Get probability from Metaforecast
    mf = _metaforecast_get_prob(qid)
    prob = mf.get("probability") if mf else None

    if prob is None:
        result = {
            "found": False,
            "probability": None,
            "reason": "no_probability_via_metaforecast",
            "best_match_title": meta_title,
            "metaculus_id": qid,
            "forecasts_count": fc_count,
            "timestamp": datetime.now().isoformat(),
        }
        cache.setdefault("metaculus", {})[cache_key] = result
        save_cache(cache)
        return result

    logger.info(
        f"[METACULUS] Match: '{meta_title[:50]}' → p={prob:.1%} "
        f"({mf.get('num_forecasts', 0)}fc, {mf.get('stars', 0)}★)"
    )

    result = {
        "found": True,
        "probability": prob,
        "question_title": meta_title,
        "url": f"https://www.metaculus.com/questions/{qid}/",
        "forecaster_count": mf.get("num_forecasts", fc_count),
        "stars": mf.get("stars", 0),
        "match_score": raw_best_score,
        "timestamp": datetime.now().isoformat(),
    }

    cache.setdefault("metaculus", {})[cache_key] = result
    save_cache(cache)
    return result


# ── Gap detection ────────────────────────────────────────────────────────────


def get_time_decay_threshold(end_date_str: str | None) -> float:
    if not end_date_str:
        return METACULUS_GAP_THRESHOLD

    try:
        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(end_dt.tzinfo) if end_dt.tzinfo else datetime.now()
        days_to_res = max(0, (end_dt - now).total_seconds() / 86400)
    except Exception:
        return METACULUS_GAP_THRESHOLD

    if days_to_res > 30:
        threshold = 0.20
    elif days_to_res >= 8:
        threshold = 0.15
    elif days_to_res >= 3:
        threshold = 0.10
    elif days_to_res >= 1:
        threshold = 0.05
    else:
        threshold = 0.03 + (days_to_res * 0.02)

    logger.info(f"[TIME-DECAY] days_to_res={days_to_res:.1f}, threshold={threshold:.2f}")
    return threshold


def check_metaculus_gap(
    market: dict, polymarket_prob: float | None = None
) -> dict | None:
    """Check if Metaculus forecast disagrees with Polymarket price.

    Returns dict compatible with check_manifold_gap / check_metaforecast_gap:
        - found: bool
        - probability: float
        - polymarket_prob: float
        - signal_strength: float
        - source: str
        - url: str
    """
    meta = get_metaculus_forecast(market["question"], market.get("end_date"))

    if not meta or not meta.get("found"):
        return None

    metaculus_prob = meta.get("probability", 0)
    price_to_use = (
        polymarket_prob if polymarket_prob is not None else market["price"]
    )

    required_gap = get_time_decay_threshold(market.get("end_date"))
    gap = metaculus_prob - price_to_use

    if gap <= required_gap:
        logger.info(
            f"[GAP-SKIP] gap={gap:.3f} <= required={required_gap:.3f} "
            f"({market['question'][:40]})"
        )
        return None

    signal_strength = min(gap / 0.15, 1.0)

    # Apply dispersion penalty if available
    dispersion_penalty = meta.get("dispersion_penalty", 1.0)
    signal_strength *= dispersion_penalty
    signal_strength = round(signal_strength, 3)

    logger.info(
        f"[GAP-APPROVED] Metaculus={metaculus_prob:.0%} vs PM={price_to_use:.0%} | "
        f"gap={gap:.3f} required={required_gap:.3f} => signal={signal_strength:.2f}"
    )

    return {
        "found": True,
        "probability": metaculus_prob,
        "metaculus_prob": metaculus_prob,
        "polymarket_prob": price_to_use,
        "gap": gap,
        "required_gap": required_gap,
        "signal_strength": round(signal_strength, 3),
        "dispersion_penalty": dispersion_penalty,
        "source": "metaculus",
        "url": meta.get("url", ""),
        "forecaster_count": meta.get("forecaster_count", 0),
        "match_score": meta.get("match_score", 0),
    }
