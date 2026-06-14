"""
manifold.py — Manifold Markets integration for external forecasting data.

Replaces Metaculus (API v2 aggregation data is broken — returns null).
Manifold Markets provides:
  - Free, no-auth API
  - Search by keywords
  - Probability, volume, URL for each market

Used for: external forecast comparison, gap detection, p_model override.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, UTC
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_env_file

load_env_file()

logger = logging.getLogger(__name__)

MANIFOLD_API = "https://manifold.markets/api/v0"
MANIFOLD_TIMEOUT = 15

# Cache
_FORECAST_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL = 3600  # 1 hour

# Match threshold for fuzzy matching
MATCH_THRESHOLD = 0.25

# Dispersion penalty threshold (for markets with low volume/liquidity)
VOLUME_THRESHOLD = 500  # Markets under $500 volume are less reliable


def _search_manifold(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search Manifold Markets for matching questions."""
    try:
        r = requests.get(
            f"{MANIFOLD_API}/search-markets",
            params={"term": query, "limit": limit},
            timeout=MANIFOLD_TIMEOUT,
        )
        if r.status_code != 200:
            logger.warning(f"[MANIFOLD] Search HTTP {r.status_code}")
            return []

        markets = r.json()
        # Filter: only binary markets with a probability
        result = []
        for m in markets:
            if m.get("outcomeType") != "BINARY":
                continue
            if m.get("isResolved"):
                continue
            prob = m.get("probability")
            if prob is None:
                continue
            result.append(m)

        return result
    except requests.exceptions.Timeout:
        logger.warning(f"[MANIFOLD] Search timeout ({MANIFOLD_TIMEOUT}s)")
        return []
    except Exception as e:
        logger.warning(f"[MANIFOLD] Search error: {type(e).__name__}: {e}")
        return []


def _calculate_match_score(pm_question: str, manifold_question: str) -> float:
    """Calculate how well a Manifold question matches a Polymarket question."""
    pm_lower = pm_question.lower()
    mf_lower = manifold_question.lower()

    # Remove common words
    stop = {
        "will", "the", "a", "an", "be", "by", "of", "in", "on", "at",
        "to", "for", "is", "are", "was", "were", "before", "after",
        "any", "this", "that", "there", "it", "or", "and",
    }
    pm_words = set(re.sub(r"[^\w\s]", " ", pm_lower).split()) - stop
    mf_words = set(re.sub(r"[^\w\s]", " ", mf_lower).split()) - stop

    if not pm_words:
        return 0.0

    # Word overlap score
    overlap = len(pm_words & mf_words)
    word_score = overlap / max(len(pm_words), 1)

    # Key phrase bonus
    key_phrases = [
        "trump", "biden", "bitcoin", "ethereum", "nuclear", "nato",
        "ukraine", "russia", "china", "taiwan", "ai", "openai",
        "fed", "rate", "election", "president", "senate", "court",
        "supreme", "climate", "carbon", "spacex", "mars",
    ]
    phrase_bonus = 0.0
    for phrase in key_phrases:
        if phrase in pm_lower and phrase in mf_lower:
            phrase_bonus += 0.10

    # Number matching (years, percentages)
    pm_nums = set(w for w in pm_words if any(c.isdigit() for c in w))
    mf_nums = set(w for w in mf_words if any(c.isdigit() for c in w))
    num_bonus = 0.0
    if pm_nums and mf_nums:
        common_nums = pm_nums & mf_nums
        if common_nums:
            num_bonus = 0.15 * len(common_nums)

    # Substring similarity
    from fuzzywuzzy import fuzz
    similarity = fuzz.partial_ratio(pm_lower, mf_lower) / 100.0
    sim_bonus = 0.0
    if similarity > 0.75:
        sim_bonus = 0.20

    return min(word_score + phrase_bonus + num_bonus + sim_bonus, 1.0)


def _generate_search_terms(question: str) -> list[str]:
    """Generate multiple search queries from a market question."""
    stop = {
        "will", "the", "a", "an", "be", "by", "of", "in", "on", "at",
        "to", "for", "is", "are", "was", "were", "before", "after",
        "any", "this", "that", "there",
    }
    words = [w for w in re.sub(r"[^\w\s]", " ", question.lower()).split()
             if w not in stop and len(w) >= 3]

    if not words:
        return [question[:50]]

    queries = []

    # Full cleaned question (truncated)
    queries.append(" ".join(words[:6]))

    # Key words only
    if len(words) >= 4:
        mid = len(words) // 2
        queries.append(" ".join(words[:3]))
        queries.append(" ".join(words[mid:mid + 3]))

    # Numbers (years)
    nums = [w for w in words if any(c.isdigit() for c in w)]
    if nums:
        queries.append(" ".join(nums[:3] + words[:2]))

    return queries[:4]


def get_manifold_forecast(pm_question: str, pm_resolve_date: str | None = None) -> dict[str, Any]:
    """Get Manifold Markets forecast for a Polymarket question.

    Returns dict with:
      - found: bool
      - probability: float | None
      - question_title: str
      - url: str
      - volume: float
      - forecaster_count: int (approximate from volume)
      - match_score: float
    """
    # Check cache
    cache_key = pm_question
    if cache_key in _FORECAST_CACHE:
        cached = _FORECAST_CACHE[cache_key]
        cached_time = cached.get("_cache_time", 0)
        if (datetime.now(UTC).timestamp() - cached_time) < _CACHE_TTL:
            return {k: v for k, v in cached.items() if not k.startswith("_")}

    search_terms = _generate_search_terms(pm_question)

    best_match = None
    best_score = 0.0

    for term in search_terms:
        results = _search_manifold(term, limit=10)
        if not results:
            continue

        for m in results:
            mf_question = m.get("question", "")
            score = _calculate_match_score(pm_question, mf_question)
            if score > best_score:
                best_score = score
                best_match = m

        if best_match and best_score >= 0.45:
            break  # Good enough match

    if not best_match or best_score < MATCH_THRESHOLD:
        result = {
            "found": False,
            "probability": None,
            "reason": "no_match" if not best_match else "low_score",
            "best_score": best_score,
        }
        _FORECAST_CACHE[cache_key] = {**result, "_cache_time": datetime.now(UTC).timestamp()}
        return result

    prob = best_match.get("probability", 0)
    volume = best_match.get("volume", 0) or 0
    title = best_match.get("question", "")
    url = best_match.get("url", "")

    # Dispersion/quality penalty based on volume
    volume_penalty = 1.0
    if volume < VOLUME_THRESHOLD:
        volume_penalty = max(0.3, volume / VOLUME_THRESHOLD)
        logger.info(
            f"[MANIFOLD] Low volume penalty: ${volume:,.0f} < ${VOLUME_THRESHOLD} "
            f"→ penalty={volume_penalty:.2f}"
        )

    result = {
        "found": True,
        "probability": prob,
        "question_title": title,
        "url": url,
        "volume": volume,
        "match_score": best_score,
        "volume_penalty": volume_penalty,
        "source": "manifold",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    logger.info(
        f"[MANIFOLD] Match: '{title[:40]}' prob={prob:.1%} vol=${volume:,.0f} "
        f"score={best_score:.2f} for '{pm_question[:40]}'"
    )

    _FORECAST_CACHE[cache_key] = {**result, "_cache_time": datetime.now(UTC).timestamp()}
    return result


def check_manifold_gap(market: dict, polymarket_prob: float | None = None) -> dict | None:
    """Check if Manifold forecast shows a significant gap with Polymarket.

    Returns gap analysis dict or None if no significant gap.
    Same interface as check_metaculus_gap().
    """
    from metaculus import get_time_decay_threshold

    mf = get_manifold_forecast(market["question"], market.get("end_date"))

    if not mf.get("found"):
        return None

    manifold_prob = mf.get("probability", 0)
    price_to_use = polymarket_prob if polymarket_prob is not None else market["price"]

    required_gap = get_time_decay_threshold(market.get("end_date"))
    gap = manifold_prob - price_to_use

    if gap <= required_gap:
        logger.info(
            f"[MANIFOLD-GAP-SKIP] gap={gap:.3f} <= required={required_gap:.3f} "
            f"({market['question'][:40]})"
        )
        return None

    volume_penalty = mf.get("volume_penalty", 1.0)
    match_score = mf.get("match_score", 0.5)

    # Signal strength: gap scaled by match quality and volume
    raw_strength = min(gap / 0.15, 1.0)
    signal_strength = raw_strength * volume_penalty * min(match_score * 2, 1.0)

    logger.info(
        f"[MANIFOLD-GAP-APPROVED] mf={manifold_prob:.0%} vs pm={price_to_use:.0%} | "
        f"gap={gap:.3f} required={required_gap:.3f} | "
        f"vol_penalty={volume_penalty:.2f} match={match_score:.2f} => signal={signal_strength:.2f}"
    )

    return {
        "source": "manifold",
        "manifold_prob": manifold_prob,
        "metaculus_prob": manifold_prob,  # Alias for downstream compatibility
        "polymarket_prob": price_to_use,
        "gap": gap,
        "required_gap": required_gap,
        "signal_strength": signal_strength,
        "dispersion_penalty": volume_penalty,  # Alias for downstream compatibility
        "url": mf.get("url", ""),
        "reasoning": (
            f"Manifold {manifold_prob:.0%} vs Polymarket {price_to_use:.0%}: "
            f"gap={gap:.0%}, volume=${mf.get('volume', 0):,.0f}, match={match_score:.2f}"
        ),
    }
