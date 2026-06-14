"""Cross-platform prediction market reference engine via Metaforecast GraphQL API.

Fetches and caches forecast data from multiple platforms:
- Good Judgment Open (professional superforecasters)
- Metaculus (when available)
- Betfair, Insight Prediction, Infer, and others
- Manifold Markets (supplements direct Manifold API)

Used as an additional external reference alongside Manifold Markets
to compute cross-platform consensus probabilities.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from fuzzywuzzy import fuzz

from config import PROJECT_ROOT
from utils import load_env_file

load_env_file()

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

_GRAPHQL_URL = "https://metaforecast.org/api/graphql"
_CACHE_PATH = Path(PROJECT_ROOT) / "data" / "metaforecast_index.json"
_CACHE_TTL = 86400  # 24 hours
_BATCH_SIZE = 500
_MAX_BATCHES = 15  # 7500 questions max
_REQUEST_TIMEOUT = 30

# Platforms we care about (higher weight = more trusted)
_PLATFORM_WEIGHTS: dict[str, float] = {
    "Good Judgment Open": 1.5,  # Professional superforecasters
    "Metaculus": 1.3,
    "Manifold Markets": 1.0,
    "Infer": 1.2,
    "Insight Prediction": 1.0,
    "Betfair": 1.1,  # Real-money markets
    "PredictIt": 1.0,
    "Kalshi": 1.1,
    "Good Judgment": 1.5,
    "Rootclaim": 1.3,
    "Peter Wildeford": 1.1,
}

# Minimum stars (quality threshold)
_MIN_STARS = 2

# Fuzzy match thresholds
_MATCH_THRESHOLD = 55  # Minimum fuzzy score to consider a match
_GOOD_MATCH_THRESHOLD = 70  # Strong match
_KEY_PHRASE_BONUS = 25  # Bonus for matching key phrases
_NUMBER_MATCH_BONUS = 30  # Bonus for matching numbers (years, prices, etc.)


# ── Index Management ─────────────────────────────────────────────────────────

class _IndexCache:
    """Manages the local JSON index of Metaforecast questions."""

    def __init__(self) -> None:
        self._questions: list[dict[str, Any]] = []
        self._last_fetch: float = 0.0
        self._lock = __import__("threading").RLock()

    @property
    def questions(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._needs_refresh():
                self._refresh()
            return self._questions

    def _needs_refresh(self) -> bool:
        if not self._questions:
            # Try loading from disk
            if _CACHE_PATH.exists():
                try:
                    data = json.loads(_CACHE_PATH.read_text())
                    self._questions = data.get("questions", [])
                    self._last_fetch = data.get("fetched_at", 0)
                    if self._questions and (time.time() - self._last_fetch < _CACHE_TTL):
                        logger.info(
                            "[METAForecast] Loaded %d questions from cache (age: %.1fh)",
                            len(self._questions),
                            (time.time() - self._last_fetch) / 3600,
                        )
                        return False
                except Exception:
                    pass
            return True
        return (time.time() - self._last_fetch) >= _CACHE_TTL

    def _refresh(self) -> None:
        """Fetch all questions from Metaforecast GraphQL API."""
        with self._lock:
            logger.info("[METAForecast] Refreshing index from API...")
            all_q: list[dict[str, Any]] = []
            cursor: str | None = None

            proxies = {
                "https": os.environ.get("ALL_PROXY", ""),
                "http": os.environ.get("ALL_PROXY", ""),
            }

            for batch_num in range(_MAX_BATCHES):
                after_arg = f', after: "{cursor}"' if cursor else ""
                query = (
                    f"{{ questions(first: {_BATCH_SIZE}{after_arg}) {{ edges {{ cursor node {{ "
                    "title platform { label } "
                    "options { name probability } "
                    "qualityIndicators { numForecasts stars volume } "
                    "url "
                    "} } } }"
                )

                try:
                    r = requests.post(
                        _GRAPHQL_URL,
                        json={"query": query},
                        proxies=proxies,
                        timeout=_REQUEST_TIMEOUT,
                        headers={"Content-Type": "application/json"},
                    )
                    r.raise_for_status()
                    data = r.json()

                    if "errors" in data:
                        logger.warning(
                            "[METAForecast] GraphQL error batch %d: %s",
                            batch_num,
                            str(data["errors"])[:120],
                        )
                        break

                    edges = (
                        data.get("data", {})
                        .get("questions", {})
                        .get("edges", [])
                    )
                    if not edges:
                        break

                    for e in edges:
                        node = e["node"]
                        opts = {
                            o["name"]: o.get("probability")
                            for o in node.get("options", [])
                            if o.get("probability") is not None
                        }
                        if not opts:
                            continue

                        qi = node.get("qualityIndicators", {}) or {}
                        all_q.append(
                            {
                                "title": node["title"],
                                "platform": node["platform"]["label"],
                                "options": opts,
                                "url": node.get("url", ""),
                                "num_forecasts": qi.get("numForecasts") or 0,
                                "stars": qi.get("stars") or 0,
                                "volume": qi.get("volume") or 0,
                            }
                        )

                    cursor = edges[-1]["cursor"]
                    logger.info(
                        "[METAForecast] Batch %d: +%d (total: %d)",
                        batch_num,
                        len(edges),
                        len(all_q),
                    )
                    time.sleep(0.5)

                except Exception as e:
                    logger.warning("[METAForecast] Fetch error batch %d: %s", batch_num, e)
                    break

            self._questions = all_q
            self._last_fetch = time.time()

            # Persist to disk
            try:
                _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CACHE_PATH.write_text(
                    json.dumps(
                        {
                            "fetched_at": self._last_fetch,
                            "questions": all_q,
                        },
                        ensure_ascii=False,
                    )
                )
                logger.info(
                    "[METAForecast] Cached %d questions to %s", len(all_q), _CACHE_PATH
                )
            except Exception as e:
                logger.warning("[METAForecast] Cache write error: %s", e)


_cache = _IndexCache()


# ── Fuzzy Matching ───────────────────────────────────────────────────────────


def _extract_numbers(text: str) -> set[str]:
    """Extract years and numbers from text."""
    return set(re.findall(r"\b(?:19|20)\d{2}\b|\$?\d[\d,]*(?:\.\d+)?[kKmMbB]?\b", text))


def _extract_key_phrases(text: str) -> set[str]:
    """Extract significant words (proper nouns, long words, entities)."""
    # Remove common stop words
    stop = {
        "the", "will", "be", "a", "an", "in", "on", "at", "to", "for",
        "of", "is", "are", "by", "before", "after", "this", "that",
        "it", "or", "and", "not", "no", "yes", "with", "from", "as",
        "any", "part", "what", "which", "who", "how", "when", "if",
        "they", "their", "there", "than", "then", "so", "do", "does",
        "was", "were", "been", "being", "have", "has", "had", "would",
        "could", "should", "may", "might", "can", "shall",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in stop}


def _fuzzy_score(polymarket_title: str, candidate_title: str) -> int:
    """Compute fuzzy match score between two titles.

    Combines token overlap, key phrase matching, number matching,
    and fuzzy string similarity.
    """
    pm_lower = polymarket_title.lower()
    cand_lower = candidate_title.lower()

    # 1. Base fuzzy string similarity
    base = fuzz.partial_ratio(pm_lower, cand_lower)

    # 2. Key phrase overlap
    pm_phrases = _extract_key_phrases(pm_lower)
    cand_phrases = _extract_key_phrases(cand_lower)
    if pm_phrases and cand_phrases:
        overlap = len(pm_phrases & cand_phrases)
        total = len(pm_phrases)
        phrase_score = int((overlap / total) * _KEY_PHRASE_BONUS) if total else 0
    else:
        phrase_score = 0

    # 3. Number matching (years, prices, etc.)
    pm_numbers = _extract_numbers(polymarket_title)
    cand_numbers = _extract_numbers(candidate_title)
    if pm_numbers:
        num_overlap = len(pm_numbers & cand_numbers)
        number_score = int((num_overlap / len(pm_numbers)) * _NUMBER_MATCH_BONUS)
    else:
        number_score = 0

    return min(100, base + phrase_score + number_score)


# ── Public API ───────────────────────────────────────────────────────────────


def get_metaforecast_forecast(
    question_title: str,
    *,
    min_stars: int = _MIN_STARS,
    max_results: int = 10,
) -> dict[str, Any] | None:
    """Search the cross-platform index for forecasts matching a question.

    Returns a dict with:
        - found: bool
        - probability: float | None  (best cross-platform probability)
        - platform: str  (which platform provided the probability)
        - all_matches: list of per-platform matches
        - url: str
        - dispersion: float  (how much platforms disagree)
    """
    questions = _cache.questions
    if not questions:
        logger.warning("[METAForecast] Index empty, cannot search")
        return {"found": False, "probability": None}

    # Score all candidates
    scored: list[tuple[int, dict[str, Any]]] = []
    pm_numbers = _extract_numbers(question_title)
    pm_phrases = _extract_key_phrases(question_title.lower())

    for q in questions:
        # Quick pre-filter: must share at least one key phrase or number
        q_phrases = _extract_key_phrases(q["title"].lower())
        q_numbers = _extract_numbers(q["title"])
        if pm_phrases and q_phrases and not (pm_phrases & q_phrases):
            continue
        if pm_numbers and q_numbers and not (pm_numbers & q_numbers) and len(pm_numbers) >= 2:
            continue

        score = _fuzzy_score(question_title, q["title"])
        if score >= _MATCH_THRESHOLD:
            scored.append((score, q))

    if not scored:
        return {"found": False, "probability": None}

    # Sort by score descending, filter by stars
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_results]

    # Filter by stars (relax threshold if no matches)
    good_quality = [(s, q) for s, q in top if q.get("stars", 0) >= min_stars]
    if not good_quality:
        good_quality = top  # Use whatever we have

    # Extract per-platform probabilities
    platform_probs: dict[str, list[tuple[float, int, str]]] = {}  # platform → [(prob, score, url)]
    for score, q in good_quality:
        platform = q["platform"]
        # Get "Yes" probability (or first option for binary)
        opts = q["options"]
        yes_prob = opts.get("Yes", opts.get("yes"))
        if yes_prob is None:
            # Try first option
            probs = [v for v in opts.values() if isinstance(v, (int, float))]
            yes_prob = max(probs) if probs else None
        if yes_prob is not None and isinstance(yes_prob, (int, float)):
            weight = _PLATFORM_WEIGHTS.get(platform, 0.8) * (1 + score / 200)
            platform_probs.setdefault(platform, []).append(
                (float(yes_prob), int(weight * 100), q.get("url", ""))
            )

    if not platform_probs:
        return {"found": False, "probability": None}

    # Compute weighted average across platforms
    total_weight = 0
    weighted_sum = 0.0
    all_matches: list[dict[str, Any]] = []

    for platform, matches in platform_probs.items():
        # Best match per platform
        best_prob, best_weight, best_url = max(matches, key=lambda x: x[1])
        total_weight += best_weight
        weighted_sum += best_prob * best_weight
        all_matches.append(
            {
                "platform": platform,
                "probability": best_prob,
                "weight": best_weight,
                "url": best_url,
            }
        )

    consensus_prob = weighted_sum / total_weight if total_weight else None

    # Dispersion (how much platforms disagree)
    probs_list = [m["probability"] for m in all_matches]
    if len(probs_list) >= 2:
        dispersion = max(probs_list) - min(probs_list)
    else:
        dispersion = 0.0

    # Best single match for URL
    best_match = max(all_matches, key=lambda x: x["weight"])

    logger.info(
        "[METAForecast] Match for '%s': p=%.1f%% (%d platforms, dispersion=%.2f)",
        question_title[:50],
        (consensus_prob or 0) * 100,
        len(all_matches),
        dispersion,
    )

    return {
        "found": True,
        "probability": consensus_prob,
        "platform": best_match["platform"],
        "url": best_match["url"],
        "all_matches": all_matches,
        "dispersion": dispersion,
        "num_platforms": len(all_matches),
    }


def check_metaforecast_gap(
    market: dict[str, Any],
    polymarket_prob: float | None = None,
) -> dict[str, Any] | None:
    """Check if cross-platform forecasts disagree with Polymarket price.

    Compatible with the Manifold/Metaculus gap-check interface.

    Returns dict with:
        - found: bool
        - probability: float
        - polymarket_prob: float
        - signal_strength: float (0-1, capped)
        - source: str
        - url: str
        - dispersion: float
    """
    question = market.get("question", "")
    if not question:
        return None

    price = polymarket_prob
    if price is None:
        price = float(market.get("best_bid", 0) + market.get("best_ask", 1)) / 2
    if price <= 0.005 or price >= 0.995:
        return None

    result = get_metaforecast_forecast(question)
    if not result or not result.get("found"):
        return {"found": False, "probability": None}

    ext_prob = result["probability"]
    if ext_prob is None:
        return {"found": False, "probability": None}

    gap = abs(ext_prob - price)
    signal_strength = min(1.0, gap / 0.30)  # 30% gap = max signal

    # Penalize for dispersion (platform disagreement)
    dispersion = result.get("dispersion", 0)
    dispersion_penalty = max(0.5, 1.0 - dispersion)
    signal_strength *= dispersion_penalty

    return {
        "found": True,
        "probability": ext_prob,
        "polymarket_prob": price,
        "signal_strength": round(signal_strength, 3),
        "source": "metaforecast",
        "url": result.get("url", ""),
        "dispersion": dispersion,
        "num_platforms": result.get("num_platforms", 1),
        "all_matches": result.get("all_matches", []),
    }


def get_index_stats() -> dict[str, Any]:
    """Return statistics about the current index."""
    questions = _cache.questions
    if not questions:
        return {"total": 0, "platforms": {}, "cache_age_hours": -1}

    from collections import Counter

    platforms = Counter(q["platform"] for q in questions)
    return {
        "total": len(questions),
        "platforms": dict(platforms.most_common()),
        "cache_age_hours": round((time.time() - _cache._last_fetch) / 3600, 1),
    }
