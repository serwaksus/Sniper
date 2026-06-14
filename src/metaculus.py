"""
metaculus.py — Metaculus integration: forecast fetching, gap detection, caching.
Extracted from signal_pipeline.py.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, UTC
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from utils import load_json, save_json, load_env_file
from config import CACHE_FILE

load_env_file()

logger = logging.getLogger(__name__)

METACULUS_API_KEY = os.environ.get("METACULUS_TOKEN", "")
METACULUS_URL = "https://www.metaculus.com/api2/questions/"
METACULUS_HEADERS = {"Authorization": f"Token {METACULUS_API_KEY}"}
DISPERSION_PENALTY_THRESHOLD = 0.25
METACULUS_GAP_THRESHOLD = 0.08
DATE_WINDOW_DAYS = 7

_QUESTIONS_CACHE: list[dict] = []
_QUESTIONS_CACHE_TIME: float = 0.0
_QUESTIONS_CACHE_TTL = 6 * 3600  # 6 hours


def load_cache():
    cache = load_json(CACHE_FILE, {"metaculus": {}, "news": {}, "last_update": None})
    now = datetime.now()
    for section in ("metaculus", "news"):
        entries = cache.get(section, {})
        stale = [k for k, v in entries.items()
                 if isinstance(v, dict) and v.get("timestamp")
                 and (now - datetime.fromisoformat(v["timestamp"])).total_seconds() > 86400]
        for k in stale:
            del entries[k]
    return cache


def save_cache(cache: dict) -> None:
    cache["last_update"] = datetime.now().isoformat()
    save_json(CACHE_FILE, cache)


def _fetch_all_open_questions() -> list[dict]:
    """Fetch all open questions via pagination. API search is broken server-side,
    so we bulk-download and search client-side."""
    import time as _t

    global _QUESTIONS_CACHE, _QUESTIONS_CACHE_TIME

    if _QUESTIONS_CACHE and (_t.time() - _QUESTIONS_CACHE_TIME < _QUESTIONS_CACHE_TTL):
        return _QUESTIONS_CACHE

    all_questions: list[dict] = []
    offset = 0
    page_size = 100
    max_pages = 30

    for _page in range(max_pages):
        try:
            resp = requests.get(
                METACULUS_URL,
                headers=METACULUS_HEADERS,
                params={"limit": page_size, "offset": offset, "status": "open"},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(f"[_fetch_all_open_questions] HTTP {resp.status_code} at offset={offset}")
                break
            results = resp.json().get("results", [])
            if not results:
                break
            all_questions.extend(results)
            if len(results) < page_size:
                break
            offset += page_size
            _t.sleep(0.3)
        except requests.exceptions.Timeout:
            logger.warning(f"[_fetch_all_open_questions] Timeout at offset={offset}")
            break
        except Exception as e:
            logger.warning(f"[_fetch_all_open_questions] {type(e).__name__}: {e}")
            break

    if all_questions:
        _QUESTIONS_CACHE = all_questions
        _QUESTIONS_CACHE_TIME = _t.time()
        logger.info(f"[METACULUS] Bulk-fetched {len(all_questions)} open questions")

    return all_questions


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


def metaculus_search(query: str, limit: int = 10) -> list[dict]:
    """Search open questions. API `search` param is broken server-side,
    so we bulk-fetch all questions and match client-side."""
    all_qs = _fetch_all_open_questions()
    if not all_qs:
        return []

    query_lower = query.lower()
    query_words = set(re.sub(r"[^\w\s]", " ", query_lower).split())

    stop = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at",
            "to", "for", "is", "are", "was", "were", "before", "after",
            "any", "this", "that", "there"}
    query_keywords = query_words - stop
    if not query_keywords:
        query_keywords = query_words

    scored: list[tuple[float, dict]] = []
    for q in all_qs:
        title = (q.get("title", "") or q.get("short_title", "")).lower()
        title_words = set(re.sub(r"[^\w\s]", " ", title).split())

        overlap = len(query_keywords & title_words)
        if overlap == 0:
            continue

        score = overlap / max(len(query_keywords), 1)

        from fuzzywuzzy import fuzz
        similarity = fuzz.partial_ratio(query_lower, title) / 100.0
        if similarity > 0.5:
            score += similarity * 0.3

        scored.append((score, q))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in scored[:limit]]


def metaculus_get_question(qid: int) -> dict | None:
    for attempt in range(3):
        try:
            resp = requests.get(f"{METACULUS_URL}{qid}/", headers=METACULUS_HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code >= 500 and attempt < 2:
                import time as _t
                _t.sleep(2 ** attempt)
                continue
        except requests.exceptions.Timeout:
            if attempt < 2:
                import time as _t
                _t.sleep(2 ** attempt)
                continue
            logger.warning(f"[metaculus_get_question] Timeout after {attempt + 1} attempts")
        except Exception as e:
            logger.warning(f"[metaculus_get_question] {type(e).__name__}: {e}")
            break
    return None


def parse_resolve_date(date_str: str | None) -> Any:
    if not date_str:
        return None
    date_str = date_str.replace("Z", "+00:00")
    try:
        from dateutil.parser import parse
        return parse(date_str)
    except Exception as e:
        logger.debug(f"[parse_resolve_date] {type(e).__name__}: {e}")
    try:
        return datetime.fromisoformat(date_str)
    except Exception as e:
        logger.debug(f"[parse_resolve_date] {type(e).__name__}: {e}")
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


def _generate_search_queries(question: str) -> list[str]:
    stop = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at",
            "to", "for", "is", "are", "was", "were", "before", "after",
            "any", "this", "that", "there"}
    words = [w for w in re.sub(r"[^\w\s]", " ", question.lower()).split()
             if w not in stop and len(w) >= 3]
    if not words:
        return []

    queries = []
    mid = len(words) // 2
    if len(words) >= 6:
        queries.append(" ".join(words[:3]))
        queries.append(" ".join(words[mid:mid+3]))
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
    from fuzzywuzzy import fuzz
    pm_lower = pm_question.lower()
    meta_title = (result.get("title", "") or result.get("short_title", "")).lower()

    pm_words = set(re.sub(r'[^\w\s]', ' ', pm_lower).split())
    meta_words = set(re.sub(r'[^\w\s]', ' ', meta_title).split())

    stop = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for", "is", "are", "was", "were", "before", "after", "any", "this", "that"}
    pm_clean = pm_words - stop
    meta_clean = meta_words - stop

    overlap = len(pm_clean & meta_clean)
    base_score = overlap / max(len(pm_clean), 1) if pm_clean else 0

    key_phrases = ["ai safety", "artificial intelligence", "anthropic", "ukraine", "nato", "nuclear", "china", "taiwan", "trump", "fed", "powell", "bitcoin"]
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


def get_metaculus_forecast(pm_question: str, pm_resolve_date: str | None = None) -> dict:
    cache = load_cache()
    cache_key = pm_question

    if cache_key in cache.get("metaculus", {}):
        cached = cache["metaculus"][cache_key]
        if cached.get("timestamp"):
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cached_time).total_seconds() < 3600:
                return cached

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

    if not best_match or raw_best_score < 0.30:
        return {"found": False, "probability": None, "reason": "no_title_match", "best_score": raw_best_score}

    meta_title = best_match.get("title", "") or best_match.get("short_title", "")

    q_data = best_match.get("question", {})
    if not q_data:
        q_data = best_match

    if pm_resolve_date:
        meta_resolve = q_data.get("scheduled_resolve_time") or best_match.get("resolve_date")
        if meta_resolve and not dates_match(pm_resolve_date, meta_resolve):
            return {"found": False, "probability": None, "reason": "date_mismatch",
                    "meta_date": meta_resolve, "pm_date": pm_resolve_date}

    qid = q_data.get("id") or best_match.get("id")
    if qid is None:
        return {"found": False, "probability": None, "reason": "no_question_id"}
    cp_reveal = q_data.get("cp_reveal_time")
    if cp_reveal:
        try:
            reveal_dt = datetime.fromisoformat(cp_reveal.replace("Z", "+00:00"))
            if datetime.now(reveal_dt.tzinfo) < reveal_dt:
                return {"found": False, "probability": None, "reason": "cp_not_revealed"}
        except Exception as e:
            logger.debug(f"[cp_reveal_parse] {type(e).__name__}: {e}")

    agg_data = q_data.get("aggregations") if q_data.get("aggregations") is not None else {}
    agg = agg_data.get("recency_weighted") if agg_data.get("recency_weighted") is not None else {}
    latest = agg.get("latest") if agg and isinstance(agg, dict) else None

    if not latest:
        full_q = metaculus_get_question(qid)
        if full_q:
            q_inner = full_q.get("question", {})
            inner_agg_data = q_inner.get("aggregations") if q_inner.get("aggregations") is not None else {}
            agg = inner_agg_data.get("recency_weighted") if inner_agg_data.get("recency_weighted") is not None else {}
            latest = agg.get("latest") if agg and isinstance(agg, dict) else None

    prob = None
    if latest:
        means = latest.get("means", [])
        if means:
            prob = float(means[0])

    if prob is None:
        pred = q_data.get("prediction") or best_match.get("prediction")
        if pred and isinstance(pred, dict):
            raw = pred.get("number")
            if raw is None:
                raw = pred.get("p_above")
            if raw is None:
                raw = pred.get("p_below")
            prob = float(raw) if raw is not None else None

    if prob is None:
        vote = best_match.get("vote", {})
        if vote and isinstance(vote, dict) and "prediction" in vote:
            prob = float(vote["prediction"])

    if prob is None:
        return {"found": False, "probability": None, "reason": "no_aggregation", "best_match_title": meta_title}
    forecaster_count = latest.get("forecaster_count", 0) if latest else 0
    title = best_match.get("title", "") or best_match.get("short_title", "")

    q1 = (latest or {}).get("q1")
    q3 = (latest or {}).get("q3")
    std = (latest or {}).get("std")
    dispersion = None
    dispersion_penalty = 1.0

    if q1 is not None and q3 is not None:
        dispersion = q3 - q1
    elif std is not None:
        dispersion = std

    if dispersion is not None and dispersion > DISPERSION_PENALTY_THRESHOLD:
        dispersion_penalty = max(0.0, 1.0 - (dispersion - DISPERSION_PENALTY_THRESHOLD))
        logger.info(f"[DISPERSION] q1={q1!r}, q3={q3!r}, dispersion={dispersion:.3f}, penalty={dispersion_penalty:.2f}")

    result = {
        "found": True,
        "probability": prob,
        "question_title": title,
        "url": f"https://www.metaculus.com/questions/{qid}/",
        "forecaster_count": forecaster_count,
        "timestamp": datetime.now().isoformat(),
        "dispersion": dispersion,
        "dispersion_penalty": dispersion_penalty
    }

    cache.setdefault("metaculus", {})[cache_key] = result
    save_cache(cache)
    return result


def get_time_decay_threshold(end_date_str: str | None) -> float:
    if not end_date_str:
        return METACULUS_GAP_THRESHOLD

    try:
        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(end_dt.tzinfo) if end_dt.tzinfo else datetime.now()
        days_to_res = max(0, (end_dt - now).total_seconds() / 86400)
    except Exception as e:
        logger.debug(f"[metaculus] {type(e).__name__}: {e}")
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


def check_metaculus_gap(market: dict, polymarket_prob: float | None = None) -> dict | None:
    meta = get_metaculus_forecast(market["question"], market.get("end_date"))

    if not meta.get("found"):
        return None

    metaculus_prob = meta.get("probability", 0)
    price_to_use = polymarket_prob if polymarket_prob is not None else market["price"]

    required_gap = get_time_decay_threshold(market.get("end_date"))

    gap = metaculus_prob - price_to_use

    if gap <= required_gap:
        logger.info(f"[GAP-SKIP] gap={gap:.3f} <= required={required_gap:.3f} ({market['question'][:40]})")
        return None

    dispersion_penalty = meta.get("dispersion_penalty", 1.0)

    raw_strength = min(gap / 0.15, 1.0)

    signal_strength = raw_strength * dispersion_penalty

    logger.info(
        f"[GAP-APPROVED] meta={metaculus_prob:.0%} vs pm={price_to_use:.0%} | "
        f"gap={gap:.3f} required={required_gap:.3f} | "
        f"disp_penalty={dispersion_penalty:.2f} => signal={signal_strength:.2f}"
    )

    return {
        "source": "metaculus",
        "metaculus_prob": metaculus_prob,
        "polymarket_prob": price_to_use,
        "gap": gap,
        "required_gap": required_gap,
        "signal_strength": signal_strength,
        "dispersion_penalty": dispersion_penalty,
        "reasoning": (
            f"Metaculus {metaculus_prob:.0%} vs Polymarket {price_to_use:.0%}: "
            f"gap={gap:.0%}, dispersion_penalty={dispersion_penalty:.2f}"
        )
    }
