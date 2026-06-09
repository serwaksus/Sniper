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


def normalize_probability(p: float | None) -> float:
    if p is None:
        return 0
    p = float(p)
    if p > 1.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))


def metaculus_search(query: str, limit: int = 10) -> list[dict]:
    try:
        resp = requests.get(METACULUS_URL, headers=METACULUS_HEADERS,
                          params={"search": query, "limit": limit, "status": "open"},
                          timeout=15)
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception as e:
        logger.warning(f"[metaculus_search] {type(e).__name__}: {e}")
    return []


def metaculus_get_question(qid: int) -> dict | None:
    try:
        resp = requests.get(f"{METACULUS_URL}{qid}/", headers=METACULUS_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"[metaculus_get_question] {type(e).__name__}: {e}")
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
    words = question.replace("?", " ").replace(",", " ").split()
    queries = []
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words)+1)):
            phrase = " ".join(words[i:j])
            if len(phrase) >= 4:
                queries.append(phrase)
                if len(queries) >= 5:
                    return queries
    return queries


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
    substring_bonus = 0
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
    raw_best_score = 0
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
            prob = pred.get("number")
            if prob is None:
                prob = pred.get("p_above")
            if prob is None:
                prob = pred.get("p_below")
            prob = float(prob) if prob is not None else 0.0

    if prob is None:
        vote = best_match.get("vote", {})
        if vote and isinstance(vote, dict):
            prob = float(vote.get("prediction", 0))

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
