#!/usr/bin/env python3
"""
Social Buzz Score module for DOTM Sniper.
Aggregates social/news attention signals from multiple sources
to detect emerging interest in prediction market topics.

Sources (by weight):
1. GDELT DOC API v2 (50%) — global news monitoring
2. Google News RSS (35%) — English-language news
3. Reddit JSON API (15%) — community discussion

Telegram removed: blocked on Russian IP, Telethon input() hangs process.
"""
import os
import sys
import json
import time
import logging
import threading
import requests
import feedparser
from datetime import datetime, timedelta, UTC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json

logger = logging.getLogger(__name__)

BUZZ_CACHE_FILE = "/root/dotm-sniper/buzz_cache.json"
BUZZ_CACHE_TTL = 3600

DEFAULT_TELEGRAM_CHANNELS = [
    "breakingaviation", "IntelSlava", "war_monitor",
    "nexta_tv", "disclosetv",
]

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_URL = "https://news.google.com/rss/search"
REDDIT_URL = "https://www.reddit.com/search.json"


from utils import load_env_file  # noqa: E402
load_env_file()



def _get_cache() -> dict:
    return load_json(BUZZ_CACHE_FILE, {"entries": {}})


def _save_cache(cache: dict):
    save_json(BUZZ_CACHE_FILE, cache)


def _get_cached(slug: str) -> dict | None:
    cache = _get_cache()
    entry = cache.get("entries", {}).get(slug)
    if not entry:
        return None
    try:
        ts = datetime.fromisoformat(entry["timestamp"])
        if (datetime.now(UTC) - ts).total_seconds() < BUZZ_CACHE_TTL:
            return entry
    except (ValueError, KeyError):
        pass
    return None


def _set_cached(slug: str, result: dict):
    cache = _get_cache()
    if not isinstance(cache, dict):
        cache = {"entries": {}}
    _utcnow = datetime.now(UTC)
    result["timestamp"] = _utcnow.isoformat()
    cache.setdefault("entries", {})[slug] = result
    cutoff = _utcnow - timedelta(seconds=BUZZ_CACHE_TTL * 2)
    entries = cache.get("entries", {})
    stale = [k for k, v in entries.items()
             if isinstance(v, dict) and v.get("timestamp")
             and datetime.fromisoformat(v["timestamp"]) < cutoff]
    for k in stale:
        del entries[k]
    _save_cache(cache)


def extract_keywords_llm(question: str) -> list[str]:
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not API_KEY:
        return _extract_keywords_simple(question)

    URL = "https://api.deepseek.com/v1/chat/completions"
    HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    prompt = f"""Extract 3-5 English search keywords from this prediction market question.
Return ONLY a JSON array of lowercase keywords, nothing else.

Question: {question}

Example: "Will US withdraw from NATO before 2027?" -> ["nato", "us withdraw", "trump nato", "withdrawal treaty"]

Keywords:"""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 100,
        }, timeout=10)

        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        import re
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            keywords = json.loads(json_match.group(0))
            if isinstance(keywords, list) and len(keywords) >= 2:
                return [k.strip().lower() for k in keywords if isinstance(k, str) and k.strip()][:5]
    except Exception as e:
        logger.debug(f"[BUZZ-LLM] Keyword extraction failed: {e}")

    return _extract_keywords_simple(question)


def _extract_keywords_simple(question: str) -> list[str]:
    import re
    stop = {"will", "the", "a", "an", "in", "on", "by", "of", "to", "for", "and",
            "or", "is", "are", "be", "it", "this", "that", "from", "with", "before",
            "after", "during", "than", "more", "less", "any", "some", "all", "no",
            "not", "do", "does", "did", "has", "have", "had", "was", "were", "been",
            "win", "first", "second", "next", "new", "become", "get", "its", "at"}
    words = re.findall(r'[a-zA-Z]{3,}', question.lower())
    keywords = [w for w in words if w not in stop]
    bigrams = []
    for i in range(len(keywords) - 1):
        bigrams.append(f"{keywords[i]} {keywords[i+1]}")
    combined = keywords[:3] + bigrams[:2]
    return combined[:5]


_GDELT_LOCK = threading.RLock()
_GDELT_LAST_FAIL = 0.0
_GDELT_COOLDOWN = 600
_GDELT_FAIL_COUNT = 0


def fetch_gdelt(keywords: list[str]) -> dict:
    global _GDELT_LAST_FAIL, _GDELT_FAIL_COUNT
    with _GDELT_LOCK:
        cooldown = min(_GDELT_COOLDOWN * (2 ** _GDELT_FAIL_COUNT), 3600)
        if (time.time() - _GDELT_LAST_FAIL) < cooldown:
            return {"count": 0, "tone": 0, "status": "cooldown"}

    query = " ".join(keywords[:3])
    try:
        resp = requests.get(GDELT_URL, params={
            "query": query,
            "mode": "artlist",
            "maxrecords": 250,
            "timespan": "24h",
            "format": "json",
            "sourcelang": "english",
        }, timeout=8, headers={"User-Agent": "DotmSniper/1.0"})
        if resp.status_code == 429:
            logger.warning("[BUZZ-GDELT] Rate limited")
            with _GDELT_LOCK:
                _GDELT_LAST_FAIL = time.time()
                _GDELT_FAIL_COUNT += 1
            return {"count": 0, "tone": 0, "status": "rate_limited"}
        if resp.status_code != 200:
            with _GDELT_LOCK:
                _GDELT_LAST_FAIL = time.time()
                _GDELT_FAIL_COUNT += 1
            return {"count": 0, "tone": 0, "status": f"error_{resp.status_code}"}

        with _GDELT_LOCK:
            _GDELT_FAIL_COUNT = 0

        data = resp.json()
        articles = data.get("articles", [])
        count = len(articles)

        tones = []
        for a in articles:
            try:
                tone = float(a.get("tone", "0|0").split("|")[0])
                tones.append(tone)
            except (ValueError, IndexError):
                pass

        avg_tone = sum(tones) / len(tones) if tones else 0

        if avg_tone > 1:
            sentiment = "positive"
        elif avg_tone < -1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {"count": count, "tone": round(avg_tone, 2), "sentiment": sentiment, "status": "ok"}
    except Exception as e:
        logger.debug(f"[BUZZ-GDELT] Error: {e}")
        with _GDELT_LOCK:
            _GDELT_LAST_FAIL = time.time()
            _GDELT_FAIL_COUNT += 1
        return {"count": 0, "tone": 0, "status": f"error: {e}"}


def fetch_google_news(keywords: list[str]) -> dict:
    query = " ".join(keywords[:3])
    try:
        resp = requests.get(GOOGLE_NEWS_URL, params={
            "q": query,
            "hl": "en",
            "gl": "US",
            "ceid": "US:en",
        }, timeout=15, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})

        if resp.status_code != 200:
            return {"count": 0, "status": f"error_{resp.status_code}"}

        feed = feedparser.parse(resp.text)
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        count = 0
        for entry in feed.entries:
            try:
                pub = entry.get("published_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6], tzinfo=UTC)
                    if pub_dt >= cutoff:
                        count += 1
                else:
                    count += 1
            except Exception:
                count += 1

        return {"count": count, "total_feed": len(feed.entries), "status": "ok"}
    except Exception as e:
        logger.debug(f"[BUZZ-GOOGLE] Error: {e}")
        return {"count": 0, "status": f"error: {e}"}


def fetch_reddit(keywords: list[str]) -> dict:
    query = " ".join(keywords[:3])
    try:
        resp = requests.get(REDDIT_URL, params={
            "q": query,
            "sort": "new",
            "t": "day",
            "limit": 100,
        }, timeout=10, headers={"User-Agent": "DotmSniper/1.0"})

        if resp.status_code != 200:
            return {"count": 0, "status": f"error_{resp.status_code}"}

        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        return {"count": len(posts), "status": "ok"}
    except Exception as e:
        logger.debug(f"[BUZZ-REDDIT] Error: {e}")
        return {"count": 0, "status": f"error: {e}"}


def compute_buzz_score(slug: str, question: str, force: bool = False) -> dict:
    if not force:
        cached = _get_cached(slug)
        if cached:
            return cached

    keywords = extract_keywords_llm(question)
    logger.info(f"[BUZZ] Keywords for {slug[:40]}...: {keywords}")

    time.sleep(0.5)
    gdelt = fetch_gdelt(keywords)
    time.sleep(0.5)
    google = fetch_google_news(keywords)
    reddit = fetch_reddit(keywords)

    sources_active = sum(1 for s in [gdelt, google, reddit]
                        if s.get("status") == "ok")

    gdelt_norm = min(gdelt.get("count", 0) / 50, 1.0)
    google_norm = min(google.get("count", 0) / 30, 1.0)
    reddit_norm = min(reddit.get("count", 0) / 20, 1.0)

    total_buzz = 0

    if gdelt.get("status") == "ok":
        total_buzz += 0.50 * gdelt_norm

    if google.get("status") == "ok":
        total_buzz += 0.35 * google_norm

    if reddit.get("status") == "ok":
        total_buzz += 0.15 * reddit_norm

    if sources_active == 1:
        total_buzz = total_buzz * 0.7

    buzz_score = round(total_buzz * 20, 1)

    tone = gdelt.get("tone", 0)
    sentiment = gdelt.get("sentiment", "neutral")

    result = {
        "slug": slug,
        "keywords": keywords,
        "buzz_score": buzz_score,
        "sentiment": sentiment,
        "tone": tone,
        "sources": {
            "gdelt": {"count": gdelt.get("count", 0), "status": gdelt.get("status")},
            "google": {"count": google.get("count", 0), "status": google.get("status")},
            "reddit": {"count": reddit.get("count", 0), "status": reddit.get("status")},
        },
        "sources_active": sources_active,
    }

    logger.info(
        f"[BUZZ] {slug[:40]}... score={buzz_score:.1f}/20 "
        f"(G:{gdelt.get('count', 0)} Go:{google.get('count', 0)} "
        f"R:{reddit.get('count', 0)} sentiment={sentiment})"
    )

    _set_cached(slug, result)
    return result
