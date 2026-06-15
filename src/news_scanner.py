#!/usr/bin/env python3
"""
News Scanner - Fresh news reality check before trading.
Uses Tavily API for real-time news search.
"""
from __future__ import annotations
import os
import re
import sys
import time
import logging
import requests

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json

TAVILY_API_URL = "https://api.tavily.com/search"

NEWS_TIME_WINDOW_HOURS = 48
MAX_NEWS_AGE_DAYS_DEFAULT = 30

# ── Circuit breaker: skip Tavily for 6h after quota/auth errors ──
_TAVILY_COOLDOWN_SECONDS = 6 * 3600
_tavily_trip_time: float = 0.0  # monotonic timestamp when breaker tripped


def _tavily_circuit_open() -> bool:
    """True if circuit breaker is open (Tavily should be skipped)."""
    if _tavily_trip_time == 0.0:
        return False
    elapsed = time.monotonic() - _tavily_trip_time
    # Half-open: allow next attempt through after cooldown
    return elapsed < _TAVILY_COOLDOWN_SECONDS


def _tavily_trip_breaker() -> None:
    """Trip the circuit breaker after quota/auth failure."""
    global _tavily_trip_time
    if _tavily_trip_time == 0.0:
        logger.info(
            f"[news_scanner] Tavily circuit breaker TRIPPED — "
            f"skipping for {_TAVILY_COOLDOWN_SECONDS // 3600}h to save VPN bandwidth"
        )
    _tavily_trip_time = time.monotonic()

def load_env() -> None:
    from config import ENV_FILE
    env_path = ENV_FILE
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), val)

load_env()

def _tavily_search(api_key: str, query: str, max_results: int, max_age_days: int) -> dict | None:
    """Try a single Tavily API key. Returns result dict on success, None on failure/quota.
    Trips circuit breaker on 401/429 to stop wasting VPN connections."""
    params = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "topic": "news",
        "days": max_age_days,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False
    }
    try:
        response = requests.post(TAVILY_API_URL, json=params, timeout=20)
        if response.ok:
            data = response.json()
            results = data.get("results", [])
            headlines = [r.get("title", "") for r in results if r.get("title")]
            sources = [r.get("url", "") for r in results if r.get("url")]
            return {
                "headlines": headlines,
                "sources": sources,
                "query": query,
                "found": len(headlines) > 0
            }
        elif response.status_code in (401, 403, 429):
            logger.warning(f"[news_scanner] Tavily quota/auth error HTTP {response.status_code}, tripping circuit breaker")
            _tavily_trip_breaker()
            return None
        else:
            logger.warning(f"[news_scanner] Tavily HTTP {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        logger.warning(f"[news_scanner] Tavily error: {type(e).__name__}: {e}")
        return None


def fetch_recent_news(market_keywords: list[str], max_results: int = 5, max_age_days: int | None = None) -> dict:
    """
    Search for recent news headlines based on market keywords.
    Chain: Tavily (primary) -> Tavily (backup) -> DuckDuckGo fallback.
    Circuit breaker skips Tavily entirely for 6h after quota/auth errors.
    """
    if max_age_days is None:
        max_age_days = MAX_NEWS_AGE_DAYS_DEFAULT

    query = " ".join(market_keywords[:5])

    # Circuit breaker: skip Tavily if recently tripped
    if not _tavily_circuit_open():
        tavily_primary = os.environ.get("TAVILY_API_KEY", "")
        tavily_backup = os.environ.get("TAVILY_API_KEY_BACKUP", "")

        for label, key in [("primary", tavily_primary), ("backup", tavily_backup)]:
            if not key or key == "your_tavily_key_here":
                continue
            result = _tavily_search(key, query, max_results, max_age_days)
            if result is not None:
                return result
            # If breaker tripped during this call, stop trying more keys
            if _tavily_circuit_open():
                logger.info("[news_scanner] Tavily circuit open, going straight to DDG fallback")
                break
            logger.info(f"[news_scanner] Tavily {label} exhausted, trying next source...")
    else:
        logger.debug("[news_scanner] Tavily circuit breaker open, using DDG directly")

    return _fetch_ddg_news_fallback(market_keywords, max_results, max_age_days)


def _fetch_ddg_news_fallback(keywords: list[str], max_results: int, max_age_days: int = 30) -> dict:
    query = " ".join(keywords[:5])
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            timelimit = "m" if max_age_days <= 30 else "y"
            results = list(ddgs.news(query, max_results=max_results, timelimit=timelimit))
            headlines = [r.get("title", "") for r in results if r.get("title")]
            sources = [r.get("url", "") for r in results if r.get("url")]
            return {
                "headlines": headlines,
                "sources": sources,
                "query": query,
                "found": len(headlines) > 0
            }
    except Exception as e:
        logger.debug(f"[news_scanner] DDG fallback failed: {type(e).__name__}: {e}")
    return {"headlines": [], "sources": [], "query": query, "found": False}


def extract_keywords(question: str) -> list[str]:
    """Extract meaningful keywords from market question for news search"""
    import re
    stop_words = {
        "will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for",
        "this", "that", "is", "are", "was", "were", "before", "after", "end",
        "happen", "occur", "take", "place"
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
    return [w for w in words if w not in stop_words][:8]


def news_sanity_check(market_title: str, news_headlines: list[str],
                      metaculus_prob: float | None = None) -> tuple[bool, str]:
    """
    LLM-powered sanity check using news headlines.

    Returns: (passed: bool, reason: str)
    - If news contain breaking confirmation/refutation of event -> BLOCK
    - If news are neutral/irrelevant -> PASS

    Only calls LLM if there are actual news headlines.
    """
    if not news_headlines:
        return True, "No recent news found, default PASS"

    headlines_text = "\n".join([f"- {h}" for h in news_headlines[:5]])

    metaculus_str = f"{metaculus_prob:.0%}" if metaculus_prob else "N/A"

    prompt = f'''Ты - риск-менеджер хедж-фонда. Твоя задача: оценить новости и определить, содержат ли они 100% подтверждение или опровержение события.

Событие: "{market_title}"
Текущая вероятность на Metaculus: {metaculus_str}

Свежие новости (последние 30 дней):
{headlines_text}

Правила:
1. Если в новостях есть "Breaking News", свершившийся факт или событие, которое ДЕЛАЕТ прогноз НЕАКТУАЛЬНЫМ - ответь "BLOCK".
2. Если новости обычные/нейтральные/не относятся напрямую к событию - ответь "PASS".
3. Отвечай ТОЛЬКО одним словом: BLOCK или PASS.

Примеры BLOCK:
- "Президент подписал закон..." (когда рынок о законе)
- "Компания объявила о банкротстве..." (когда рынок о её будущем)
- "Выборы завершились..." (когда рынок о выборах)

Примеры PASS:
- Общие рыночные новости
- Новости о похожих, но других событиях
- Прогнозы аналитиков без свершившихся фактов'''

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return True, "No LLM API key, default PASS"

    model = "deepseek-chat"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 50
            },
            timeout=30
        )

        if not resp.ok:
            return True, f"LLM API returned {resp.status_code}, default PASS"

        data = resp.json()
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        content = raw_content.strip().upper() if raw_content else ""

        if re.search(r'\bBLOCK\b', content):
            return False, "News contain breaking confirmation/refutation"
        elif re.search(r'\bPASS\b', content):
            return True, "News neutral, trade allowed"
        else:
            return True, f"Ambiguous LLM response: {content[:20]}, default PASS"

    except Exception as e:
        logger.debug(f"[news_scanner] {type(e).__name__}: {e}")
        return True, f"LLM error: {str(e)[:50]}, default PASS"


def check_market_news(market: dict) -> tuple[bool, str]:
    keywords = extract_keywords(market.get("question", ""))
    if not keywords:
        return True, "No keywords extracted"

    news_data = fetch_recent_news(keywords)
    headlines = news_data.get("headlines", [])

    clusters = market.get("clusters", ["other"])
    cluster_key = clusters[0] if clusters else "other"

    if not headlines:
        return True, "No fresh news — no contradiction found"

    metaculus_prob = market.get("metaculus_prob")
    passed, reason = news_sanity_check(
        market_title=market.get("question", ""),
        news_headlines=headlines,
        metaculus_prob=metaculus_prob
    )

    try:
        from config import CACHE_FILE
        cache = load_json(CACHE_FILE, {})
        cache.setdefault("news", {}).setdefault(cluster_key, {})["passed"] = passed
        save_json(CACHE_FILE, cache)
    except Exception as e:
        logger.debug(f"[news_scanner] {type(e).__name__}: {e}")
        pass

    return passed, reason


if __name__ == "__main__":
    test_question = "Will U.S. enact AI safety bill before 2027?"
    keywords = extract_keywords(test_question)
    print(f"Keywords: {keywords}")

    news = fetch_recent_news(keywords)
    print(f"\nNews found: {len(news.get('headlines', []))}")
    for h in news.get("headlines", [])[:3]:
        print(f"  - {h[:80]}...")

    if news.get("headlines"):
        passed, reason = news_sanity_check(test_question, news["headlines"], metaculus_prob=0.48)
        print(f"\nSanity check: {'PASS' if passed else 'BLOCK'}")
        print(f"Reason: {reason}")
