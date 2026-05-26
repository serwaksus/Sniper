#!/usr/bin/env python3
"""
News Scanner - Fresh news reality check before trading.
Uses Tavily API for real-time news search.
"""
import os
import requests
import subprocess
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

TAVILY_API_URL = "https://api.tavily.com/search"

NEWS_TIME_WINDOW_HOURS = 48
MAX_NEWS_AGE_DAYS_DEFAULT = 30

def load_env():
    """Load environment variables from .env file"""
    env_path = "/root/dotm-sniper/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

load_env()

def fetch_recent_news(market_keywords: List[str], max_results: int = 5, max_age_days: int = None) -> Dict:
    """
    Search for recent news headlines based on market keywords.
    Returns dict with headlines, sources, and timestamps.
    max_age_days: only return news published within this many days (default 30).
    """
    if max_age_days is None:
        max_age_days = MAX_NEWS_AGE_DAYS_DEFAULT

    tavily_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_key or tavily_key == "your_tavily_key_here":
        return _fetch_ddg_news_fallback(market_keywords, max_results, max_age_days)

    query = " ".join(market_keywords[:5])
    params = {
        "api_key": tavily_key,
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
    except Exception:
        pass

    return _fetch_ddg_news_fallback(market_keywords, max_results, max_age_days)


def _fetch_ddg_news_fallback(keywords: List[str], max_results: int, max_age_days: int = 30) -> Dict:
    """
    Fallback using DuckDuckGo HTML news search
    when Tavily API is not available.
    Applies freshness filter: only results within max_age_days.
    """
    query = "+".join(keywords[:3])
    url = f"https://duckduckgo.com/html/?q={query}+news&df=m&ia=news"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.ok:
            import re
            html = response.text
            headlines = re.findall(r'<a class="result__a"[^>]*href="[^"]*"[^>]*>([^<]+)</a>', html)[:max_results]
            if not headlines:
                headlines = re.findall(r'<h2[^>]*>([^<]+)</h2>', html)[:max_results]
            return {
                "headlines": [h.strip() for h in headlines if len(h.strip()) > 20],
                "sources": [],
                "query": " ".join(keywords),
                "found": len(headlines) > 0
            }
    except Exception:
        pass

    return {"headlines": [], "sources": [], "query": "", "found": False}


def extract_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from market question for news search"""
    import re
    stop_words = {
        "will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for",
        "this", "that", "is", "are", "was", "were", "before", "after", "end",
        "happen", "occur", "take", "place"
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
    return [w for w in words if w not in stop_words][:8]


def news_sanity_check(market_title: str, news_headlines: List[str],
                      metaculus_prob: Optional[float] = None) -> Tuple[bool, str]:
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

        data = resp.json()
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        content = raw_content.strip().upper() if raw_content else ""

        if "BLOCK" in content:
            return False, f"News contain breaking confirmation/refutation"
        elif "PASS" in content:
            return True, "News neutral, trade allowed"
        else:
            return True, f"Ambiguous LLM response: {content[:20]}, default PASS"

    except Exception as e:
        return True, f"LLM error: {str(e)[:50]}, default PASS"


def check_market_news(market: Dict) -> Tuple[bool, str]:
    """
    Main entry point for news check on a market.
    Returns (passed: bool, reason: str)
    Anti-FUD: if no fresh news found, the trade is BLOCKED (safety measure).
    """
    keywords = extract_keywords(market.get("question", ""))
    if not keywords:
        return True, "No keywords extracted"

    news_data = fetch_recent_news(keywords)
    headlines = news_data.get("headlines", [])

    if not headlines:
        return False, "No fresh news — blocked by Anti-FUD v5.3.2"

    metaculus_prob = market.get("metaculus_prob")
    passed, reason = news_sanity_check(
        market_title=market.get("question", ""),
        news_headlines=headlines,
        metaculus_prob=metaculus_prob
    )

    if passed:
        return True, reason

    if not headlines or len(headlines) < 2:
        return True, f"Insufficient news ({len(headlines)} headlines), default PASS"

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