#!/usr/bin/env python3
"""
Bayesian posterior updater for DOTM Sniper.
Maintains log-odds prior per position, updates with news likelihood ratios.
Replaces expensive LLM calls with fast Bayesian computation.
"""
from __future__ import annotations
import os
import sys
import math
import time
import logging
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json
from config import BAYESIAN_STATE_FILE

logger = logging.getLogger(__name__)

_bayes_lock = threading.RLock()


NEWS_LIKELIHOOD = {
    "confirms_impossible": {"p_yes_given_news": 0.02, "label": "News confirms outcome impossible"},
    "strongly_contradicts": {"p_yes_given_news": 0.10, "label": "Strong contradiction"},
    "moderately_contradicts": {"p_yes_given_news": 0.40, "label": "Moderate contradiction"},
    "neutral": {"p_yes_given_news": 0.50, "label": "Neutral/no relevant news"},
    "moderately_supports": {"p_yes_given_news": 0.65, "label": "Moderate support"},
    "strongly_supports": {"p_yes_given_news": 0.85, "label": "Strong support"},
    "confirms_inevitable": {"p_yes_given_news": 0.95, "label": "News confirms outcome"},
}

_ADAPTIVE_CACHE = {"data": None, "loaded_at": 0}


def _get_effective_likelihoods() -> dict:
    now = time.time()
    if _ADAPTIVE_CACHE["data"] and (now - _ADAPTIVE_CACHE["loaded_at"]) < 300:
        return _ADAPTIVE_CACHE["data"]

    base = dict(NEWS_LIKELIHOOD)
    try:
        from hermes_memory import get_adaptive_likelihoods
        adapted = get_adaptive_likelihoods(min_samples=5)
        if adapted:
            for cat, vals in adapted.items():
                if cat in base:
                    base[cat] = {"p_yes_given_news": vals["p_yes_given_news"],
                                 "label": base[cat]["label"], "adapted": True, "samples": vals["samples"]}
    except Exception as e:
        logger.debug(f"[bayesian_init] {type(e).__name__}: {e}")

    _ADAPTIVE_CACHE["data"] = base
    _ADAPTIVE_CACHE["loaded_at"] = now
    return base


def _prob_to_logodds(p: float) -> float:
    p = max(1e-8, min(1 - 1e-8, p))
    return math.log(p / (1 - p))


def _logodds_to_prob(lo: float) -> float:
    if lo > 500:
        return 1.0
    if lo < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-lo))


def init_posterior(slug: str, p_model: float, p_market: float) -> None:
    with _bayes_lock:
        state = load_json(BAYESIAN_STATE_FILE, {"positions": {}})
        if not isinstance(state, dict):
            state = {"positions": {}}

        prior_logodds = _prob_to_logodds(p_model)
        market_logodds = _prob_to_logodds(p_market)

        state["positions"][slug] = {
            "p_model_entry": round(p_model, 6),
            "p_market_entry": round(p_market, 6),
            "prior_logodds": round(prior_logodds, 4),
            "posterior_logodds": round(prior_logodds, 4),
            "posterior_prob": round(p_model, 6),
            "market_logodds": round(market_logodds, 4),
            "updates": 0,
            "last_update": datetime.now().isoformat(),
            "history": [{"t": datetime.now().isoformat(), "posterior": round(p_model, 6), "event": "init"}],
        }

        save_json(BAYESIAN_STATE_FILE, state)
    logger.info(f"[BAYES] Init {slug[:40]}... prior={p_model:.1%}, logodds={prior_logodds:.2f}")


def update_posterior(slug: str, news_category: str, llm_assessment: str | None = None) -> float | None:
    with _bayes_lock:
        state = load_json(BAYESIAN_STATE_FILE, {"positions": {}})
        if not isinstance(state, dict):
            state = {"positions": {}}

        pos = state["positions"].get(slug)
        if not pos:
            return None

        likelihoods = _get_effective_likelihoods()
        likelihood = likelihoods.get(news_category, likelihoods.get("neutral", NEWS_LIKELIHOOD["neutral"]))
        p_yes_given_news = likelihood["p_yes_given_news"]
        p_no_given_news = 1 - p_yes_given_news

        prior = pos["posterior_logodds"]
        p_prior = _logodds_to_prob(prior)

        lr = math.log(max(p_yes_given_news, 1e-8) / max(p_no_given_news, 1e-8))
        new_posterior = prior + lr

        pos["posterior_logodds"] = round(new_posterior, 4)
        pos["posterior_prob"] = round(_logodds_to_prob(new_posterior), 6)
        pos["updates"] = pos.get("updates", 0) + 1
        pos["last_update"] = datetime.now().isoformat()

        if len(pos.get("history", [])) > 100:
            pos["history"] = pos["history"][-100:]
        pos.setdefault("history", []).append({
            "t": datetime.now().isoformat(),
            "posterior": round(_logodds_to_prob(new_posterior), 6),
            "event": news_category,
            "lr": round(lr, 4),
        })

        state["positions"][slug] = pos
        save_json(BAYESIAN_STATE_FILE, state)

    new_prob = _logodds_to_prob(new_posterior)
    p_entry = pos.get("p_model_entry", 0.5)
    drop_ratio = new_prob / p_entry if p_entry > 0 else 0

    logger.info(
        f"[BAYES] {slug[:40]}... prior={p_prior:.1%} → posterior={new_prob:.1%} "
        f"(news={news_category}, lr={lr:.2f}, drop={drop_ratio:.2f})"
    )

    return new_prob


def should_exit(slug: str, threshold_ratio: float = 0.40) -> tuple[bool, str]:
    state = load_json(BAYESIAN_STATE_FILE, {"positions": {}})
    if not isinstance(state, dict):
        return False, "no_state"

    pos = state.get("positions", {}).get(slug)
    if not pos:
        return False, "no_position"

    posterior = pos.get("posterior_prob", 0.5)
    entry_p = pos.get("p_model_entry", 0.5)

    if entry_p <= 0:
        return False, "invalid_entry"

    drop_ratio = posterior / entry_p

    if drop_ratio <= threshold_ratio:
        return True, f"bayesian_posterior={posterior:.1%} <= {threshold_ratio:.0%} of entry={entry_p:.1%}"

    if posterior < 0.02:
        return True, f"bayesian_posterior={posterior:.1%} near zero"

    return False, f"posterior={posterior:.1%} ok (ratio={drop_ratio:.2f})"


def classify_news_with_llm(question: str, headlines: list) -> str:
    if not headlines:
        return "neutral"

    news_text = "\n".join(f"- {h}" for h in headlines[:5])

    prompt = f"""Classify the impact of these news headlines on this prediction market question.

Question: {question}

News:
{news_text}

Return ONLY one category:
- confirms_impossible: News proves the YES outcome is now impossible
- strongly_contradicts: Major evidence against YES outcome
- moderately_contradicts: Some negative evidence for YES
- neutral: No relevant news or balanced evidence
- moderately_supports: Some positive evidence for YES
- strongly_supports: Major evidence favoring YES
- confirms_inevitable: News proves YES outcome is certain

Category:"""

    try:
        import requests
        import os
        API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
        URL = "https://api.deepseek.com/v1/chat/completions"
        HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

        resp = requests.post(URL, headers=HEADERS, json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 20,
        }, timeout=15)

        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
        for cat in NEWS_LIKELIHOOD:
            if cat in content:
                return cat
    except Exception as e:
        logger.warning(f"[BAYES-LLM] Classification failed: {e}")

    return "neutral"


def cleanup_slug(slug: str) -> None:
    with _bayes_lock:
        state = load_json(BAYESIAN_STATE_FILE, {"positions": {}})
        if not isinstance(state, dict):
            return
        state.get("positions", {}).pop(slug, None)
        save_json(BAYESIAN_STATE_FILE, state)
