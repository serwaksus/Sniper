"""
signal_pipeline.py — Signal generation and market analysis pipeline (orchestrator).
Extracted from dotm_sniper.py v5.3.0.
Modules extracted: metaculus.py, market_fetcher.py, signal_scorer.py.
"""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import requests
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import sanitize_for_prompt
from config import MAX_P_MODEL_RATIO, SNIPER_LOG
from schema import HYP_CLUSTERS, HYP_CONFIDENCE, HYP_FACTORS, HYP_P_MODEL, HYP_SLUG
from utils import load_env_file
from db import load_settings as _db_load_settings

load_env_file()

LOG_FILE = SNIPER_LOG
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ── DeepSeek API ─────────────────────────────────────────────
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_MAIN = "deepseek-chat"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
URL = "https://api.deepseek.com/v1/chat/completions"

# ── Signal thresholds ────────────────────────────────────────
MIN_PROB_RATIO = 2.0
MIN_P_MODEL = 0.03
MIN_CONFIDENCE = 0.65
BATCH_SIZE = 6
ADVISOR_MODEL = "deepseek-reasoner"
ADVISOR_MIN_CONFIDENCE = 0.70

_llm_call_times: list[float] = []


def get_settings():
    return _db_load_settings() or {}


def _check_llm_circuit_breaker():
    now = time.time()
    _llm_call_times[:] = [t for t in _llm_call_times if now - t < 3600]
    if len(_llm_call_times) >= 60:
        logger.warning("[LLM-CB] Circuit breaker: 60 calls/hour limit reached, skipping")
        return False
    _llm_call_times.append(now)
    return True


# ── Re-exports from extracted modules ────────────────────────
from metaculus import (
    METACULUS_API_KEY,
    METACULUS_URL,
    METACULUS_HEADERS,
    DISPERSION_PENALTY_THRESHOLD,
    METACULUS_GAP_THRESHOLD,
    DATE_WINDOW_DAYS,
    load_cache,
    save_cache,
    normalize_probability,
    metaculus_search,
    metaculus_get_question,
    parse_resolve_date,
    dates_match,
    _generate_search_queries,
    _calculate_metaculus_match,
    get_metaculus_forecast,
    get_time_decay_threshold,
    check_metaculus_gap,
)

from market_fetcher import (
    MIN_VOLUME,
    MIN_TTL_HOURS,
    MAX_PRICE,
    ALLOWED_CLUSTERS,
    BANNED_CLUSTERS,
    PRE_FILTER_OTHER_MIN_VOLUME,
    fetch_markets,
    fetch_gamma_dotm_candidates,
    pre_filter_before_batching,
)

from signal_scorer import (
    CALIBRATION_DAMPING_FACTOR,
    CALIBRATION_DOTM_THRESHOLD,
    CALIBRATION_AGGRESSIVE_PMODEL,
    CALIBRATION_METACULUS_LOW,
    CLUSTER_SCORE_ADJUSTMENTS,
    _count_resolved_hypotheses,
    calibrate_prediction,
    _cluster_score_adjustment,
    _compute_signal_score,
    full_market_analysis,
)

__all__ = [
    "ADVISOR_MIN_CONFIDENCE",
    "ADVISOR_MODEL",
    "ALLOWED_CLUSTERS",
    "API_KEY",
    "BANNED_CLUSTERS",
    "BATCH_SIZE",
    "CALIBRATION_AGGRESSIVE_PMODEL",
    "CALIBRATION_DAMPING_FACTOR",
    "CALIBRATION_DOTM_THRESHOLD",
    "CALIBRATION_METACULUS_LOW",
    "CLUSTER_SCORE_ADJUSTMENTS",
    "DATE_WINDOW_DAYS",
    "DISPERSION_PENALTY_THRESHOLD",
    "HEADERS",
    "MAX_PRICE",
    "MAX_P_MODEL_RATIO",
    "METACULUS_API_KEY",
    "METACULUS_GAP_THRESHOLD",
    "METACULUS_HEADERS",
    "METACULUS_URL",
    "MIN_CONFIDENCE",
    "MIN_PROB_RATIO",
    "MIN_P_MODEL",
    "MIN_TTL_HOURS",
    "MIN_VOLUME",
    "MODEL_MAIN",
    "PRE_FILTER_OTHER_MIN_VOLUME",
    "URL",
    "_build_batch_results",
    "_calculate_metaculus_match",
    "_check_llm_circuit_breaker",
    "_cluster_score_adjustment",
    "_compute_signal_score",
    "_count_resolved_hypotheses",
    "_generate_search_queries",
    "_parse_batch_response",
    "advisor_pre_check",
    "batch_analyze_markets",
    "calibrate_prediction",
    "check_metaculus_gap",
    "dates_match",
    "fetch_gamma_dotm_candidates",
    "fetch_markets",
    "full_market_analysis",
    "get_metaculus_forecast",
    "get_settings",
    "get_time_decay_threshold",
    "load_cache",
    "metaculus_get_question",
    "metaculus_search",
    "normalize_probability",
    "parse_resolve_date",
    "pre_filter_before_batching",
    "save_cache",
]


# ═══════════════════════════════════════════════════════════════
# GROUP E — Batch orchestration + Advisor
# ═══════════════════════════════════════════════════════════════

def batch_analyze_markets(markets):
    if not markets:
        return []

    metaculus_cache = {}
    for m in markets:
        if m.get("price", 0) < 0.35:
            slug = m.get(HYP_SLUG, "")
            question = m.get("question", "")
            end_date = m.get("end_date")
            meta = get_metaculus_forecast(question, end_date)
            if meta.get("found"):
                metaculus_cache[slug] = meta.get("probability")
            else:
                metaculus_cache[slug] = None
        else:
            metaculus_cache[m.get(HYP_SLUG)] = None

    batch_items = []
    for m in markets:
        slug = m.get(HYP_SLUG, "")
        question = m.get("question", "")
        price = m.get("price", 0)
        volume = m.get("volume", 0)
        ttl_hours = m.get("ttl_hours", 999)
        cluster = m.get(HYP_CLUSTERS, ["other"])[0]
        batch_items.append({
            HYP_SLUG: slug,
            "question": question,
            "market_price": round(price, 4),
            "volume": round(volume, 0),
            "ttl_hours": round(ttl_hours, 0),
            "cluster": cluster,
        })

    for item in batch_items:
        item["question"] = sanitize_for_prompt(item["question"])
    items_json = json.dumps(batch_items, indent=2)

    prompt = f"""Prediction market analyst. Analyze these DOTM (deep out-the-money) markets where the crowd may underestimate probability.

CRITICAL DOTM METHODOLOGY: These markets are priced very low (often 1-30 cents). The crowd systematically underestimates tail risks due to recency bias, black swan blindness, and time premium. For each market:
1. List 2-3 SPECIFIC plausible scenarios that could make the event happen
2. Estimate probability for each scenario (even 1-5% each adds up)
3. Sum scenario probabilities to get TRUE probability
4. If total > market price, the crowd is underestimating

MARKETS (JSON array):
{items_json}

For EACH market, identify 2-3 specific scenarios/factors and estimate the TRUE probability independently from the market price.

Return a JSON ARRAY with one object per market (same order as input):
[
  {{
    "slug": "market-slug",
    "factors": [{{"factor": "description", "direction": "supports/opposes", "weight": "high/medium/low", "source": "source"}}],
    "estimated_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "brief explanation"
  }}
]

Rules:
- estimated_probability: decimal 0.0-1.0 (NOT percentage)
- Do NOT anchor on the market price - provide independent assessment based on scenarios
- Conservative on low-volume (<$10K) markets
- If estimate >3x price, explain what crowd is missing
- Return exactly {len(batch_items)} items matching the input slugs

CRITICAL REGULATION FOR CONFIDENCE SCORING: Do NOT default to a flat 0.65 confidence across multiple markets just to pass filters. You must utilize the full analytical spectrum from 0.65 to 1.0 based on the robustness of available data, market volume, and time to resolution. If a market has weak evidence, score it near 0.65. If the evidence is solid and aligned with predictive markets, score it between 0.80 and 0.95. Generating a flat 0.65 across the entire batch will break the downstream composite scoring and will be treated as an invalid evaluation."""

    resp = None
    if not _check_llm_circuit_breaker():
        logger.warning("[BATCH] Circuit breaker tripped, falling back to individual analysis")
        resp = None
    else:
        for _attempt in range(3):
            try:
                resp = requests.post(URL, headers=HEADERS, json={
                    "model": MODEL_MAIN,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }, timeout=60)
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as _e:
                if _attempt < 2:
                    _backoff = 2 ** (_attempt + 1)
                    logger.warning(f"[BATCH] LLM retry {_attempt+1}/3 in {_backoff}s: {_e}")
                    time.sleep(_backoff)
                else:
                    logger.error(f"[BATCH] LLM error after 3 retries: {_e}")
                    resp = None
    try:
        if resp is None:
            raise Exception("LLM unavailable after retries")
        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if not content:
            content = msg.get("reasoning") or ""

        batch_results = _parse_batch_response(content, batch_items, metaculus_cache)
        if batch_results and len(batch_results) == len(markets):
            logger.info(f"[BATCH] Successfully parsed batch of {len(batch_results)} markets")
            return batch_results
        else:
            logger.warning(f"[BATCH] Batch parse returned {len(batch_results) if batch_results else 0}/{len(markets)} items, falling back to individual")
    except Exception as e:
        logger.error(f"[BATCH] LLM error: {e}, falling back to individual analysis")

    results = []
    for m in markets:
        results.append(full_market_analysis(m))
    return results


def _parse_batch_response(content, batch_items, metaculus_cache=None):
    if metaculus_cache is None:
        metaculus_cache = {}

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    for prefix in ("```json", "```JSON", "```"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    cleaned = cleaned.strip()
    if cleaned.startswith(":"):
        cleaned = cleaned[1:].strip()

    start = cleaned.find('[')
    if start == -1:
        logger.warning(f"[BATCH-PARSE] No '[' found in LLM response: {content[:200]}")
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    arr = json.loads(candidate)
                    if isinstance(arr, list):
                        return _build_batch_results(arr, batch_items, metaculus_cache)
                except json.JSONDecodeError:
                    pass
                for end in range(i, len(cleaned)):
                    if cleaned[end] == ']':
                        try:
                            arr = json.loads(cleaned[start:end + 1])
                            if isinstance(arr, list):
                                return _build_batch_results(arr, batch_items, metaculus_cache)
                        except json.JSONDecodeError:
                            continue
                break

    fallback = re.search(r'\[[\s\S]*\]', cleaned)
    if fallback:
        try:
            arr = json.loads(fallback.group(0))
            if isinstance(arr, list):
                return _build_batch_results(arr, batch_items, metaculus_cache)
        except json.JSONDecodeError:
            pass

    individual_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
    if individual_objects:
        try:
            arr = [json.loads(obj) for obj in individual_objects]
            arr = [a for a in arr if isinstance(a, dict)]
            if arr:
                logger.info(f"[BATCH-PARSE] Recovered {len(arr)} individual objects from failed batch")
                return _build_batch_results(arr, batch_items, metaculus_cache)
        except json.JSONDecodeError:
            pass

    logger.warning(f"[BATCH-PARSE] All parsing methods failed. Response preview: {content[:300]}")

    return None


def _build_batch_results(parsed_array, batch_items, metaculus_cache=None):
    slug_to_item = {it[HYP_SLUG]: it for it in batch_items}

    results_map: dict[str, dict[str, Any]] = {}
    for item in parsed_array:
        if not isinstance(item, dict):
            continue

        slug = item.get(HYP_SLUG, "")
        if slug not in slug_to_item:
            if len(results_map) < len(batch_items):
                unmatched_idx = len(results_map)
                slug = batch_items[unmatched_idx][HYP_SLUG]
            else:
                continue

        bi = slug_to_item.get(slug)
        if not bi:
            continue

        market_price = bi["market_price"]
        cluster = bi["cluster"]

        p_model_llm = normalize_probability(item.get("estimated_probability", market_price * 2))
        confidence = min(max(float(item.get(HYP_CONFIDENCE, 0.6)), 0.1), 0.95)
        factors = item.get(HYP_FACTORS, [])

        max_p_model = market_price * MAX_P_MODEL_RATIO
        p_model = min(p_model_llm, max_p_model)

        metaculus_prob = None
        if metaculus_cache:
            metaculus_prob = metaculus_cache.get(slug)

        p_model, _ = calibrate_prediction(p_model, market_price, metaculus_prob, cluster=cluster)

        settings = get_settings()
        min_p_model = settings.get("min_p_model", MIN_P_MODEL)
        if p_model < min_p_model - 0.001:
            results_map[slug] = {
                "question": bi["question"],
                HYP_SLUG: slug,
                "market_price": market_price,
                HYP_P_MODEL: p_model,
                "prob_ratio": 0,
                HYP_CONFIDENCE: confidence,
                "action": "SKIP",
                HYP_FACTORS: [],
                "source_signal": "default",
                "reasoning": f"p_model too low ({p_model:.1%})",
                "best_ask": None,
            }
            continue

        metaculus_prob_val = metaculus_cache.get(slug) if metaculus_cache else None
        signal_score, prob_ratio, supporting, high_weight, metaculus_alignment, buzz_score, ratio_score, factor_score, vol_score, time_score, ttl_days, orderbook_score, _ob_reason = _compute_signal_score(
            p_model, market_price, factors, bi.get("volume", 0), bi.get("ttl_hours", 999),
            cluster, slug=slug, question=bi.get("question", ""),
            metaculus_prob_val=metaculus_prob_val, settings=settings,
            condition_token_id=bi.get("condition_token_id", "")
        )

        base_threshold = settings.get("signal_threshold", 55)
        if ttl_days > 90:
            min_signal = settings.get("signal_threshold_long_horizon", base_threshold + 10)
        elif ttl_days >= 31:
            min_signal = settings.get("signal_threshold_medium_horizon", base_threshold + 5)
        else:
            min_signal = base_threshold

        source_signal = "default"
        metaculus_prob_val = metaculus_cache.get(slug) if metaculus_cache else None
        if metaculus_prob_val is not None and metaculus_prob_val > market_price:
            gap = metaculus_prob_val - market_price
            signal_strength = min(gap / 0.15, 1.0)
            dispersion = None
            meta_cache_entry = None
            cache = load_cache()
            for _q, cached_meta in cache.get("metaculus", {}).items():
                if cached_meta.get("probability") == metaculus_prob_val:
                    meta_cache_entry = cached_meta
                    break
            if meta_cache_entry:
                dispersion = meta_cache_entry.get("dispersion")
            if dispersion is not None and dispersion < 0.10:
                signal_strength *= dispersion / 0.10
            if signal_strength > 0.3:
                if metaculus_prob_val > p_model:
                    p_model = metaculus_prob_val
                else:
                    p_model = 0.6 * metaculus_prob_val + 0.4 * p_model
                source_signal = "metaculus_override"
                confidence = min(confidence + 0.10, 0.95)
                min_signal = max(min_signal - 10, 35)
                logger.info(f"[META-OVERRIDE-BATCH] {slug[:30]}... p_model={p_model:.1%} from metaculus={metaculus_prob_val:.1%}")

        prob_ratio = p_model / market_price if market_price > 0 else 0
        ratio_score = min(prob_ratio / 3.0, 1.0) * 30
        sm_score_batch = 0
        ct_id = bi.get("condition_token_id", "")
        if ct_id:
            try:
                from smart_money import check_smart_money_activity
                sm_result = check_smart_money_activity(ct_id)
                sm_score_batch = sm_result.get("signal_score", 0)
                if sm_score_batch > 0:
                    logger.info(f"[SMART_MONEY-BATCH] {slug[:30]}... score=+{sm_score_batch}")
            except Exception:
                pass
        signal_score = ratio_score + factor_score + vol_score + time_score + metaculus_alignment + _cluster_score_adjustment(cluster, settings) + buzz_score + orderbook_score + sm_score_batch + orderbook_score

        action = "BUY" if signal_score >= min_signal and confidence >= settings.get("min_confidence", MIN_CONFIDENCE) and prob_ratio >= MIN_PROB_RATIO else "SKIP"

        logger.info(
            f"[SIGNAL-BATCH] ratio={prob_ratio:.2f}x -> {ratio_score:.0f}, factors={len(supporting)}/{len(high_weight)} -> {factor_score:.0f}, "
            f"vol=${bi.get('volume',0):,.0f} -> {vol_score:.0f}, ttl={ttl_days:.0f}d -> {time_score:.0f}, "
            f"buzz={buzz_score:.1f}, ob={orderbook_score} "
            f"= {signal_score:.0f}/{min_signal} => {action}"
        )

        results_map[slug] = {
            "question": bi["question"],
            HYP_SLUG: slug,
            "market_price": market_price,
            HYP_P_MODEL: p_model,
            "prob_ratio": prob_ratio,
            HYP_CONFIDENCE: confidence,
            "action": action,
            HYP_FACTORS: factors,
            "source_signal": source_signal,
            "signal_score": signal_score,
            "reasoning": f"score={signal_score:.0f}/{min_signal}(batch), ratio={prob_ratio:.2f}x, conf={confidence:.2f}, src={source_signal}",
            "best_ask": None,
        }

    results = []
    for bi in batch_items:
        if bi[HYP_SLUG] in results_map:
            results.append(results_map[bi[HYP_SLUG]])
        else:
            results.append({
                "question": bi["question"],
                HYP_SLUG: bi[HYP_SLUG],
                "market_price": bi["market_price"],
                HYP_P_MODEL: bi["market_price"] * 2,
                "prob_ratio": 2.0,
                HYP_CONFIDENCE: 0.5,
                "action": "SKIP",
                HYP_FACTORS: [],
                "source_signal": "default",
                "reasoning": "batch_parse_fallback",
                "best_ask": None,
            })

    return results


def advisor_pre_check(market, analysis, estimated_size=0, balance=1):
    question = market.get("question", "")
    slug = market.get(HYP_SLUG, "")
    price = market.get("price", 0)
    p_model = analysis.get(HYP_P_MODEL, 0)
    factors = analysis.get(HYP_FACTORS, [])
    score = analysis.get("signal_score", 0)
    reasoning = analysis.get("reasoning", "")

    factors_text = "\n".join(
        f"  - [{f.get('weight', '?')}] {sanitize_for_prompt(f.get('factor', ''))} ({f.get('direction', '')})"
        for f in factors[:5]
    ) if factors else "  (none)"

    prompt = f"""You are DOTM Advisor - an independent risk analyst verifying a trade before execution.
Use Chain-of-Thought reasoning to evaluate the thesis.

MARKET: {sanitize_for_prompt(question)}
SLUG: {slug}
MARKET PRICE: ${price:.3f} ({price*100:.1f}%)
BOT P_MODEL (estimated true probability): {p_model:.1%}
PROBABILITY RATIO: {f"{p_model/price:.2f}" if price > 0 else "N/A"}x vs market
COMPOSITE SIGNAL SCORE: {score:.0f}/100
BOT REASONING: {sanitize_for_prompt(reasoning)}

SUPPORTING FACTORS IDENTIFIED BY BOT:
{factors_text}

YOUR TASK:
1. Think step-by-step about whether the bot's thesis is sound.
2. Consider: Is the probability estimate realistic? Are the factors genuine or generic?
3. Check for hallucination patterns: p_model >> market price without concrete catalyst.
4. Is this a DOTM market where the crowd truly underestimates probability?

Return ONLY JSON:
{{"p_estimate": 0.XX, "confidence": 0.XX, "factors": ["factor1", "factor2"], "verdict": "CONFIRM/DIVERGE/WARNING/UNKNOWN"}}

Rules:
- CONFIRM: You agree the trade has positive expected value. confidence >= 0.70 required.
- DIVERGE: Your analysis contradicts the bot (e.g., you think probability is LOWER).
- WARNING: Significant risk factor the bot missed. Do NOT trade.
- UNKNOWN: Insufficient information to verify. Default safe choice."""

    try:
        if not _check_llm_circuit_breaker():
            size_pct = estimated_size / balance if balance > 0 else 1
            if size_pct <= 0.02:
                logger.info("[ADVISOR] Circuit breaker, but micro-position <=2%, allowing")
                return True, "UNKNOWN", 0.0, "advisor_cb_micro_override"
            logger.warning("[ADVISOR] Circuit breaker tripped, blocking trade")
            return False, "UNKNOWN", 0.0, "advisor_circuit_breaker"
        resp = requests.post(URL, headers=HEADERS, json={
            "model": ADVISOR_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000
        }, timeout=60)

        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

        if not content and reasoning:
            json_match = re.search(r'\{[^{}]*\}', reasoning)
            if json_match:
                content = json_match.group(0)
                logger.info("[ADVISOR] Extracted JSON from reasoning_content (content was empty)")

        if not content:
            size_pct = estimated_size / balance if balance > 0 else 1
            if size_pct <= 0.02:
                logger.info("[ADVISOR] Empty response, but micro-position <=2%, allowing")
                return True, "UNKNOWN", 0.0, "advisor_empty_micro_override"
            logger.warning("[ADVISOR] Empty response from reasoner, blocking trade")
            return False, "UNKNOWN", 0.0, "advisor_empty_response"

        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from advisor_script import parse_llm_advisor_response
        result, parse_err = parse_llm_advisor_response(content, log_label="ADVISOR-PRE")
        if result is None:
            size_pct = estimated_size / balance if balance > 0 else 1
            if size_pct <= 0.02:
                logger.info(f"[ADVISOR] Parse failed but micro-position, allowing: {parse_err}")
                return True, "UNKNOWN", 0.0, "advisor_parse_micro_override"
            logger.warning(f"[ADVISOR] Parse failed: {parse_err}, blocking trade")
            return False, "UNKNOWN", 0.0, f"advisor_parse_error: {parse_err}"

        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get(HYP_CONFIDENCE, 0.0)
        advisor_p = result.get("p_estimate", 0)
        advisor_factors = result.get(HYP_FACTORS, [])

        logger.info(
            f"[ADVISOR] verdict={verdict} conf={confidence:.2f} "
            f"advisor_p={advisor_p:.1%} vs bot_p={p_model:.1%} | "
            f"factors: {advisor_factors[:2]}"
        )

        if verdict == "CONFIRM" and confidence >= ADVISOR_MIN_CONFIDENCE:
            logger.info(f"[ADVISOR] ✅ Trade APPROVED by advisor ({verdict}, conf={confidence:.2f})")
            return True, verdict, confidence, "approved"
        elif verdict == "WARNING" and confidence < ADVISOR_MIN_CONFIDENCE:
            logger.info("[ADVISOR] ⚠️ Trade WARNING (advisor uncertain), allowing small position")
            return True, verdict, confidence, "advisor_warning_allowed"
        elif verdict == "DIVERGE":
            size_pct = estimated_size / balance if balance > 0 else 1
            advisor_agrees_direction = advisor_p >= 0.5 * p_model
            if size_pct <= 0.02:
                logger.info(f"[ADVISOR] 🔄 DIVERGE override: micro-position {size_pct:.1%} ≤ 2%, allowing")
                return True, verdict, confidence, "diverge_micro_override"
            if advisor_agrees_direction:
                logger.info(f"[ADVISOR] 🔄 DIVERGE override: advisor_p={advisor_p:.1%} > price={price:.1%}, direction agrees")
                return True, verdict, confidence, "diverge_direction_agrees"
            reason = f"advisor_veto: verdict={verdict}, conf={confidence:.2f}"
            logger.info(f"[ADVISOR] 🚫 Trade BLOCKED: {reason}")
            print(f"   🚫 Advisor veto: {verdict} (conf={confidence:.2f})")
            return False, verdict, confidence, reason
        else:
            reason = f"advisor_veto: verdict={verdict}, conf={confidence:.2f}"
            logger.info(f"[ADVISOR] 🚫 Trade BLOCKED: {reason}")
            print(f"   🚫 Advisor veto: {verdict} (conf={confidence:.2f})")
            return False, verdict, confidence, reason

    except requests.exceptions.Timeout:
        logger.warning("[ADVISOR] Timeout, blocking trade")
        return False, "UNKNOWN", 0.0, "advisor_timeout"
    except Exception as e:
        logger.error(f"[ADVISOR] Error: {e}")
        return False, "UNKNOWN", 0.0, f"advisor_error: {str(e)[:80]}"
