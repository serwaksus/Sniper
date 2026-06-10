"""
signal_scorer.py — Signal scoring, calibration, and individual market analysis.
Extracted from signal_pipeline.py.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from utils import sanitize_for_prompt, parse_llm_json
from config import MAX_P_MODEL_RATIO
from schema import HYP_CLUSTERS, HYP_CONFIDENCE, HYP_FACTORS, HYP_P_MODEL, HYP_SLUG
import hypotheses_db
from db import load_settings as _db_load_settings
from metaculus import normalize_probability, check_metaculus_gap

logger = logging.getLogger(__name__)

MIN_PROB_RATIO = 2.0
MIN_P_MODEL = 0.03
MIN_CONFIDENCE = 0.65
CALIBRATION_DAMPING_FACTOR = 0.65
CALIBRATION_DOTM_THRESHOLD = 0.10
CALIBRATION_AGGRESSIVE_PMODEL = 0.20
CALIBRATION_METACULUS_LOW = 0.10
CLUSTER_SCORE_ADJUSTMENTS = {
    "other": 15,
    "crypto": -25,
    "sports_nba": -15,
}


def get_settings():
    return _db_load_settings() or {}


def _count_resolved_hypotheses():
    try:
        return hypotheses_db.count_resolved()
    except Exception as e:
        logger.debug(f"[signal_scorer] {type(e).__name__}: {e}")
        return 0


def calibrate_prediction(p_model: float, market_price: float, metaculus_prob: float | None = None, cluster: str | None = None) -> tuple[float, bool]:
    if not isinstance(p_model, (int, float)) or p_model <= 0 or p_model >= 1:
        return p_model, False
    if not isinstance(market_price, (int, float)) or market_price < 0:
        return p_model, False

    from calibration import get_calibrator

    p_calibrated = p_model
    method = "raw"

    resolved_count = _count_resolved_hypotheses()
    if resolved_count >= 20:
        try:
            from calibration_tracker import get_platt_calibrated
            p_platt = get_platt_calibrated(p_model, cluster or "other")
            if p_platt is not None:
                p_calibrated = p_platt
                method = "platt"
        except Exception as e:
            logger.warning(f"[calibrate_platt] {type(e).__name__}: {e}")

    if method == "raw" and resolved_count >= 20:
        calibrator = get_calibrator()
        if calibrator.is_fitted:
            p_calibrated = calibrator.predict(p_model, cluster or "other")
            method = "isotonic"
            if p_calibrated >= 0.85 and (p_calibrated - p_model) > 0.30:
                p_calibrated = p_model
                method = "raw_overfit_guard"

    if method == "raw":
        if resolved_count < 50:
            p_calibrated = min(p_model * 1.05, 0.35)
            method = "soft_extremize"
        else:
            p_calibrated = p_model
            method = "raw"

    if market_price < 0.40:
        p_ext = min(p_calibrated, 0.85)
    else:
        p_ext = p_calibrated

    if method != "raw" or abs(p_ext - p_model) > 0.02:
        logger.info(
            f"[CALIBRATION] p_model={p_model:.1%} -> {p_calibrated:.1%} -> {p_ext:.1%} "
            f"(method={method}, cluster={cluster})"
        )

    return p_ext, method != "raw"


def _cluster_score_adjustment(cluster: str, settings: dict | None = None) -> float:
    if settings is None:
        settings = get_settings()
    adjustments = settings.get("cluster_score_adjustments", CLUSTER_SCORE_ADJUSTMENTS)
    adj = adjustments.get(cluster, 0)
    if adj != 0:
        logger.info(f"[CLUSTER-ADJ] {cluster}: {adj:+d} to signal_score")
    return adj


def _compute_signal_score(p_model: float, market_price: float, factors: list[dict], volume: float, ttl_hours: float, cluster: str, slug: str = "", question: str = "", metaculus_prob_val: float | None = None, settings: dict | None = None, condition_token_id: str = "") -> tuple[float, float, list[dict], list[dict], float, float, float, float, float, float, float, int, str]:
    if settings is None:
        settings = get_settings()

    prob_ratio = p_model / market_price if market_price > 0 else 0
    supporting = [f for f in factors if f.get("direction") == "supports"]
    high_weight = [f for f in supporting if f.get("weight") == "high"]

    ratio_score = min(prob_ratio / 3.0, 1.0) * 30

    metaculus_alignment = 0
    if metaculus_prob_val is not None:
        diff_model_meta = abs(p_model - metaculus_prob_val)
        diff_meta_pm = abs(metaculus_prob_val - market_price)
        if diff_model_meta < 0.05:
            metaculus_alignment = 10
        elif p_model > metaculus_prob_val + 0.10 and diff_meta_pm < 0.03:
            metaculus_alignment = -20

    factor_score = min((len(supporting) + len(high_weight)) / 4, 1.0) * 20
    vol_score = min(volume / 500_000, 1.0) * 20
    ttl_days = ttl_hours / 24 if ttl_hours else 999 / 24
    if ttl_days > 180:
        time_score = 20
    elif ttl_days > 90:
        time_score = 15
    elif ttl_days > 30:
        time_score = 12
    elif ttl_days > 14:
        time_score = 8
    elif ttl_days >= 2:
        time_score = 5
    else:
        time_score = 0

    signal_score = ratio_score + factor_score + vol_score + time_score + metaculus_alignment + _cluster_score_adjustment(cluster, settings)

    buzz_score = 0
    try:
        from social_buzz import compute_buzz_score
        buzz = compute_buzz_score(slug, question)
        buzz_score = buzz.get("buzz_score", 0)
    except Exception as e:
        logger.debug(f"[signal_scorer] {type(e).__name__}: {e}")
        pass

    orderbook_score = 0
    ob_reason = ""
    if condition_token_id:
        try:
            from orderbook_analyzer import analyze_orderbook_depth
            ob = analyze_orderbook_depth(condition_token_id, market_price)
            orderbook_score = ob.get("signal_score", 0)
            ob_reason = ob.get("reason", "")
            if orderbook_score > 0:
                logger.info(f"[ORDERBOOK] {slug[:30]}... score=+{orderbook_score} ({ob_reason})")
        except Exception as e:
            logger.debug(f"[signal_scorer] orderbook: {type(e).__name__}: {e}")

    sm_score = 0
    if condition_token_id:
        try:
            from smart_money import check_smart_money_activity
            sm = check_smart_money_activity(condition_token_id)
            sm_score = sm.get("signal_score", 0)
            if sm_score > 0:
                logger.info(f"[SMART_MONEY] {slug[:30]}... score=+{sm_score}")
        except Exception as e:
            logger.debug(f"[signal_scorer] smart_money: {type(e).__name__}: {e}")

    return signal_score + buzz_score + orderbook_score + sm_score, prob_ratio, supporting, high_weight, metaculus_alignment, buzz_score, ratio_score, factor_score, vol_score, time_score, ttl_days, orderbook_score, ob_reason


def full_market_analysis(market: dict) -> dict:
    from signal_pipeline import _check_llm_circuit_breaker, URL, HEADERS, MODEL_MAIN, get_settings as _sp_get_settings

    from order_manager import get_best_ask

    cluster = market.get(HYP_CLUSTERS, ["other"])[0]
    is_geopol = cluster in ["venezuela", "russia_ukraine", "usa_politics"]

    best_ask = None
    polymarket_prob = market["price"]

    if market["price"] < 0.35:
        best_ask = get_best_ask(market[HYP_SLUG])
        polymarket_prob = best_ask if best_ask is not None else market["price"]

    if best_ask is not None and market["price"] < 0.10:
        ask_ratio = best_ask / market["price"] if market["price"] > 0 else 0
        if ask_ratio > 10:
            logger.info(f"[LIQUIDITY-SKIP] {market.get(HYP_SLUG, '')[:40]}... ask={best_ask:.4f} is {ask_ratio:.1f}x price={market['price']:.4f}, no real liquidity")
            return {
                "question": market["question"],
                HYP_SLUG: market[HYP_SLUG],
                "market_price": market["price"],
                HYP_P_MODEL: 0,
                "prob_ratio": 0,
                HYP_CONFIDENCE: 0,
                "action": "SKIP",
                HYP_FACTORS: [],
                "source_signal": "default",
                "min_signal": 999,
                "reasoning": f"no_liquidity: ask={best_ask:.4f} is {ask_ratio:.1f}x price",
                "best_ask": best_ask,
            }

    metaculus_gap = None
    if market["price"] < 0.35:
        metaculus_gap = check_metaculus_gap(market, polymarket_prob)

    source_signal = "default"
    if metaculus_gap:
        source_signal = "metaculus"

    confidence = 0.60
    if source_signal == "metaculus":
        confidence = 0.80
    elif is_geopol:
        confidence = 0.70

    confidence = min(confidence, 0.95)

    gap_info = ""
    if metaculus_gap:
        gap_info = f"- Metaculus forecast: {metaculus_gap['metaculus_prob']:.0%} vs Polymarket {metaculus_gap['polymarket_prob']:.0%}\n"

    prompt = f"""Prediction market analyst. Your job is to find DOTM (deep out-the-money) events where the crowd SIGNIFICANTLY underestimates probability.

Market: {sanitize_for_prompt(market['question'])}
Price: ${market['price']:.3f} ({market['price']*100:.1f}%) | Volume: ${market.get('volume', 0):,.0f} | Resolution: {market.get('ttl_hours', 999) / 24:.0f}d | Category: {cluster}
Best Ask: ${polymarket_prob:.3f}
{gap_info}

CRITICAL DOTM METHODOLOGY: This market is priced at {market['price']*100:.1f} cents, meaning the crowd assigns only {market['price']*100:.1f}% chance. For DOTM markets, the crowd systematically underestimates tail risks because:
1. Recency bias - people overweight the status quo
2. Black swan blindness - people discount unprecedented but plausible events
3. Time premium - long-dated markets have more time for surprise developments

STEP 1: List 2-3 SPECIFIC plausible scenarios that could make this event happen. Think creatively about catalysts, tipping points, and nonlinear dynamics.
STEP 2: For each scenario, estimate its probability (even 1-5% each adds up).
STEP 3: Sum the scenario probabilities to get total TRUE probability.
STEP 4: If total probability > market price, the crowd is underestimating.

ANCHORING WARNING: Do NOT simply return a probability near the market price. The market price already reflects the crowd. You must independently assess the TRUE probability based on the underlying event scenarios.

Return ONLY JSON:
{{"factors": [{{"factor": "description", "direction": "supports/opposes", "weight": "high/medium/low", "source": "source"}}], "estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "brief"}}

Rules:
- estimated_probability: decimal 0.0-1.0 (NOT percentage)
- For markets with >180d to resolution, uncertainty favors YES
- Conservative on low-volume (<$10K) markets
- If estimate >3x price, explain what crowd is missing
- IMPORTANT: For DOTM markets (price < 5%), even small probability increases are significant"""

    resp = None
    if not _check_llm_circuit_breaker():
        p_model_llm = market["price"] * 2
        factors: list[dict[str, Any]] = []
        resp = None
    else:
        for _attempt in range(3):
            try:
                resp = requests.post(URL, headers=HEADERS, json={
                    "model": MODEL_MAIN,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                }, timeout=60)
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as _e:
                if _attempt < 2:
                    _backoff = 2 ** (_attempt + 1)
                    logger.warning(f"[ANALYSIS] LLM retry {_attempt+1}/3 in {_backoff}s: {_e}")
                    time.sleep(_backoff)
                else:
                    logger.error(f"[ANALYSIS] LLM error after 3 retries: {_e}")
                    p_model_llm = market["price"] * 2
                    factors = []
                    resp = None
    try:
        if resp is None:
            raise Exception("LLM unavailable after retries")
        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if not content:
            content = msg.get("reasoning") or ""

        result = parse_llm_json(content)
        if result:
            p_model_llm = normalize_probability(result.get("estimated_probability", market["price"] * 2))
            confidence = min(max(float(result.get(HYP_CONFIDENCE, confidence)), 0.1), 0.95)
            factors = result.get(HYP_FACTORS, [])
        else:
            p_model_llm = market["price"] * 2
            factors = []
    except Exception as e:
        logger.error(f"[ANALYSIS] LLM error: {e}")
        p_model_llm = market["price"] * 2
        factors = []

    if metaculus_gap and metaculus_gap.get("signal_strength", 0) > 0.3:
        p_model_metaculus = metaculus_gap["metaculus_prob"]
        if p_model_metaculus > p_model_llm:
            p_model = p_model_metaculus
            logger.info(
                f"[METACULUS-OVERRIDE] LLM={p_model_llm:.1%} < Metaculus={p_model_metaculus:.1%}, "
                f"using Metaculus (gap={metaculus_gap['gap']:.1%}, signal={metaculus_gap['signal_strength']:.2f})"
            )
        else:
            p_model = 0.6 * p_model_metaculus + 0.4 * p_model_llm
            logger.info(
                f"[METACULUS-BLEND] LLM={p_model_llm:.1%} > Metaculus={p_model_metaculus:.1%}, "
                f"blended to {p_model:.1%} (60/40 meta/llm)"
            )
        source_signal = "metaculus_override"
        confidence = min(confidence + 0.10, 0.95)
    else:
        p_model = p_model_llm

    max_p_model = market["price"] * MAX_P_MODEL_RATIO
    if p_model > max_p_model:
        logger.info(f"[ANALYSIS] p_model={p_model:.1%} > max={max_p_model:.1%} (price={market['price']:.3f}), capping")
        p_model = max_p_model

    metaculus_prob_val = metaculus_gap.get("metaculus_prob") if metaculus_gap else None
    p_model, _was_dampened = calibrate_prediction(p_model, market["price"], metaculus_prob_val, cluster=cluster)

    settings = _sp_get_settings()

    min_p_model = settings.get("min_p_model", MIN_P_MODEL)
    if p_model < min_p_model - 0.001:
        logger.info(f"[ANALYSIS] p_model={p_model:.1%} < MIN_P_MODEL={min_p_model:.1%}, skipping")
        return {
            "question": market["question"],
            HYP_SLUG: market[HYP_SLUG],
            "market_price": market["price"],
            HYP_P_MODEL: p_model,
            "prob_ratio": 0,
            HYP_CONFIDENCE: confidence,
            "action": "SKIP",
            HYP_FACTORS: [],
            "source_signal": "default",
            "reasoning": f"p_model too low ({p_model:.1%})",
            "best_ask": best_ask
        }

    signal_score, prob_ratio, supporting, high_weight, metaculus_alignment, buzz_score, ratio_score, factor_score, vol_score, time_score, ttl_days, orderbook_score, _ob_reason = _compute_signal_score(
        p_model, market["price"], factors, market.get("volume", 0), market.get("ttl_hours", 999),
        cluster, slug=market.get(HYP_SLUG, ""), question=market.get("question", ""),
        metaculus_prob_val=metaculus_prob_val, settings=settings,
        condition_token_id=market.get("condition_token_id", "")
    )

    base_threshold = _sp_get_settings().get("signal_threshold", 55)
    if ttl_days > 90:
        min_signal = _sp_get_settings().get("signal_threshold_long_horizon", base_threshold + 10)
    elif ttl_days >= 31:
        min_signal = _sp_get_settings().get("signal_threshold_medium_horizon", base_threshold + 5)
    else:
        min_signal = base_threshold
    if source_signal == "metaculus_override":
        min_signal = max(min_signal - 10, 35)

    action = "BUY" if signal_score >= min_signal and confidence >= _sp_get_settings().get("min_confidence", MIN_CONFIDENCE) and prob_ratio >= MIN_PROB_RATIO else "SKIP"

    if action == "BUY":
        with contextlib.suppress(Exception):
            from db import record_trade as _record_simulated
            _record_simulated(mode="simulated", slug=market[HYP_SLUG], action="buy",
                              price=market["price"],
                              size_usd=0,
                              p_model=p_model,
                              confidence=confidence,
                              signal_score=signal_score,
                              prob_ratio=prob_ratio,
                              reason=f"score={signal_score:.0f}/{min_signal}",
                              cluster=cluster,
                              source=source_signal,
                              metadata={"factors": [f.get("factor", "") for f in supporting],
                                        "source_signal": source_signal})

    if supporting:
        print(f"   📊 Factors: {len(supporting)} supporting ({len(high_weight)} high)")

    logger.info(
        f"[SIGNAL] ratio={prob_ratio:.2f}x -> {ratio_score:.0f}, factors={len(supporting)} -> {factor_score:.0f}, "
        f"vol=${market.get('volume',0):,.0f} -> {vol_score:.0f}, ttl={ttl_days:.0f}d -> {time_score:.0f}, "
        f"buzz={buzz_score:.1f}, ob={orderbook_score} "
        f"= {signal_score:.0f}/{min_signal} => {action}"
    )

    return {
        "question": market["question"],
        HYP_SLUG: market[HYP_SLUG],
        "market_price": market["price"],
        HYP_P_MODEL: p_model,
        "prob_ratio": prob_ratio,
        HYP_CONFIDENCE: confidence,
        "action": action,
        HYP_FACTORS: factors,
        "source_signal": source_signal,
        "signal_score": signal_score,
        "min_signal": min_signal,
        "reasoning": f"score={signal_score:.0f}/{min_signal}(horizon), ratio={prob_ratio:.2f}x, conf={confidence:.2f}, src={source_signal}, meta_align={metaculus_alignment:+d}",
        "best_ask": best_ask
    }
