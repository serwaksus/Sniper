"""
model_council.py — Multi-model AI council for prediction market analysis.

Replaces single-LLM estimation with a weighted ensemble of models:
  - DeepSeek (primary, existing)
  - OVH AI Endpoints models (Mistral, gpt-oss, Llama, Qwen)

OVH rate limit: 2 req/min → 31s between calls (global, shared across all OVH models).
Council aggregation: confidence-weighted average with disagreement detection.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── OVH AI Endpoints configuration ───────────────────────────
OVH_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
OVH_API_KEY = os.environ.get("OVH_API_KEY", "")

# Council models (diverse architectures, good JSON compliance)
OVH_MODELS: list[str] = [
    "Mistral-Small-3.2-24B-Instruct-2506",
    "gpt-oss-120b",
]

# Rate limit: OVH free tier = 2 req/min → 31s between calls
OVH_MIN_INTERVAL = 31.0

# How many OVH models to query per batch (limited by rate limit)
MAX_OVH_CALLS_PER_BATCH = 2

# Per-model call timeout
OVH_TIMEOUT = 90

# Consensus weights (DeepSeek is primary)
DEEPSEEK_WEIGHT = 1.0
OVH_MODEL_WEIGHT = 0.7

# Disagreement threshold (std dev of estimates)
DISAGREEMENT_THRESHOLD = 0.15

# ── Rate limiter (global, process-wide) ──────────────────────
_OVH_LOCK = threading.RLock()
_OVH_LAST_CALL: float = 0.0
_OVH_ENABLED: bool | None = None


def is_ovh_enabled() -> bool:
    """Check if OVH API is configured and should be used."""
    global _OVH_ENABLED
    if _OVH_ENABLED is None:
        _OVH_ENABLED = bool(OVH_API_KEY)
    return _OVH_ENABLED


def _ovh_rate_limit_wait() -> float:
    """Sleep if needed to maintain OVH rate limit. Returns wait time."""
    global _OVH_LAST_CALL
    with _OVH_LOCK:
        elapsed = time.time() - _OVH_LAST_CALL
        wait = max(0.0, OVH_MIN_INTERVAL - elapsed)
        if wait > 0:
            time.sleep(wait)
        _OVH_LAST_CALL = time.time()
        return wait


def _call_ovh_model(
    model: str, prompt: str, max_tokens: int = 2000
) -> str | None:
    """Call a single OVH AI Endpoints model. Returns raw text or None."""
    if not is_ovh_enabled():
        return None

    waited = _ovh_rate_limit_wait()
    if waited > 0:
        logger.debug(f"[COUNCIL-OVH] Rate limit wait: {waited:.1f}s before {model}")

    try:
        r = requests.post(
            f"{OVH_BASE_URL}/chat/completions",
            headers={"api-key": OVH_API_KEY, "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=OVH_TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            # OVH models use either "content" (standard) or "reasoning" (thinking models)
            content = msg.get("content") or ""
            if not content:
                reasoning = msg.get("reasoning") or ""
                # For thinking models, try to extract JSON from reasoning
                content = _extract_json_from_reasoning(reasoning)
            if not content:
                logger.warning(f"[COUNCIL-OVH] {model}: empty response")
                return None
            logger.info(f"[COUNCIL-OVH] {model}: OK ({len(content)} chars)")
            return content
        elif r.status_code == 429:
            retry_after = r.headers.get("retry-after", "?")
            logger.warning(
                f"[COUNCIL-OVH] {model}: rate limited (retry-after={retry_after}s)"
            )
            return None
        else:
            logger.warning(f"[COUNCIL-OVH] {model}: HTTP {r.status_code} {r.text[:120]}")
            return None
    except requests.exceptions.Timeout:
        logger.warning(f"[COUNCIL-OVH] {model}: timeout ({OVH_TIMEOUT}s)")
        return None
    except Exception as e:
        logger.warning(f"[COUNCIL-OVH] {model}: {type(e).__name__}: {e}")
        return None


def _extract_json_from_reasoning(reasoning: str) -> str:
    """Extract JSON content from thinking-model reasoning output."""
    # Look for the final JSON answer after reasoning
    # Thinking models often output reasoning then switch to answer
    # Try to find a JSON array or object in the reasoning text
    for pattern in [
        r'(\[[\s\S]*\])',          # JSON array
        r'(\{[\s\S]*\})',          # JSON object
    ]:
        matches = re.findall(pattern, reasoning)
        if matches:
            # Return the last match (most likely the final answer)
            candidate = matches[-1].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return ""


def _parse_ovh_batch(content: str) -> list[dict[str, Any]] | None:
    """Parse OVH model response into list of market estimates.

    Handles various JSON formats: array, individual objects, markdown fences.
    Returns list of dicts with keys: slug, estimated_probability, confidence, reasoning.
    """
    if not content or not content.strip():
        return None

    cleaned = content.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Remove leading colons
    while cleaned.startswith(":"):
        cleaned = cleaned[1:].strip()

    # Try parsing as JSON array
    start = cleaned.find("[")
    if start != -1:
        # Find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(cleaned)):
            c = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        arr = json.loads(candidate)
                        if isinstance(arr, list):
                            return [a for a in arr if isinstance(a, dict)]
                    except json.JSONDecodeError:
                        pass
                    break

    # Fallback: regex
    fallback = re.search(r"\[[\s\S]*\]", cleaned)
    if fallback:
        try:
            arr = json.loads(fallback.group(0))
            if isinstance(arr, list):
                return [a for a in arr if isinstance(a, dict)]
        except json.JSONDecodeError:
            pass

    # Fallback: individual objects
    individual = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned)
    if individual:
        try:
            arr = [json.loads(obj) for obj in individual]
            arr = [a for a in arr if isinstance(a, dict)]
            if arr:
                return arr
        except json.JSONDecodeError:
            pass

    return None


def _parse_single_ovh(content: str) -> dict[str, Any] | None:
    """Parse OVH single-market response into estimate dict."""
    if not content or not content.strip():
        return None

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    while cleaned.startswith(":"):
        cleaned = cleaned[1:].strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break

    return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Parse a value to float safely."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip().rstrip("%")) / (100.0 if "%" in val else 1.0)
        except (ValueError, TypeError):
            return default
    return default


def council_batch_consensus(
    prompt: str,
    batch_slugs: list[str],
    deepseek_results: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Run council on a batch of markets.

    Calls OVH models and merges their estimates with DeepSeek results.

    Args:
        prompt: The batch prompt (same as sent to DeepSeek).
        batch_slugs: List of market slugs in batch order.
        deepseek_results: Parsed DeepSeek results (list of dicts with slug,
            estimated_probability, etc.). Can be None if DeepSeek failed.

    Returns:
        Tuple of (merged_results, council_meta):
        - merged_results: Same format as deepseek_results but with
          consensus estimated_probability. None if all models failed.
        - council_meta: Dict with council stats (models_queried, models_ok, etc.)
    """
    meta: dict[str, Any] = {
        "models_queried": [],
        "models_ok": [],
        "models_failed": [],
        "consensus_applied": False,
    }

    if not is_ovh_enabled():
        return deepseek_results, meta

    # Collect estimates per slug from all models
    # Structure: {slug: [{"model": name, "p": float, "confidence": float}, ...]}
    all_estimates: dict[str, list[dict[str, float | str]]] = {}

    # Add DeepSeek estimates
    if deepseek_results:
        for item in deepseek_results:
            slug = item.get("slug", "")
            if not slug:
                continue
            p = _safe_float(item.get("estimated_probability"), -1)
            conf = _safe_float(item.get("confidence"), 0.6)
            if p >= 0:
                all_estimates.setdefault(slug, []).append({
                    "model": "deepseek-chat",
                    "p": p,
                    "confidence": conf,
                    "weight": DEEPSEEK_WEIGHT,
                })

    # Call OVH models
    models_to_call = OVH_MODELS[:MAX_OVH_CALLS_PER_BATCH]
    for model in models_to_call:
        meta["models_queried"].append(model)
        content = _call_ovh_model(model, prompt, max_tokens=2000)
        if content is None:
            meta["models_failed"].append(model)
            continue

        parsed = _parse_ovh_batch(content)
        if not parsed:
            logger.warning(f"[COUNCIL] Failed to parse {model} response")
            meta["models_failed"].append(model)
            continue

        meta["models_ok"].append(model)
        for item in parsed:
            slug = item.get("slug", "")
            if not slug and len(parsed) == len(batch_slugs):
                idx = parsed.index(item)
                if idx < len(batch_slugs):
                    slug = batch_slugs[idx]
            if not slug:
                continue
            p = _safe_float(item.get("estimated_probability"), -1)
            conf = _safe_float(item.get("confidence"), 0.6)
            if p >= 0:
                all_estimates.setdefault(slug, []).append({
                    "model": model,
                    "p": p,
                    "confidence": conf,
                    "weight": OVH_MODEL_WEIGHT,
                })

    if not all_estimates:
        return deepseek_results, meta

    # Compute consensus per slug
    consensus_map: dict[str, dict[str, Any]] = {}
    for slug, estimates in all_estimates.items():
        if len(estimates) <= 1:
            # Only one model — no consensus needed
            consensus_map[slug] = {
                "p": estimates[0]["p"],
                "disagreement": 0.0,
                "models": [e["model"] for e in estimates],
            }
            continue

        # Weighted average
        ps = [e["p"] for e in estimates]
        ws = [e["weight"] * e["confidence"] for e in estimates]
        total_w = sum(ws)
        if total_w > 0:
            consensus_p = sum(p * w for p, w in zip(ps, ws, strict=True)) / total_w
        else:
            consensus_p = sum(ps) / len(ps)

        # Disagreement (std dev)
        mean_p = sum(ps) / len(ps)
        variance = sum((p - mean_p) ** 2 for p in ps) / len(ps)
        std_p = variance ** 0.5

        consensus_map[slug] = {
            "p": consensus_p,
            "disagreement": std_p,
            "models": [e["model"] for e in estimates],
        }

        if std_p > DISAGREEMENT_THRESHOLD:
            est_str = ", ".join(f"{e['model']}:{e['p']:.2f}" for e in estimates)
            logger.info(
                f"[COUNCIL] {slug[:30]}.. disagreement={std_p:.3f} "
                f"estimates=[{est_str}]"
            )

    # Merge consensus into DeepSeek results
    if not deepseek_results:
        # DeepSeek failed — build results from OVH
        merged = []
        for slug in batch_slugs:
            c = consensus_map.get(slug)
            if c:
                merged.append({
                    "slug": slug,
                    "estimated_probability": c["p"],
                    "confidence": 0.6,
                    "reasoning": f"Council consensus ({', '.join(c['models'])})",
                    "factors": [],
                })
        if merged:
            meta["consensus_applied"] = True
            return merged, meta
        return None, meta

    # Override DeepSeek estimates with consensus
    merged = []
    for item in deepseek_results:
        slug = item.get("slug", "")
        c = consensus_map.get(slug)
        if c and len(c.get("models", [])) > 1:
            old_p = _safe_float(item.get("estimated_probability"), -1)
            item["estimated_probability"] = round(c["p"], 4)
            item["_council_disagreement"] = round(c["disagreement"], 4)
            item["_council_models"] = c["models"]
            if old_p >= 0:
                logger.info(
                    f"[COUNCIL] {slug[:30]}.. p: {old_p:.3f} → {c['p']:.3f} "
                    f"(models: {', '.join(c['models'])}, disagreement: {c['disagreement']:.3f})"
                )
        merged.append(item)

    meta["consensus_applied"] = any(
        "_council_models" in item for item in merged
    )
    return merged, meta


def council_single_consensus(
    prompt: str,
    slug: str,
    deepseek_p: float | None,
    deepseek_confidence: float = 0.6,
) -> tuple[float | None, dict[str, Any]]:
    """Run council for a single market analysis.

    Args:
        prompt: Single-market prompt.
        slug: Market slug.
        deepseek_p: DeepSeek's estimated probability (or None if failed).

    Returns:
        Tuple of (consensus_p, meta):
        - consensus_p: Consensus probability, or deepseek_p if no OVH models available.
        - meta: Dict with council stats.
    """
    meta: dict[str, Any] = {
        "models_queried": [],
        "models_ok": [],
        "models_failed": [],
        "consensus_applied": False,
    }

    if not is_ovh_enabled():
        return deepseek_p, meta

    estimates: list[dict[str, float | str]] = []

    if deepseek_p is not None:
        estimates.append({
            "model": "deepseek-chat",
            "p": deepseek_p,
            "confidence": deepseek_confidence,
            "weight": DEEPSEEK_WEIGHT,
        })

    models_to_call = OVH_MODELS[:MAX_OVH_CALLS_PER_BATCH]
    for model in models_to_call:
        meta["models_queried"].append(model)
        content = _call_ovh_model(model, prompt, max_tokens=500)
        if content is None:
            meta["models_failed"].append(model)
            continue

        parsed = _parse_single_ovh(content)
        if not parsed:
            meta["models_failed"].append(model)
            continue

        p = _safe_float(
            parsed.get("estimated_probability")
            or parsed.get("p")
            or parsed.get("probability"),
            -1,
        )
        conf = _safe_float(parsed.get("confidence") or parsed.get("c"), 0.6)
        if p >= 0:
            estimates.append({
                "model": model,
                "p": p,
                "confidence": conf,
                "weight": OVH_MODEL_WEIGHT,
            })
            meta["models_ok"].append(model)

    if len(estimates) <= 1:
        return (estimates[0]["p"] if estimates else None), meta

    # Weighted consensus
    ps = [e["p"] for e in estimates]
    ws = [e["weight"] * e["confidence"] for e in estimates]
    total_w = sum(ws)
    consensus_p = sum(p * w for p, w in zip(ps, ws, strict=True)) / total_w if total_w > 0 else sum(ps) / len(ps)

    mean_p = sum(ps) / len(ps)
    variance = sum((p - mean_p) ** 2 for p in ps) / len(ps)
    std_p = variance ** 0.5

    meta["consensus_applied"] = True
    meta["disagreement"] = round(std_p, 4)
    meta["estimates"] = {e["model"]: round(e["p"], 4) for e in estimates}

    if std_p > DISAGREEMENT_THRESHOLD:
        logger.info(
            f"[COUNCIL-SINGLE] {slug[:30]}.. disagreement={std_p:.3f} "
            f"estimates={meta['estimates']}"
        )

    return round(consensus_p, 4), meta
