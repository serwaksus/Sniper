"""
model_council.py — Multi-model AI council with judge for prediction market analysis.

Architecture:
  Round 1 — Council (9 advisors, equal weight):
    - DeepSeek + 8 OVH models provide independent probability estimates
  Round 2 — Judge (1 model, strongest reasoning):
    - Qwen3.5-397B-A17B (397B MoE thinking model) receives all estimates
    - Synthesizes council data and makes FINAL decision

OVH rate limit: 2 req/min → 31s between calls (global, shared).
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

# ── Council advisors (8 OVH models — NOT including the judge) ─
# These provide independent estimates in Round 1
OVH_ADVISORS: list[str] = [
    "gpt-oss-120b",                               # 120B OpenAI architecture
    "Meta-Llama-3_3-70B-Instruct",               # 70B Meta (note: underscore in 3_3)
    "Mistral-Small-3.2-24B-Instruct-2506",       # 24B, fast, good JSON
    "Qwen3-32B",                                  # 32B
    "Qwen3.6-27B",                               # 27B
    "Qwen3-Coder-30B-A3B-Instruct",              # 30B MoE coder
    "Mistral-Nemo-Instruct-2407",                # 12B
    "gpt-oss-20b",                               # 20B
]

# ── Judge model — makes FINAL decision based on council data ─
# Qwen3.5-397B-A17B: 397B MoE thinking model, strongest reasoning available
# Stable, chain-of-thought reduces hallucination, largest model on platform
JUDGE_MODEL = "Qwen3.5-397B-A17B"

# Rate limit: OVH free tier = 2 req/min → 31s between calls
OVH_MIN_INTERVAL = 31.0
OVH_TIMEOUT = 120  # Longer for thinking model

# Disagreement threshold for logging (std dev of advisor estimates)
DISAGREEMENT_THRESHOLD = 0.15

# ── Rate limiter (global, process-wide) ──────────────────────
_OVH_LOCK = threading.RLock()
_OVH_LAST_CALL: float = 0.0
_OVH_ENABLED: bool | None = None


def is_ovh_enabled() -> bool:
    """Check if OVH API is configured and should be used."""
    global _OVH_ENABLED
    if _OVH_ENABLED is None:
        _OVH_ENABLED = bool(OVH_API_KEY) and not os.environ.get("COUNCIL_DISABLED")
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
            content = msg.get("content") or ""
            if not content:
                reasoning = msg.get("reasoning") or ""
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
    for pattern in [r'(\[[\s\S]*\])', r'(\{[\s\S]*\})']:
        matches = re.findall(pattern, reasoning)
        if matches:
            candidate = matches[-1].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return ""


def _parse_json_array(content: str) -> list[dict[str, Any]] | None:
    """Parse raw text into JSON array of dicts."""
    if not content or not content.strip():
        return None

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    while cleaned.startswith(":"):
        cleaned = cleaned[1:].strip()

    start = cleaned.find("[")
    if start == -1:
        # No array — try individual objects
        individual = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned)
        if individual:
            try:
                    arr = [json.loads(obj) for obj in individual]
                    return [a for a in arr if isinstance(a, dict)]
            except json.JSONDecodeError:
                    pass
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
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    arr = json.loads(candidate)
                    if isinstance(arr, list):
                        return [a for a in arr if isinstance(a, dict)]
                except json.JSONDecodeError:
                    pass
                break

    fallback = re.search(r"\[[\s\S]*\]", cleaned)
    if fallback:
        try:
            arr = json.loads(fallback.group(0))
            if isinstance(arr, list):
                return [a for a in arr if isinstance(a, dict)]
        except json.JSONDecodeError:
            pass

    individual = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned)
    if individual:
        try:
            arr = [json.loads(obj) for obj in individual]
            return [a for a in arr if isinstance(a, dict)]
        except json.JSONDecodeError:
            pass
    return None


def _parse_json_object(content: str) -> dict[str, Any] | None:
    """Parse raw text into single JSON object."""
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
                try:
                    obj = json.loads(cleaned[start:i + 1])
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


def _build_judge_prompt_batch(
    estimates_by_slug: dict[str, list[dict[str, Any]]],
    batch_items: list[dict[str, Any]],
) -> str:
    """Build prompt for the judge model with all council estimates."""
    lines = [
        "You are the lead analyst of a prediction market council.",
        "Multiple AI models have independently estimated probabilities for these markets.",
        "Your job is to synthesize their estimates into the BEST final probability.",
        "",
    ]

    market_sections = []
    for item in batch_items:
        slug = item.get("slug", "")
        question = item.get("question", "")
        price = item.get("market_price", 0)

        estimates = estimates_by_slug.get(slug, [])
        if not estimates:
            continue

        ps = [e["p"] for e in estimates]
        mean_p = sum(ps) / len(ps)
        sorted_ps = sorted(ps)
        median_p = sorted_ps[len(sorted_ps) // 2]
        variance = sum((p - mean_p) ** 2 for p in ps) / len(ps) if ps else 0
        std_p = variance ** 0.5

        section = f"### Market: {question}\n"
        section += f"Slug: {slug}\n"
        section += f"Market price: {price:.4f} ({price*100:.1f}%)\n\n"
        section += "Council estimates:\n"
        for i, est in enumerate(estimates, 1):
            section += f"  {i}. {est['model']}: {est['p']*100:.1f}% (conf: {est['confidence']*100:.0f}%)\n"
        section += f"\nStatistics: mean={mean_p*100:.1f}%, median={median_p*100:.1f}%, std={std_p*100:.1f}%\n"
        section += f"Council size: {len(estimates)} models\n"
        market_sections.append(section)

    lines.extend(market_sections)
    lines.extend([
        "",
        "Synthesize ALL council estimates into YOUR final probability for each market.",
        "Do NOT simply average — use your analytical judgment:",
        "- Weight models by confidence",
        "- Consider which estimates have stronger reasoning",
        "- High disagreement (std) means more uncertainty",
        "- The market price reflects crowd wisdom — consider if council disagrees with crowd",
        "",
        "Return ONLY a JSON ARRAY (one object per market):",
        '[',
        '  {"slug": "market-slug", "estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "brief"}',
        ']',
        "",
        f"Return exactly {len(batch_items)} items matching the input slugs.",
    ])

    return "\n".join(lines)


def _build_judge_prompt_single(
    slug: str,
    question: str,
    price: float,
    estimates: list[dict[str, Any]],
) -> str:
    """Build judge prompt for a single market."""
    ps = [e["p"] for e in estimates]
    mean_p = sum(ps) / len(ps)
    sorted_ps = sorted(ps)
    median_p = sorted_ps[len(sorted_ps) // 2]
    variance = sum((p - mean_p) ** 2 for p in ps) / len(ps) if ps else 0
    std_p = variance ** 0.5

    lines = [
        "You are the lead analyst of a prediction market council.",
        "Multiple AI models have independently estimated the probability for this market.",
        "Your job is to synthesize their estimates into the BEST final probability.",
        "",
        f"Market: {question}",
        f"Market price: {price:.4f} ({price*100:.1f}%)",
        "",
        "Council estimates:",
    ]
    for i, est in enumerate(estimates, 1):
        lines.append(f"  {i}. {est['model']}: {est['p']*100:.1f}% (conf: {est['confidence']*100:.0f}%)")

    lines.extend([
        "",
        f"Statistics: mean={mean_p*100:.1f}%, median={median_p*100:.1f}%, std={std_p*100:.1f}%",
        f"Council size: {len(estimates)} models",
        "",
        "Synthesize ALL council estimates into YOUR final probability.",
        "Do NOT simply average — use your analytical judgment.",
        "",
        "Return ONLY JSON:",
        '{"estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "brief"}',
    ])
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────

def council_batch_consensus(
    prompt: str,
    batch_slugs: list[str],
    deepseek_results: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Run council + judge on a batch of markets.

    Round 1: Collect estimates from DeepSeek + OVH advisors
    Round 2: Judge (Qwen3.5-397B) synthesizes all estimates → final decision

    Args:
        prompt: The batch prompt (sent to OVH advisors, same as DeepSeek).
        batch_slugs: List of market slugs in batch order.
        deepseek_results: Parsed DeepSeek results (list of dicts).

    Returns:
        Tuple of (merged_results, council_meta).
        merged_results has same format as deepseek_results but with
        judge's final estimated_probability.
    """
    meta: dict[str, Any] = {
        "advisors_queried": [],
        "advisors_ok": [],
        "advisors_failed": [],
        "judge_called": False,
        "judge_ok": False,
    }

    if not is_ovh_enabled():
        return deepseek_results, meta

    # ── Round 1: Collect all advisor estimates ───────────────
    # Structure: {slug: [{model, p, confidence}, ...]}
    all_estimates: dict[str, list[dict[str, Any]]] = {}

    # DeepSeek estimates
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
                })
                meta["advisors_ok"].append("deepseek-chat")

    # OVH advisor estimates
    for model in OVH_ADVISORS:
        meta["advisors_queried"].append(model)
        content = _call_ovh_model(model, prompt, max_tokens=2000)
        if content is None:
            meta["advisors_failed"].append(model)
            continue

        parsed = _parse_json_array(content)
        if not parsed:
            logger.warning(f"[COUNCIL] Failed to parse {model} response")
            meta["advisors_failed"].append(model)
            continue

        meta["advisors_ok"].append(model)
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
                })

    if not all_estimates:
        return deepseek_results, meta

    # Log disagreement
    for slug, estimates in all_estimates.items():
        if len(estimates) > 1:
            ps = [e["p"] for e in estimates]
            mean_p = sum(ps) / len(ps)
            std_p = (sum((p - mean_p) ** 2 for p in ps) / len(ps)) ** 0.5
            if std_p > DISAGREEMENT_THRESHOLD:
                est_str = ", ".join(f"{e['model']}:{e['p']:.2f}" for e in estimates)
                logger.info(
                    f"[COUNCIL] {slug[:30]}.. disagreement={std_p:.3f} [{est_str}]"
                )

    # ── Round 2: Judge makes final decision ──────────────────
    batch_items_for_judge = []
    for slug in batch_slugs:
        estimates = all_estimates.get(slug, [])
        if estimates:
            batch_items_for_judge.append({"slug": slug, "estimates": estimates})

    if not batch_items_for_judge:
        return deepseek_results, meta

    # Build batch_items for judge prompt (need question + price from deepseek_results)
    slug_to_ds = {}
    if deepseek_results:
        for item in deepseek_results:
            s = item.get("slug", "")
            if s:
                slug_to_ds[s] = item

    judge_batch_items = []
    for slug in batch_slugs:
        estimates = all_estimates.get(slug, [])
        if not estimates:
            continue
        ds = slug_to_ds.get(slug, {})
        judge_batch_items.append({
            "slug": slug,
            "question": ds.get("reasoning", slug),  # Best available context
            "market_price": _safe_float(ds.get("market_price", 0.05)),
            "estimates": estimates,
        })

    if not judge_batch_items:
        return deepseek_results, meta

    judge_prompt = _build_judge_prompt_batch(all_estimates, judge_batch_items)
    meta["judge_called"] = True
    judge_content = _call_ovh_model(JUDGE_MODEL, judge_prompt, max_tokens=2000)

    if judge_content is None:
        logger.warning(f"[COUNCIL] Judge {JUDGE_MODEL} failed — falling back to advisor average")
        # Fallback: simple confidence-weighted average
        return _fallback_average(deepseek_results, all_estimates, batch_slugs, meta)

    judge_parsed = _parse_json_array(judge_content)
    if not judge_parsed:
        logger.warning("[COUNCIL] Judge response unparseable — falling back to advisor average")
        return _fallback_average(deepseek_results, all_estimates, batch_slugs, meta)

    meta["judge_ok"] = True
    logger.info(f"[COUNCIL] Judge {JUDGE_MODEL}: OK ({len(judge_parsed)} items)")

    # ── Merge judge's decisions into results ─────────────────
    judge_map: dict[str, dict[str, Any]] = {}
    for item in judge_parsed:
        slug = item.get("slug", "")
        if slug:
            judge_map[slug] = item

    if not judge_map and deepseek_results:
        # Positional fallback
        for i, item in enumerate(judge_parsed):
            if i < len(batch_slugs):
                judge_map[batch_slugs[i]] = item

    if not deepseek_results:
        # Build from scratch using judge's estimates
        merged = []
        for slug in batch_slugs:
            j = judge_map.get(slug, {})
            estimates = all_estimates.get(slug, [])
            merged.append({
                "slug": slug,
                "estimated_probability": _safe_float(
                    j.get("estimated_probability", 0.05), 0.05
                ),
                "confidence": min(max(_safe_float(j.get("confidence", 0.6), 0.6), 0.1), 0.95),
                "reasoning": j.get("reasoning", f"Judge verdict ({len(estimates)} advisors)"),
                "factors": [],
                "_council_judge": JUDGE_MODEL,
                "_council_advisors": [e["model"] for e in estimates],
            })
        return merged, meta

    # Override DeepSeek estimates with judge's verdict
    merged = []
    for item in deepseek_results:
        slug = item.get("slug", "")
        j = judge_map.get(slug)
        estimates = all_estimates.get(slug, [])
        if j and len(estimates) > 1:
            old_p = _safe_float(item.get("estimated_probability"), -1)
            new_p = _safe_float(j.get("estimated_probability"), old_p)
            item["estimated_probability"] = round(new_p, 4)
            item["confidence"] = min(max(
                _safe_float(j.get("confidence"), item.get("confidence", 0.6)),
                0.1), 0.95)
            item["_council_judge"] = JUDGE_MODEL
            item["_council_advisors"] = [e["model"] for e in estimates]
            if old_p >= 0:
                logger.info(
                    f"[COUNCIL-JUDGE] {slug[:30]}.. "
                    f"{old_p:.3f} → {new_p:.3f} "
                    f"(judge={JUDGE_MODEL}, {len(estimates)} advisors)"
                )
        merged.append(item)

    return merged, meta


def _fallback_average(
    deepseek_results: list[dict[str, Any]] | None,
    all_estimates: dict[str, list[dict[str, Any]]],
    batch_slugs: list[str],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Fallback: confidence-weighted average when judge is unavailable."""
    for slug, estimates in all_estimates.items():
        if len(estimates) <= 1:
            continue
        ps = [e["p"] for e in estimates]
        confs = [e["confidence"] for e in estimates]
        total_w = sum(confs)
        avg_p = sum(p * c for p, c in zip(ps, confs, strict=True)) / total_w if total_w > 0 else sum(ps) / len(ps)

        if deepseek_results:
            for item in deepseek_results:
                if item.get("slug") == slug:
                    old_p = _safe_float(item.get("estimated_probability"), -1)
                    item["estimated_probability"] = round(avg_p, 4)
                    item["_council_advisors"] = [e["model"] for e in estimates]
                    if old_p >= 0:
                        logger.info(
                            f"[COUNCIL-FALLBACK] {slug[:30]}.. "
                            f"{old_p:.3f} → {avg_p:.3f} (avg, judge unavailable)"
                        )
                    break

    return deepseek_results, meta


def council_single_consensus(
    prompt: str,
    slug: str,
    deepseek_p: float | None,
    deepseek_confidence: float = 0.6,
    question: str = "",
    price: float = 0.0,
) -> tuple[float | None, dict[str, Any]]:
    """Run council + judge for a single market.

    Args:
        prompt: Single-market prompt (sent to OVH advisors).
        slug: Market slug.
        deepseek_p: DeepSeek's estimated probability (or None).
        deepseek_confidence: DeepSeek's confidence.
        question: Market question (for judge prompt).
        price: Market price (for judge prompt).

    Returns:
        Tuple of (final_p, meta) where final_p is the JUDGE's verdict.
    """
    meta: dict[str, Any] = {
        "advisors_queried": [],
        "advisors_ok": [],
        "advisors_failed": [],
        "judge_called": False,
        "judge_ok": False,
        "consensus_applied": False,
    }

    if not is_ovh_enabled():
        return deepseek_p, meta

    # ── Round 1: Collect advisor estimates ───────────────────
    estimates: list[dict[str, Any]] = []

    if deepseek_p is not None:
        estimates.append({
            "model": "deepseek-chat",
            "p": deepseek_p,
            "confidence": deepseek_confidence,
        })
        meta["advisors_ok"].append("deepseek-chat")

    for model in OVH_ADVISORS:
        meta["advisors_queried"].append(model)
        content = _call_ovh_model(model, prompt, max_tokens=2000)
        if content is None:
            meta["advisors_failed"].append(model)
            continue

        parsed = _parse_json_object(content)
        if not parsed:
            meta["advisors_failed"].append(model)
            continue

        p = _safe_float(
            parsed.get("estimated_probability")
            or parsed.get("p")
            or parsed.get("probability"),
            -1,
        )
        conf = _safe_float(parsed.get("confidence") or parsed.get("c"), 0.6)
        if p >= 0:
            estimates.append({"model": model, "p": p, "confidence": conf})
            meta["advisors_ok"].append(model)

    if len(estimates) <= 1:
        return (estimates[0]["p"] if estimates else None), meta

    # Log disagreement
    ps = [e["p"] for e in estimates]
    mean_p = sum(ps) / len(ps)
    std_p = (sum((p - mean_p) ** 2 for p in ps) / len(ps)) ** 0.5
    if std_p > DISAGREEMENT_THRESHOLD:
        est_str = ", ".join(f"{e['model']}:{e['p']:.2f}" for e in estimates)
        logger.info(f"[COUNCIL-SINGLE] {slug[:30]}.. disagreement={std_p:.3f} [{est_str}]")

    # ── Round 2: Judge ───────────────────────────────────────
    judge_prompt = _build_judge_prompt_single(slug, question or slug, price, estimates)
    meta["judge_called"] = True
    judge_content = _call_ovh_model(JUDGE_MODEL, judge_prompt, max_tokens=2000)

    if judge_content is None:
        # Fallback: confidence-weighted average
        total_w = sum(e["confidence"] for e in estimates)
        avg_p = sum(e["p"] * e["confidence"] for e in estimates) / total_w if total_w > 0 else sum(ps) / len(ps)
        meta["consensus_applied"] = True
        meta["estimates"] = {e["model"]: round(e["p"], 4) for e in estimates}
        return round(avg_p, 4), meta

    judge_parsed = _parse_json_object(judge_content)
    if not judge_parsed:
        total_w = sum(e["confidence"] for e in estimates)
        avg_p = sum(e["p"] * e["confidence"] for e in estimates) / total_w if total_w > 0 else sum(ps) / len(ps)
        meta["consensus_applied"] = True
        return round(avg_p, 4), meta

    meta["judge_ok"] = True
    meta["consensus_applied"] = True
    meta["estimates"] = {e["model"]: round(e["p"], 4) for e in estimates}

    final_p = _safe_float(
        judge_parsed.get("estimated_probability")
        or judge_parsed.get("final_probability")
        or judge_parsed.get("p"),
        -1,
    )
    if final_p < 0:
        total_w = sum(e["confidence"] for e in estimates)
        final_p = sum(e["p"] * e["confidence"] for e in estimates) / total_w if total_w > 0 else sum(ps) / len(ps)

    logger.info(
        f"[COUNCIL-JUDGE-SINGLE] {slug[:30]}.. final={final_p:.3f} "
        f"(judge={JUDGE_MODEL}, advisors={len(estimates)})"
    )

    return round(final_p, 4), meta
