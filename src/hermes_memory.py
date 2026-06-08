#!/usr/bin/env python3
"""
Hermes Self-Improvement Memory Module.

Tracks predictions, generates skills from outcomes, adapts likelihood ratios,
and provides learned context for LLM prompts.
"""
import os
import sys
import json
import math
import logging
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json

MEMORY_FILE = "/root/dotm-sniper/hermes_memory.json"
SKILLS_FILE = "/root/dotm-sniper/hermes_skills.json"

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY = {
    "predictions": {},
    "resolved": [],
    "calibration": {
        "by_verdict": {},
        "by_cluster": {},
        "by_category": {},
    },
    "adaptive_likelihood": {},
    "skill_generation_runs": 0,
    "last_skill_generation": None,
}

_MAX_PREDICTIONS = 500
_MAX_RESOLVED = 200


def _load_memory():
    data = load_json(MEMORY_FILE, _DEFAULT_MEMORY)
    if not isinstance(data, dict):
        data = dict(_DEFAULT_MEMORY)
    for k, v in _DEFAULT_MEMORY.items():
        data.setdefault(k, v)
    return data


def _save_memory(data):
    if len(data.get("resolved", [])) > _MAX_RESOLVED:
        data["resolved"] = data["resolved"][-_MAX_RESOLVED:]
    save_json(MEMORY_FILE, data)


def log_prediction(slug, question, p_bot, p_hermes, verdict, status,
                   reason="", cluster="", news_category=""):
    m = _load_memory()
    m["predictions"][slug] = {
        "question": question,
        "p_bot": round(p_bot, 4),
        "p_hermes": round(p_hermes, 4),
        "verdict": verdict,
        "status": status,
        "reason": reason[:200],
        "cluster": cluster,
        "news_category": news_category,
        "timestamp": datetime.now().isoformat(),
    }
    if len(m["predictions"]) > _MAX_PREDICTIONS:
        slugs = list(m["predictions"].keys())
        for s in slugs[:len(slugs) - _MAX_PREDICTIONS]:
            del m["predictions"][s]
    _save_memory(m)


def resolve_prediction(slug, actual_outcome):
    m = _load_memory()
    pred = m["predictions"].pop(slug, None)
    if not pred:
        return

    p_hermes = pred.get("p_hermes", 0.5)
    if actual_outcome == "yes":
        correct = p_hermes >= 0.5
    else:
        correct = p_hermes < 0.5
    abs_error = abs(p_hermes - (1.0 if actual_outcome == "yes" else 0.0))

    entry = {
        "slug": slug,
        "question": pred.get("question", ""),
        "p_bot": pred.get("p_bot", 0),
        "p_hermes": pred.get("p_hermes", 0),
        "verdict": pred.get("verdict", ""),
        "cluster": pred.get("cluster", ""),
        "news_category": pred.get("news_category", ""),
        "actual_outcome": actual_outcome,
        "correct": correct,
        "abs_error": round(abs_error, 4),
        "predicted_at": pred.get("timestamp", ""),
        "resolved_at": datetime.now().isoformat(),
    }
    m["resolved"].append(entry)

    _update_calibration(m, entry)
    _update_adaptive_likelihood(m, entry)

    _save_memory(m)
    logger.info(
        f"[HERMES-MEMORY] Resolved {slug[:40]}... "
        f"p_hermes={p_hermes:.1%} actual={actual_outcome} "
        f"{'CORRECT' if correct else 'WRONG'} (error={abs_error:.1%})"
    )


def _update_calibration(memory, entry):
    cal = memory["calibration"]

    for key, val in [
        ("by_verdict", entry.get("verdict", "unknown")),
        ("by_cluster", entry.get("cluster", "unknown")),
        ("by_category", entry.get("news_category", "unknown")),
    ]:
        bucket = cal.setdefault(key, {}).setdefault(val, {
            "count": 0, "correct": 0, "total_error": 0.0
        })
        bucket["count"] += 1
        bucket["total_error"] = round(bucket["total_error"] + entry["abs_error"], 4)
        if entry["correct"]:
            bucket["correct"] += 1


def _update_adaptive_likelihood(memory, entry):
    al = memory.setdefault("adaptive_likelihood", {})
    cat = entry.get("news_category", "")
    if not cat:
        return

    bucket = al.setdefault(cat, {
        "count": 0,
        "yes_count": 0,
        "avg_p_hermes_when_yes": 0.0,
        "avg_p_hermes_when_no": 0.0,
        "total_p_yes": 0.0,
        "total_p_no": 0.0,
    })
    bucket["count"] += 1
    if entry["actual_outcome"] == "yes":
        bucket["yes_count"] += 1
        bucket["total_p_yes"] += entry["p_hermes"]
    else:
        bucket["total_p_no"] += entry["p_hermes"]

    n_yes = bucket["yes_count"]
    n_no = bucket["count"] - n_yes
    if n_yes > 0:
        bucket["avg_p_hermes_when_yes"] = round(bucket["total_p_yes"] / n_yes, 4)
    if n_no > 0:
        bucket["avg_p_hermes_when_no"] = round(bucket["total_p_no"] / n_no, 4)


def get_adaptive_likelihoods(min_samples=5):
    m = _load_memory()
    al = m.get("adaptive_likelihood", {})
    adapted = {}
    for cat, bucket in al.items():
        count = bucket.get("count", 0)
        if count < min_samples:
            continue
        p_yes = max(0.01, min(0.99, bucket["yes_count"] / count))
        adapted[cat] = {"p_yes_given_news": p_yes, "samples": count}
    return adapted


def generate_skills():
    m = _load_memory()
    resolved = m.get("resolved", [])
    if len(resolved) < 10:
        logger.info("[HERMES-SKILLS] Too few resolved predictions for skill generation")
        return []

    skills = []

    cluster_stats = defaultdict(lambda: {"count": 0, "correct": 0, "total_error": 0.0})
    verdict_stats = defaultdict(lambda: {"count": 0, "correct": 0, "total_error": 0.0})
    category_stats = defaultdict(lambda: {"count": 0, "correct": 0, "total_error": 0.0})
    overconfident_yes = 0
    underconfident_yes = 0
    overconfident_no = 0
    underconfident_no = 0

    for r in resolved:
        c = r.get("cluster", "unknown")
        cluster_stats[c]["count"] += 1
        cluster_stats[c]["total_error"] += r.get("abs_error", 0)
        if r.get("correct"):
            cluster_stats[c]["correct"] += 1

        v = r.get("verdict", "unknown")
        verdict_stats[v]["count"] += 1
        if r.get("correct"):
            verdict_stats[v]["correct"] += 1

        cat = r.get("news_category", "unknown")
        category_stats[cat]["count"] += 1
        category_stats[cat]["total_error"] += r.get("abs_error", 0)

        p_h = r.get("p_hermes", 0.5)
        actual = r.get("actual_outcome", "")
        if actual == "yes" and p_h < 0.3:
            underconfident_yes += 1
        elif actual == "yes" and p_h > 0.8:
            overconfident_yes += 1
        elif actual == "no" and p_h > 0.7:
            overconfident_no += 1
        elif actual == "no" and p_h < 0.1:
            underconfident_no += 1

    for cluster, stats in cluster_stats.items():
        if stats["count"] < 3:
            continue
        accuracy = stats["correct"] / stats["count"]
        avg_error = stats["total_error"] / stats["count"]
        if accuracy < 0.4:
            skills.append({
                "type": "cluster_bias",
                "cluster": cluster,
                "rule": f"LOW ACCURACY cluster '{cluster}': {accuracy:.0%} correct ({stats['count']} cases). "
                        f"Be more cautious with probability estimates for this domain. "
                        f"Average error: {avg_error:.0%}. Consider wider uncertainty bands.",
                "accuracy": round(accuracy, 3),
                "samples": stats["count"],
            })
        elif accuracy > 0.75 and avg_error < 0.2:
            skills.append({
                "type": "cluster_strength",
                "cluster": cluster,
                "rule": f"HIGH ACCURACY cluster '{cluster}': {accuracy:.0%} correct ({stats['count']} cases). "
                        f"This domain is well-calibrated. Trust p_hermes estimates here.",
                "accuracy": round(accuracy, 3),
                "samples": stats["count"],
            })

    for verdict, stats in verdict_stats.items():
        if stats["count"] < 3:
            continue
        accuracy = stats["correct"] / stats["count"]
        if verdict in ("DIVERGENCE", "RED") and accuracy < 0.5:
            skills.append({
                "type": "verdict_false_alarm",
                "verdict": verdict,
                "rule": f"FALSE ALARM pattern: '{verdict}' verdict was wrong in "
                        f"{1 - accuracy:.0%} of cases ({stats['count']} total). "
                        f"Require stronger evidence before issuing {verdict}.",
                "accuracy": round(accuracy, 3),
                "samples": stats["count"],
            })

    total = len(resolved)
    if underconfident_yes / max(total, 1) > 0.15:
        skills.append({
            "type": "calibration_bias",
            "rule": f"UNDERCONFIDENT on YES outcomes: {underconfident_yes}/{total} times "
                    f"p_hermes was <30% but outcome was YES. "
                    f"Shift probability estimates upward when evidence is ambiguous.",
        })
    if overconfident_no / max(total, 1) > 0.15:
        skills.append({
            "type": "calibration_bias",
            "rule": f"OVERCONFIDENT on NO outcomes: {overconfident_no}/{total} times "
                    f"p_hermes was >70% but outcome was NO. "
                    f"Reduce probability estimates when contradicting evidence exists but is not definitive.",
        })

    for cat, stats in category_stats.items():
        if stats["count"] < 5:
            continue
        avg_error = stats["total_error"] / stats["count"]
        if avg_error > 0.4:
            skills.append({
                "type": "news_category_miscalibration",
                "category": cat,
                "rule": f"News category '{cat}' has high avg error ({avg_error:.0%}, {stats['count']} cases). "
                        f"This news type is unreliable for probability estimation. "
                        f"Weight it less in p_hermes calculations.",
            })

    skills_data = {
        "skills": skills,
        "generated_at": datetime.now().isoformat(),
        "resolved_count": len(resolved),
        "generation_run": m.get("skill_generation_runs", 0) + 1,
    }
    save_json(SKILLS_FILE, skills_data)

    m["skill_generation_runs"] = m.get("skill_generation_runs", 0) + 1
    m["last_skill_generation"] = datetime.now().isoformat()
    _save_memory(m)

    if skills:
        logger.info(f"[HERMES-SKILLS] Generated {len(skills)} skills from {len(resolved)} resolved predictions")
    else:
        logger.info(f"[HERMES-SKILLS] No actionable skills from {len(resolved)} predictions (well-calibrated)")

    return skills


def load_skills_for_prompt(max_skills=5):
    data = load_json(SKILLS_FILE, {"skills": []})
    skills = data.get("skills", [])
    if not skills:
        return ""
    recent = skills[-max_skills:]
    lines = ["\nLearned patterns from past predictions (apply these when relevant):"]
    for i, s in enumerate(recent, 1):
        rule = s.get("rule", "")
        samples = s.get("samples", "")
        suffix = f" [{samples} samples]" if samples else ""
        lines.append(f"{i}. {rule}{suffix}")
    return "\n".join(lines)


def get_calibration_summary():
    m = _load_memory()
    resolved = m.get("resolved", [])
    if not resolved:
        return "No resolved predictions yet"

    total = len(resolved)
    correct = sum(1 for r in resolved if r.get("correct"))
    avg_error = sum(r.get("abs_error", 0) for r in resolved) / total

    by_cluster = {}
    for r in resolved:
        c = r.get("cluster", "unknown")
        by_cluster.setdefault(c, {"count": 0, "correct": 0})
        by_cluster[c]["count"] += 1
        if r.get("correct"):
            by_cluster[c]["correct"] += 1

    lines = [
        f"Hermes accuracy: {correct}/{total} ({correct/total:.0%}), avg error={avg_error:.0%}",
        "By cluster:",
    ]
    for c, s in sorted(by_cluster.items(), key=lambda x: -x[1]["count"]):
        acc = s["correct"] / s["count"] if s["count"] > 0 else 0
        lines.append(f"  {c}: {s['correct']}/{s['count']} ({acc:.0%})")

    return "\n".join(lines)


def get_stats():
    m = _load_memory()
    resolved = m.get("resolved", [])
    active = len(m.get("predictions", {}))
    skills_data = load_json(SKILLS_FILE, {"skills": []})
    skills_count = len(skills_data.get("skills", []))
    return {
        "active_predictions": active,
        "resolved_predictions": len(resolved),
        "total_skills": skills_count,
        "last_skill_generation": m.get("last_skill_generation"),
    }
