#!/usr/bin/env python3
"""
Calibration Feedback Module for DOTM Sniper.
Tracks p_model vs actual outcomes, computes calibration metrics,
detects over/underestimation, and alerts on model drift.
"""
import json
import os
import sys
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json

CALIBRATION_LOG = "/root/dotm-sniper/calibration_log.json"
HYPOTHESIS_DB = "/root/dotm-sniper/hypothesis_db.json"

logger = logging.getLogger(__name__)


def log_calibration_entry(slug: str, question: str, p_model: float,
                          p_calibrated: float, market_price: float,
                          actual_outcome: str, cluster: str,
                          entry_price: float = 0, exit_price: float = 0,
                          pnl_pct: float = 0):
    log = load_json(CALIBRATION_LOG, {"entries": []})
    if not isinstance(log, dict):
        log = {"entries": []}

    actual_bin = 1.0 if actual_outcome == "YES" else 0.0

    log["entries"].append({
        "timestamp": datetime.now().isoformat(),
        "slug": slug,
        "question": question[:80],
        "p_model": round(p_model, 4),
        "p_calibrated": round(p_calibrated, 4),
        "market_price": round(market_price, 4),
        "actual_outcome": actual_outcome,
        "actual_bin": actual_bin,
        "cluster": cluster,
        "entry_price": round(entry_price, 4) if entry_price else 0,
        "exit_price": round(exit_price, 4) if exit_price else 0,
        "pnl_pct": round(pnl_pct, 4) if pnl_pct else 0,
    })

    if len(log["entries"]) > 5000:
        log["entries"] = log["entries"][-5000:]

    save_json(CALIBRATION_LOG, log)


def compute_calibration_curve(bins: int = 10) -> Dict:
    log = load_json(CALIBRATION_LOG, {"entries": []})
    if not isinstance(log, dict):
        log = {"entries": []}
    entries = log.get("entries", [])
    if not entries:
        return {"error": "no data", "entries": 0}

    actual_outcomes = [e for e in entries if e.get("actual_outcome") in ("YES", "NO")]
    if not actual_outcomes:
        return {"error": "no resolved outcomes", "entries": len(entries)}

    bucket_size = 1.0 / bins
    buckets = []
    for i in range(bins):
        low = i * bucket_size
        high = (i + 1) * bucket_size
        in_bucket = [e for e in actual_outcomes if low <= e["p_model"] < high]
        if in_bucket:
            avg_predicted = sum(e["p_model"] for e in in_bucket) / len(in_bucket)
            avg_actual = sum(e["actual_bin"] for e in in_bucket) / len(in_bucket)
            buckets.append({
                "range": f"{low:.0%}-{high:.0%}",
                "count": len(in_bucket),
                "avg_predicted": round(avg_predicted, 4),
                "avg_actual": round(avg_actual, 4),
                "bias": round(avg_actual - avg_predicted, 4),
                "abs_error": round(abs(avg_actual - avg_predicted), 4),
            })
        else:
            buckets.append({
                "range": f"{low:.0%}-{high:.0%}",
                "count": 0,
                "avg_predicted": 0,
                "avg_actual": 0,
                "bias": 0,
                "abs_error": 0,
            })

    brier = sum((e["p_model"] - e["actual_bin"]) ** 2 for e in actual_outcomes) / len(actual_outcomes)
    brier_cal = sum((e["p_calibrated"] - e["actual_bin"]) ** 2 for e in actual_outcomes if e.get("p_calibrated")) / max(1, len([e for e in actual_outcomes if e.get("p_calibrated")]))

    low_p = [e for e in actual_outcomes if e["p_model"] < 0.15]
    if low_p:
        low_p_predicted = sum(e["p_model"] for e in low_p) / len(low_p)
        low_p_actual = sum(e["actual_bin"] for e in low_p) / len(low_p)
        overestimation = round((low_p_predicted - low_p_actual) / max(low_p_predicted, 0.01), 2)
    else:
        low_p_predicted = 0
        low_p_actual = 0
        overestimation = 0

    by_cluster = defaultdict(list)
    for e in actual_outcomes:
        by_cluster[e.get("cluster", "other")].append(e)

    cluster_stats = {}
    for cluster, centries in by_cluster.items():
        wins = sum(1 for e in centries if e["pnl_pct"] > 0)
        losses = sum(1 for e in centries if e["pnl_pct"] < 0)
        avg_pnl = sum(e["pnl_pct"] for e in centries) / len(centries) if centries else 0
        cluster_brier = sum((e["p_model"] - e["actual_bin"]) ** 2 for e in centries) / len(centries)
        cluster_stats[cluster] = {
            "count": len(centries),
            "win_rate": round(wins / max(wins + losses, 1), 3),
            "avg_pnl": round(avg_pnl, 3),
            "brier": round(cluster_brier, 4),
        }

    total = len(actual_outcomes)
    correct_direction = 0
    for e in actual_outcomes:
        pred_yes = e["p_model"] > 0.5
        actual_yes = e["actual_outcome"] == "YES"
        if pred_yes == actual_yes:
            correct_direction += 1

    return {
        "entries": len(entries),
        "resolved": total,
        "brier_raw": round(brier, 4),
        "brier_calibrated": round(brier_cal, 4),
        "improvement": round(brier - brier_cal, 4),
        "overestimation_low_p": overestimation,
        "direction_accuracy": round(correct_direction / total, 3) if total else 0,
        "curve": buckets,
        "clusters": cluster_stats,
    }


def detect_model_drift(window_days: int = 90, min_trades: int = 10) -> Optional[str]:
    log = load_json(CALIBRATION_LOG, {"entries": []})
    if not isinstance(log, dict):
        return None
    entries = log.get("entries", [])
    if not entries:
        return None

    cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
    recent = [e for e in entries if e.get("timestamp", "") >= cutoff and e.get("actual_outcome") in ("YES", "NO")]

    if len(recent) < min_trades:
        return None

    recent_brier = sum((e["p_model"] - e["actual_bin"]) ** 2 for e in recent) / len(recent)
    older = [e for e in entries if e.get("timestamp", "") < cutoff and e.get("actual_outcome") in ("YES", "NO")]
    if len(older) < min_trades:
        return None

    older_brier = sum((e["p_model"] - e["actual_bin"]) ** 2 for e in older) / len(older)
    degradation = recent_brier - older_brier

    if degradation > 0.05:
        return (f"MODEL DRIFT DETECTED: Brier degraded from {older_brier:.3f} to {recent_brier:.3f} "
                f"(+{degradation:.3f}) over last {window_days}d ({len(recent)} trades)")
    return None


def sync_from_hypothesis_db():
    db = load_json(HYPOTHESIS_DB, {"hypotheses": [], "resolved": []})
    if not isinstance(db, dict):
        return 0

    log = load_json(CALIBRATION_LOG, {"entries": []})
    if not isinstance(log, dict):
        log = {"entries": []}
    existing_slugs = {e["slug"] for e in log["entries"]}

    added = 0
    for h in db.get("resolved", []):
        if h.get("slug") in existing_slugs:
            continue
        if h.get("outcome") not in ("YES", "NO"):
            continue
        if h.get("p_model") is None:
            continue

        log["entries"].append({
            "timestamp": h.get("resolved_at", datetime.now().isoformat()),
            "slug": h["slug"],
            "question": h.get("question", "")[:80],
            "p_model": h.get("p_model", 0),
            "p_calibrated": 0,
            "market_price": h.get("market_price", 0),
            "actual_outcome": h["outcome"],
            "actual_bin": 1.0 if h["outcome"] == "YES" else 0.0,
            "cluster": h.get("clusters", ["other"])[0] if h.get("clusters") else "other",
            "entry_price": h.get("market_price", 0),
            "exit_price": h.get("exit_price", 0),
            "pnl_pct": h.get("pnl_at_exit", 0),
        })
        added += 1

    if added:
        save_json(CALIBRATION_LOG, log)
        logger.info(f"[CAL-TRACK] Synced {added} entries from hypothesis_db")

    return added


def get_edge_report() -> Dict:
    stats = compute_calibration_curve()
    if "error" in stats:
        return stats

    drift = detect_model_drift()

    entries = stats.get("resolved", 0)
    low_p_buckets = [b for b in stats.get("curve", []) if b.get("count", 0) > 0 and float(b["range"].split("-")[0].rstrip("%")) < 20]

    return {
        "total_trades": entries,
        "brier_raw": stats.get("brier_raw"),
        "brier_calibrated": stats.get("brier_calibrated"),
        "direction_accuracy": stats.get("direction_accuracy"),
        "overestimation_low_p": stats.get("overestimation_low_p"),
        "drift_alert": drift,
        "clusters": stats.get("clusters", {}),
        "low_p_calibration": low_p_buckets,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="Sync from hypothesis_db")
    parser.add_argument("--report", action="store_true", help="Show calibration report")
    parser.add_argument("--drift", action="store_true", help="Check for model drift")
    args = parser.parse_args()

    if args.sync:
        added = sync_from_hypothesis_db()
        print(f"Synced {added} entries")

    if args.report:
        import pprint
        report = compute_calibration_curve()
        pprint.pprint(report)

    if args.drift:
        alert = detect_model_drift()
        if alert:
            print(f"ALERT: {alert}")
        else:
            print("No drift detected")


if __name__ == "__main__":
    main()
