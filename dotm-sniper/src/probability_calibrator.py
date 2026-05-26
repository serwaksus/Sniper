#!/usr/bin/env python3
"""
ProbabilityCalibrator — isotonic regression per-cluster.

Trains on historical backtest results (p_model vs actual resolution),
produces a monotonic calibration curve that maps raw p_model → calibrated
probability.  Persists model as JSON (no pickle/sklearn needed at inference).

Usage:
    python3 src/probability_calibrator.py train
    python3 src/probability_calibrator.py evaluate
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression

CALIBRATOR_MODEL_PATH = "/root/dotm-sniper/calibrator_model.json"
BACKTEST_BASELINE_PATH = "/root/dotm-sniper/backtest_stats_v533_baseline.json"
MIN_CLUSTER_SAMPLES = 30
MIN_GLOBAL_SAMPLES = 100


class ProbabilityCalibrator:
    def __init__(self):
        self.global_model = None
        self.cluster_models = {}
        self.metadata = {}

    def fit(self, results):
        all_pairs = []
        by_cluster = defaultdict(list)

        for r in results:
            if r.get("status") != "resolved":
                continue
            p_raw = r.get("p_model_raw", r.get("p_model", 0.5))
            outcome = 1.0 if r.get("resolution") == "YES" else 0.0
            cluster = (r.get("clusters") or ["other"])[0]

            all_pairs.append((p_raw, outcome))
            by_cluster[cluster].append((p_raw, outcome))

        if len(all_pairs) >= MIN_GLOBAL_SAMPLES:
            self.global_model = self._fit_isotonic(all_pairs)

        for cluster, pairs in by_cluster.items():
            if len(pairs) >= MIN_CLUSTER_SAMPLES:
                self.cluster_models[cluster] = self._fit_isotonic(pairs)

        self.metadata = {
            "n_total": len(all_pairs),
            "n_clusters": len(self.cluster_models),
            "clusters": list(self.cluster_models.keys()),
        }
        return self

    @staticmethod
    def _fit_isotonic(pairs):
        X = np.array([p[0] for p in pairs], dtype=np.float64)
        y = np.array([p[1] for p in pairs], dtype=np.float64)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(X, y)
        return {
            "X_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "n_samples": len(pairs),
        }

    def calibrate(self, p_model, cluster="other"):
        model = self.cluster_models.get(cluster) or self.global_model
        if model is None:
            return p_model
        X = model["X_thresholds"]
        y = model["y_thresholds"]
        return float(np.interp(p_model, X, y))

    def save(self, path=CALIBRATOR_MODEL_PATH):
        data = {
            "global_model": self.global_model,
            "cluster_models": self.cluster_models,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path=CALIBRATOR_MODEL_PATH):
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            data = json.load(f)
        self.global_model = data.get("global_model")
        self.cluster_models = data.get("cluster_models", {})
        self.metadata = data.get("metadata", {})
        return True

    def is_loaded(self):
        return self.global_model is not None or bool(self.cluster_models)


def load_calibrator(path=CALIBRATOR_MODEL_PATH):
    cal = ProbabilityCalibrator()
    if cal.load(path):
        return cal
    return None


def train_from_baseline(baseline_path=BACKTEST_BASELINE_PATH,
                        output_path=CALIBRATOR_MODEL_PATH):
    with open(baseline_path, "r") as f:
        stats = json.load(f)
    results = stats.get("results", [])
    resolved = [r for r in results if r.get("status") == "resolved"]

    print(f"Training calibrator on {len(resolved)} resolved markets...")

    cal = ProbabilityCalibrator()
    cal.fit(resolved)
    cal.save(output_path)

    print(f"  Global model:    {'YES' if cal.global_model else 'NO'}")
    print(f"  Cluster models:  {list(cal.cluster_models.keys())}")
    for c, m in cal.cluster_models.items():
        print(f"    {c}: {m['n_samples']} samples, {len(m['X_thresholds'])} thresholds")
    print(f"  Saved to: {output_path}")
    return cal


def evaluate_improvement(baseline_path=BACKTEST_BASELINE_PATH,
                         model_path=CALIBRATOR_MODEL_PATH):
    with open(baseline_path, "r") as f:
        stats = json.load(f)
    results = [r for r in stats.get("results", []) if r.get("status") == "resolved"]

    cal = load_calibrator(model_path)
    if cal is None:
        print("No calibrator model found. Run `train` first.")
        return

    brier_before = []
    brier_after = []
    brier_per_cluster = defaultdict(lambda: {"before": [], "after": []})

    wins_before = 0
    wins_after = 0
    losses_before = 0
    losses_after = 0
    total_traded = 0

    for r in results:
        p_raw = r.get("p_model_raw", r.get("p_model", 0.5))
        p_model = r.get("p_model", p_raw)
        actual = 1.0 if r["resolution"] == "YES" else 0.0
        cluster = (r.get("clusters") or ["other"])[0]

        b_before = (p_model - actual) ** 2
        p_cal = cal.calibrate(p_raw, cluster)
        b_after = (p_cal - actual) ** 2

        brier_before.append(b_before)
        brier_after.append(b_after)
        brier_per_cluster[cluster]["before"].append(b_before)
        brier_per_cluster[cluster]["after"].append(b_after)

        if r.get("action") == "BUY":
            total_traded += 1
            if r["resolution"] == "YES":
                wins_before += 1
            else:
                losses_before += 1

    avg_before = sum(brier_before) / len(brier_before)
    avg_after = sum(brier_after) / len(brier_after)

    print("=" * 60)
    print("  CALIBRATION EVALUATION")
    print("=" * 60)
    print(f"  Markets:           {len(results)}")
    print(f"  Brier (raw model): {avg_before:.4f}")
    print(f"  Brier (calibrated):{avg_after:.4f}")
    print(f"  Improvement:       {avg_before - avg_after:+.4f} ({(avg_before-avg_after)/avg_before*100:+.1f}%)")

    base_rate = sum(1 for r in results if r["resolution"] == "YES") / len(results)
    brier_baseline = base_rate * (1 - base_rate) ** 2 + (1 - base_rate) * base_rate ** 2
    print(f"  Brier (base rate): {brier_baseline:.4f}")
    print(f"  Calibrated vs base rate: {'BETTER' if avg_after < brier_baseline else 'WORSE'}")

    print(f"\n  Per-cluster Brier improvement:")
    for cluster in sorted(brier_per_cluster, key=lambda c: -len(brier_per_cluster[c]["before"])):
        bb = brier_per_cluster[cluster]["before"]
        ba = brier_per_cluster[cluster]["after"]
        avg_b = sum(bb) / len(bb)
        avg_a = sum(ba) / len(ba)
        print(f"    {cluster:20s} n={len(bb):4d}  {avg_b:.4f} → {avg_a:.4f}  ({avg_b-avg_a:+.4f})")

    print(f"\n  Calibration curve (global):")
    if cal.global_model:
        X = cal.global_model["X_thresholds"]
        y = cal.global_model["y_thresholds"]
        for i in range(0, len(X), max(1, len(X) // 12)):
            print(f"    p_raw={X[i]:.3f} → p_cal={y[i]:.3f}")
        if len(X) > 1:
            print(f"    p_raw={X[-1]:.3f} → p_cal={y[-1]:.3f}")

    return {
        "brier_before": avg_before,
        "brier_after": avg_after,
        "improvement": avg_before - avg_after,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/probability_calibrator.py [train|evaluate]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        train_from_baseline()
    elif cmd == "evaluate":
        evaluate_improvement()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
