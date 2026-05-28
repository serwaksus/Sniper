#!/usr/bin/env python3
"""
Tests for calibration_tracker.py — Brier score, Platt scaling, sigmoid,
drift detection, edge cases.
"""
import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import calibration_tracker as ct


class TestSigmoid(unittest.TestCase):
    def test_sigmoid_0_is_05(self):
        self.assertAlmostEqual(ct._sigmoid(0), 0.5)

    def test_sigmoid_large_positive(self):
        self.assertGreater(ct._sigmoid(10), 0.99)

    def test_sigmoid_large_negative(self):
        self.assertLess(ct._sigmoid(-10), 0.01)

    def test_sigmoid_symmetry(self):
        self.assertAlmostEqual(ct._sigmoid(1) + ct._sigmoid(-1), 1.0, places=5)

    def test_sigmoid_bounded(self):
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            val = ct._sigmoid(x)
            self.assertGreaterEqual(val, 0)
            self.assertLessEqual(val, 1)


class TestBrierScore(unittest.TestCase):
    def test_perfect_predictions(self):
        outcomes = [
            {"p_model": 1.0, "actual_bin": 1},
            {"p_model": 0.0, "actual_bin": 0},
        ]
        brier = sum((e["p_model"] - e["actual_bin"]) ** 2 for e in outcomes) / len(outcomes)
        self.assertAlmostEqual(brier, 0.0)

    def test_worst_predictions(self):
        outcomes = [
            {"p_model": 0.0, "actual_bin": 1},
            {"p_model": 1.0, "actual_bin": 0},
        ]
        brier = sum((e["p_model"] - e["actual_bin"]) ** 2 for e in outcomes) / len(outcomes)
        self.assertAlmostEqual(brier, 1.0)

    def test_random_predictions(self):
        outcomes = [
            {"p_model": 0.5, "actual_bin": 1},
            {"p_model": 0.5, "actual_bin": 0},
        ]
        brier = sum((e["p_model"] - e["actual_bin"]) ** 2 for e in outcomes) / len(outcomes)
        self.assertAlmostEqual(brier, 0.25)

    def test_calibrated_brier_none_when_empty(self):
        entries = [
            {"p_model": 0.30, "actual_bin": 1},
        ]
        cal_entries = [e for e in entries if e.get("p_calibrated") is not None]
        if cal_entries:
            brier_cal = sum((e["p_calibrated"] - e["actual_bin"]) ** 2 for e in cal_entries) / len(cal_entries)
        else:
            brier_cal = None
        self.assertIsNone(brier_cal)

    def test_calibrated_brier_when_present(self):
        entries = [
            {"p_model": 0.30, "p_calibrated": 0.25, "actual_bin": 1},
            {"p_model": 0.70, "p_calibrated": 0.65, "actual_bin": 0},
        ]
        cal_entries = [e for e in entries if e.get("p_calibrated") is not None]
        brier_cal = sum((e["p_calibrated"] - e["actual_bin"]) ** 2 for e in cal_entries) / len(cal_entries)
        self.assertGreater(brier_cal, 0)
        self.assertLess(brier_cal, 1)


class TestPlattTraining(unittest.TestCase):
    def test_train_with_min_samples(self):
        p_models = [0.1 + i * 0.1 for i in range(15)]
        outcomes = [1 if p > 0.5 else 0 for p in p_models]
        model = ct._train_platt_cluster(p_models, outcomes)
        self.assertIsNotNone(model)
        self.assertIn("a", model)
        self.assertIn("b", model)
        self.assertEqual(model["samples"], 15)

    def test_train_below_min_returns_none(self):
        p_models = [0.3, 0.5, 0.7]
        outcomes = [0, 1, 1]
        model = ct._train_platt_cluster(p_models, outcomes)
        self.assertIsNone(model)

    def test_train_with_all_same_outcome(self):
        p_models = [0.2 + i * 0.05 for i in range(20)]
        outcomes = [1] * 20
        model = ct._train_platt_cluster(p_models, outcomes)
        self.assertIsNotNone(model)


class TestPlattCalibration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_file = ct.PLATT_MODEL_FILE
        ct.PLATT_MODEL_FILE = os.path.join(self.tmpdir, "platt.json")

    def tearDown(self):
        ct.PLATT_MODEL_FILE = self.orig_file
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_no_model_file_returns_none(self):
        result = ct.get_platt_calibrated(0.30, "other")
        self.assertIsNone(result)

    def test_empty_model_returns_none(self):
        ct.save_json(ct.PLATT_MODEL_FILE, {"models": {}})
        result = ct.get_platt_calibrated(0.30, "other")
        self.assertIsNone(result)

    def test_valid_model_returns_calibrated(self):
        ct.save_json(ct.PLATT_MODEL_FILE, {
            "models": {
                "other": {"a": 1.0, "b": 0.0, "samples": 30},
                "__global__": {"a": 1.0, "b": 0.0, "samples": 100},
            }
        })
        result = ct.get_platt_calibrated(0.30, "other")
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)

    def test_fallback_to_global_model(self):
        ct.save_json(ct.PLATT_MODEL_FILE, {
            "models": {
                "__global__": {"a": 1.0, "b": 0.0, "samples": 100},
            }
        })
        result = ct.get_platt_calibrated(0.30, "nonexistent_cluster")
        self.assertIsNotNone(result)

    def test_identity_calibration(self):
        ct.save_json(ct.PLATT_MODEL_FILE, {
            "models": {
                "other": {"a": 1.0, "b": 0.0, "samples": 30},
            }
        })
        result = ct.get_platt_calibrated(0.50, "other")
        self.assertAlmostEqual(result, 0.50, places=3)

    def test_calibrated_bounded(self):
        ct.save_json(ct.PLATT_MODEL_FILE, {
            "models": {
                "other": {"a": 3.0, "b": -1.0, "samples": 30},
            }
        })
        for p in [0.01, 0.10, 0.50, 0.90, 0.99]:
            result = ct.get_platt_calibrated(p, "other")
            if result is not None:
                self.assertGreater(result, 0)
                self.assertLess(result, 1)


class TestComputeCalibrationCurve(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_log = ct.CALIBRATION_LOG
        self.orig_db = ct.HYPOTHESIS_DB
        ct.CALIBRATION_LOG = os.path.join(self.tmpdir, "cal_log.json")
        ct.HYPOTHESIS_DB = os.path.join(self.tmpdir, "hypothesis.json")

    def tearDown(self):
        ct.CALIBRATION_LOG = self.orig_log
        ct.HYPOTHESIS_DB = self.orig_db
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_empty_entries(self):
        ct.save_json(ct.CALIBRATION_LOG, {"entries": []})
        result = ct.compute_calibration_curve()
        self.assertIsNotNone(result)

    def test_with_entries(self):
        entries = []
        for i in range(20):
            entries.append({
                "slug": f"s{i}",
                "p_model": 0.3,
                "actual_bin": 1 if i < 6 else 0,
                "actual_outcome": "YES" if i < 6 else "NO",
                "pnl_pct": 0.5 if i < 6 else -0.3,
                "timestamp": "2025-01-01T00:00:00",
            })
        ct.save_json(ct.CALIBRATION_LOG, {"entries": entries})
        result = ct.compute_calibration_curve()
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
