#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from probability_calibrator import ProbabilityCalibrator, load_calibrator


def _make_results(n, cluster="test", base_p=0.5, yes_ratio=0.5):
    results = []
    for i in range(n):
        resolution = "YES" if i < int(n * yes_ratio) else "NO"
        results.append({
            "status": "resolved",
            "p_model": base_p + (i / n) * 0.4,
            "p_model_raw": base_p + (i / n) * 0.4,
            "resolution": resolution,
            "clusters": [cluster],
        })
    return results


class TestProbabilityCalibratorFit(unittest.TestCase):
    def test_fit_global_model_with_enough_data(self):
        cal = ProbabilityCalibrator()
        results = _make_results(120, yes_ratio=0.5)
        cal.fit(results)
        self.assertIsNotNone(cal.global_model)
        self.assertIn("X_thresholds", cal.global_model)
        self.assertIn("y_thresholds", cal.global_model)

    def test_fit_no_global_model_below_threshold(self):
        cal = ProbabilityCalibrator()
        results = _make_results(50, yes_ratio=0.5)
        cal.fit(results)
        self.assertIsNone(cal.global_model)

    def test_fit_per_cluster_model(self):
        cal = ProbabilityCalibrator()
        results = _make_results(35, cluster="sports", yes_ratio=0.4)
        cal.fit(results)
        self.assertIn("sports", cal.cluster_models)

    def test_fit_cluster_below_threshold_skipped(self):
        cal = ProbabilityCalibrator()
        results = _make_results(10, cluster="tiny", yes_ratio=0.5)
        cal.fit(results)
        self.assertNotIn("tiny", cal.cluster_models)

    def test_fit_metadata(self):
        cal = ProbabilityCalibrator()
        results = _make_results(120, cluster="c1", yes_ratio=0.5)
        cal.fit(results)
        self.assertEqual(cal.metadata["n_total"], 120)
        self.assertIn("c1", cal.metadata["clusters"])


class TestProbabilityCalibratorCalibrate(unittest.TestCase):
    def test_calibrate_with_model(self):
        cal = ProbabilityCalibrator()
        results = _make_results(120, yes_ratio=0.5)
        cal.fit(results)
        p_cal = cal.calibrate(0.5, "other")
        self.assertIsInstance(p_cal, float)
        self.assertGreaterEqual(p_cal, 0.0)
        self.assertLessEqual(p_cal, 1.0)

    def test_calibrate_no_model_returns_raw(self):
        cal = ProbabilityCalibrator()
        p_cal = cal.calibrate(0.7, "other")
        self.assertAlmostEqual(p_cal, 0.7)

    def test_calibrate_cluster_model_preferred(self):
        cal = ProbabilityCalibrator()
        results = _make_results(120, cluster="target", yes_ratio=0.3)
        cal.fit(results)
        p_global = cal.calibrate(0.5, "nonexistent")
        p_cluster = cal.calibrate(0.5, "target")
        self.assertIsInstance(p_cluster, float)
        self.assertIsInstance(p_global, float)


class TestSaveLoad(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "cal_model.json")
        cal = ProbabilityCalibrator()
        results = _make_results(120, yes_ratio=0.5)
        cal.fit(results)
        cal.save(path)

        cal2 = ProbabilityCalibrator()
        loaded = cal2.load(path)
        self.assertTrue(loaded)
        self.assertIsNotNone(cal2.global_model)
        self.assertEqual(cal2.metadata["n_total"], 120)

        p1 = cal.calibrate(0.5, "other")
        p2 = cal2.calibrate(0.5, "other")
        self.assertAlmostEqual(p1, p2, places=5)

    def test_load_nonexistent_returns_false(self):
        cal = ProbabilityCalibrator()
        self.assertFalse(cal.load("/tmp/nonexistent_cal_test.json"))

    def test_is_loaded_after_load(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "cal_model.json")
        cal = ProbabilityCalibrator()
        results = _make_results(120, yes_ratio=0.5)
        cal.fit(results)
        cal.save(path)

        cal2 = ProbabilityCalibrator()
        self.assertFalse(cal2.is_loaded())
        cal2.load(path)
        self.assertTrue(cal2.is_loaded())


class TestLoadCalibratorFunction(unittest.TestCase):
    def test_returns_none_for_missing(self):
        result = load_calibrator("/tmp/nonexistent_cal_func.json")
        self.assertIsNone(result)


class TestEdgeCases(unittest.TestCase):
    def test_fit_empty_results(self):
        cal = ProbabilityCalibrator()
        cal.fit([])
        self.assertIsNone(cal.global_model)
        self.assertEqual(cal.metadata.get("n_total", 0), 0)

    def test_fit_unresolved_skipped(self):
        cal = ProbabilityCalibrator()
        cal.fit([{"status": "open", "p_model": 0.5, "resolution": "YES", "clusters": ["x"]}] * 50)
        self.assertIsNone(cal.global_model)

    def test_calibrate_extreme_values(self):
        cal = ProbabilityCalibrator()
        results = _make_results(120, yes_ratio=0.5)
        cal.fit(results)
        for p in [0.0, 1.0]:
            p_cal = cal.calibrate(p, "other")
            self.assertGreaterEqual(p_cal, 0.0)
            self.assertLessEqual(p_cal, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
