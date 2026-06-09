#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import calibration_tracker as ct
from utils import save_json, load_json


class TestLogCalibrationEntry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_log = ct.CALIBRATION_LOG
        ct.CALIBRATION_LOG = os.path.join(self.tmpdir, "cal_log.json")

    def tearDown(self):
        ct.CALIBRATION_LOG = self.orig_log

    def test_adds_entry_to_empty_log(self):
        ct.log_calibration_entry("slug-1", "Will X happen?", 0.7, 0.65, 0.60, "YES", "politics")
        log = load_json(ct.CALIBRATION_LOG, {"entries": []})
        self.assertEqual(len(log["entries"]), 1)
        e = log["entries"][0]
        self.assertEqual(e["slug"], "slug-1")
        self.assertEqual(e["actual_bin"], 1.0)
        self.assertEqual(e["p_model"], 0.7)

    def test_no_outcome_gives_bin_zero(self):
        ct.log_calibration_entry("slug-2", "Q?", 0.4, 0.35, 0.45, "NO", "sports")
        log = load_json(ct.CALIBRATION_LOG, {"entries": []})
        self.assertEqual(log["entries"][0]["actual_bin"], 0.0)

    def test_duplicate_slug_not_added(self):
        ct.log_calibration_entry("dup-slug", "Q?", 0.5, 0.5, 0.5, "YES", "other")
        ct.log_calibration_entry("dup-slug", "Q?", 0.6, 0.6, 0.6, "NO", "other")
        log = load_json(ct.CALIBRATION_LOG, {"entries": []})
        self.assertEqual(len(log["entries"]), 1)

    def test_caps_at_5000_entries(self):
        entries = [{"slug": f"s{i}", "p_model": 0.5, "actual_outcome": "YES", "actual_bin": 1.0} for i in range(4999)]
        save_json(ct.CALIBRATION_LOG, {"entries": entries})
        ct.log_calibration_entry("new-one", "Q?", 0.5, 0.5, 0.5, "YES", "other")
        log = load_json(ct.CALIBRATION_LOG, {"entries": []})
        self.assertLessEqual(len(log["entries"]), 5000)


class TestDetectModelDrift(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_log = ct.CALIBRATION_LOG
        ct.CALIBRATION_LOG = os.path.join(self.tmpdir, "cal_log.json")

    def tearDown(self):
        ct.CALIBRATION_LOG = self.orig_log

    def _make_entries(self, n_recent, recent_brier_shift, n_older=15, older_brier_shift=0.0):
        now = datetime.now()
        entries = []
        for i in range(n_older):
            p = 0.5 + older_brier_shift
            entries.append({
                "timestamp": (now - timedelta(days=120 + i)).isoformat(),
                "p_model": p, "actual_bin": 1.0 if i % 2 == 0 else 0.0,
                "actual_outcome": "YES" if i % 2 == 0 else "NO",
            })
        for i in range(n_recent):
            p = 0.5 + recent_brier_shift
            entries.append({
                "timestamp": (now - timedelta(days=30 + i)).isoformat(),
                "p_model": p, "actual_bin": 0.0 if i % 2 == 0 else 1.0,
                "actual_outcome": "NO" if i % 2 == 0 else "YES",
            })
        return entries

    def test_no_drift_when_good(self):
        entries = self._make_entries(15, 0.0, 15, 0.0)
        save_json(ct.CALIBRATION_LOG, {"entries": entries})
        result = ct.detect_model_drift(window_days=90, min_trades=10)
        self.assertIsNone(result)

    def test_drift_detected_when_degraded(self):
        now = datetime.now()
        older = []
        for i in range(15):
            older.append({
                "timestamp": (now - timedelta(days=120 + i)).isoformat(),
                "p_model": 0.5, "actual_bin": 0.5, "actual_outcome": "YES",
            })
        recent = []
        for i in range(15):
            recent.append({
                "timestamp": (now - timedelta(days=10 + i)).isoformat(),
                "p_model": 0.1 if i % 2 == 0 else 0.9,
                "actual_bin": 1.0 if i % 2 == 0 else 0.0,
                "actual_outcome": "YES" if i % 2 == 0 else "NO",
            })
        save_json(ct.CALIBRATION_LOG, {"entries": older + recent})
        result = ct.detect_model_drift(window_days=90, min_trades=10)
        self.assertIsNotNone(result)
        self.assertIn("MODEL DRIFT", result)

    def test_returns_none_with_few_entries(self):
        save_json(ct.CALIBRATION_LOG, {"entries": []})
        self.assertIsNone(ct.detect_model_drift())


class TestSyncFromHypothesisDb(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_log = ct.CALIBRATION_LOG
        self.orig_db = ct.HYPOTHESIS_DB
        ct.CALIBRATION_LOG = os.path.join(self.tmpdir, "cal_log.json")
        ct.HYPOTHESIS_DB = os.path.join(self.tmpdir, "hypothesis.json")

    def tearDown(self):
        ct.CALIBRATION_LOG = self.orig_log
        ct.HYPOTHESIS_DB = self.orig_db

    @patch("calibration_tracker.hypotheses_db")
    def test_sync_adds_resolved(self, mock_hdb):
        mock_hdb.load_all.return_value = {
            "hypotheses": [{
                "slug": "sync-1", "resolved": True, "outcome": "YES",
                "p_model": 0.7, "market_price": 0.6, "clusters": ["politics"],
            }]
        }
        save_json(ct.CALIBRATION_LOG, {"entries": []})
        added = ct.sync_from_hypothesis_db()
        self.assertEqual(added, 1)

    @patch("calibration_tracker.hypotheses_db")
    def test_skips_unresolved(self, mock_hdb):
        mock_hdb.load_all.return_value = {
            "hypotheses": [{"slug": "u1", "resolved": False}]
        }
        save_json(ct.CALIBRATION_LOG, {"entries": []})
        self.assertEqual(ct.sync_from_hypothesis_db(), 0)

    @patch("calibration_tracker.hypotheses_db")
    def test_skips_existing_slug(self, mock_hdb):
        mock_hdb.load_all.return_value = {
            "hypotheses": [{
                "slug": "exist", "resolved": True, "outcome": "YES",
                "p_model": 0.5, "clusters": [],
            }]
        }
        save_json(ct.CALIBRATION_LOG, {"entries": [{"slug": "exist"}]})
        self.assertEqual(ct.sync_from_hypothesis_db(), 0)


class TestTrainPlattModels(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_log = ct.CALIBRATION_LOG
        self.orig_platt = ct.PLATT_MODEL_FILE
        self.orig_db = ct.HYPOTHESIS_DB
        ct.CALIBRATION_LOG = os.path.join(self.tmpdir, "cal_log.json")
        ct.PLATT_MODEL_FILE = os.path.join(self.tmpdir, "platt.json")
        ct.HYPOTHESIS_DB = os.path.join(self.tmpdir, "hypothesis.json")

    def tearDown(self):
        ct.CALIBRATION_LOG = self.orig_log
        ct.PLATT_MODEL_FILE = self.orig_platt
        ct.HYPOTHESIS_DB = self.orig_db

    @patch("calibration_tracker.hypotheses_db")
    def test_trains_and_saves(self, mock_hdb):
        mock_hdb.load_all.return_value = {"hypotheses": []}
        entries = []
        for i in range(20):
            entries.append({
                "slug": f"t{i}", "p_model": 0.1 + i * 0.04,
                "actual_outcome": "YES" if i > 10 else "NO",
                "actual_bin": 1.0 if i > 10 else 0.0,
                "cluster": "test",
            })
        save_json(ct.CALIBRATION_LOG, {"entries": entries})
        models = ct.train_platt_models()
        self.assertIn("__global__", models)
        saved = load_json(ct.PLATT_MODEL_FILE, {})
        self.assertIn("models", saved)

    @patch("calibration_tracker.hypotheses_db")
    def test_empty_log_no_crash(self, mock_hdb):
        mock_hdb.load_all.return_value = {"hypotheses": []}
        save_json(ct.CALIBRATION_LOG, {"entries": []})
        models = ct.train_platt_models()
        self.assertEqual(models, {})


class TestGetEdgeReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_log = ct.CALIBRATION_LOG
        ct.CALIBRATION_LOG = os.path.join(self.tmpdir, "cal_log.json")

    def tearDown(self):
        ct.CALIBRATION_LOG = self.orig_log

    def test_empty_data(self):
        save_json(ct.CALIBRATION_LOG, {"entries": []})
        report = ct.get_edge_report()
        self.assertIn("error", report)

    def test_with_data(self):
        entries = []
        for i in range(20):
            entries.append({
                "slug": f"r{i}", "p_model": 0.3, "actual_outcome": "YES" if i < 6 else "NO",
                "actual_bin": 1.0 if i < 6 else 0.0, "pnl_pct": 0.5 if i < 6 else -0.3,
                "timestamp": "2025-01-01T00:00:00",
            })
        save_json(ct.CALIBRATION_LOG, {"entries": entries})
        report = ct.get_edge_report()
        self.assertIn("brier_raw", report)
        self.assertIn("direction_accuracy", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
