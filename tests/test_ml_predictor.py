#!/usr/bin/env python3
"""Tests for ml_predictor.py — LightGBM-based p_model predictor."""
import json
import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import ml_predictor


class TestBuildFeatures(unittest.TestCase):
    def _sample_market(self, **overrides):
        base = {
            "p_model": 0.15,
            "confidence": 0.80,
            "metaculus_prob": 0.12,
            "metaculus_n": 25,
            "buzz_score": 5.0,
            "ob_imbalance": 0.3,
            "smart_money_detected": True,
            "end_date_iso": (datetime.now(UTC) + timedelta(days=60)).isoformat(),
            "price": 0.07,
        }
        base.update(overrides)
        return base

    def test_all_fields_present(self):
        features = ml_predictor.build_features(self._sample_market())
        self.assertEqual(features.shape, (10,))
        self.assertEqual(features.dtype, np.float32)
        self.assertAlmostEqual(features[0], 0.15)
        self.assertAlmostEqual(features[1], 0.80)
        self.assertAlmostEqual(features[2], 0.12)
        self.assertAlmostEqual(features[3], 25)
        self.assertAlmostEqual(features[4], 5.0)
        self.assertAlmostEqual(features[5], 0.3)
        self.assertAlmostEqual(features[6], 1.0)
        self.assertGreater(features[7], 0)
        self.assertAlmostEqual(features[8], 0.07)
        self.assertAlmostEqual(features[9], 0.15 / 0.07, places=3)

    def test_missing_fields_default(self):
        features = ml_predictor.build_features({})
        self.assertEqual(features.shape, (10,))
        self.assertAlmostEqual(features[0], 0.0)
        self.assertAlmostEqual(features[6], 0.0)
        self.assertAlmostEqual(features[7], 30.0)

    def test_none_fields_default(self):
        market = self._sample_market(metaculus_prob=None, metaculus_n=None, buzz_score=None)
        features = ml_predictor.build_features(market)
        self.assertAlmostEqual(features[2], 0.0)
        self.assertAlmostEqual(features[3], 0.0)
        self.assertAlmostEqual(features[4], 0.0)

    def test_smart_money_false(self):
        features = ml_predictor.build_features(self._sample_market(smart_money_detected=False))
        self.assertAlmostEqual(features[6], 0.0)

    def test_nan_handling(self):
        features = ml_predictor.build_features({"p_model": float("nan"), "price": 0.05})
        self.assertTrue(np.isnan(features[0]))

    def test_time_to_expiry_future(self):
        future = (datetime.now(UTC) + timedelta(days=90)).isoformat()
        features = ml_predictor.build_features(self._sample_market(end_date_iso=future))
        self.assertGreater(features[7], 89)
        self.assertLess(features[7], 91)

    def test_time_to_expiry_past(self):
        past = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        features = ml_predictor.build_features(self._sample_market(end_date_iso=past))
        self.assertAlmostEqual(features[7], 0.0)

    def test_time_to_expiry_default(self):
        features = ml_predictor.build_features({"p_model": 0.1, "price": 0.05})
        self.assertAlmostEqual(features[7], 30.0)

    def test_prob_ratio_with_zero_price(self):
        features = ml_predictor.build_features({"p_model": 0.10, "price": 0})
        expected = 0.10 / 0.001
        self.assertAlmostEqual(features[9], expected)

    def test_market_price_fallback(self):
        features = ml_predictor.build_features({"p_model": 0.1, "market_price": 0.08})
        self.assertAlmostEqual(features[8], 0.08)

    def test_feature_count_matches_names(self):
        self.assertEqual(len(ml_predictor.FEATURE_NAMES), 10)
        features = ml_predictor.build_features(self._sample_market())
        self.assertEqual(len(features), len(ml_predictor.FEATURE_NAMES))


class TestPredict(unittest.TestCase):
    def test_no_model_file_returns_false(self):
        with patch.object(ml_predictor, "MODEL_PATH", "/tmp/nonexistent_model_12345.txt"):
            pred, used = ml_predictor.predict({"p_model": 0.5, "price": 0.1})
            self.assertAlmostEqual(pred, 0.0)
            self.assertFalse(used)

    def test_with_mocked_model_file(self):
        import lightgbm as lgb

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "predictor.txt")
            X = np.random.rand(20, 10).astype(np.float32)
            y = np.random.randint(0, 2, 20).astype(np.float32)
            train_data = lgb.Dataset(X, label=y, feature_name=ml_predictor.FEATURE_NAMES)
            model = lgb.train(
                {"objective": "binary", "verbosity": -1, "n_jobs": 1, "seed": 42},
                train_data,
                num_boost_round=5,
            )
            model.save_model(model_path)

            with patch.object(ml_predictor, "MODEL_PATH", model_path):
                pred, used = ml_predictor.predict({"p_model": 0.15, "confidence": 0.8, "price": 0.05})
                self.assertTrue(used)
                self.assertGreaterEqual(pred, 0.0)
                self.assertLessEqual(pred, 1.0)

    def test_predict_output_clamped(self):
        import lightgbm as lgb

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "predictor.txt")
            X = np.random.rand(20, 10).astype(np.float32)
            y = np.random.randint(0, 2, 20).astype(np.float32)
            train_data = lgb.Dataset(X, label=y, feature_name=ml_predictor.FEATURE_NAMES)
            model = lgb.train(
                {"objective": "binary", "verbosity": -1, "n_jobs": 1, "seed": 42},
                train_data,
                num_boost_round=5,
            )
            model.save_model(model_path)

            with patch.object(ml_predictor, "MODEL_PATH", model_path):
                pred, used = ml_predictor.predict({"p_model": 0.5, "confidence": 0.5, "price": 0.3})
                self.assertTrue(used)
                self.assertGreaterEqual(pred, 0.0)
                self.assertLessEqual(pred, 1.0)


class TestTrainModel(unittest.TestCase):
    def _make_samples(self, n, seed=42):
        rng = np.random.RandomState(seed)
        samples = []
        for _ in range(n):
            samples.append({
                "p_model": rng.uniform(0.01, 0.40),
                "confidence": rng.uniform(0.5, 0.95),
                "metaculus_prob": rng.uniform(0.01, 0.30),
                "metaculus_n": float(rng.randint(1, 100)),
                "buzz_score": rng.uniform(0, 15),
                "ob_imbalance": rng.uniform(-1, 1),
                "smart_money_detected": bool(rng.randint(0, 2)),
                "end_date_iso": (datetime.now(UTC) + timedelta(days=rng.randint(10, 200))).isoformat(),
                "price": rng.uniform(0.01, 0.20),
                "target": int(rng.randint(0, 2)),
            })
        return samples

    def test_insufficient_samples_returns_none(self):
        samples = self._make_samples(10)
        result = ml_predictor.train_model(samples)
        self.assertIsNone(result)

    def test_train_with_enough_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "predictor.txt")
            with patch.object(ml_predictor, "MODEL_DIR", tmpdir), \
                 patch.object(ml_predictor, "MODEL_PATH", model_path):
                samples = self._make_samples(60)
                result = ml_predictor.train_model(samples)
                self.assertIsNotNone(result)
                self.assertEqual(result["n_samples"], 60)
                self.assertIn("brier_score", result)
                self.assertIn("accuracy", result)
                self.assertIn("feature_importance", result)
                self.assertIn("trained_at", result)
                self.assertTrue(os.path.exists(model_path))

    def test_model_file_saved_and_loadable(self):
        import lightgbm as lgb

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "predictor.txt")
            with patch.object(ml_predictor, "MODEL_DIR", tmpdir), \
                 patch.object(ml_predictor, "MODEL_PATH", model_path):
                samples = self._make_samples(60)
                ml_predictor.train_model(samples)
                loaded = lgb.Booster(model_file=model_path)
                self.assertIsNotNone(loaded)

    def test_feature_importance_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "predictor.txt")
            with patch.object(ml_predictor, "MODEL_DIR", tmpdir), \
                 patch.object(ml_predictor, "MODEL_PATH", model_path):
                samples = self._make_samples(60)
                result = ml_predictor.train_model(samples)
                importance = result["feature_importance"]
                self.assertEqual(set(importance.keys()), set(ml_predictor.FEATURE_NAMES))

    def test_metrics_json_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "predictor.txt")
            with patch.object(ml_predictor, "MODEL_DIR", tmpdir), \
                 patch.object(ml_predictor, "MODEL_PATH", model_path):
                samples = self._make_samples(60)
                ml_predictor.train_model(samples)
                metrics_path = os.path.join(tmpdir, "metrics.json")
                self.assertTrue(os.path.exists(metrics_path))
                with open(metrics_path) as f:
                    metrics = json.load(f)
                self.assertIn("brier_score", metrics)
                self.assertIn("accuracy", metrics)


class TestCollectTrainingSamples(unittest.TestCase):
    def test_with_mocked_db(self):
        mock_conn = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self_, key: {
            "slug": "test-market",
            "data": json.dumps({
                "resolved": True,
                "outcome": "YES",
                "p_model": 0.20,
                "confidence": 0.85,
                "market_price": 0.08,
            }),
        }[key]
        mock_conn.execute.return_value.fetchall.return_value = [mock_row]

        with patch("db._get_conn", return_value=mock_conn):
            samples = ml_predictor.collect_training_samples()
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["target"], 1)
            self.assertAlmostEqual(samples[0]["p_model"], 0.20)

    def test_no_outcome_skipped(self):
        mock_conn = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self_, key: {
            "slug": "test-market",
            "data": json.dumps({"resolved": True}),
        }[key]
        mock_conn.execute.return_value.fetchall.return_value = [mock_row]

        with patch("db._get_conn", return_value=mock_conn):
            samples = ml_predictor.collect_training_samples()
            self.assertEqual(len(samples), 0)

    def test_sold_positive_pnl_treated_as_yes(self):
        mock_conn = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self_, key: {
            "slug": "test-market",
            "data": json.dumps({
                "resolved": True,
                "outcome": "SOLD",
                "pnl_pct": 0.15,
                "p_model": 0.10,
            }),
        }[key]
        mock_conn.execute.return_value.fetchall.return_value = [mock_row]

        with patch("db._get_conn", return_value=mock_conn):
            samples = ml_predictor.collect_training_samples()
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["target"], 1)


class TestGetModelInfo(unittest.TestCase):
    def test_no_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ml_predictor, "MODEL_DIR", tmpdir), \
                 patch.object(ml_predictor, "MODEL_PATH", os.path.join(tmpdir, "predictor.txt")):
                mock_conn = MagicMock()
                mock_conn.execute.return_value.fetchall.return_value = []
                with patch("db._get_conn", return_value=mock_conn):
                    info = ml_predictor.get_model_info()
                    self.assertFalse(info["model_available"])
                    self.assertEqual(info["current_samples"], 0)
                    self.assertEqual(info["metrics"], {})

    def test_model_exists(self):
        import lightgbm as lgb

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "predictor.txt")
            X = np.random.rand(20, 10).astype(np.float32)
            y = np.random.randint(0, 2, 20).astype(np.float32)
            train_data = lgb.Dataset(X, label=y, feature_name=ml_predictor.FEATURE_NAMES)
            model = lgb.train(
                {"objective": "binary", "verbosity": -1, "n_jobs": 1, "seed": 42},
                train_data,
                num_boost_round=5,
            )
            model.save_model(model_path)

            metrics = {"brier_score": 0.2, "accuracy": 0.7}
            with open(os.path.join(tmpdir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

            with patch.object(ml_predictor, "MODEL_DIR", tmpdir), \
                 patch.object(ml_predictor, "MODEL_PATH", model_path):
                mock_conn = MagicMock()
                mock_conn.execute.return_value.fetchall.return_value = []
                with patch("db._get_conn", return_value=mock_conn):
                    info = ml_predictor.get_model_info()
                    self.assertTrue(info["model_available"])
                    self.assertEqual(info["metrics"]["brier_score"], 0.2)


class TestTrainIfReady(unittest.TestCase):
    def test_not_enough_samples_returns_none(self):
        with patch("ml_predictor.collect_training_samples", return_value=[]):
            result = ml_predictor.train_if_ready()
            self.assertIsNone(result)


class TestMinSamples(unittest.TestCase):
    def test_min_samples_is_50(self):
        self.assertEqual(ml_predictor.MIN_SAMPLES, 50)


if __name__ == "__main__":
    unittest.main()
