#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import correlation_matrix as cm
from utils import save_json


class TestGetCluster(unittest.TestCase):
    def test_returns_first_cluster(self):
        pos = {"clusters": ["politics", "fed"]}
        self.assertEqual(cm._get_cluster(pos), "politics")

    def test_empty_clusters_returns_other(self):
        self.assertEqual(cm._get_cluster({"clusters": []}), "other")

    def test_missing_clusters_returns_other(self):
        self.assertEqual(cm._get_cluster({}), "other")


class TestComputePairwiseCorrelation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_ph = cm.PRICE_HISTORY_FILE
        cm.PRICE_HISTORY_FILE = os.path.join(self.tmpdir, "prices.json")

    def tearDown(self):
        cm.PRICE_HISTORY_FILE = self.orig_ph

    def test_returns_none_insufficient_data(self):
        save_json(cm.PRICE_HISTORY_FILE, {"a": [{"p": 0.5}] * 2, "b": [{"p": 0.5}] * 2})
        result = cm.compute_pairwise_correlation("a", "b")
        self.assertIsNone(result)

    def test_returns_correlation_for_valid_series(self):
        prices_a = [{"p": 0.5 + i * 0.01} for i in range(10)]
        prices_b = [{"p": 0.5 + i * 0.01} for i in range(10)]
        save_json(cm.PRICE_HISTORY_FILE, {"a": prices_a, "b": prices_b})
        result = cm.compute_pairwise_correlation("a", "b")
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_missing_slug_returns_none(self):
        save_json(cm.PRICE_HISTORY_FILE, {"a": [{"p": 0.5}] * 10})
        result = cm.compute_pairwise_correlation("a", "missing")
        self.assertIsNone(result)

    def test_constant_series_returns_none(self):
        prices_a = [{"p": 0.5}] * 10
        prices_b = [{"p": 0.5}] * 10
        save_json(cm.PRICE_HISTORY_FILE, {"a": prices_a, "b": prices_b})
        result = cm.compute_pairwise_correlation("a", "b")
        self.assertIsNone(result)


class TestGetCorrelatedExposure(unittest.TestCase):
    def test_empty_positions(self):
        self.assertEqual(cm.get_correlated_exposure({}, 1000), {})

    def test_zero_balance(self):
        self.assertEqual(cm.get_correlated_exposure({"s": {"clusters": ["politics"], "entry_price": 1, "shares": 10}}, 0), {})

    def test_calculates_group_exposure(self):
        positions = {
            "s1": {"clusters": ["usa_politics"], "entry_price": 5, "shares": 10},
            "s2": {"clusters": ["geopolitics"], "entry_price": 5, "shares": 10},
        }
        result = cm.get_correlated_exposure(positions, balance=1000)
        self.assertIn("trump_admin_politics", result)
        self.assertAlmostEqual(result["trump_admin_politics"], 0.1)

    def test_no_correlated_group_gives_empty(self):
        positions = {"s1": {"clusters": ["lonely_cluster"], "entry_price": 5, "shares": 10}}
        result = cm.get_correlated_exposure(positions, balance=1000)
        self.assertEqual(result, {})


class TestCheckCorrelationLimit(unittest.TestCase):
    def test_empty_positions_ok(self):
        ok, _msg = cm.check_correlation_limit("politics", {}, 1000)
        self.assertTrue(ok)

    def test_single_cluster_over_limit(self):
        positions = {"s1": {"clusters": ["politics"], "entry_price": 3, "shares": 100}}
        ok, msg = cm.check_correlation_limit("politics", positions, 500, new_investment=150)
        self.assertFalse(ok)
        self.assertIn("single_cluster", msg)

    def test_correlated_group_over_limit(self):
        positions = {
            "s1": {"clusters": ["usa_politics"], "entry_price": 2, "shares": 50},
        }
        ok, msg = cm.check_correlation_limit("russia_ukraine", positions, 400, new_investment=50)
        self.assertFalse(ok)
        self.assertIn("correlated_group", msg)

    def test_within_limits_ok(self):
        positions = {"s1": {"clusters": ["politics"], "entry_price": 1, "shares": 10}}
        ok, _msg = cm.check_correlation_limit("sports_nba", positions, 1000, new_investment=10)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
