"""Tests for orderbook_analyzer.py — CLOB API order book depth analysis."""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import orderbook_analyzer as ob


class TestComputeImbalance(unittest.TestCase):
    def test_balanced_book_returns_near_zero(self):
        bids = [{"price": "0.10", "size": "1000"}] * 5
        asks = [{"price": "0.10", "size": "1000"}] * 5
        result = ob.compute_imbalance(bids, asks)
        self.assertAlmostEqual(result, 0.0)

    def test_bid_heavy_returns_positive(self):
        bids = [{"price": "0.10", "size": "2000"}] * 5
        asks = [{"price": "0.10", "size": "500"}] * 5
        result = ob.compute_imbalance(bids, asks)
        self.assertGreater(result, 0.4)

    def test_ask_heavy_returns_negative(self):
        bids = [{"price": "0.10", "size": "500"}] * 5
        asks = [{"price": "0.10", "size": "2000"}] * 5
        result = ob.compute_imbalance(bids, asks)
        self.assertLess(result, -0.4)

    def test_empty_book_returns_zero(self):
        self.assertEqual(ob.compute_imbalance([], []), 0.0)

    def test_only_bids_returns_one(self):
        bids = [{"price": "0.10", "size": "1000"}]
        result = ob.compute_imbalance(bids, [])
        self.assertAlmostEqual(result, 1.0)

    def test_only_asks_returns_negative_one(self):
        asks = [{"price": "0.10", "size": "1000"}]
        result = ob.compute_imbalance([], asks)
        self.assertAlmostEqual(result, -1.0)

    def test_uses_top_20_only(self):
        bids = [{"price": "0.10", "size": "100"}] * 30
        asks = [{"price": "0.10", "size": "1"}] * 30
        result = ob.compute_imbalance(bids, asks)
        self.assertGreater(result, 0.9)

    def test_handles_missing_keys(self):
        bids = [{"price": "0.10"}, {"size": "100"}]
        asks = [{"price": "0.10", "size": "100"}]
        result = ob.compute_imbalance(bids, asks)
        self.assertIsInstance(result, float)


class TestDetectBidWall(unittest.TestCase):
    def test_large_single_bid_triggers_wall(self):
        bids = [{"price": "0.10", "size": "60000"}]
        has_wall, wall_size = ob.detect_bid_wall(bids, 0.10)
        self.assertTrue(has_wall)
        self.assertGreater(wall_size, 5000)

    def test_small_bids_no_wall(self):
        bids = [{"price": "0.10", "size": "100"}] * 3
        has_wall, _wall_size = ob.detect_bid_wall(bids, 0.10)
        self.assertFalse(has_wall)

    def test_empty_bids_no_wall(self):
        has_wall, wall_size = ob.detect_bid_wall([], 0.10)
        self.assertFalse(has_wall)
        self.assertEqual(wall_size, 0.0)

    def test_zero_price_no_wall(self):
        bids = [{"price": "0.10", "size": "60000"}]
        has_wall, _wall_size = ob.detect_bid_wall(bids, 0.0)
        self.assertFalse(has_wall)

    def test_top_three_aggregate_wall(self):
        bids = [{"price": "0.10", "size": "20000"}] * 3
        has_wall, wall_size = ob.detect_bid_wall(bids, 0.10)
        self.assertTrue(has_wall)
        self.assertGreaterEqual(wall_size, 5000)


class TestAnalyzeOrderbookDepth(unittest.TestCase):
    def setUp(self):
        ob._book_cache.clear()

    @patch("orderbook_analyzer.fetch_order_book")
    def test_high_imbalance_gives_15_points(self, mock_fetch):
        mock_fetch.return_value = {
            "bids": [{"price": "0.10", "size": "20000"}] * 10,
            "asks": [{"price": "0.10", "size": "1000"}] * 5,
        }
        result = ob.analyze_orderbook_depth("token123", 0.10)
        self.assertEqual(result["signal_score"], 15)
        self.assertIn("imbalance", result["reason"])

    @patch("orderbook_analyzer.fetch_order_book")
    def test_no_imbalance_no_score(self, mock_fetch):
        mock_fetch.return_value = {
            "bids": [{"price": "0.10", "size": "1000"}] * 5,
            "asks": [{"price": "0.10", "size": "1000"}] * 5,
        }
        result = ob.analyze_orderbook_depth("token123", 0.10)
        self.assertEqual(result["signal_score"], 0)

    @patch("orderbook_analyzer.fetch_order_book", return_value=None)
    def test_fetch_failed_returns_empty(self, mock_fetch):
        result = ob.analyze_orderbook_depth("token123", 0.10)
        self.assertEqual(result["signal_score"], 0)
        self.assertEqual(result["reason"], "fetch_failed")

    @patch("orderbook_analyzer.fetch_order_book", return_value={"bids": [], "asks": []})
    def test_empty_book_returns_empty(self, mock_fetch):
        result = ob.analyze_orderbook_depth("token123", 0.10)
        self.assertEqual(result["signal_score"], 0)
        self.assertEqual(result["reason"], "empty_book")

    @patch("orderbook_analyzer.fetch_order_book")
    def test_bid_wall_gives_12_points(self, mock_fetch):
        mock_fetch.return_value = {
            "bids": [{"price": "0.10", "size": "60000"}],
            "asks": [{"price": "0.10", "size": "2000"}] * 3,
        }
        result = ob.analyze_orderbook_depth("token123", 0.10)
        self.assertIn(result["signal_score"], (12, 15))
        self.assertTrue(result["has_bid_wall"])

    @patch("orderbook_analyzer.fetch_order_book")
    def test_result_structure(self, mock_fetch):
        mock_fetch.return_value = {
            "bids": [{"price": "0.10", "size": "1000"}],
            "asks": [{"price": "0.10", "size": "1000"}],
        }
        result = ob.analyze_orderbook_depth("token123", 0.10)
        for key in ("imbalance", "bid_volume_usd", "ask_volume_usd", "has_bid_wall", "wall_size_usd", "signal_score", "reason"):
            self.assertIn(key, result)


class TestCacheBehavior(unittest.TestCase):
    def setUp(self):
        ob._book_cache.clear()

    @patch("orderbook_analyzer.fetch_order_book")
    def test_cache_hits_within_ttl(self, mock_fetch):
        mock_fetch.return_value = {
            "bids": [{"price": "0.10", "size": "1000"}],
            "asks": [{"price": "0.10", "size": "1000"}],
        }
        ob.analyze_orderbook_depth("cached_token", 0.10)
        ob.analyze_orderbook_depth("cached_token", 0.10)
        mock_fetch.assert_called_once()

    @patch("orderbook_analyzer.fetch_order_book")
    def test_cache_expires_after_ttl(self, mock_fetch):
        mock_fetch.return_value = {
            "bids": [{"price": "0.10", "size": "1000"}],
            "asks": [{"price": "0.10", "size": "1000"}],
        }
        ob.analyze_orderbook_depth("expire_token", 0.10)
        ob._book_cache["expire_token"] = (time.time() - 600, ob._book_cache["expire_token"][1])
        ob.analyze_orderbook_depth("expire_token", 0.10)
        self.assertEqual(mock_fetch.call_count, 2)


class TestEmptyResult(unittest.TestCase):
    def test_empty_result_structure(self):
        result = ob._empty_result("test_reason")
        self.assertEqual(result["signal_score"], 0)
        self.assertEqual(result["reason"], "test_reason")
        self.assertEqual(result["imbalance"], 0.0)
        self.assertFalse(result["has_bid_wall"])

    def test_empty_result_reason_propagated(self):
        result = ob._empty_result("fetch_failed")
        self.assertEqual(result["reason"], "fetch_failed")


class TestFetchOrderBook(unittest.TestCase):
    @patch("orderbook_analyzer.requests.get")
    def test_success_returns_json(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"bids": [], "asks": []}
        mock_get.return_value = mock_resp
        result = ob.fetch_order_book("token123")
        self.assertIsNotNone(result)

    @patch("orderbook_analyzer.requests.get")
    def test_non_200_returns_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp
        self.assertIsNone(ob.fetch_order_book("token123"))

    @patch("orderbook_analyzer.requests.get", side_effect=Exception("timeout"))
    def test_exception_returns_none(self, mock_get):
        self.assertIsNone(ob.fetch_order_book("token123"))
