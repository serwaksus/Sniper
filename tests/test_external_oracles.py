"""
test_external_oracles.py — Tests for the three free oracle sources.

Tests cover:
- Unit tests for each oracle (with mocked HTTP)
- Cache TTL behavior
- Keyword extraction logic
- Macro alignment heuristics
- Integration with signal_scorer (oracle bonus appears in score)
- Fault injection (timeouts, malformed responses don't crash)
- Edge cases (empty data, extreme values)
"""
from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest


class TestFearGreedIndex(unittest.TestCase):
    """Alternative.me Fear & Greed Index tests."""

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    @patch("external_oracles._save_cache")
    def test_fetch_success(self, mock_save, mock_load, mock_get):
        """FNG index parses correctly from API response."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "data": [{"value": "25", "value_classification": "Extreme Fear"}]
            }),
        )
        result = external_oracles.get_fear_greed_index()
        self.assertEqual(result, 25)

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_http_error_returns_none(self, mock_load, mock_get):
        """HTTP error returns None, not crash."""
        import external_oracles

        mock_get.return_value = MagicMock(status_code=500)
        result = external_oracles.get_fear_greed_index()
        self.assertIsNone(result)

    @patch("external_oracles.requests.get", side_effect=Exception("network down"))
    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_exception_returns_none(self, mock_load, mock_get):
        """Network exception returns None."""
        import external_oracles

        result = external_oracles.get_fear_greed_index()
        self.assertIsNone(result)

    @patch("external_oracles._load_cache", return_value=42)
    def test_cache_hit_no_api_call(self, mock_load):
        """Cache hit returns cached value without API call."""
        import external_oracles

        with patch("external_oracles.requests.get") as mock_get:
            result = external_oracles.get_fear_greed_index()
            self.assertEqual(result, 42)
            mock_get.assert_not_called()

    @patch("external_oracles.get_fear_greed_index", return_value=25)
    def test_bonus_extreme_fear_crypto(self, mock_fng):
        """+5 for crypto cluster when FNG < 30."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("crypto"), 5)

    @patch("external_oracles.get_fear_greed_index", return_value=25)
    def test_bonus_extreme_fear_ai_tech(self, mock_fng):
        """+5 for ai_tech cluster when FNG < 30."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("ai_tech"), 5)

    @patch("external_oracles.get_fear_greed_index", return_value=25)
    def test_bonus_extreme_fear_tech(self, mock_fng):
        """+5 for tech cluster when FNG < 30."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("tech"), 5)

    @patch("external_oracles.get_fear_greed_index", return_value=25)
    def test_no_bonus_wrong_cluster(self, mock_fng):
        """0 bonus for non-tech cluster even in extreme fear."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("geopolitical"), 0)

    @patch("external_oracles.get_fear_greed_index", return_value=55)
    def test_no_bonus_greed_mode(self, mock_fng):
        """0 bonus when index ≥ 30 (not extreme fear)."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("crypto"), 0)

    @patch("external_oracles.get_fear_greed_index", return_value=None)
    def test_no_bonus_fetch_failed(self, mock_fng):
        """0 bonus when FNG fetch fails."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("crypto"), 0)

    @patch("external_oracles.get_fear_greed_index", return_value=29)
    def test_bonus_boundary_29(self, mock_fng):
        """+5 at FNG=29 (just below threshold)."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("tech"), 5)

    @patch("external_oracles.get_fear_greed_index", return_value=30)
    def test_no_bonus_boundary_30(self, mock_fng):
        """0 at FNG=30 (exactly at threshold)."""
        from external_oracles import fear_greed_bonus
        self.assertEqual(fear_greed_bonus("tech"), 0)


class TestManifoldArbitrage(unittest.TestCase):
    """Manifold Markets cross-platform arbitrage tests."""

    @patch("external_oracles.requests.get")
    def test_arbitrage_found(self, mock_get):
        """+15 when Manifold prob is ≥15% higher than Polymarket price."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"question": "Will BTC hit 200k?", "probability": 0.30,
                 "outcomeType": "BINARY", "isResolved": False},
            ]),
        )
        # Polymarket price = 0.08, Manifold = 0.30, gap = 0.22 ≥ 0.15
        result = external_oracles.check_manifold_arbitrage(
            "Will Bitcoin reach $200k?", 0.08,
        )
        self.assertEqual(result, 15)

    @patch("external_oracles.requests.get")
    def test_no_arbitrage_small_gap(self, mock_get):
        """0 when gap < 15%."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"question": "Will BTC hit 200k?", "probability": 0.15,
                 "outcomeType": "BINARY", "isResolved": False},
            ]),
        )
        # Polymarket price = 0.08, Manifold = 0.15, gap = 0.07 < 0.15
        result = external_oracles.check_manifold_arbitrage(
            "Will Bitcoin reach $200k?", 0.08,
        )
        self.assertEqual(result, 0)

    @patch("external_oracles.requests.get")
    def test_boundary_gap_exactly_15pct(self, mock_get):
        """+15 when gap is exactly 15%."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"question": "Test", "probability": 0.23,
                 "outcomeType": "BINARY", "isResolved": False},
            ]),
        )
        # gap = 0.23 - 0.08 = 0.15 ≥ 0.15
        result = external_oracles.check_manifold_arbitrage(
            "Test question", 0.08,
        )
        self.assertEqual(result, 15)

    @patch("external_oracles.requests.get")
    def test_skip_non_binary(self, mock_get):
        """Non-binary markets are skipped."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"question": "Multi-choice", "probability": 0.90,
                 "outcomeType": "MULTIPLE_CHOICE", "isResolved": False},
                {"question": "Binary", "probability": 0.25,
                 "outcomeType": "BINARY", "isResolved": False},
            ]),
        )
        result = external_oracles.check_manifold_arbitrage("Test", 0.05)
        # Second market (binary) has gap = 0.20 ≥ 0.15 → +15
        self.assertEqual(result, 15)

    @patch("external_oracles.requests.get")
    def test_skip_resolved(self, mock_get):
        """Resolved markets are skipped."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"question": "Resolved", "probability": 0.95,
                 "outcomeType": "BINARY", "isResolved": True},
                {"question": "Open", "probability": 0.95,
                 "outcomeType": "BINARY", "isResolved": False},
            ]),
        )
        result = external_oracles.check_manifold_arbitrage("Test", 0.05)
        # Only second market (unresolved) qualifies, gap=0.90 ≥ 0.15 → +15
        self.assertEqual(result, 15)

    @patch("external_oracles.requests.get")
    def test_empty_results(self, mock_get):
        """Empty Manifold results → 0 bonus."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[]),
        )
        result = external_oracles.check_manifold_arbitrage("Test", 0.05)
        self.assertEqual(result, 0)

    @patch("external_oracles.requests.get", side_effect=__import__("requests").exceptions.Timeout("slow"))
    def test_timeout_returns_zero(self, mock_get):
        """Timeout → 0, not crash."""
        import external_oracles
        result = external_oracles.check_manifold_arbitrage("Test", 0.05)
        self.assertEqual(result, 0)

    @patch("external_oracles.requests.get")
    def test_http_500_returns_zero(self, mock_get):
        """HTTP 500 → 0."""
        import external_oracles
        mock_get.return_value = MagicMock(status_code=500)
        result = external_oracles.check_manifold_arbitrage("Test", 0.05)
        self.assertEqual(result, 0)

    @patch("external_oracles.requests.get")
    def test_cache_hit_skips_api(self, mock_get):
        """Second call with same slug uses cache (no API call)."""
        import external_oracles

        # Clear cache
        external_oracles._manifold_cache.clear()

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"question": "Test", "probability": 0.30,
                 "outcomeType": "BINARY", "isResolved": False},
            ]),
        )
        # First call
        r1 = external_oracles.check_manifold_arbitrage("Test", 0.08, slug="test-slug")
        self.assertEqual(r1, 15)
        self.assertEqual(mock_get.call_count, 1)

        # Second call — should use cache
        r2 = external_oracles.check_manifold_arbitrage("Test", 0.08, slug="test-slug")
        self.assertEqual(r2, 15)
        self.assertEqual(mock_get.call_count, 1, "Second call should hit cache, not API")

    @patch("external_oracles.requests.get")
    def test_null_probability_skipped(self, mock_get):
        """Market with null probability is skipped."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"question": "Null prob", "probability": None,
                 "outcomeType": "BINARY", "isResolved": False},
            ]),
        )
        result = external_oracles.check_manifold_arbitrage("Test", 0.05)
        self.assertEqual(result, 0)


class TestKeywordExtraction(unittest.TestCase):
    """Keyword extraction from Polymarket questions."""

    def test_extracts_meaningful_words(self):
        from external_oracles import _extract_keywords
        result = _extract_keywords("Will Bitcoin reach $200,000 before 2028?")
        self.assertIn("bitcoin", result)
        self.assertIn("reach", result)

    def test_strips_stop_words(self):
        from external_oracles import _extract_keywords
        result = _extract_keywords("Will the a an in of on at to be this that")
        # All stop words should be stripped
        words = result.split()
        for w in words:
            self.assertNotIn(w, ["will", "the", "that", "this"])

    def test_strips_years(self):
        from external_oracles import _extract_keywords
        result = _extract_keywords("Will something happen by 2027?")
        self.assertNotIn("2027", result.split())

    def test_strips_months(self):
        from external_oracles import _extract_keywords
        result = _extract_keywords("Will event happen before January 2028?")
        self.assertNotIn("january", result.split())

    def test_fallback_on_empty(self):
        from external_oracles import _extract_keywords
        result = _extract_keywords("???")
        self.assertTrue(len(result) > 0)

    def test_max_three_keywords(self):
        from external_oracles import _extract_keywords
        result = _extract_keywords("Alpha beta gamma delta epsilon zeta eta")
        words = result.split()
        self.assertLessEqual(len(words), 3)


class TestDBnomics(unittest.TestCase):
    """DBnomics macroeconomic data tests."""

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    @patch("external_oracles._save_cache")
    def test_fetch_fed_funds_success(self, mock_save, mock_load, mock_get):
        """Fed Funds Rate parses correctly."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01", "2024-02"], "value": [5.25, 5.5]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertEqual(result, 5.5)

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_handles_na_values(self, mock_load, mock_get):
        """NA values are skipped, latest valid value returned."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01", "2024-02", "2024-03"], "value": [5.25, "NA", 5.5]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertEqual(result, 5.5)

    @patch("external_oracles.requests.get", side_effect=Exception("network"))
    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_exception_returns_none(self, mock_load, mock_get):
        """Network exception returns None."""
        import external_oracles
        result = external_oracles.get_fed_funds_rate()
        self.assertIsNone(result)

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_empty_docs(self, mock_load, mock_get):
        """Empty docs → None."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"series": {"docs": []}}),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertIsNone(result)

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_all_na_values(self, mock_load, mock_get):
        """All values NA → None."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01"], "value": ["NA"]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertIsNone(result)


class TestMacroAlignment(unittest.TestCase):
    """Macro trend alignment heuristics."""

    def test_rate_cut_aligned_with_high_rate(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will there be a rate cut in 2025?", 5.25, None))

    def test_rate_cut_not_aligned_with_low_rate(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will there be a rate cut?", 1.5, None))

    def test_rate_hike_aligned_with_low_rate(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will the Fed hike rates?", 1.0, None))

    def test_rate_hike_not_aligned_with_high_rate(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will the Fed hike rates?", 5.0, None))

    def test_inflation_aligned_with_high_cpi(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will inflation increase?", None, 315.0))

    def test_inflation_not_aligned_with_low_cpi(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will inflation increase?", None, 250.0))

    def test_recession_aligned_with_high_rate(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will there be a recession in 2025?", 5.0, None))

    def test_no_keywords_returns_false(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will it rain?", 5.0, 300.0))

    def test_none_data_returns_false(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will there be a rate cut?", None, None))


class TestDBnomicsBonus(unittest.TestCase):
    """DBnomics bonus computation."""

    @patch("external_oracles.get_fed_funds_rate", return_value=5.25)
    @patch("external_oracles.get_cpi_inflation", return_value=310.0)
    def test_bonus_for_fed_fomc_rate_cut(self, mock_cpi, mock_fed):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("fed_fomc", "Will the Fed cut rates in 2025?")
        self.assertEqual(result, 10)

    @patch("external_oracles.get_fed_funds_rate", return_value=5.25)
    @patch("external_oracles.get_cpi_inflation", return_value=310.0)
    def test_bonus_for_us_economic_inflation(self, mock_cpi, mock_fed):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("us_economic", "Will inflation rise further?")
        self.assertEqual(result, 10)

    @patch("external_oracles.get_fed_funds_rate", return_value=5.25)
    @patch("external_oracles.get_cpi_inflation", return_value=310.0)
    def test_no_bonus_wrong_cluster(self, mock_cpi, mock_fed):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("crypto", "Will inflation rise?")
        self.assertEqual(result, 0)

    @patch("external_oracles.get_fed_funds_rate", return_value=5.25)
    @patch("external_oracles.get_cpi_inflation", return_value=310.0)
    def test_no_bonus_no_alignment(self, mock_cpi, mock_fed):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("fed_fomc", "Will it snow in Alaska?")
        self.assertEqual(result, 0)


class TestComputeOracleBonus(unittest.TestCase):
    """Unified entry point tests."""

    def setUp(self):
        """Temporarily enable oracles."""
        self._old_val = os.environ.get("ORACLES_DISABLED")
        os.environ["ORACLES_DISABLED"] = "0"

    def tearDown(self):
        if self._old_val is None:
            os.environ.pop("ORACLES_DISABLED", None)
        else:
            os.environ["ORACLES_DISABLED"] = self._old_val

    @patch("external_oracles.fear_greed_bonus", return_value=5)
    @patch("external_oracles.check_manifold_arbitrage", return_value=15)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=0)
    def test_all_three_combined(self, mock_dbn, mock_mf, mock_fng):
        from external_oracles import compute_oracle_bonus
        total, breakdown = compute_oracle_bonus(
            "crypto", "Will BTC reach 200k?", 0.08, "btc-200k",
        )
        self.assertEqual(total, 20)
        self.assertEqual(breakdown["fng"], 5)
        self.assertEqual(breakdown["manifold_arb"], 15)
        self.assertEqual(breakdown["dbnomics"], 0)

    @patch("external_oracles.fear_greed_bonus", return_value=0)
    @patch("external_oracles.check_manifold_arbitrage", return_value=0)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=0)
    def test_no_bonus_anywhere(self, mock_dbn, mock_mf, mock_fng):
        from external_oracles import compute_oracle_bonus
        total, breakdown = compute_oracle_bonus(
            "other", "Random question", 0.10, "random",
        )
        self.assertEqual(total, 0)

    @patch("external_oracles.fear_greed_bonus", side_effect=Exception("crash"))
    @patch("external_oracles.check_manifold_arbitrage", return_value=15)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=0)
    def test_fng_crash_doesnt_break_others(self, mock_dbn, mock_mf, mock_fng):
        """FNG crash → 0, but other oracles still work."""
        from external_oracles import compute_oracle_bonus
        total, breakdown = compute_oracle_bonus(
            "crypto", "Test", 0.05, "test",
        )
        self.assertEqual(breakdown["fng"], 0)
        self.assertEqual(breakdown["manifold_arb"], 15)
        self.assertEqual(total, 15)

    @patch("external_oracles.check_manifold_arbitrage", side_effect=Exception("crash"))
    @patch("external_oracles.fear_greed_bonus", return_value=5)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=10)
    def test_manifold_crash_doesnt_break_others(self, mock_dbn, mock_mf, mock_fng):
        """Manifold crash → 0, but other oracles still work."""
        from external_oracles import compute_oracle_bonus
        total, breakdown = compute_oracle_bonus(
            "fed_fomc", "Test rate cut", 0.05, "test",
        )
        self.assertEqual(breakdown["manifold_arb"], 0)
        self.assertEqual(breakdown["fng"], 5)
        self.assertEqual(breakdown["dbnomics"], 10)
        self.assertEqual(total, 15)


class TestSignalScorerIntegration(unittest.TestCase):
    """Verify oracle bonus integrates into _compute_signal_score."""

    def setUp(self):
        """Temporarily enable oracles for integration tests."""
        self._old_val = os.environ.get("ORACLES_DISABLED")
        os.environ["ORACLES_DISABLED"] = "0"

    def tearDown(self):
        if self._old_val is None:
            os.environ.pop("ORACLES_DISABLED", None)
        else:
            os.environ["ORACLES_DISABLED"] = self._old_val

    @patch("external_oracles.fear_greed_bonus", return_value=5)
    @patch("external_oracles.check_manifold_arbitrage", return_value=15)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=0)
    @patch("social_buzz.compute_buzz_score", return_value={"buzz_score": 0})
    def test_oracle_bonus_adds_to_signal_score(self, mock_buzz, mock_dbn, mock_mf, mock_fng):
        """Oracle bonus is included in the final signal score."""
        from signal_scorer import _compute_signal_score

        # Run with oracle bonus
        score_with_oracle, _, _, _, _, _, _, _, _, _, _, _, _ = _compute_signal_score(
            p_model=0.20, market_price=0.08, factors=[],
            volume=500_000, ttl_hours=720, cluster="crypto",
            slug="test-crypto", question="Will BTC reach 200k?",
        )

        # Now patch oracle to return 0
        with patch("external_oracles.fear_greed_bonus", return_value=0), \
             patch("external_oracles.check_manifold_arbitrage", return_value=0):
            score_without_oracle, _, _, _, _, _, _, _, _, _, _, _, _ = _compute_signal_score(
                p_model=0.20, market_price=0.08, factors=[],
                volume=500_000, ttl_hours=720, cluster="crypto",
                slug="test-crypto2", question="Will BTC reach 200k?",
            )

        # Difference should be 5 + 15 = 20
        diff = score_with_oracle - score_without_oracle
        self.assertEqual(diff, 20,
            f"Oracle bonus should add 20 points, got diff={diff}")


class TestCacheInfrastructure(unittest.TestCase):
    """File-based cache behavior."""

    def test_cache_save_and_load(self):
        from external_oracles import _save_cache, _load_cache
        import tempfile as tf
        import json

        # Use unique filename to avoid conflicts
        fname = f"test_cache_{int(time.time())}.json"
        try:
            _save_cache(fname, {"test": 42})
            loaded = _load_cache(fname, 3600)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["test"], 42)
        finally:
            # Cleanup
            cache_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", fname,
            )
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def test_cache_expired_returns_none(self):
        """Expired cache returns None."""
        from external_oracles import _save_cache, _load_cache

        fname = f"test_expired_{int(time.time())}.json"
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", fname,
        )
        try:
            _save_cache(fname, 42)
            # Manually set the timestamp to 2 hours ago
            with open(cache_path, "r") as f:
                data = json.load(f)
            data["_cached_at"] = time.time() - 7200  # 2h ago
            with open(cache_path, "w") as f:
                json.dump(data, f)

            # TTL = 1 hour → expired
            result = _load_cache(fname, 3600)
            self.assertIsNone(result)
        finally:
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def test_cache_missing_file_returns_none(self):
        from external_oracles import _load_cache
        result = _load_cache("nonexistent_cache_file.json", 3600)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
