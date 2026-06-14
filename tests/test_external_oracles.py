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
    """DBnomics macroeconomic data tests (v2: trend-aware)."""

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    @patch("external_oracles._save_cache")
    def test_fetch_fed_funds_success_with_trend(self, mock_save, mock_load, mock_get):
        """Fed Funds Rate parses correctly with trend info."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01", "2024-02"], "value": [5.25, 5.5]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertIsNotNone(result)
        self.assertEqual(result["latest"], 5.5)
        self.assertEqual(result["previous"], 5.25)
        self.assertAlmostEqual(result["delta"], 0.25)
        self.assertEqual(result["trend"], "rising")
        self.assertEqual(result["period"], "2024-02")

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    @patch("external_oracles._save_cache")
    def test_fetch_trend_falling(self, mock_save, mock_load, mock_get):
        """Trend is 'falling' when latest < previous."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01", "2024-02"], "value": [5.5, 5.25]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertEqual(result["trend"], "falling")
        self.assertAlmostEqual(result["delta"], -0.25)

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    @patch("external_oracles._save_cache")
    def test_fetch_trend_stable(self, mock_save, mock_load, mock_get):
        """Trend is 'stable' when values are identical."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01", "2024-02"], "value": [5.25, 5.25]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertEqual(result["trend"], "stable")
        self.assertAlmostEqual(result["delta"], 0.0)

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    @patch("external_oracles._save_cache")
    def test_fetch_single_value_unknown_trend(self, mock_save, mock_load, mock_get):
        """Single value → trend='unknown', previous=None."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01"], "value": [5.5]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertEqual(result["latest"], 5.5)
        self.assertIsNone(result["previous"])
        self.assertIsNone(result["delta"])
        self.assertEqual(result["trend"], "unknown")

    @patch("external_oracles.requests.get")
    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_handles_na_values(self, mock_load, mock_get):
        """NA values are skipped, latest valid pair returned."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "series": {"docs": [{"period": ["2024-01", "2024-02", "2024-03"], "value": [5.25, "NA", 5.5]}]}
            }),
        )
        result = external_oracles.get_fed_funds_rate()
        self.assertEqual(result["latest"], 5.5)
        self.assertEqual(result["previous"], 5.25)

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

    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_unemployment_rate(self, mock_load):
        """Unemployment rate getter calls correct series."""
        import external_oracles

        with patch("external_oracles.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "series": {"docs": [{"period": ["2024-04", "2024-05"], "value": [3.9, 4.0]}]}
                }),
            )
            result = external_oracles.get_unemployment_rate()
            self.assertIsNotNone(result)
            self.assertEqual(result["latest"], 4.0)
            self.assertEqual(result["trend"], "rising")

    @patch("external_oracles._load_cache", return_value=None)
    def test_fetch_gdp_growth(self, mock_load):
        """GDP growth getter calls correct series."""
        import external_oracles

        with patch("external_oracles.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "series": {"docs": [{"period": ["2024-Q1", "2024-Q2"], "value": [2.5, -1.0]}]}
                }),
            )
            result = external_oracles.get_gdp_growth()
            self.assertIsNotNone(result)
            self.assertEqual(result["latest"], -1.0)
            self.assertEqual(result["trend"], "falling")


# ── Helper dicts for alignment tests ──────────────────────────────────

_FED_FALLING = {"latest": 4.5, "previous": 5.25, "delta": -0.75, "trend": "falling", "period": "2024-07"}
_FED_RISING = {"latest": 5.5, "previous": 5.25, "delta": 0.25, "trend": "rising", "period": "2024-07"}
_FED_STABLE_HIGH = {"latest": 5.25, "previous": 5.25, "delta": 0.0, "trend": "stable", "period": "2024-07"}
_FED_STABLE_LOW = {"latest": 1.5, "previous": 1.5, "delta": 0.0, "trend": "stable", "period": "2024-07"}
_CPI_RISING = {"latest": 315.0, "previous": 310.0, "delta": 5.0, "trend": "rising", "period": "2024-06"}
_CPI_FALLING = {"latest": 310.0, "previous": 315.0, "delta": -5.0, "trend": "falling", "period": "2024-06"}
_UNEMP_RISING = {"latest": 4.5, "previous": 3.8, "delta": 0.7, "trend": "rising", "period": "2024-06"}
_UNEMP_STABLE = {"latest": 3.8, "previous": 3.8, "delta": 0.0, "trend": "stable", "period": "2024-06"}
_GDP_NEGATIVE = {"latest": -2.5, "previous": 1.0, "delta": -3.5, "trend": "falling", "period": "2024-Q2"}
_GDP_NORMAL = {"latest": 2.5, "previous": 2.0, "delta": 0.5, "trend": "rising", "period": "2024-Q2"}
_GDP_SLOW = {"latest": 0.5, "previous": 2.0, "delta": -1.5, "trend": "falling", "period": "2024-Q2"}


class TestMacroAlignment(unittest.TestCase):
    """Macro trend alignment heuristics (v2: trend-based)."""

    # ── Rate cut alignment ────────────────────────────────────────────

    def test_rate_cut_aligned_falling(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will there be a rate cut?", _FED_FALLING, None))

    def test_rate_cut_aligned_high_stable(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will there be a rate cut?", _FED_STABLE_HIGH, None))

    def test_rate_cut_not_aligned_rising(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will there be a rate cut?", _FED_RISING, None))

    def test_rate_cut_not_aligned_low_stable(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will there be a rate cut?", _FED_STABLE_LOW, None))

    # ── Rate hike alignment ───────────────────────────────────────────

    def test_rate_hike_aligned_rising(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will the Fed hike rates?", _FED_RISING, None))

    def test_rate_hike_not_aligned_falling(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will the Fed hike rates?", _FED_FALLING, None))

    # ── Inflation alignment ───────────────────────────────────────────

    def test_inflation_aligned_cpi_rising(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will inflation increase?", None, _CPI_RISING))

    def test_inflation_not_aligned_cpi_falling(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will inflation increase?", None, _CPI_FALLING))

    def test_inflation_not_aligned_cpi_stable_high(self):
        """CPI that was >300 in old test — now needs trend, not just level."""
        from external_oracles import _check_macro_alignment
        stable_cpi = {"latest": 315.0, "previous": 315.0, "delta": 0.0, "trend": "stable", "period": "2024-06"}
        self.assertFalse(_check_macro_alignment("Will inflation increase?", None, stable_cpi))

    # ── Recession alignment (multi-signal) ────────────────────────────

    def test_recession_aligned_gdp_negative(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will there be a recession?", None, None, None, _GDP_NEGATIVE))

    def test_recession_aligned_unemployment_rising(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will there be a recession?", None, None, _UNEMP_RISING, None))

    def test_recession_aligned_high_rising_rates(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will there be a recession?", _FED_RISING, None, None, None))

    def test_recession_not_aligned_normal(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will there be a recession?", _FED_STABLE_HIGH, None, _UNEMP_STABLE, _GDP_NORMAL))

    # ── Unemployment alignment ────────────────────────────────────────

    def test_unemployment_aligned_rising(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will unemployment claims rise?", None, None, _UNEMP_RISING))

    def test_unemployment_not_aligned_stable(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will unemployment rise?", None, None, _UNEMP_STABLE))

    # ── GDP alignment ─────────────────────────────────────────────────

    def test_gdp_aligned_negative_growth(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will GDP growth slow down?", None, None, None, _GDP_NEGATIVE))

    def test_gdp_aligned_decelerating(self):
        from external_oracles import _check_macro_alignment
        self.assertTrue(_check_macro_alignment("Will we see economic stagnation?", None, None, None, _GDP_SLOW))

    # ── Edge cases ────────────────────────────────────────────────────

    def test_no_keywords_returns_false(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will it rain?", _FED_STABLE_HIGH, _CPI_RISING))

    def test_none_data_returns_false(self):
        from external_oracles import _check_macro_alignment
        self.assertFalse(_check_macro_alignment("Will there be a rate cut?", None, None))


class TestDBnomicsBonus(unittest.TestCase):
    """DBnomics bonus computation (v2: trend-aware)."""

    @patch("external_oracles.get_gdp_growth", return_value=None)
    @patch("external_oracles.get_unemployment_rate", return_value=None)
    @patch("external_oracles.get_cpi_inflation", return_value=None)
    @patch("external_oracles.get_fed_funds_rate", return_value=_FED_FALLING)
    def test_bonus_for_fed_fomc_rate_cut(self, mock_fed, mock_cpi, mock_unemp, mock_gdp):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("fed_fomc", "Will the Fed cut rates in 2025?")
        self.assertEqual(result, 10)

    @patch("external_oracles.get_gdp_growth", return_value=None)
    @patch("external_oracles.get_unemployment_rate", return_value=None)
    @patch("external_oracles.get_cpi_inflation", return_value=_CPI_RISING)
    @patch("external_oracles.get_fed_funds_rate", return_value=None)
    def test_bonus_for_us_economic_inflation(self, mock_fed, mock_cpi, mock_unemp, mock_gdp):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("us_economic", "Will inflation rise further?")
        self.assertEqual(result, 10)

    @patch("external_oracles.get_gdp_growth", return_value=_GDP_NEGATIVE)
    @patch("external_oracles.get_unemployment_rate", return_value=_UNEMP_RISING)
    @patch("external_oracles.get_cpi_inflation", return_value=None)
    @patch("external_oracles.get_fed_funds_rate", return_value=None)
    def test_bonus_for_usa_politics_recession(self, mock_fed, mock_cpi, mock_unemp, mock_gdp):
        """usa_politics cluster now triggers DBnomics."""
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("usa_politics", "Will the US enter a recession before the 2028 election?")
        self.assertEqual(result, 10)

    @patch("external_oracles.get_gdp_growth", return_value=None)
    @patch("external_oracles.get_unemployment_rate", return_value=_UNEMP_RISING)
    @patch("external_oracles.get_cpi_inflation", return_value=None)
    @patch("external_oracles.get_fed_funds_rate", return_value=None)
    def test_bonus_for_unemployment_question(self, mock_fed, mock_cpi, mock_unemp, mock_gdp):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("us_economic", "Will unemployment claims spike?")
        self.assertEqual(result, 10)

    @patch("external_oracles.get_gdp_growth", return_value=None)
    @patch("external_oracles.get_unemployment_rate", return_value=None)
    @patch("external_oracles.get_cpi_inflation", return_value=None)
    @patch("external_oracles.get_fed_funds_rate", return_value=None)
    def test_no_bonus_wrong_cluster(self, mock_fed, mock_cpi, mock_unemp, mock_gdp):
        from external_oracles import dbnomics_macro_bonus
        result = dbnomics_macro_bonus("crypto", "Will inflation rise?")
        self.assertEqual(result, 0)

    @patch("external_oracles.get_gdp_growth", return_value=_GDP_NORMAL)
    @patch("external_oracles.get_unemployment_rate", return_value=_UNEMP_STABLE)
    @patch("external_oracles.get_cpi_inflation", return_value=None)
    @patch("external_oracles.get_fed_funds_rate", return_value=_FED_RISING)
    def test_no_bonus_no_alignment(self, mock_fed, mock_cpi, mock_unemp, mock_gdp):
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
        total, _breakdown = compute_oracle_bonus(
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
            with open(cache_path) as f:
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


# ═══════════════════════════════════════════════════════════════════
# YAHOO FINANCE (yfinance) TESTS
# ═══════════════════════════════════════════════════════════════════


class TestTickerDetection(unittest.TestCase):
    """Ticker extraction from Polymarket questions."""

    def test_detect_spy(self):
        from external_oracles import _detect_ticker
        self.assertEqual(_detect_ticker("Will the S&P 500 drop below 5000?"), "SPY")

    def test_detect_qqq(self):
        from external_oracles import _detect_ticker
        self.assertEqual(_detect_ticker("Will Nasdaq hit a new high?"), "QQQ")

    def test_detect_btc(self):
        from external_oracles import _detect_ticker
        self.assertEqual(_detect_ticker("Will Bitcoin reach $200k?"), "BTC-USD")

    def test_detect_eth(self):
        from external_oracles import _detect_ticker
        self.assertEqual(_detect_ticker("Will Ethereum 2.0 launch successfully?"), "ETH-USD")

    def test_detect_tesla(self):
        from external_oracles import _detect_ticker
        self.assertEqual(_detect_ticker("Will Tesla stock fall below $150?"), "TSLA")

    def test_detect_gold(self):
        from external_oracles import _detect_ticker
        self.assertEqual(_detect_ticker("Will gold prices surge?"), "GLD")

    def test_detect_vix(self):
        from external_oracles import _detect_ticker
        self.assertEqual(_detect_ticker("Will the VIX spike above 30?"), "^VIX")

    def test_no_ticker_found(self):
        from external_oracles import _detect_ticker
        self.assertIsNone(_detect_ticker("Will it rain in London?"))

    def test_no_ticker_generic_economic(self):
        from external_oracles import _detect_ticker
        self.assertIsNone(_detect_ticker("Will GDP growth exceed 3%?"))


class TestPriceTargetExtraction(unittest.TestCase):
    """Price target extraction from question text."""

    def test_below_pattern(self):
        from external_oracles import _extract_price_target
        self.assertAlmostEqual(_extract_price_target("Will S&P 500 drop below 5000?"), 5000.0)

    def test_under_pattern(self):
        from external_oracles import _extract_price_target
        self.assertAlmostEqual(_extract_price_target("Will Tesla fall under $150?"), 150.0)

    def test_above_pattern(self):
        from external_oracles import _extract_price_target
        self.assertAlmostEqual(_extract_price_target("Will Bitcoin surge above 100000?"), 100000.0)

    def test_dollar_amount(self):
        from external_oracles import _extract_price_target
        self.assertAlmostEqual(_extract_price_target("Will BTC hit $200000?"), 200000.0)

    def test_comma_numbers(self):
        from external_oracles import _extract_price_target
        self.assertAlmostEqual(_extract_price_target("Will S&P 500 drop below 5,000?"), 5000.0)

    def test_no_target_found(self):
        from external_oracles import _extract_price_target
        self.assertIsNone(_extract_price_target("Will the market crash?"))

    def test_drop_to_pattern(self):
        from external_oracles import _extract_price_target
        self.assertAlmostEqual(_extract_price_target("Will SPY drop to 450?"), 450.0)

    def test_bare_dollar_no_keyword(self):
        from external_oracles import _extract_price_target
        self.assertAlmostEqual(_extract_price_target("Will it reach $500 level?"), 500.0)


class TestYFinanceBonus(unittest.TestCase):
    """yfinance_bonus integration tests."""

    @patch("external_oracles._get_current_price", return_value=495.0)
    def test_bonus_price_near_target(self, mock_price):
        """+8 when current price is within 10% of target."""
        from external_oracles import yfinance_bonus
        # SPY at $495, target = $500 (from "below 500") → proximity = 1% → +8
        result = yfinance_bonus("us_economic", "Will the S&P 500 drop below 500 by Friday?")
        self.assertEqual(result, 8)

    @patch("external_oracles._get_current_price", return_value=400.0)
    def test_no_bonus_price_far_from_target(self, mock_price):
        """0 when price is far from target (>10%)."""
        from external_oracles import yfinance_bonus
        # SPY at $400, target = $500 → proximity = 20% → 0
        result = yfinance_bonus("us_economic", "Will the S&P 500 drop below 5000?")
        self.assertEqual(result, 0)

    @patch("external_oracles._get_current_price", return_value=None)
    def test_no_bonus_price_fetch_failed(self, mock_price):
        """0 when price fetch fails."""
        from external_oracles import yfinance_bonus
        result = yfinance_bonus("us_economic", "Will the S&P 500 drop below 5000?")
        self.assertEqual(result, 0)

    def test_no_bonus_no_ticker(self):
        """0 when no ticker detected."""
        from external_oracles import yfinance_bonus
        result = yfinance_bonus("other", "Will it rain in London?")
        self.assertEqual(result, 0)

    def test_no_bonus_no_target(self):
        """0 when ticker found but no price target."""
        from external_oracles import yfinance_bonus
        result = yfinance_bonus("us_economic", "Will the S&P 500 do something?")
        self.assertEqual(result, 0)

    @patch("external_oracles._get_current_price", return_value=505.0)
    def test_bonus_boundary_within_10pct(self, mock_price):
        """+8 at exactly 10% proximity boundary."""
        from external_oracles import yfinance_bonus
        # target = 550 (from "below 550"), current = 505 → proximity = 8.2% → +8
        result = yfinance_bonus("us_economic", "Will the S&P 500 drop below 550?")
        self.assertEqual(result, 8)

    @patch("external_oracles._get_current_price", return_value=195000.0)
    def test_bonus_btc_near_target(self, mock_price):
        """+8 for Bitcoin near target."""
        from external_oracles import yfinance_bonus
        # BTC at $195k, target = $200k → proximity = 2.5% → +8
        result = yfinance_bonus("crypto", "Will Bitcoin reach $200000?")
        self.assertEqual(result, 8)


class TestYFinanceCurrentPrice(unittest.TestCase):
    """_get_current_price caching and fault tolerance."""

    def test_cache_hit_no_yf_call(self):
        """Second call uses cache, no yfinance import needed."""
        import external_oracles
        external_oracles._yf_cache.clear()
        external_oracles._yf_cache["SPY"] = (time.time(), 495.0)

        result = external_oracles._get_current_price("SPY")
        self.assertEqual(result, 495.0)

    def test_cache_expired_refetches(self):
        """Expired cache triggers refetch."""
        import external_oracles
        external_oracles._yf_cache["TEST"] = (time.time() - 7200, 100.0)  # 2h ago

        # Will try to fetch (and fail), returns None
        with patch("external_oracles.yfinance", create=True):
            result = external_oracles._get_current_price("NONEXISTENT_TICKER_XYZ")
        self.assertIsNone(result)

    def test_yfinance_import_error_returns_none(self):
        """yfinance not installed → returns None, not crash."""
        import external_oracles

        external_oracles._yf_cache.clear()

        # Simulate import failure
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = external_oracles._get_current_price("SPY")
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════
# WIKIPEDIA PAGEVIEWS SPIKE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestEntityExtraction(unittest.TestCase):
    """Entity extraction from Polymarket questions."""

    def test_extract_politician_name(self):
        from external_oracles import _extract_entities
        entities = _extract_entities("Will Donald Trump win the election?")
        self.assertIn("Donald Trump", entities)

    def test_extract_company_name(self):
        from external_oracles import _extract_entities
        entities = _extract_entities("Will Tesla announce bankruptcy?")
        self.assertIn("Tesla", entities)

    def test_extract_multi_word_entity(self):
        from external_oracles import _extract_entities
        entities = _extract_entities("Will the Federal Reserve cut rates?")
        self.assertTrue(any("Federal Reserve" in e for e in entities))

    def test_no_entities_for_generic_question(self):
        from external_oracles import _extract_entities
        entities = _extract_entities("Will it rain tomorrow?")
        # Should find no meaningful entities
        self.assertEqual(len(entities), 0)

    def test_strips_leading_will(self):
        from external_oracles import _extract_entities
        entities = _extract_entities("Will Elon Musk buy Twitter?")
        self.assertIn("Elon Musk", entities)
        # "Will" should NOT be an entity
        self.assertNotIn("Will", entities)

    def test_max_three_entities(self):
        from external_oracles import _extract_entities
        entities = _extract_entities("Will Joe Biden meet Vladimir Putin and Xi Jinping?")
        self.assertLessEqual(len(entities), 3)

    def test_strips_month_names(self):
        from external_oracles import _extract_entities
        entities = _extract_entities("Will Donald Trump resign January 2025?")
        # January should not appear as entity
        for e in entities:
            self.assertNotIn("January", e)


class TestWikipediaSearch(unittest.TestCase):
    """Wikipedia article search tests."""

    @patch("external_oracles.requests.get")
    def test_search_success(self, mock_get):
        """Search returns article title."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "query": {"search": [{"title": "Donald Trump"}]}
            }),
        )
        result = external_oracles._search_wikipedia("Donald Trump")
        self.assertEqual(result, "Donald Trump")

    @patch("external_oracles.requests.get")
    def test_search_no_results(self, mock_get):
        """No results returns None."""
        import external_oracles

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"query": {"search": []}}),
        )
        result = external_oracles._search_wikipedia("nonexistent xyzzy 123")
        self.assertIsNone(result)

    @patch("external_oracles.requests.get")
    def test_search_http_error(self, mock_get):
        """HTTP error returns None."""
        import external_oracles

        mock_get.return_value = MagicMock(status_code=500)
        result = external_oracles._search_wikipedia("test")
        self.assertIsNone(result)

    @patch("external_oracles.requests.get", side_effect=Exception("network"))
    def test_search_exception_returns_none(self, mock_get):
        """Exception returns None."""
        import external_oracles
        result = external_oracles._search_wikipedia("test")
        self.assertIsNone(result)


class TestPageviewSpikeDetection(unittest.TestCase):
    """Pageviews spike detection tests."""

    @patch("external_oracles._fetch_pageviews")
    def test_spike_detected(self, mock_fetch):
        """Spike when recent median > 2x baseline median."""
        import external_oracles
        external_oracles._wiki_cache.clear()

        # 20 days baseline at ~100/day, 3 days recent at ~300/day
        baseline = [100] * 20
        recent = [300, 350, 280]
        mock_fetch.return_value = baseline + recent

        result = external_oracles._detect_wiki_spike("Donald_Trump")
        self.assertTrue(result)

    @patch("external_oracles._fetch_pageviews")
    def test_no_spike_stable(self, mock_fetch):
        """No spike when views are stable."""
        import external_oracles
        external_oracles._wiki_cache.clear()

        # All days at ~100/day → no spike
        mock_fetch.return_value = [100] * 23
        result = external_oracles._detect_wiki_spike("Some_Article")
        self.assertFalse(result)

    @patch("external_oracles._fetch_pageviews")
    def test_no_spike_low_baseline(self, mock_fetch):
        """No spike when baseline is very low (<10 views/day)."""
        import external_oracles
        external_oracles._wiki_cache.clear()

        # Low baseline → not meaningful even with spike
        baseline = [5] * 20
        recent = [50, 60, 40]
        mock_fetch.return_value = baseline + recent
        result = external_oracles._detect_wiki_spike("Obscure_Article")
        self.assertFalse(result)

    @patch("external_oracles._fetch_pageviews")
    def test_insufficient_data(self, mock_fetch):
        """No spike when insufficient data returned."""
        import external_oracles
        external_oracles._wiki_cache.clear()

        mock_fetch.return_value = [100, 200]  # Only 2 data points
        result = external_oracles._detect_wiki_spike("Test_Article")
        self.assertFalse(result)

    @patch("external_oracles._fetch_pageviews", return_value=[])
    def test_empty_pageviews(self, mock_fetch):
        """Empty pageviews → no spike."""
        import external_oracles
        external_oracles._wiki_cache.clear()
        result = external_oracles._detect_wiki_spike("Test_Article")
        self.assertFalse(result)

    def test_cache_hit_skips_fetch(self):
        """Cache hit returns without fetching."""
        import external_oracles
        external_oracles._wiki_cache.clear()
        external_oracles._wiki_cache["Cached_Article"] = (time.time(), True)

        with patch("external_oracles._fetch_pageviews") as mock_fetch:
            result = external_oracles._detect_wiki_spike("Cached_Article")
            self.assertTrue(result)
            mock_fetch.assert_not_called()


class TestWikipediaBonus(unittest.TestCase):
    """wikipedia_bonus end-to-end tests."""

    @patch("external_oracles._detect_wiki_spike", return_value=True)
    @patch("external_oracles._search_wikipedia", return_value="Donald Trump")
    @patch("external_oracles._extract_entities", return_value=["Donald Trump"])
    def test_bonus_with_spike(self, mock_extract, mock_search, mock_spike):
        """+7 when entity has pageviews spike."""
        from external_oracles import wikipedia_bonus
        result = wikipedia_bonus("Will Donald Trump win the election?")
        self.assertEqual(result, 7)

    @patch("external_oracles._detect_wiki_spike", return_value=False)
    @patch("external_oracles._search_wikipedia", return_value="Donald Trump")
    @patch("external_oracles._extract_entities", return_value=["Donald Trump"])
    def test_no_bonus_no_spike(self, mock_extract, mock_search, mock_spike):
        """0 when no spike detected."""
        from external_oracles import wikipedia_bonus
        result = wikipedia_bonus("Will Donald Trump win the election?")
        self.assertEqual(result, 0)

    @patch("external_oracles._extract_entities", return_value=[])
    def test_no_bonus_no_entities(self, mock_extract):
        """0 when no entities extracted."""
        from external_oracles import wikipedia_bonus
        result = wikipedia_bonus("Will it rain?")
        self.assertEqual(result, 0)

    @patch("external_oracles._search_wikipedia", return_value=None)
    @patch("external_oracles._extract_entities", return_value=["Some Entity"])
    def test_no_bonus_article_not_found(self, mock_extract, mock_search):
        """0 when Wikipedia article not found."""
        from external_oracles import wikipedia_bonus
        result = wikipedia_bonus("Will Some Entity do something?")
        self.assertEqual(result, 0)

    @patch("external_oracles._detect_wiki_spike", side_effect=[False, True])
    @patch("external_oracles._search_wikipedia", side_effect=["Entity One", "Entity Two"])
    @patch("external_oracles._extract_entities", return_value=["Entity One", "Entity Two"])
    def test_checks_multiple_entities(self, mock_extract, mock_search, mock_spike):
        """Checks multiple entities until one has a spike."""
        from external_oracles import wikipedia_bonus
        result = wikipedia_bonus("Will Entity One or Entity Two win?")
        self.assertEqual(result, 7)


class TestComputeOracleBonusFiveSources(unittest.TestCase):
    """Updated unified entry point with all 5 sources."""

    def setUp(self):
        self._old_val = os.environ.get("ORACLES_DISABLED")
        os.environ["ORACLES_DISABLED"] = "0"

    def tearDown(self):
        if self._old_val is None:
            os.environ.pop("ORACLES_DISABLED", None)
        else:
            os.environ["ORACLES_DISABLED"] = self._old_val

    @patch("external_oracles.wikipedia_bonus", return_value=7)
    @patch("external_oracles.yfinance_bonus", return_value=8)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=0)
    @patch("external_oracles.check_manifold_arbitrage", return_value=0)
    @patch("external_oracles.fear_greed_bonus", return_value=0)
    def test_yfinance_and_wiki_combined(self, mock_fng, mock_mf, mock_dbn, mock_yf, mock_wiki):
        """yfinance + Wikipedia combined = 15."""
        from external_oracles import compute_oracle_bonus
        total, breakdown = compute_oracle_bonus(
            "us_economic", "Will the S&P 500 drop below 5000?", 0.08, "test",
        )
        self.assertEqual(total, 15)
        self.assertEqual(breakdown["yfinance"], 8)
        self.assertEqual(breakdown["wiki"], 7)

    @patch("external_oracles.wikipedia_bonus", return_value=0)
    @patch("external_oracles.yfinance_bonus", return_value=0)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=0)
    @patch("external_oracles.check_manifold_arbitrage", return_value=0)
    @patch("external_oracles.fear_greed_bonus", return_value=5)
    def test_all_five_in_breakdown(self, mock_fng, mock_mf, mock_dbn, mock_yf, mock_wiki):
        """Breakdown dict has all 5 source keys."""
        from external_oracles import compute_oracle_bonus
        _total, breakdown = compute_oracle_bonus(
            "crypto", "Will BTC hit 200k?", 0.05, "test",
        )
        self.assertIn("fng", breakdown)
        self.assertIn("manifold_arb", breakdown)
        self.assertIn("dbnomics", breakdown)
        self.assertIn("yfinance", breakdown)
        self.assertIn("wiki", breakdown)

    @patch("external_oracles.wikipedia_bonus", side_effect=Exception("crash"))
    @patch("external_oracles.yfinance_bonus", return_value=8)
    @patch("external_oracles.dbnomics_macro_bonus", return_value=0)
    @patch("external_oracles.check_manifold_arbitrage", return_value=0)
    @patch("external_oracles.fear_greed_bonus", return_value=5)
    def test_wiki_crash_doesnt_break_others(self, mock_fng, mock_mf, mock_dbn, mock_yf, mock_wiki):
        """Wikipedia crash → 0, but other oracles still work."""
        from external_oracles import compute_oracle_bonus
        total, breakdown = compute_oracle_bonus(
            "crypto", "Will BTC hit 200k?", 0.05, "test",
        )
        self.assertEqual(breakdown["wiki"], 0)
        self.assertEqual(breakdown["yfinance"], 8)
        self.assertEqual(breakdown["fng"], 5)
        self.assertEqual(total, 13)


if __name__ == "__main__":
    unittest.main()
