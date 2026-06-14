"""Property-based invariant tests — verify fundamental properties ALWAYS hold.

These tests verify mathematical/logical invariants that must be true
for ANY input, not just specific test cases. They catch:

- p_model out of [0, 1] range
- signal_score negative
- prob_ratio < 0
- confidence > 1.0
- Division by zero
- NaN propagation

Uses simple boundary testing instead of hypothesis library (no extra deps).
Each test pushes inputs to extremes and verifies outputs stay valid.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from typing import Any


class TestProbabilityInvariants(unittest.TestCase):
    """Probability values must ALWAYS be in [0, 1]."""

    def test_normalize_probability_boundaries(self):
        from metaculus import normalize_probability
        cases = [0, 0.0, 0.5, 1.0, 1, -0.5, 2.0, 100, 100.0, 50, "50%", "0.5", "", None]
        for p in cases:
            result = normalize_probability(p)
            self.assertGreaterEqual(result, 0.0,
                f"normalize_probability({p!r}) = {result} < 0")
            self.assertLessEqual(result, 1.0,
                f"normalize_probability({p!r}) = {result} > 1")

    def test_calibrate_prediction_stays_in_range(self):
        """calibrate_prediction must return p_model in [0, 1] for any input."""
        from signal_scorer import calibrate_prediction
        test_cases = [
            (0.01, 0.01), (0.50, 0.50), (0.99, 0.99),
            (0.001, 0.001), (0.999, 0.999),
            (0.05, 0.50), (0.50, 0.05),
            (0.30, 0.10, 0.35),  # with metaculus_prob
            (0.10, 0.30, 0.50),
        ]
        for case in test_cases:
            p_model, market_price = case[0], case[1]
            meta_prob = case[2] if len(case) > 2 else None
            result, dampened = calibrate_prediction(p_model, market_price, meta_prob)
            self.assertGreaterEqual(result, 0.0,
                f"calibrate_prediction{case} = {result} < 0")
            self.assertLessEqual(result, 1.0,
                f"calibrate_prediction{case} = {result} > 1")


class TestSignalScoreInvariants(unittest.TestCase):
    """Signal score must ALWAYS be non-negative and finite."""

    def test_signal_score_non_negative(self):
        """_compute_signal_score must return score >= 0 for any input."""
        from signal_scorer import _compute_signal_score
        # Extreme edge cases
        edge_cases = [
            # (p_model, price, factors, volume, ttl, cluster)
            (0.01, 0.001, [], 0, 0, "geopolitical"),
            (0.99, 0.99, [], 0, 0, "other"),
            (0.50, 0.01, [], 1_000_000, 1, "ai_tech"),
            (0.50, 0.50, [], 1, 100_000, "election"),
        ]
        for p_model, price, factors, vol, ttl, cluster in edge_cases:
            try:
                result = _compute_signal_score(
                    p_model, price, factors, vol, ttl, cluster,
                    slug="test", question="test",
                )
                score = result[0]
                self.assertGreaterEqual(
                    score, 0,
                    f"signal_score negative for p={p_model}, price={price}, vol={vol}, ttl={ttl}"
                )
                self.assertFalse(
                    score != score,  # NaN check
                    f"signal_score is NaN for p={p_model}, price={price}"
                )
            except (ZeroDivisionError, ValueError) as e:
                self.fail(f"_compute_signal_score crashed for edge case: {e}")

    def test_prob_ratio_non_negative(self):
        """prob_ratio must be >= 0 for any p_model / price combination."""
        from signal_scorer import _compute_signal_score
        for p_model in [0.01, 0.05, 0.10, 0.30, 0.50, 0.99]:
            for price in [0.01, 0.05, 0.10, 0.30, 0.50]:
                result = _compute_signal_score(
                    p_model, price, [], 100_000, 720, "other",
                    slug="test", question="test",
                )
                prob_ratio = result[1]
                self.assertGreaterEqual(
                    prob_ratio, 0,
                    f"prob_ratio negative: p_model={p_model}, price={price}"
                )


class TestManifoldFuzzyInvariants(unittest.TestCase):
    """Fuzzy matching must produce consistent scores."""

    def test_identical_strings_score_max(self):
        from metaforecast import _fuzzy_score
        score = _fuzzy_score("Will AI happen?", "Will AI happen?")
        self.assertGreaterEqual(score, 80,
            msg=f"Identical strings should score >= 80, got {score}")

    def test_unrelated_strings_score_low(self):
        from metaforecast import _fuzzy_score
        score = _fuzzy_score("Will AI happen?", "What color is the sky?")
        self.assertLess(score, 60,
            msg=f"Unrelated strings should score < 60, got {score}")

    def test_score_symmetric(self):
        """Fuzzy score should be roughly symmetric."""
        from metaforecast import _fuzzy_score
        s1 = _fuzzy_score("Will Bitcoin reach $200k?", "Bitcoin price $200000")
        s2 = _fuzzy_score("Bitcoin price $200000", "Will Bitcoin reach $200k?")
        self.assertAlmostEqual(s1, s2, delta=15,
            msg=f"Fuzzy scores not symmetric: {s1} vs {s2}")


class TestMetaforecastInvariants(unittest.TestCase):
    """Metaforecast matching must produce consistent results."""

    def test_weighted_average_in_probability_range(self):
        """Consensus probability must be in [0, 1] for any match combination."""
        from metaforecast import get_metaforecast_forecast
        # Mock with extreme probabilities
        import metaforecast
        mock_questions = [
            {"title": "Test question unique xyzzy",
             "platform": "Manifold Markets",
             "options": {"Yes": 0.99},
             "qualityIndicators": {"numForecasts": 100, "stars": 3}},
        ]
        with patch.object(type(metaforecast._cache), 'questions', mock_questions):
            result = get_metaforecast_forecast("Test question unique xyzzy")
            if result.get("found"):
                prob = result["probability"]
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_no_match_returns_not_found(self):
        """Garbage query must return {found: False}."""
        from metaforecast import get_metaforecast_forecast
        import metaforecast
        # Mock questions property to return empty list (property auto-reloads otherwise)
        with patch.object(type(metaforecast._cache), 'questions', []):
            result = get_metaforecast_forecast("zzzqqqxxx123 nonexistent garbage")
            self.assertFalse(result.get("found"))


class TestDateTimeInvariants(unittest.TestCase):
    """Date parsing and comparison must handle edge cases."""

    def test_none_dates_dont_match(self):
        from metaculus import dates_match
        self.assertFalse(dates_match(None, None))
        self.assertFalse(dates_match("2025-01-01", None))
        self.assertFalse(dates_match(None, "2025-01-01"))

    def test_far_apart_dates_dont_match(self):
        from metaculus import dates_match
        self.assertFalse(dates_match("2025-01-01T00:00:00Z", "2030-01-01T00:00:00Z"))

    def test_close_dates_match(self):
        from metaculus import dates_match
        self.assertTrue(dates_match("2025-01-01T00:00:00Z", "2025-01-03T00:00:00Z"))

    def test_garbage_dates_dont_crash(self):
        from metaculus import dates_match
        # Must not crash on garbage input
        self.assertFalse(dates_match("garbage", "also garbage"))
        self.assertFalse(dates_match("", ""))


# Helper import
from unittest.mock import patch


if __name__ == "__main__":
    unittest.main()
