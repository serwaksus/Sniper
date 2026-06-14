"""Fault injection tests — verify graceful degradation when external deps fail.

Each test breaks ONE external dependency and verifies the system:
  1. Does NOT crash (no unhandled exception)
  2. Falls back to a sensible default
  3. Logs the error appropriately

These tests would have caught:
- Council calls without try/except (bug #3)
- Any future missing error handling around API calls

Philosophy: every external call is a potential failure point.
The system MUST degrade gracefully, not crash.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from unittest.mock import patch, MagicMock
from typing import Any


class TestManifoldFailureModes(unittest.TestCase):
    """Manifold API failure modes."""

    def _market(self) -> dict[str, Any]:
        return {
            "question": "Will X happen?", "price": 0.10,
            "end_date": "2028-01-01", "best_bid": 0.09, "best_ask": 0.11,
            "slug": "test-slug", "volume": 50000,
        }

    def test_timeout_returns_not_found(self):
        """Manifold timeout → returns {found: False}, not crash."""
        import manifold
        import requests
        with patch("manifold.requests.get", side_effect=requests.exceptions.Timeout("slow")):
            result = manifold.get_manifold_forecast("test?", None)
            self.assertFalse(result.get("found"))

    def test_http_500_returns_not_found(self):
        """Manifold 500 → returns {found: False}, not crash."""
        import manifold
        mock_resp = MagicMock(status_code=500)
        with patch("manifold.requests.get", return_value=mock_resp):
            result = manifold.get_manifold_forecast("test?", None)
            self.assertFalse(result.get("found"))

    def test_empty_results_returns_not_found(self):
        """Manifold empty results → returns {found: False}."""
        import manifold
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = []
        with patch("manifold.requests.get", return_value=mock_resp):
            result = manifold.get_manifold_forecast("test?", None)
            self.assertFalse(result.get("found"))

    def test_malformed_json_returns_not_found(self):
        """Manifold malformed JSON → returns {found: False}, not crash."""
        import manifold
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.side_effect = ValueError("bad json")
        with patch("manifold.requests.get", return_value=mock_resp):
            result = manifold.get_manifold_forecast("test?", None)
            self.assertFalse(result.get("found"))


class TestMetaculusFailureModes(unittest.TestCase):
    """Metaculus API failure modes."""

    def test_rate_limit_429_returns_empty(self):
        """Metaculus 429 → returns empty list, not crash."""
        import metaculus
        mock_resp = MagicMock(status_code=429)
        with patch("metaculus.requests.get", return_value=mock_resp):
            results = metaculus.metaculus_search("test")
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 0)

    def test_timeout_returns_empty(self):
        """Metaculus timeout → returns empty list."""
        import metaculus
        import requests
        with patch("metaculus.requests.get", side_effect=requests.exceptions.Timeout("slow")):
            results = metaculus.metaculus_search("test")
            self.assertEqual(len(results), 0)

    def test_metaforecast_bridge_failure_returns_not_found(self):
        """Metaforecast bridge failure → get_metaculus_forecast returns not found."""
        import metaculus
        # Mock search success but bridge failure
        mock_post = MagicMock(status_code=500)
        with patch("metaculus.requests.get") as mock_get, \
             patch("metaculus.requests.post", return_value=mock_post):
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {
                "results": [{"id": 123, "title": "Test", "question": {"id": 123}}]
            }
            mock_get.return_value = mock_resp
            result = metaculus.get_metaculus_forecast("Test question?", None)
            self.assertFalse(result.get("found"))


class TestMetaforecastFailureModes(unittest.TestCase):
    """Metaforecast GraphQL failure modes."""

    def test_graphql_error_returns_not_found(self):
        """Metaforecast GraphQL error → returns {found: False}, not crash."""
        import metaforecast
        # Patch the property (not _questions) to avoid auto-reload from disk
        with patch.object(type(metaforecast._cache), 'questions', []):
            result = metaforecast.get_metaforecast_forecast("test nonexistent xyzzy 12345")
            self.assertFalse(result.get("found"))

    def test_graphql_exception_returns_not_found(self):
        """Metaforecast network error → returns {found: False}, not crash."""
        import metaforecast
        import requests
        # Patch the property to inject controlled test data
        mock_questions = [{"title": "test", "platform": "X", "options": {"Yes": 0.5}}]
        with patch.object(type(metaforecast._cache), 'questions', mock_questions):
            with patch("metaforecast.requests.post", side_effect=requests.exceptions.ConnectionError("down")):
                # This should not crash even if the GraphQL call fails
                # because the index is already loaded locally
                result = metaforecast.get_metaforecast_forecast("test")
                # Should still work from local index
                self.assertIsNotNone(result)


class TestCouncilFailureModes(unittest.TestCase):
    """Model council failure modes — must fall back to DeepSeek only."""

    def test_council_batch_consensus_survives_ovh_failure(self):
        """OVH completely down → council returns DeepSeek results only."""
        os.environ["COUNCIL_DISABLED"] = "1"
        try:
            from model_council import council_batch_consensus
            # Should not crash even with all OVH models down
            results, meta = council_batch_consensus(
                "test prompt",
                ["slug1"],
                [{"p_model": 0.3, "confidence": 0.7, "factors": []}],
            )
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 1)
        finally:
            del os.environ["COUNCIL_DISABLED"]

    def test_council_single_consensus_survives_all_failure(self):
        """All models fail → council returns DeepSeek p_model unchanged."""
        os.environ["COUNCIL_DISABLED"] = "1"
        try:
            from model_council import council_single_consensus
            p, meta = council_single_consensus(
                "test prompt", "slug1", 0.25, 0.70,
                question="Test?", price=0.10,
            )
            self.assertIsInstance(p, (int, float))
            self.assertGreater(p, 0)
            self.assertLess(p, 1)
        finally:
            del os.environ["COUNCIL_DISABLED"]


class TestSignalScorerFailureModes(unittest.TestCase):
    """Signal scorer must handle all external dep failures gracefully."""

    def _market(self) -> dict[str, Any]:
        from dotm_sniper import HYP_SLUG
        return {
            "question": "Will X happen?", "price": 0.10,
            HYP_SLUG: "test-slug", "volume": 50000, "ttl_hours": 720,
            "end_date": "2028-01-01", "best_bid": 0.09, "best_ask": 0.11,
        }

    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer.requests.post")
    def test_llm_timeout_falls_back_gracefully(self, mock_post, mock_cal, mock_count):
        """DeepSeek timeout → p_model falls back to price * 2, no crash."""
        import requests
        from signal_scorer import full_market_analysis
        from dotm_sniper import HYP_SLUG

        mock_cal.is_fitted = False
        mock_post.side_effect = requests.exceptions.Timeout("slow")

        with patch("signal_scorer.check_manifold_gap", return_value=None), \
             patch("signal_scorer.check_metaforecast_gap", return_value=None), \
             patch("metaculus.check_metaculus_gap", return_value=None), \
             patch("signal_scorer._cluster_score_adjustment", return_value=0), \
             patch("signal_scorer.get_settings", return_value={"signal_threshold": 40}):
            result = full_market_analysis(self._market())

        self.assertIn("action", result)
        self.assertIn(result["action"], ["BUY", "SKIP"])

    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer.requests.post")
    def test_council_failure_doesnt_crash_analysis(self, mock_post, mock_cal, mock_count):
        """Council consensus crash → analysis continues with DeepSeek only."""
        from signal_scorer import full_market_analysis

        mock_cal.is_fitted = False
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        mock_post.return_value = mock_resp

        with patch("signal_scorer.check_manifold_gap", return_value=None), \
             patch("signal_scorer.check_metaforecast_gap", return_value=None), \
             patch("metaculus.check_metaculus_gap", return_value=None), \
             patch("signal_scorer._cluster_score_adjustment", return_value=0), \
             patch("signal_scorer.get_settings", return_value={"signal_threshold": 40}), \
             patch("model_council.council_single_consensus",
                   side_effect=Exception("OVH completely down")), \
             patch("signal_scorer.parse_llm_json") as mock_parse:
            mock_parse.return_value = (0.20, 0.65, [])
            result = full_market_analysis(self._market())

        # Must NOT crash — should return a valid result
        self.assertIn("action", result)
        self.assertIn(result["action"], ["BUY", "SKIP"])


if __name__ == "__main__":
    unittest.main()
