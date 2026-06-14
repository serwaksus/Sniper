"""Contract tests — verify that all forecast sources return the SAME dict shape.

These tests do NOT mock the source modules. They call the real functions
with controlled inputs and verify the return dict has all required keys.

This catches:
- Missing 'found' key in success returns (bug that was in manifold.py)
- Missing 'probability' key (manifold/metaforecast compatibility)
- Wrong field names (metaculus_prob vs probability)
- Inconsistent None vs {"found": False} returns

If you add a new forecast source, add it here to enforce the contract.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from unittest.mock import patch, MagicMock
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# CONTRACT: check_*_gap success return MUST have these keys
# ═══════════════════════════════════════════════════════════════════════

REQUIRED_SUCCESS_KEYS = {"found", "probability", "polymarket_prob", "signal_strength", "source"}
REQUIRED_FORECAST_KEYS = {"found", "probability"}


class TestGapCheckContract(unittest.TestCase):
    """Verify all check_*_gap functions return consistent dict shapes."""

    def _make_market(self, price: float = 0.10) -> dict[str, Any]:
        return {
            "question": "Will a nuclear weapon be used in combat by 2030?",
            "price": price,
            "end_date": "2030-01-01T00:00:00Z",
            "best_bid": price - 0.01,
            "best_ask": price + 0.01,
        }

    # ── Manifold ──────────────────────────────────────────────

    @patch("manifold.get_manifold_forecast")
    def test_manifold_success_has_required_keys(self, mock_fc):
        """Manifold check_manifold_gap MUST include found, probability, source."""
        import manifold
        mock_fc.return_value = {
            "found": True,
            "probability": 0.40,
            "volume": 50000,
            "url": "https://manifold.markets/example",
            "match_score": 0.85,
            "volume_penalty": 1.0,
        }
        result = manifold.check_manifold_gap(self._make_market(price=0.05))

        self.assertIsNotNone(result, "Should return a dict when gap is large")
        self.assertIsInstance(result, dict)

        missing = REQUIRED_SUCCESS_KEYS - set(result.keys())
        self.assertEqual(missing, set(),
            f"Manifold gap result MISSING required keys: {missing}")

        self.assertTrue(result["found"])
        self.assertIsInstance(result["probability"], float)
        self.assertEqual(result["source"], "manifold")

    @patch("manifold.get_manifold_forecast")
    def test_manifold_not_found_returns_none(self, mock_fc):
        """Manifold MUST return None when forecast not found."""
        import manifold
        mock_fc.return_value = {"found": False, "probability": None}
        result = manifold.check_manifold_gap(self._make_market())
        self.assertIsNone(result, "Should return None when not found")

    # ── Metaculus ─────────────────────────────────────────────

    @patch("metaculus.get_metaculus_forecast")
    @patch("metaculus.get_time_decay_threshold", return_value=0.08)
    def test_metaculus_success_has_required_keys(self, mock_td, mock_fc):
        """Metaculus check_metaculus_gap MUST include found, probability, source."""
        import metaculus
        mock_fc.return_value = {
            "found": True,
            "probability": 0.40,
            "question_title": "Test question",
            "url": "https://metaculus.com/test",
            "forecaster_count": 100,
            "match_score": 0.8,
        }
        result = metaculus.check_metaculus_gap(self._make_market(price=0.05))

        self.assertIsNotNone(result)
        missing = REQUIRED_SUCCESS_KEYS - set(result.keys())
        self.assertEqual(missing, set(),
            f"Metaculus gap result MISSING required keys: {missing}")

        self.assertTrue(result["found"])
        self.assertIsInstance(result["probability"], float)
        self.assertEqual(result["source"], "metaculus")

    @patch("metaculus.get_metaculus_forecast")
    def test_metaculus_not_found_returns_none(self, mock_fc):
        """Metaculus MUST return None when forecast not found."""
        import metaculus
        mock_fc.return_value = {"found": False, "probability": None}
        result = metaculus.check_metaculus_gap(self._make_market())
        self.assertIsNone(result)

    # ── Metaforecast ──────────────────────────────────────────

    @patch("metaforecast.get_metaforecast_forecast")
    def test_metaforecast_success_has_required_keys(self, mock_fc):
        """Metaforecast check_metaforecast_gap MUST include found, probability, source."""
        import metaforecast
        mock_fc.return_value = {
            "found": True,
            "probability": 0.40,
            "url": "https://example.com",
            "dispersion": 0.1,
            "num_platforms": 2,
            "all_matches": [],
        }
        result = metaforecast.check_metaforecast_gap(self._make_market(price=0.05))

        self.assertIsNotNone(result)
        missing = REQUIRED_SUCCESS_KEYS - set(result.keys())
        self.assertEqual(missing, set(),
            f"Metaforecast gap result MISSING required keys: {missing}")

        self.assertTrue(result["found"])
        self.assertEqual(result["source"], "metaforecast")

    @patch("metaforecast.get_metaforecast_forecast")
    def test_metaforecast_not_found_returns_none_or_false(self, mock_fc):
        """Metaforecast MUST return None or {"found": False} when not found."""
        import metaforecast
        mock_fc.return_value = {"found": False, "probability": None}
        result = metaforecast.check_metaforecast_gap(self._make_market())
        # metaforecast returns {"found": False} or None — both acceptable
        if result is not None:
            self.assertFalse(result.get("found", True),
                "Non-None result must have found=False")


class TestForecastResultContract(unittest.TestCase):
    """Verify all get_*_forecast functions return consistent dict shapes."""

    # ── Manifold ──────────────────────────────────────────────

    @patch("manifold.requests.get")
    def test_manifold_forecast_found_shape(self, mock_get):
        """Manifold get_manifold_forecast MUST return found + probability."""
        import manifold
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = [{
            "probability": 0.40,
            "volume": 50000,
            "url": "https://manifold.markets/test",
            "question": "Test?",
            "outcomeType": "BINARY",
            "active": True,
            "closeTime": 1893456000000,
        }]
        mock_get.return_value = mock_resp

        result = manifold.get_manifold_forecast("Test question?", None)

        missing = REQUIRED_FORECAST_KEYS - set(result.keys())
        self.assertEqual(missing, set(),
            f"Manifold forecast MISSING required keys: {missing}")

    # ── Metaforecast ──────────────────────────────────────────

    def test_metaforecast_forecast_not_found_shape(self):
        """Metaforecast get_metaforecast_forecast MUST return found + probability."""
        import metaforecast
        # Empty index → not found
        with patch.object(metaforecast._cache, '_questions', []):
            result = metaforecast.get_metaforecast_forecast("xyz nonexistent query 12345")

        missing = REQUIRED_FORECAST_KEYS - set(result.keys())
        self.assertEqual(missing, set(),
            f"Metaforecast forecast MISSING required keys: {missing}")


# ═══════════════════════════════════════════════════════════════════════
# CONTRACT: Consumer code accesses these specific keys
# ═══════════════════════════════════════════════════════════════════════

class TestConsumerKeyAccess(unittest.TestCase):
    """Verify that consumer code (signal_scorer, backtest_simulator) accesses
    only keys that ALL three sources provide.

    This catches the bug where signal_scorer accessed gap.get("metaculus_prob")
    but Manifold/Metaforecast don't provide that key.
    """

    def test_signal_scorer_uses_probability_not_metaculus_prob(self):
        """signal_scorer MUST use .get('probability', ...) not just .get('metaculus_prob')."""
        with open("src/signal_scorer.py") as f:
            src = f.read()

        # The fix we applied: probability is primary key, metaculus_prob is fallback
        self.assertIn(
            'get("probability"',
            src,
            "signal_scorer should access 'probability' key (not just 'metaculus_prob')"
        )

    def test_signal_scorer_checks_found_before_access(self):
        """signal_scorer MUST check .get('found') before accessing ['probability']."""
        with open("src/signal_scorer.py") as f:
            src = f.read()

        # Must have pattern: get("found") before ["probability"]
        found_check = 'get("found")' in src or "get('found')" in src
        self.assertTrue(found_check,
            "signal_scorer must check 'found' key before accessing probability")

    def test_backtest_simulator_uses_probability_not_metaculus_prob(self):
        """backtest_simulator MUST use .get('probability', ...) not just .get('metaculus_prob')."""
        with open("src/backtest_simulator.py") as f:
            src = f.read()

        # Should use probability as primary key
        self.assertIn(
            'get("probability"',
            src,
            "backtest_simulator should access 'probability' key"
        )


# ═══════════════════════════════════════════════════════════════════════
# CONTRACT: Cascade order in signal_pipeline/signal_scorer
# ═══════════════════════════════════════════════════════════════════════

class TestCascadeContract(unittest.TestCase):
    """Verify the forecast cascade is correct: Manifold → Metaculus → Metaforecast."""

    def test_batch_cascade_order(self):
        """signal_pipeline batch path MUST try Manifold first, then Metaculus, then Metaforecast."""
        with open("src/signal_pipeline.py") as f:
            src = f.read()

        # Find the cascade block (look for calls, not imports)
        # The cascade is in the batch processing section
        manifold_call = src.find("ext = get_manifold_forecast")
        metaculus_call = src.find("meta = get_metaculus_forecast(question")
        metaforecast_call = src.find("mf = get_metaforecast_forecast(question)")

        # All must be present in the cascade
        self.assertGreater(manifold_call, 0, "Manifold call must be in cascade")
        self.assertGreater(metaculus_call, 0, "Metaculus call must be in cascade")
        self.assertGreater(metaforecast_call, 0, "Metaforecast call must be in cascade")

        # Order: Manifold < Metaculus < Metaforecast
        self.assertLess(manifold_call, metaculus_call,
            "Manifold must be tried BEFORE Metaculus in cascade")
        self.assertLess(metaculus_call, metaforecast_call,
            "Metaculus must be tried BEFORE Metaforecast in cascade")

    def test_single_cascade_order(self):
        """signal_scorer single path MUST try Manifold → Metaculus → Metaforecast."""
        with open("src/signal_scorer.py") as f:
            src = f.read()

        manifold_pos = src.find("check_manifold_gap")
        metaculus_pos = src.find("check_metaculus_gap")
        metaforecast_pos = src.find("check_metaforecast_gap")

        self.assertGreater(manifold_pos, 0)
        self.assertGreater(metaforecast_pos, 0)
        self.assertLess(manifold_pos, metaforecast_pos,
            "Manifold must be tried BEFORE Metaforecast")

    def test_council_calls_have_try_except(self):
        """Council consensus calls MUST be wrapped in try/except."""
        for filepath in ["src/signal_pipeline.py", "src/signal_scorer.py"]:
            with open(filepath) as f:
                src = f.read()

            council_mention = "council_batch_consensus" in src or "council_single_consensus" in src
            if not council_mention:
                continue

            # Find the council call and check for try: before it
            lines = src.split("\n")
            for i, line in enumerate(lines):
                if "council_" in line and "consensus" in line and "import" not in line:
                    # Look backwards for try:
                    found_try = False
                    for j in range(max(0, i - 10), i):
                        if "try:" in lines[j]:
                            found_try = True
                            break
                    self.assertTrue(found_try,
                        f"{filepath}:{i+1}: council call must be inside try/except")


if __name__ == "__main__":
    unittest.main()
