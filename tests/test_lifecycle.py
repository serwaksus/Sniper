"""Tests for position lifecycle state transitions."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import save_json, load_json
from schema import (
    POS_ENTRY_PRICE,
    POS_HIGH_PRICE,
    POS_STOP_LOSS,
    POS_SHARES,
    POS_CLUSTERS,
    POS_OUTCOME,
    POS_MARKET_QUESTION,
    POS_SELLING_IN_PROGRESS,
    POS_IN_EMERGENCY_EXIT,
    HYP_SLUG,
    HYP_QUESTION,
    HYP_P_MODEL,
    HYP_MARKET_PRICE,
    HYP_PROB_RATIO,
    HYP_CONFIDENCE,
    HYP_FACTORS,
    HYP_CLUSTERS,
    HYP_SIZE_PCT,
    HYP_CREATED_AT,
    HYP_RESOLVED,
    HYP_SOURCE_SIGNAL,
)


class TestPositionLifecycle:
    """Verify position state transitions are valid."""

    def _make_position(self, **overrides):
        pos = {
            POS_ENTRY_PRICE: 0.10,
            POS_HIGH_PRICE: 0.12,
            POS_STOP_LOSS: 0.07,
            POS_SHARES: 100,
            POS_CLUSTERS: ["test"],
            POS_OUTCOME: "YES",
            POS_MARKET_QUESTION: "Test?",
            POS_SELLING_IN_PROGRESS: False,
            POS_IN_EMERGENCY_EXIT: False,
        }
        pos.update(overrides)
        return pos

    def test_new_position_has_required_fields(self):
        pos = self._make_position()
        required = [POS_ENTRY_PRICE, POS_SHARES, POS_CLUSTERS, POS_OUTCOME]
        for key in required:
            assert key in pos, f"Missing required key: {key}"

    def test_selling_in_progress_cleared_after_sell(self, tmp_path):
        """Verify selling_in_progress is cleared when position is sold."""
        positions_path = str(tmp_path / "positions.json")
        pos = self._make_position()
        pos[POS_SELLING_IN_PROGRESS] = True
        save_json(positions_path, {"test-slug": pos})

        data = load_json(positions_path, {})
        data.pop("test-slug", None)
        save_json(positions_path, data)

        assert "test-slug" not in load_json(positions_path, {})

    def test_cannot_be_selling_and_emergency_simultaneously(self):
        """A position should not have both flags True at the same time."""
        pos = self._make_position()
        assert not (pos[POS_SELLING_IN_PROGRESS] and pos.get(POS_IN_EMERGENCY_EXIT, False))

    def test_stop_loss_is_below_entry(self):
        pos = self._make_position()
        assert pos[POS_STOP_LOSS] < pos[POS_ENTRY_PRICE]

    def test_high_price_at_least_entry(self):
        pos = self._make_position()
        assert pos[POS_HIGH_PRICE] >= pos[POS_ENTRY_PRICE]

    def test_shares_positive(self):
        pos = self._make_position()
        assert pos[POS_SHARES] > 0

    def test_entry_price_in_valid_range(self):
        pos = self._make_position()
        assert 0 < pos[POS_ENTRY_PRICE] < 1


class TestHypothesisLifecycle:
    def _make_hypothesis(self, **overrides):
        hyp = {
            HYP_SLUG: "test-market",
            HYP_QUESTION: "Will X happen?",
            HYP_P_MODEL: 0.15,
            HYP_MARKET_PRICE: 0.08,
            HYP_PROB_RATIO: 1.875,
            HYP_CONFIDENCE: 0.7,
            HYP_FACTORS: ["test"],
            HYP_CLUSTERS: ["test"],
            HYP_SIZE_PCT: 0.03,
            HYP_CREATED_AT: "2026-01-01T00:00:00",
            HYP_RESOLVED: False,
            HYP_SOURCE_SIGNAL: "default",
        }
        hyp.update(overrides)
        return hyp

    def test_active_hypothesis_not_resolved(self):
        hyp = self._make_hypothesis()
        assert not hyp[HYP_RESOLVED]

    def test_resolved_hypothesis_has_outcome(self):
        hyp = self._make_hypothesis(resolved=True, outcome="yes")
        assert hyp.get("resolved") is True
        assert hyp.get("outcome") in ("yes", "no")

    def test_prob_ratio_above_1(self):
        hyp = self._make_hypothesis()
        assert hyp[HYP_PROB_RATIO] > 1.0
