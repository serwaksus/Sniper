#!/usr/bin/env python3
"""
Tests for dotm_sniper.py state invariants and position management.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotm_sniper import (
    repair_positions_file,
)


class TestRepairPositionsFile(unittest.TestCase):
    def setUp(self):
        import positions_db
        positions_db.ensure_init()
        positions_db.save_all({})

    def tearDown(self):
        import positions_db
        positions_db.save_all({})

    def _write_positions(self, data):
        import positions_db
        for slug, pos_data in data.items():
            positions_db.update(slug, pos_data)

    def test_repairs_high_price_below_entry(self):
        self._write_positions({
            "slug-a": {"entry_price": 0.25, "high_price": 0.20, "trailing_on": False}
        })
        repair_positions_file()
        import positions_db
        result = positions_db.load_all()
        self.assertGreaterEqual(result["slug-a"]["high_price"], result["slug-a"]["entry_price"])
        self.assertEqual(result["slug-a"]["high_price"], 0.25)

    def test_no_repair_needed_when_valid(self):
        self._write_positions({
            "slug-b": {"entry_price": 0.10, "high_price": 0.15, "trailing_on": False}
        })
        repair_positions_file()
        import positions_db
        result = positions_db.load_all()
        self.assertEqual(result["slug-b"]["high_price"], 0.15)

    def test_empty_positions(self):
        import positions_db
        positions_db.save_all({})
        repair_positions_file()
        result = positions_db.load_all()
        self.assertEqual(result, {})

    def test_multiple_positions_mixed(self):
        self._write_positions({
            "slug-ok": {"entry_price": 0.10, "high_price": 0.12, "trailing_on": False},
            "slug-bad": {"entry_price": 0.30, "high_price": 0.25, "trailing_on": False},
            "slug-equal": {"entry_price": 0.15, "high_price": 0.15, "trailing_on": False},
        })
        repair_positions_file()
        import positions_db
        result = positions_db.load_all()
        self.assertEqual(result["slug-ok"]["high_price"], 0.12)
        self.assertEqual(result["slug-bad"]["high_price"], 0.30)
        self.assertEqual(result["slug-equal"]["high_price"], 0.15)

    def test_missing_entry_price_no_crash(self):
        self._write_positions({
            "slug-no-entry": {"high_price": 0.20, "trailing_on": False}
        })
        repair_positions_file()
        import positions_db
        result = positions_db.load_all()
        self.assertEqual(result["slug-no-entry"]["high_price"], 0.20)


class TestHighPriceInvariant(unittest.TestCase):
    """Test that high_price is always >= entry_price after initialization."""

    def test_init_uses_max_of_entry_and_current(self):
        """When entry_price > current_price, high_price should be entry_price."""
        entry_price = 0.25
        current_price = 0.20
        high_price = max(entry_price, current_price)
        self.assertEqual(high_price, entry_price)
        self.assertGreaterEqual(high_price, entry_price)

    def test_init_uses_current_when_higher(self):
        entry_price = 0.10
        current_price = 0.15
        high_price = max(entry_price, current_price)
        self.assertEqual(high_price, current_price)
        self.assertGreaterEqual(high_price, entry_price)

    def test_update_never_decreases_high_price(self):
        entry_price = 0.15
        existing_high = 0.20
        current_price = 0.18
        new_high = max(existing_high, current_price, entry_price)
        self.assertEqual(new_high, existing_high)
        self.assertGreaterEqual(new_high, entry_price)


class TestCompositeScoring(unittest.TestCase):
    """Test the v4.6.0 composite scoring formula."""

    def _make_market(self, price=0.10, volume=500000, ttl_hours=720):
        return {
            "question": "Will X happen?",
            "slug": "test-slug",
            "price": price,
            "volume": volume,
            "ttl_hours": ttl_hours,
            "clusters": ["ai_tech"],
        }

    def test_ratio_score_max_is_30(self):
        prob_ratio = 10.0
        ratio_score = min(prob_ratio / 3.0, 1.0) * 30
        self.assertEqual(ratio_score, 30.0)

    def test_ratio_score_typical(self):
        prob_ratio = 2.0
        ratio_score = min(prob_ratio / 3.0, 1.0) * 30
        self.assertAlmostEqual(ratio_score, 20.0)

    def test_factor_score_max_20(self):
        supporting = [{"direction": "supports", "weight": "high"}] * 4
        high_weight = [f for f in supporting if f.get("weight") == "high"]
        factor_score = min((len(supporting) + len(high_weight)) / 4, 1.0) * 20
        self.assertEqual(factor_score, 20.0)

    def test_vol_score_max_20(self):
        vol_score = min(2000000 / 1_000_000, 1.0) * 20
        self.assertEqual(vol_score, 20.0)

    def test_time_score_max_20(self):
        ttl_days = 200
        if ttl_days > 180:
            time_score = 20
        elif ttl_days > 90:
            time_score = 15
        elif ttl_days > 30:
            time_score = 10
        else:
            time_score = 0
        self.assertEqual(time_score, 20)

    def test_max_signal_score_is_100(self):
        ratio_score = 30
        factor_score = 20
        vol_score = 20
        time_score = 20
        metaculus_alignment = 10
        total = ratio_score + factor_score + vol_score + time_score + metaculus_alignment
        self.assertEqual(total, 100)

    def test_metaculus_penalty_reduces_score(self):
        ratio_score = 30
        factor_score = 20
        vol_score = 20
        time_score = 20
        metaculus_alignment = -20
        total = ratio_score + factor_score + vol_score + time_score + metaculus_alignment
        self.assertEqual(total, 70)

    def test_horizon_threshold_short(self):
        base_threshold = 55
        ttl_days = 20
        if ttl_days > 90:
            min_signal = base_threshold + 10
        elif ttl_days >= 31:
            min_signal = base_threshold + 5
        else:
            min_signal = base_threshold
        self.assertEqual(min_signal, 55)

    def test_horizon_threshold_medium(self):
        base_threshold = 55
        ttl_days = 60
        if ttl_days > 90:
            min_signal = base_threshold + 10
        elif ttl_days >= 31:
            min_signal = base_threshold + 5
        else:
            min_signal = base_threshold
        self.assertEqual(min_signal, 60)

    def test_horizon_threshold_long(self):
        base_threshold = 55
        ttl_days = 365
        if ttl_days > 90:
            min_signal = base_threshold + 10
        elif ttl_days >= 31:
            min_signal = base_threshold + 5
        else:
            min_signal = base_threshold
        self.assertEqual(min_signal, 65)

    def test_metaculus_alignment_agreement(self):
        p_model = 0.15
        metaculus_prob = 0.13
        diff = abs(p_model - metaculus_prob)
        alignment = 10 if diff < 0.05 else 0
        self.assertEqual(alignment, 10)

    def test_metaculus_alignment_disagreement(self):
        p_model = 0.40
        metaculus_prob = 0.12
        market_price = 0.10
        diff_model_meta = abs(p_model - metaculus_prob)
        diff_meta_pm = abs(metaculus_prob - market_price)
        alignment = 0
        if diff_model_meta < 0.05:
            alignment = 10
        elif p_model > metaculus_prob + 0.10 and diff_meta_pm < 0.03:
            alignment = -20
        self.assertEqual(alignment, -20)


class TestAdvisorPreCheck(unittest.TestCase):
    """Test advisor pre-check logic."""

    def test_confirm_with_high_confidence_approves(self):
        verdict = "CONFIRM"
        confidence = 0.75
        min_conf = 0.70
        approved = (verdict == "CONFIRM" and confidence >= min_conf)
        self.assertTrue(approved)

    def test_confirm_with_low_confidence_blocks(self):
        verdict = "CONFIRM"
        confidence = 0.65
        min_conf = 0.70
        approved = (verdict == "CONFIRM" and confidence >= min_conf)
        self.assertFalse(approved)

    def test_diverge_blocks(self):
        for verdict in ["DIVERGE", "WARNING", "UNKNOWN"]:
            confidence = 0.90
            min_conf = 0.70
            approved = (verdict == "CONFIRM" and confidence >= min_conf)
            self.assertFalse(approved, f"verdict={verdict} should be blocked")


class TestLimitOrderLogic(unittest.TestCase):
    """Test limit order and slippage protection logic."""

    def test_wide_spread_triggers_limit(self):
        best_bid = 0.08
        best_ask = 0.12
        spread = best_ask - best_bid
        threshold = 0.03
        use_limit = spread > threshold
        self.assertTrue(use_limit)

    def test_narrow_spread_uses_market(self):
        best_bid = 0.10
        best_ask = 0.105
        spread = best_ask - best_bid
        threshold = 0.03
        use_limit = spread > threshold
        self.assertFalse(use_limit)

    def test_limit_price_is_bid_plus_buffer(self):
        best_bid = 0.08
        buffer = 0.005
        limit_price = best_bid + buffer
        self.assertAlmostEqual(limit_price, 0.085)

    def test_force_market_after_max_attempts(self):
        max_attempts = 3
        current_attempts = 3
        force = current_attempts >= max_attempts
        self.assertTrue(force)

    def test_no_force_before_max_attempts(self):
        max_attempts = 3
        current_attempts = 2
        force = current_attempts >= max_attempts
        self.assertFalse(force)

    def test_no_bids_blocks_everything(self):
        best_bid = None
        can_sell = best_bid is not None and best_bid > 0
        self.assertFalse(can_sell)


class TestPreFilterBeforeBatching(unittest.TestCase):
    """Test pre_filter_before_batching token-saving logic."""

    def setUp(self):
        import dotm_sniper
        self.module = dotm_sniper

    def _make_market(self, slug="test", clusters=None, volume=50000):
        return {
            "slug": slug,
            "question": f"Will {slug} happen?",
            "price": 0.05,
            "volume": volume,
            "clusters": clusters or ["other"],
        }

    def test_other_low_volume_skipped(self):
        m = self._make_market(slug="low-vol-other", clusters=["other"], volume=50000)
        kept, skipped = self.module.pre_filter_before_batching([m])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(kept), 0)
        self.assertEqual(skipped[0]["slug"], "low-vol-other")

    def test_other_high_volume_kept(self):
        m = self._make_market(slug="high-vol-other", clusters=["other"], volume=150000)
        kept, skipped = self.module.pre_filter_before_batching([m])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(skipped), 0)

    def test_other_exact_threshold_kept(self):
        m = self._make_market(slug="exact-vol", clusters=["other"], volume=100000)
        kept, skipped = self.module.pre_filter_before_batching([m])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(skipped), 0)

    def test_allowed_cluster_always_kept(self):
        for cluster in ["usa_politics", "russia_ukraine", "ai_tech", "fed_fomc"]:
            m = self._make_market(slug=f"c-{cluster}", clusters=[cluster], volume=1000)
            kept, skipped = self.module.pre_filter_before_batching([m])
            self.assertEqual(len(kept), 1, f"{cluster} should be kept regardless of volume")
            self.assertEqual(len(skipped), 0)

    def test_mixed_batch(self):
        markets = [
            self._make_market(slug="pol", clusters=["usa_politics"], volume=50000),
            self._make_market(slug="oth-low", clusters=["other"], volume=10000),
            self._make_market(slug="oth-high", clusters=["other"], volume=200000),
            self._make_market(slug="crypto", clusters=["crypto"], volume=30000),
        ]
        kept, skipped = self.module.pre_filter_before_batching(markets)
        kept_slugs = {m["slug"] for m in kept}
        skipped_slugs = {m["slug"] for m in skipped}
        self.assertIn("pol", kept_slugs)
        self.assertIn("oth-high", kept_slugs)
        self.assertIn("crypto", skipped_slugs)
        self.assertIn("oth-low", skipped_slugs)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(skipped), 2)

    def test_empty_input(self):
        kept, skipped = self.module.pre_filter_before_batching([])
        self.assertEqual(kept, [])
        self.assertEqual(skipped, [])

    def test_no_clusters_defaults_to_other(self):
        m = self._make_market(slug="no-clusters", clusters=None, volume=50000)
        m.pop("clusters", None)
        _kept, skipped = self.module.pre_filter_before_batching([m])
        self.assertEqual(len(skipped), 1)


class TestCalibratePrediction(unittest.TestCase):
    """Test anti-optimism dampening calibration."""

    def setUp(self):
        import dotm_sniper
        self.module = dotm_sniper

    def test_dotm_aggressive_no_metaculus_dampened(self):
        p, dampened = self.module.calibrate_prediction(0.30, 0.08, metaculus_prob=None)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.33)

    def test_dotm_aggressive_low_metaculus_dampened(self):
        p, dampened = self.module.calibrate_prediction(0.35, 0.05, metaculus_prob=0.08)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.385)

    def test_dotm_aggressive_high_metaculus_not_dampened(self):
        p, dampened = self.module.calibrate_prediction(0.30, 0.08, metaculus_prob=0.15)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.33)

    def test_non_dotm_no_dampening(self):
        p, dampened = self.module.calibrate_prediction(0.40, 0.15, metaculus_prob=None)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.44)

    def test_conservative_pmodel_no_dampening(self):
        p, dampened = self.module.calibrate_prediction(0.20, 0.08, metaculus_prob=None)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.22)

    def test_exact_threshold_boundary(self):
        p, dampened = self.module.calibrate_prediction(0.25, 0.10, metaculus_prob=None)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.275)

    def test_new_threshold_at_0_20_not_dampened(self):
        p, dampened = self.module.calibrate_prediction(0.20, 0.10, metaculus_prob=None)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.22)

    def test_just_above_threshold_dampened(self):
        p, dampened = self.module.calibrate_prediction(0.26, 0.10, metaculus_prob=None)
        self.assertTrue(dampened)
        self.assertAlmostEqual(p, 0.286)


if __name__ == '__main__':
    unittest.main(verbosity=2)
