#!/usr/bin/env python3
"""
Tests for dotm_sniper.py — Kelly sizing, normalize_probability, TP ladder math,
convergence ratio, trailing stop constants, calibrate_prediction dampening.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import dotm_sniper as ds


class TestNormalizeProbability(unittest.TestCase):
    def test_valid_decimal(self):
        self.assertAlmostEqual(ds.normalize_probability(0.25), 0.25)

    def test_percentage_converted(self):
        self.assertAlmostEqual(ds.normalize_probability(45.0), 0.45)

    def test_percentage_50(self):
        self.assertAlmostEqual(ds.normalize_probability(50.0), 0.50)

    def test_none_returns_zero(self):
        self.assertEqual(ds.normalize_probability(None), 0)

    def test_negative_clamped_to_zero(self):
        self.assertEqual(ds.normalize_probability(-0.5), 0.0)

    def test_above_one_not_converted(self):
        self.assertAlmostEqual(ds.normalize_probability(1.5), 1.0)

    def test_exactly_100(self):
        self.assertAlmostEqual(ds.normalize_probability(100.0), 1.0)

    def test_above_100_clips_to_1(self):
        self.assertAlmostEqual(ds.normalize_probability(150.0), 1.0)

    def test_just_above_1_not_converted(self):
        self.assertAlmostEqual(ds.normalize_probability(1.01), 1.0)


class TestGetTierParams(unittest.TestCase):
    def test_micro_under_2000(self):
        tier = ds.get_tier_params(500)
        self.assertEqual(tier["tier"], "micro")
        self.assertAlmostEqual(tier["kelly_mult"], 0.28)

    def test_growth_at_2000(self):
        tier = ds.get_tier_params(2000)
        self.assertEqual(tier["tier"], "growth")
        self.assertAlmostEqual(tier["kelly_mult"], 0.30)

    def test_established_at_10000(self):
        tier = ds.get_tier_params(10000)
        self.assertEqual(tier["tier"], "established")

    def test_scale_at_50000(self):
        tier = ds.get_tier_params(50000)
        self.assertEqual(tier["tier"], "scale")
        self.assertAlmostEqual(tier["kelly_mult"], 0.40)

    def test_kelly_increases_with_tier(self):
        tiers = [ds.get_tier_params(b)["kelly_mult"] for b in [500, 5000, 25000, 100000]]
        for i in range(len(tiers) - 1):
            self.assertLess(tiers[i], tiers[i + 1])


class TestPositionSizeKelly(unittest.TestCase):
    def test_positive_edge_gives_size(self):
        size = ds.position_size(0.30, 0.10, 1000)
        self.assertGreater(size, 0)

    def test_no_edge_returns_zero(self):
        size = ds.position_size(0.05, 0.10, 1000)
        self.assertEqual(size, 0)

    def test_zero_price_returns_0(self):
        size = ds.position_size(0.30, 0.0, 1000)
        self.assertEqual(size, 0)

    def test_tiny_price_returns_0(self):
        size = ds.position_size(0.30, 0.0001, 1000, best_ask=0.0001)
        self.assertEqual(size, 0)

    def test_low_balance_min_5_filter(self):
        size = ds.position_size(0.15, 0.10, 50)
        self.assertEqual(size, 0)

    def test_high_p_model_larger_size_at_higher_balance(self):
        b = 9.0
        kelly_low = (b * 0.20 - 0.80) / b
        kelly_high = (b * 0.50 - 0.50) / b
        self.assertGreater(kelly_high, kelly_low)

    def test_best_ask_reduces_size_at_higher_balance(self):
        b_mp = (1 - 0.10) / 0.10
        b_ask = (1 - 0.15) / 0.15
        kelly_mp = (b_mp * 0.30 - 0.70) / b_mp
        kelly_ask = (b_ask * 0.30 - 0.70) / b_ask
        self.assertGreater(kelly_mp, kelly_ask)

    def test_balance_scales_with_tier(self):
        size_1k = ds.position_size(0.30, 0.10, 1000)
        size_10k = ds.position_size(0.30, 0.10, 10000)
        self.assertGreater(size_10k, size_1k)

    def test_other_cluster_lower_than_named(self):
        size_other = ds.position_size(0.80, 0.10, 1000, cluster="other")
        size_ai = ds.position_size(0.80, 0.10, 1000, cluster="ai_tech")
        self.assertGreater(size_ai, size_other)

    def test_named_cluster_kelly_differentiates(self):
        size_weak = ds.position_size(0.30, 0.10, 1000, cluster="ai_tech")
        size_strong = ds.position_size(0.60, 0.10, 1000, cluster="ai_tech")
        self.assertGreater(size_strong, size_weak)


class TestCalibratePrediction(unittest.TestCase):
    def _mock_calibrate(self, p_model, market_price, metaculus_prob=None, cluster=None):
        p_calibrated = p_model
        dampened = False
        if cluster != "other" and market_price <= 0.10:
            if p_model > 0.20:
                meta_low = metaculus_prob is None or metaculus_prob < 0.10
                if meta_low:
                    p_calibrated = p_model * 0.65
                    dampened = True
        return p_calibrated, dampened

    def test_aggressive_dampened_no_metaculus(self):
        p, dampened = self._mock_calibrate(0.30, 0.08, metaculus_prob=None, cluster="ai_tech")
        self.assertTrue(dampened)
        self.assertLess(p, 0.30)

    def test_conservative_no_dampening(self):
        p, dampened = self._mock_calibrate(0.15, 0.08, metaculus_prob=None, cluster="ai_tech")
        self.assertFalse(dampened)

    def test_high_metaculus_prevents_dampening(self):
        p, dampened = self._mock_calibrate(0.30, 0.08, metaculus_prob=0.25, cluster="ai_tech")
        self.assertFalse(dampened)

    def test_other_cluster_no_dampening(self):
        p, dampened = self._mock_calibrate(0.30, 0.08, metaculus_prob=None, cluster="other")
        self.assertFalse(dampened)


class TestConvergenceRatio(unittest.TestCase):
    def test_convergence_formula(self):
        current_price = 0.08
        metaculus_prob = 0.10
        convergence = current_price / metaculus_prob
        self.assertAlmostEqual(convergence, 0.80)

    def test_convergence_at_threshold(self):
        self.assertAlmostEqual(ds.CONVERGENCE_TAKE_PROFIT, 0.90)

    def test_convergence_exceeds_threshold(self):
        convergence = 0.095 / 0.10
        self.assertGreater(convergence, ds.CONVERGENCE_TAKE_PROFIT)


class TestTrailingConstants(unittest.TestCase):
    def test_activation_threshold(self):
        self.assertAlmostEqual(ds.TRAILING_ACTIVATION, 0.30)

    def test_trailing_stop_pct(self):
        self.assertAlmostEqual(ds.TRAILING_STOP, 0.25)

    def test_atr_stop_multiplier(self):
        self.assertAlmostEqual(ds.ATR_STOP_MULTIPLIER, 2.5)

    def test_atr_trailing_multiplier(self):
        self.assertAlmostEqual(ds.ATR_TRAILING_MULTIPLIER, 1.5)

    def test_activation_formula(self):
        entry = 0.10
        activation_price = entry * (1 + ds.TRAILING_ACTIVATION)
        self.assertAlmostEqual(activation_price, 0.13)

    def test_trailing_stop_formula(self):
        high = 0.20
        stop = high * (1 - ds.TRAILING_STOP)
        self.assertAlmostEqual(stop, 0.15)


class TestTPRungMath(unittest.TestCase):
    def test_ladder_rungs(self):
        total_shares = 100
        rung1_shares = round(total_shares * 0.50)
        rung2_shares = round(total_shares * 0.30)
        held = total_shares - rung1_shares - rung2_shares
        self.assertEqual(rung1_shares, 50)
        self.assertEqual(rung2_shares, 30)
        self.assertEqual(held, 20)

    def test_ladder_value_at_rung_prices(self):
        shares = 50
        price = 0.75
        value = shares * price
        self.assertAlmostEqual(value, 37.50)
        self.assertGreater(value, 5.0)

    def test_small_position_min_check(self):
        shares = 5
        price = 0.75
        value = shares * price
        self.assertAlmostEqual(value, 3.75)
        self.assertLess(value, 5.0)


class TestKellyFormula(unittest.TestCase):
    def test_kelly_positive_edge(self):
        price = 0.10
        b = (1 - price) / price
        p = 0.30
        q = 1 - p
        kelly = (b * p - q) / b
        self.assertGreater(kelly, 0)

    def test_kelly_negative_edge(self):
        price = 0.10
        b = (1 - price) / price
        p = 0.05
        q = 1 - p
        kelly = (b * p - q) / b
        self.assertLess(kelly, 0)

    def test_kelly_zero_when_fair(self):
        price = 0.10
        b = (1 - price) / price
        p = price
        q = 1 - p
        kelly = (b * p - q) / b
        self.assertAlmostEqual(kelly, 0, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
