#!/usr/bin/env python3
"""
Tests for bayesian_updater.py — posterior update, exit thresholds, log-odds math.
"""
import os
import tempfile
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import bayesian_updater as bu


class TestLogOddsConversions(unittest.TestCase):
    def test_prob_to_logodds_0_5(self):
        lo = bu._prob_to_logodds(0.5)
        self.assertAlmostEqual(lo, 0.0, places=5)

    def test_prob_to_logodds_0_9(self):
        lo = bu._prob_to_logodds(0.9)
        self.assertGreater(lo, 0)

    def test_prob_to_logodds_0_1(self):
        lo = bu._prob_to_logodds(0.1)
        self.assertLess(lo, 0)

    def test_logodds_roundtrip(self):
        for p in [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
            lo = bu._prob_to_logodds(p)
            p_back = bu._logodds_to_prob(lo)
            self.assertAlmostEqual(p, p_back, places=5)

    def test_extreme_prob_clamped(self):
        lo = bu._prob_to_logodds(0.0)
        p = bu._logodds_to_prob(lo)
        self.assertGreater(p, 0)
        self.assertLess(p, 0.01)

    def test_extreme_prob_1_clamped(self):
        lo = bu._prob_to_logodds(1.0)
        p = bu._logodds_to_prob(lo)
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.99)


class TestNewsLikelihood(unittest.TestCase):
    def test_confirms_impossible_very_low(self):
        p = bu.NEWS_LIKELIHOOD["confirms_impossible"]["p_yes_given_news"]
        self.assertAlmostEqual(p, 0.02)

    def test_confirms_inevitable_very_high(self):
        p = bu.NEWS_LIKELIHOOD["confirms_inevitable"]["p_yes_given_news"]
        self.assertAlmostEqual(p, 0.95)

    def test_neutral_is_0_5(self):
        p = bu.NEWS_LIKELIHOOD["neutral"]["p_yes_given_news"]
        self.assertAlmostEqual(p, 0.50)

    def test_all_categories_present(self):
        expected = {"confirms_impossible", "strongly_contradicts",
                    "moderately_contradicts", "neutral",
                    "moderately_supports", "strongly_supports",
                    "confirms_inevitable"}
        self.assertEqual(set(bu.NEWS_LIKELIHOOD.keys()), expected)

    def test_monotonic_ordering(self):
        cats = ["confirms_impossible", "strongly_contradicts",
                "moderately_contradicts", "neutral",
                "moderately_supports", "strongly_supports",
                "confirms_inevitable"]
        probs = [bu.NEWS_LIKELIHOOD[c]["p_yes_given_news"] for c in cats]
        for i in range(len(probs) - 1):
            self.assertLess(probs[i], probs[i + 1])


class TestInitPosterior(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_file = bu.BAYESIAN_STATE_FILE
        bu.BAYESIAN_STATE_FILE = os.path.join(self.tmpdir, "bayesian.json")

    def tearDown(self):
        bu.BAYESIAN_STATE_FILE = self.orig_file
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_init_creates_state(self):
        bu.init_posterior("test-slug", 0.25, 0.10)
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        self.assertIn("test-slug", state["positions"])
        pos = state["positions"]["test-slug"]
        self.assertAlmostEqual(pos["p_model_entry"], 0.25)
        self.assertAlmostEqual(pos["posterior_prob"], 0.25)

    def test_init_logodds_correct(self):
        bu.init_posterior("test-slug", 0.30, 0.10)
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        pos = state["positions"]["test-slug"]
        expected_lo = bu._prob_to_logodds(0.30)
        self.assertAlmostEqual(pos["prior_logodds"], expected_lo, places=3)
        self.assertAlmostEqual(pos["posterior_logodds"], expected_lo, places=3)

    def test_init_history_has_entry(self):
        bu.init_posterior("test-slug", 0.20, 0.10)
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        pos = state["positions"]["test-slug"]
        self.assertEqual(len(pos["history"]), 1)
        self.assertEqual(pos["history"][0]["event"], "init")


class TestUpdatePosterior(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_file = bu.BAYESIAN_STATE_FILE
        bu.BAYESIAN_STATE_FILE = os.path.join(self.tmpdir, "bayesian.json")

    def tearDown(self):
        bu.BAYESIAN_STATE_FILE = self.orig_file
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_contradicting_news_lowers_posterior(self):
        bu.init_posterior("s1", 0.30, 0.10)
        new_p = bu.update_posterior("s1", "strongly_contradicts")
        self.assertLess(new_p, 0.30)

    def test_supporting_news_raises_posterior(self):
        bu.init_posterior("s1", 0.30, 0.10)
        new_p = bu.update_posterior("s1", "strongly_supports")
        self.assertGreater(new_p, 0.30)

    def test_neutral_news_no_change(self):
        bu.init_posterior("s1", 0.30, 0.10)
        new_p = bu.update_posterior("s1", "neutral")
        self.assertAlmostEqual(new_p, 0.30, places=4)

    def test_multiple_updates_accumulate(self):
        bu.init_posterior("s1", 0.50, 0.10)
        bu.update_posterior("s1", "strongly_contradicts")
        bu.update_posterior("s1", "strongly_contradicts")
        bu.update_posterior("s1", "strongly_contradicts")
        new_p = bu.update_posterior("s1", "confirms_impossible")
        self.assertLess(new_p, 0.05)

    def test_missing_slug_returns_none(self):
        result = bu.update_posterior("nonexistent", "neutral")
        self.assertIsNone(result)

    def test_updates_counter_increments(self):
        bu.init_posterior("s1", 0.30, 0.10)
        bu.update_posterior("s1", "neutral")
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        self.assertEqual(state["positions"]["s1"]["updates"], 1)
        bu.update_posterior("s1", "neutral")
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        self.assertEqual(state["positions"]["s1"]["updates"], 2)

    def test_history_appends(self):
        bu.init_posterior("s1", 0.30, 0.10)
        bu.update_posterior("s1", "strongly_contradicts")
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        self.assertEqual(len(state["positions"]["s1"]["history"]), 2)


class TestShouldExit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_file = bu.BAYESIAN_STATE_FILE
        bu.BAYESIAN_STATE_FILE = os.path.join(self.tmpdir, "bayesian.json")

    def tearDown(self):
        bu.BAYESIAN_STATE_FILE = self.orig_file
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_no_state_no_exit(self):
        exit_flag, _reason = bu.should_exit("nonexistent")
        self.assertFalse(exit_flag)

    def test_high_posterior_no_exit(self):
        bu.init_posterior("s1", 0.30, 0.10)
        exit_flag, _reason = bu.should_exit("s1")
        self.assertFalse(exit_flag)

    def test_dropped_posterior_triggers_exit(self):
        bu.init_posterior("s1", 0.50, 0.10)
        for _ in range(8):
            bu.update_posterior("s1", "confirms_impossible")
        exit_flag, _reason = bu.should_exit("s1")
        self.assertTrue(exit_flag)

    def test_near_zero_posterior_exits(self):
        bu.init_posterior("s1", 0.05, 0.10)
        for _ in range(5):
            bu.update_posterior("s1", "confirms_impossible")
        exit_flag, _reason = bu.should_exit("s1")
        self.assertTrue(exit_flag)

    def test_threshold_ratio_default(self):
        bu.init_posterior("s1", 0.30, 0.10)
        bu.update_posterior("s1", "strongly_contradicts")
        exit_flag, _reason = bu.should_exit("s1", threshold_ratio=0.40)
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        posterior = state["positions"]["s1"]["posterior_prob"]
        entry_p = state["positions"]["s1"]["p_model_entry"]
        ratio = posterior / entry_p
        self.assertEqual(exit_flag, ratio <= 0.40)

    def test_custom_threshold(self):
        bu.init_posterior("s1", 0.30, 0.10)
        exit_flag_strict, _ = bu.should_exit("s1", threshold_ratio=0.10)
        self.assertFalse(exit_flag_strict)


class TestPosteriorBounds(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_file = bu.BAYESIAN_STATE_FILE
        bu.BAYESIAN_STATE_FILE = os.path.join(self.tmpdir, "bayesian.json")

    def tearDown(self):
        bu.BAYESIAN_STATE_FILE = self.orig_file
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_posterior_stays_below_1(self):
        bu.init_posterior("s1", 0.90, 0.10)
        for _ in range(20):
            bu.update_posterior("s1", "confirms_inevitable")
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        self.assertLessEqual(state["positions"]["s1"]["posterior_prob"], 1.0)

    def test_posterior_stays_above_0(self):
        bu.init_posterior("s1", 0.10, 0.10)
        for _ in range(20):
            bu.update_posterior("s1", "confirms_impossible")
        state = bu.load_json(bu.BAYESIAN_STATE_FILE, {})
        self.assertGreaterEqual(state["positions"]["s1"]["posterior_prob"], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
