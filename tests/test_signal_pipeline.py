"""
Tests for signal_pipeline.py — market analysis, calibration, metaculus integration,
circuit breaker, batch parsing, and signal scoring.
"""
import json
import os
import sys
import time
import unittest
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import signal_pipeline as sp
from schema import HYP_CLUSTERS, HYP_CONFIDENCE, HYP_FACTORS, HYP_P_MODEL, HYP_SLUG

MOCK_SETTINGS = {"signal_threshold": 50, "min_confidence": 0.65, "min_p_model": 0.03}


def _mock_settings(return_value=None):
    if return_value is None:
        return_value = MOCK_SETTINGS
    return patch("signal_pipeline.get_settings", return_value=return_value)


# ═══════════════════════════════════════════════════════════════════
# normalize_probability
# ═══════════════════════════════════════════════════════════════════
class TestNormalizeProbability(unittest.TestCase):
    def test_fractional_input_015(self):
        self.assertAlmostEqual(sp.normalize_probability(0.15), 0.15)

    def test_fractional_input_05(self):
        self.assertAlmostEqual(sp.normalize_probability(0.5), 0.5)

    def test_percentage_50(self):
        self.assertAlmostEqual(sp.normalize_probability(50), 0.50)

    def test_percentage_3(self):
        self.assertAlmostEqual(sp.normalize_probability(3), 0.03)

    def test_capped_at_1(self):
        self.assertAlmostEqual(sp.normalize_probability(150), 1.0)

    def test_negative_floored(self):
        self.assertAlmostEqual(sp.normalize_probability(-0.1), 0.0)

    def test_string_input(self):
        self.assertAlmostEqual(sp.normalize_probability("0.5"), 0.5)

    def test_none_returns_zero(self):
        self.assertEqual(sp.normalize_probability(None), 0.0)

    def test_zero_returns_zero(self):
        self.assertAlmostEqual(sp.normalize_probability(0.0), 0.0)

    def test_one_returns_one(self):
        self.assertAlmostEqual(sp.normalize_probability(1.0), 1.0)

    def test_exactly_1_not_divided(self):
        self.assertAlmostEqual(sp.normalize_probability(1.0), 1.0)

    def test_percentage_100(self):
        self.assertAlmostEqual(sp.normalize_probability(100), 1.0)

    def test_tiny_fractional(self):
        self.assertAlmostEqual(sp.normalize_probability(0.001), 0.001)

    def test_large_negative(self):
        self.assertAlmostEqual(sp.normalize_probability(-999), 0.0)


# ═══════════════════════════════════════════════════════════════════
# _check_llm_circuit_breaker
# ═══════════════════════════════════════════════════════════════════
class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        sp._llm_call_times.clear()

    def tearDown(self):
        sp._llm_call_times.clear()

    def test_under_limit_allows(self):
        for _ in range(59):
            sp._llm_call_times.append(time.time())
        self.assertTrue(sp._check_llm_circuit_breaker())

    def test_at_limit_blocks(self):
        for _ in range(60):
            sp._llm_call_times.append(time.time())
        self.assertFalse(sp._check_llm_circuit_breaker())

    def test_empty_allows(self):
        self.assertTrue(sp._check_llm_circuit_breaker())
        self.assertEqual(len(sp._llm_call_times), 1)

    def test_prunes_old_entries(self):
        old = time.time() - 4000
        for _ in range(65):
            sp._llm_call_times.append(old)
        self.assertTrue(sp._check_llm_circuit_breaker())

    def test_single_old_pruned(self):
        old = time.time() - 4000
        sp._llm_call_times.append(old)
        self.assertTrue(sp._check_llm_circuit_breaker())
        self.assertEqual(len(sp._llm_call_times), 1)

    def test_at_59_then_one_more_reaches_60(self):
        for _ in range(59):
            sp._llm_call_times.append(time.time())
        self.assertTrue(sp._check_llm_circuit_breaker())
        self.assertEqual(len(sp._llm_call_times), 60)
        self.assertFalse(sp._check_llm_circuit_breaker())

    def test_mixed_old_and_recent(self):
        old = time.time() - 4000
        sp._llm_call_times.append(old)
        for _ in range(60):
            sp._llm_call_times.append(time.time())
        self.assertFalse(sp._check_llm_circuit_breaker())


# ═══════════════════════════════════════════════════════════════════
# parse_resolve_date
# ═══════════════════════════════════════════════════════════════════
class TestParseResolveDate(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(sp.parse_resolve_date(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(sp.parse_resolve_date(""))

    def test_iso_format(self):
        result = sp.parse_resolve_date("2025-06-01T00:00:00+00:00")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2025)
        self.assertEqual(result.month, 6)

    def test_z_suffix(self):
        result = sp.parse_resolve_date("2025-12-31T23:59:59Z")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2025)

    def test_invalid_string(self):
        self.assertIsNone(sp.parse_resolve_date("not-a-date"))


# ═══════════════════════════════════════════════════════════════════
# dates_match
# ═══════════════════════════════════════════════════════════════════
class TestDatesMatch(unittest.TestCase):
    def test_same_date_matches(self):
        self.assertTrue(sp.dates_match("2025-06-15T00:00:00Z", "2025-06-15T12:00:00Z"))

    def test_within_window(self):
        self.assertTrue(sp.dates_match("2025-06-15T00:00:00Z", "2025-06-20T00:00:00Z"))

    def test_outside_window(self):
        self.assertFalse(sp.dates_match("2025-06-01T00:00:00Z", "2025-07-01T00:00:00Z"))

    def test_none_date_returns_false(self):
        self.assertFalse(sp.dates_match(None, "2025-06-01T00:00:00Z"))
        self.assertFalse(sp.dates_match("2025-06-01T00:00:00Z", None))

    def test_both_none_returns_false(self):
        self.assertFalse(sp.dates_match(None, None))

    def test_custom_window(self):
        self.assertTrue(sp.dates_match("2025-06-01T00:00:00Z", "2025-06-10T00:00:00Z", window_days=10))
        self.assertFalse(sp.dates_match("2025-06-01T00:00:00Z", "2025-06-12T00:00:00Z", window_days=10))


# ═══════════════════════════════════════════════════════════════════
# _generate_search_queries
# ═══════════════════════════════════════════════════════════════════
class TestGenerateSearchQueries(unittest.TestCase):
    def test_basic_question(self):
        queries = sp._generate_search_queries("Will AI be regulated?")
        self.assertIsInstance(queries, list)
        self.assertGreater(len(queries), 0)
        self.assertTrue(all(len(q) >= 4 for q in queries))

    def test_max_five_queries(self):
        queries = sp._generate_search_queries("Will something happen with things stuff more words here?")
        self.assertLessEqual(len(queries), 5)

    def test_removes_question_mark(self):
        queries = sp._generate_search_queries("Will AI? be regulated?")
        for q in queries:
            self.assertNotIn("?", q)

    def test_short_question_few_queries(self):
        queries = sp._generate_search_queries("Hi?")
        self.assertEqual(len(queries), 0)

    def test_single_word(self):
        queries = sp._generate_search_queries("X?")
        self.assertEqual(len(queries), 0)


# ═══════════════════════════════════════════════════════════════════
# _calculate_metaculus_match
# ═══════════════════════════════════════════════════════════════════
class TestCalculateMetaculusMatch(unittest.TestCase):
    def _run(self, pm_q, result):
        mock_fuzz = MagicMock()
        mock_fuzz.partial_ratio.return_value = 80
        with patch.dict("sys.modules", {"fuzzywuzzy": MagicMock(fuzz=mock_fuzz)}):
            with patch("metaculus._calculate_metaculus_match.__code__", sp._calculate_metaculus_match.__code__):
                pass
        with patch("fuzzywuzzy.fuzz.partial_ratio", return_value=80):
            return sp._calculate_metaculus_match(pm_q, result)

    def test_exact_match_high_score(self):
        mock_fuzz = MagicMock()
        mock_fuzz.partial_ratio.return_value = 95
        with patch.dict("sys.modules", {"fuzzywuzzy": MagicMock(fuzz=mock_fuzz), "fuzzywuzzy.fuzz": mock_fuzz}):
            result = {"title": "Will AI be regulated by 2025?", "short_title": ""}
            score = sp._calculate_metaculus_match("Will AI be regulated by 2025?", result)
            self.assertGreater(score, 0.5)

    def test_no_overlap_low_score(self):
        mock_fuzz = MagicMock()
        mock_fuzz.partial_ratio.return_value = 10
        with patch.dict("sys.modules", {"fuzzywuzzy": MagicMock(fuzz=mock_fuzz), "fuzzywuzzy.fuzz": mock_fuzz}):
            result = {"title": "How many bananas will be exported?", "short_title": ""}
            score = sp._calculate_metaculus_match("Will Russia use nuclear weapons?", result)
            self.assertLess(score, 0.5)

    def test_key_phrase_bonus(self):
        mock_fuzz = MagicMock()
        mock_fuzz.partial_ratio.return_value = 20
        with patch.dict("sys.modules", {"fuzzywuzzy": MagicMock(fuzz=mock_fuzz), "fuzzywuzzy.fuzz": mock_fuzz}):
            r1 = {"title": "Will AI safety regulations pass?", "short_title": ""}
            score1 = sp._calculate_metaculus_match("Will AI safety regulations pass?", r1)
            r2 = {"title": "What is the GDP of France?", "short_title": ""}
            score2 = sp._calculate_metaculus_match("How many people live in Japan?", r2)
            self.assertGreater(score1, score2)

    def test_number_overlap_bonus(self):
        mock_fuzz = MagicMock()
        mock_fuzz.partial_ratio.return_value = 60
        with patch.dict("sys.modules", {"fuzzywuzzy": MagicMock(fuzz=mock_fuzz), "fuzzywuzzy.fuzz": mock_fuzz}):
            result = {"title": "Will GDP grow by 5% in 2025?", "short_title": ""}
            score = sp._calculate_metaculus_match("Will GDP grow by 5% in 2025?", result)
            self.assertGreater(score, 0)

    def test_uses_short_title_fallback(self):
        mock_fuzz = MagicMock()
        mock_fuzz.partial_ratio.return_value = 90
        with patch.dict("sys.modules", {"fuzzywuzzy": MagicMock(fuzz=mock_fuzz), "fuzzywuzzy.fuzz": mock_fuzz}):
            result = {"title": "", "short_title": "Will AI win?"}
            score = sp._calculate_metaculus_match("Will AI win?", result)
            self.assertGreater(score, 0.3)

    def test_score_bounded_at_1(self):
        mock_fuzz = MagicMock()
        mock_fuzz.partial_ratio.return_value = 100
        with patch.dict("sys.modules", {"fuzzywuzzy": MagicMock(fuzz=mock_fuzz), "fuzzywuzzy.fuzz": mock_fuzz}):
            result = {"title": "AI safety AI safety AI safety", "short_title": ""}
            score = sp._calculate_metaculus_match("AI safety AI safety AI safety", result)
            self.assertLessEqual(score, 1.0)


# ═══════════════════════════════════════════════════════════════════
# get_time_decay_threshold
# ═══════════════════════════════════════════════════════════════════
class TestGetTimeDecayThreshold(unittest.TestCase):
    def test_none_returns_default(self):
        self.assertEqual(sp.get_time_decay_threshold(None), sp.METACULUS_GAP_THRESHOLD)

    def test_empty_returns_default(self):
        self.assertEqual(sp.get_time_decay_threshold(""), sp.METACULUS_GAP_THRESHOLD)

    def test_far_future_30plus_days(self):
        future = (datetime.now(UTC) + timedelta(days=60)).isoformat()
        self.assertAlmostEqual(sp.get_time_decay_threshold(future), 0.20)

    def test_medium_8_to_30_days(self):
        future = (datetime.now(UTC) + timedelta(days=15)).isoformat()
        self.assertAlmostEqual(sp.get_time_decay_threshold(future), 0.15)

    def test_short_3_to_7_days(self):
        future = (datetime.now(UTC) + timedelta(days=5)).isoformat()
        self.assertAlmostEqual(sp.get_time_decay_threshold(future), 0.10)

    def test_near_1_to_2_days(self):
        future = (datetime.now(UTC) + timedelta(days=1, hours=12)).isoformat()
        self.assertAlmostEqual(sp.get_time_decay_threshold(future), 0.05)

    def test_sub_1_day(self):
        future = (datetime.now(UTC) + timedelta(hours=12)).isoformat()
        threshold = sp.get_time_decay_threshold(future)
        self.assertGreaterEqual(threshold, 0.03)
        self.assertLessEqual(threshold, 0.05)

    def test_past_date(self):
        past = (datetime.now(UTC) - timedelta(days=5)).isoformat()
        threshold = sp.get_time_decay_threshold(past)
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.03)

    def test_invalid_date_returns_default(self):
        self.assertEqual(sp.get_time_decay_threshold("not-a-date"), sp.METACULUS_GAP_THRESHOLD)


# ═══════════════════════════════════════════════════════════════════
# check_manifold_gap
# ═══════════════════════════════════════════════════════════════════
class TestCheckMetaculusGap(unittest.TestCase):
    def _make_market(self, price=0.10, question="Will X happen?", end_date=None):
        return {
            "question": question,
            "price": price,
            "end_date": end_date or (datetime.now(UTC) + timedelta(days=30)).isoformat(),
        }

    @patch("metaculus.get_time_decay_threshold", return_value=0.08)
    @patch("metaculus.get_metaculus_forecast")
    def test_no_metaculus_data_returns_none(self, mock_meta, mock_threshold):
        mock_meta.return_value = {"found": False}
        self.assertIsNone(sp.check_metaculus_gap(self._make_market()))

    @patch("metaculus.get_time_decay_threshold", return_value=0.08)
    @patch("metaculus.get_metaculus_forecast")
    def test_large_gap_returns_signal(self, mock_meta, mock_threshold):
        mock_meta.return_value = {
            "found": True,
            "probability": 0.40,
            "dispersion_penalty": 1.0,
        }
        result = sp.check_metaculus_gap(self._make_market(price=0.10))
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "metaculus")
        self.assertGreater(result["gap"], 0)

    @patch("metaculus.get_time_decay_threshold", return_value=0.08)
    @patch("metaculus.get_metaculus_forecast")
    def test_small_gap_returns_none(self, mock_meta, mock_threshold):
        mock_meta.return_value = {
            "found": True,
            "probability": 0.12,
            "dispersion_penalty": 1.0,
        }
        result = sp.check_metaculus_gap(self._make_market(price=0.10))
        self.assertIsNone(result)

    @patch("metaculus.get_time_decay_threshold", return_value=0.08)
    @patch("metaculus.get_metaculus_forecast")
    def test_uses_polymarket_prob_kwarg(self, mock_meta, mock_threshold):
        mock_meta.return_value = {
            "found": True,
            "probability": 0.30,
            "dispersion_penalty": 1.0,
        }
        result = sp.check_metaculus_gap(self._make_market(price=0.10), polymarket_prob=0.05)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["polymarket_prob"], 0.05)

    @patch("metaculus.get_time_decay_threshold", return_value=0.08)
    @patch("metaculus.get_metaculus_forecast")
    def test_dispersion_penalty_reduces_strength(self, mock_meta, mock_threshold):
        mock_meta.return_value = {
            "found": True,
            "probability": 0.40,
            "dispersion_penalty": 0.5,
        }
        result = sp.check_metaculus_gap(self._make_market(price=0.10))
        self.assertIsNotNone(result)
        self.assertLess(result["signal_strength"], 1.0)

    @patch("metaculus.get_time_decay_threshold", return_value=0.08)
    @patch("metaculus.get_metaculus_forecast")
    def test_signal_strength_capped_at_1(self, mock_meta, mock_threshold):
        mock_meta.return_value = {
            "found": True,
            "probability": 0.90,
            "dispersion_penalty": 1.0,
        }
        result = sp.check_metaculus_gap(self._make_market(price=0.05))
        self.assertIsNotNone(result)
        self.assertLessEqual(result["signal_strength"], 1.0)


# ═══════════════════════════════════════════════════════════════════
# calibrate_prediction
# ═══════════════════════════════════════════════════════════════════
class TestCalibratePrediction(unittest.TestCase):
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_soft_extremize_low_p(self, mock_count):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        with patch("calibration.get_calibrator", return_value=mock_cal):
            p, calibrated = sp.calibrate_prediction(0.10, 0.05)
        self.assertAlmostEqual(p, 0.105)
        self.assertTrue(calibrated)

    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_soft_extremize_capped_at_50(self, mock_count):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        with patch("calibration.get_calibrator", return_value=mock_cal):
            p, calibrated = sp.calibrate_prediction(0.80, 0.10)
        self.assertAlmostEqual(p, 0.50)
        self.assertTrue(calibrated)

    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_low_market_price_caps_at_85(self, mock_count):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        with patch("calibration.get_calibrator", return_value=mock_cal):
            p, _ = sp.calibrate_prediction(0.45, 0.20)
        self.assertLessEqual(p, 0.85)

    def test_invalid_p_model_returns_unchanged(self):
        p, calibrated = sp.calibrate_prediction(0, 0.10)
        self.assertEqual(p, 0)
        self.assertFalse(calibrated)

    def test_p_model_1_returns_unchanged(self):
        p, calibrated = sp.calibrate_prediction(1.0, 0.10)
        self.assertEqual(p, 1.0)
        self.assertFalse(calibrated)

    def test_negative_p_model_returns_unchanged(self):
        p, calibrated = sp.calibrate_prediction(-0.1, 0.10)
        self.assertAlmostEqual(p, -0.1)
        self.assertFalse(calibrated)

    def test_invalid_market_price_returns_unchanged(self):
        p, calibrated = sp.calibrate_prediction(0.10, "invalid")
        self.assertAlmostEqual(p, 0.10)
        self.assertFalse(calibrated)

    def test_negative_market_price_returns_unchanged(self):
        p, calibrated = sp.calibrate_prediction(0.10, -0.05)
        self.assertAlmostEqual(p, 0.10)
        self.assertFalse(calibrated)

    @patch("signal_scorer._count_resolved_hypotheses", return_value=25)
    def test_isotonic_calibration_used_when_fitted(self, mock_count):
        mock_cal = MagicMock()
        mock_cal.is_fitted = True
        mock_cal.predict.return_value = 0.15
        with patch("calibration.get_calibrator", return_value=mock_cal):
            with patch("calibration_tracker.get_platt_calibrated", return_value=None):
                _p, calibrated = sp.calibrate_prediction(0.10, 0.05)
        self.assertTrue(calibrated)

    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_high_market_price_no_85_cap(self, mock_count):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        with patch("calibration.get_calibrator", return_value=mock_cal):
            p, _ = sp.calibrate_prediction(0.40, 0.50)
        self.assertAlmostEqual(p, 0.42)


# ═══════════════════════════════════════════════════════════════════
# _cluster_score_adjustment
# ═══════════════════════════════════════════════════════════════════
class TestClusterScoreAdjustment(unittest.TestCase):
    def test_other_cluster_default(self):
        settings = {"cluster_score_adjustments": sp.CLUSTER_SCORE_ADJUSTMENTS}
        adj = sp._cluster_score_adjustment("other", settings)
        self.assertEqual(adj, 15)

    def test_crypto_cluster_negative(self):
        settings = {"cluster_score_adjustments": sp.CLUSTER_SCORE_ADJUSTMENTS}
        adj = sp._cluster_score_adjustment("crypto", settings)
        self.assertEqual(adj, -25)

    def test_unknown_cluster_zero(self):
        settings = {"cluster_score_adjustments": sp.CLUSTER_SCORE_ADJUSTMENTS}
        adj = sp._cluster_score_adjustment("ai_tech", settings)
        self.assertEqual(adj, 0)

    def test_custom_settings_override(self):
        custom = {"custom_cluster": 42}
        settings = {"cluster_score_adjustments": custom}
        adj = sp._cluster_score_adjustment("custom_cluster", settings)
        self.assertEqual(adj, 42)


# ═══════════════════════════════════════════════════════════════════
# pre_filter_before_batching
# ═══════════════════════════════════════════════════════════════════
class TestPreFilterBeforeBatching(unittest.TestCase):
    def test_banned_cluster_skipped(self):
        markets = [{HYP_CLUSTERS: ["crypto"], HYP_SLUG: "crypto-test", "volume": 500000}]
        kept, skipped = sp.pre_filter_before_batching(markets)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(kept), 0)

    def test_other_low_volume_skipped(self):
        markets = [{HYP_CLUSTERS: ["other"], HYP_SLUG: "low-vol", "volume": 50000}]
        kept, skipped = sp.pre_filter_before_batching(markets)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(kept), 0)

    def test_other_high_volume_kept(self):
        markets = [{HYP_CLUSTERS: ["other"], HYP_SLUG: "high-vol", "volume": 200000}]
        kept, skipped = sp.pre_filter_before_batching(markets)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(skipped), 0)

    def test_allowed_cluster_kept(self):
        markets = [{HYP_CLUSTERS: ["ai_tech"], HYP_SLUG: "ai-test", "volume": 50000}]
        kept, _skipped = sp.pre_filter_before_batching(markets)
        self.assertEqual(len(kept), 1)

    def test_mixed_markets(self):
        markets = [
            {HYP_CLUSTERS: ["ai_tech"], HYP_SLUG: "a1", "volume": 50000},
            {HYP_CLUSTERS: ["crypto"], HYP_SLUG: "c1", "volume": 500000},
            {HYP_CLUSTERS: ["other"], HYP_SLUG: "o1", "volume": 50000},
            {HYP_CLUSTERS: ["other"], HYP_SLUG: "o2", "volume": 200000},
        ]
        kept, skipped = sp.pre_filter_before_batching(markets)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(skipped), 2)

    def test_empty_input(self):
        kept, skipped = sp.pre_filter_before_batching([])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(skipped), 0)

    def test_no_clusters_defaults_to_other(self):
        markets = [{HYP_SLUG: "no-clusters", "volume": 50000}]
        _kept, skipped = sp.pre_filter_before_batching(markets)
        self.assertEqual(len(skipped), 1)


# ═══════════════════════════════════════════════════════════════════
# load_cache / save_cache
# ═══════════════════════════════════════════════════════════════════
class TestCacheHelpers(unittest.TestCase):
    @patch("metaculus.save_json")
    @patch("metaculus.load_json")
    def test_load_cache_removes_stale_entries(self, mock_load, mock_save):
        old_ts = (datetime.now() - timedelta(hours=25)).isoformat()
        recent_ts = (datetime.now() - timedelta(hours=1)).isoformat()
        mock_load.return_value = {
            "metaculus": {
                "old_key": {"timestamp": old_ts, "data": "old"},
                "new_key": {"timestamp": recent_ts, "data": "new"},
            },
            "news": {},
            "last_update": None,
        }
        cache = sp.load_cache()
        self.assertNotIn("old_key", cache["metaculus"])
        self.assertIn("new_key", cache["metaculus"])

    @patch("metaculus.load_json")
    def test_load_cache_no_timestamp_kept(self, mock_load):
        mock_load.return_value = {
            "metaculus": {"no_ts": {"data": "value"}},
            "news": {},
            "last_update": None,
        }
        cache = sp.load_cache()
        self.assertIn("no_ts", cache["metaculus"])

    @patch("metaculus.save_json")
    def test_save_cache_sets_timestamp(self, mock_save):
        sp.save_cache({"metaculus": {}, "news": {}})
        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        self.assertIn("last_update", args[1])


# ═══════════════════════════════════════════════════════════════════
# _parse_batch_response
# ═══════════════════════════════════════════════════════════════════
class TestParseBatchResponse(unittest.TestCase):
    def _batch_items(self):
        return [
            {HYP_SLUG: "slug-a", "question": "Q1?", "market_price": 0.10, "volume": 100000, "ttl_hours": 720, "cluster": "ai_tech"},
            {HYP_SLUG: "slug-b", "question": "Q2?", "market_price": 0.05, "volume": 50000, "ttl_hours": 360, "cluster": "other"},
        ]

    @patch("signal_pipeline._build_batch_results")
    def test_valid_json_array(self, mock_build):
        mock_build.return_value = [{"slug": "a"}, {"slug": "b"}]
        content = '[{"slug": "a", "estimated_probability": 0.3, "confidence": 0.8}, {"slug": "b", "estimated_probability": 0.2, "confidence": 0.7}]'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)

    @patch("signal_pipeline._build_batch_results")
    def test_json_with_code_fence(self, mock_build):
        mock_build.return_value = [{"slug": "a"}]
        content = '```json\n[{"slug": "a", "estimated_probability": 0.3}]\n```'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)

    @patch("signal_pipeline._build_batch_results")
    def test_json_with_surrounding_text(self, mock_build):
        mock_build.return_value = [{"slug": "a"}]
        content = 'Here are the results:\n[{"slug": "a", "estimated_probability": 0.3}]\nEnd of results.'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)

    def test_no_array_returns_none(self):
        content = "No valid response here, just text."
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNone(result)

    @patch("signal_pipeline._build_batch_results")
    def test_empty_array_returns_result(self, mock_build):
        mock_build.return_value = []
        content = '[]'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)

    @patch("signal_pipeline._build_batch_results")
    def test_malformed_json_fallback_regex(self, mock_build):
        mock_build.return_value = [{"slug": "a"}]
        content = 'Results: [{"slug": "a", "estimated_probability": 0.3}]'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)

    @patch("signal_pipeline._build_batch_results")
    def test_individual_objects_recovery(self, mock_build):
        mock_build.return_value = [{"slug": "a"}]
        content = '[{"slug": "a", "estimated_probability": 0.3}]'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)

    @patch("signal_pipeline._build_batch_results")
    def test_json_with_colon_prefix(self, mock_build):
        mock_build.return_value = [{"slug": "a"}]
        content = ':[{"slug": "a", "estimated_probability": 0.3}]'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)

    @patch("signal_pipeline._build_batch_results")
    def test_nested_json_in_strings(self, mock_build):
        mock_build.return_value = [{"slug": "a"}]
        content = '[{"slug": "a", "reasoning": "the [array] inside"}]'
        result = sp._parse_batch_response(content, self._batch_items())
        self.assertIsNotNone(result)


# ═══════════════════════════════════════════════════════════════════
# _build_batch_results — signal scoring composite
# ═══════════════════════════════════════════════════════════════════
class TestBuildBatchResults(unittest.TestCase):
    def _batch_items(self, price=0.10):
        return [
            {
                HYP_SLUG: "slug-a", "question": "Will X happen?", "market_price": price,
                "volume": 500000, "ttl_hours": 4320, "cluster": "ai_tech",
            },
        ]

    @patch("signal_pipeline._cluster_score_adjustment", return_value=0)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_buy_when_signals_align(self, mock_count, mock_get_cal, mock_cluster_adj):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        with _mock_settings():
            parsed = [{
                HYP_SLUG: "slug-a",
                "estimated_probability": 0.30,
                HYP_CONFIDENCE: 0.80,
                HYP_FACTORS: [
                    {"factor": "strong evidence", "direction": "supports", "weight": "high", "source": "news"},
                    {"factor": "momentum", "direction": "supports", "weight": "medium", "source": "data"},
                ],
                "reasoning": "test",
            }]
            results = sp._build_batch_results(parsed, self._batch_items())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["action"], "BUY")
            self.assertGreater(results[0]["signal_score"], 55)

    @patch("signal_pipeline._cluster_score_adjustment", return_value=0)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_skip_when_signals_weak(self, mock_count, mock_get_cal, mock_cluster_adj):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        with _mock_settings():
            parsed = [{
                HYP_SLUG: "slug-a",
                "estimated_probability": 0.04,
                HYP_CONFIDENCE: 0.40,
                HYP_FACTORS: [],
                "reasoning": "test",
            }]
            results = sp._build_batch_results(parsed, self._batch_items())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["action"], "SKIP")

    @patch("signal_pipeline._cluster_score_adjustment", return_value=0)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_p_model_below_min_skipped(self, mock_count, mock_get_cal, mock_cluster_adj):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        with _mock_settings():
            parsed = [{
                HYP_SLUG: "slug-a",
                "estimated_probability": 0.02,
                HYP_CONFIDENCE: 0.80,
                HYP_FACTORS: [],
                "reasoning": "test",
            }]
            results = sp._build_batch_results(parsed, self._batch_items())
            self.assertEqual(results[0]["action"], "SKIP")
            self.assertEqual(results[0]["prob_ratio"], 0)

    @patch("signal_pipeline._cluster_score_adjustment", return_value=0)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_missing_slug_uses_order_fallback(self, mock_count, mock_get_cal, mock_cluster_adj):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        with _mock_settings():
            parsed = [{
                "estimated_probability": 0.30,
                HYP_CONFIDENCE: 0.80,
                HYP_FACTORS: [{"factor": "test", "direction": "supports", "weight": "high", "source": "test"}],
                "reasoning": "test",
            }]
            results = sp._build_batch_results(parsed, self._batch_items())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][HYP_SLUG], "slug-a")

    @patch("signal_pipeline._cluster_score_adjustment", return_value=0)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_max_p_model_ratio_enforced(self, mock_count, mock_get_cal, mock_cluster_adj):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        items = self._batch_items(price=0.05)
        with _mock_settings():
            parsed = [{
                HYP_SLUG: "slug-a",
                "estimated_probability": 0.90,
                HYP_CONFIDENCE: 0.90,
                HYP_FACTORS: [{"factor": "test", "direction": "supports", "weight": "high", "source": "test"}],
                "reasoning": "test",
            }]
            results = sp._build_batch_results(parsed, items)
            max_p = 0.05 * sp.MAX_P_MODEL_RATIO
            calibrated_max = min(max_p * 1.1, 0.85)
            self.assertLessEqual(results[0][HYP_P_MODEL], calibrated_max)


# ═══════════════════════════════════════════════════════════════════
# metaculus_search / metaculus_get_question
# ═══════════════════════════════════════════════════════════════════
class TestMetaculusSearch(unittest.TestCase):
    @patch("metaculus._fetch_all_open_questions")
    def test_success_returns_results(self, mock_fetch):
        mock_fetch.return_value = [
            {"id": 1, "title": "Will AI safety become a major concern?", "short_title": "AI safety"},
            {"id": 2, "title": "Will Bitcoin reach $200k?", "short_title": "Bitcoin"},
        ]
        result = sp.metaculus_search("AI safety")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], 1)

    @patch("metaculus._fetch_all_open_questions", return_value=[])
    def test_empty_cache_returns_empty(self, mock_fetch):
        self.assertEqual(sp.metaculus_search("test"), [])


class TestMetaculusGetQuestion(unittest.TestCase):
    @patch("metaculus.requests.get")
    def test_success_returns_data(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"id": 123, "title": "Test?"}
        result = sp.metaculus_get_question(123)
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 123)

    @patch("metaculus.requests.get")
    def test_non_200_returns_none(self, mock_get):
        mock_get.return_value = MagicMock(status_code=403)
        self.assertIsNone(sp.metaculus_get_question(999))

    @patch("metaculus.requests.get", side_effect=Exception("err"))
    def test_exception_returns_none(self, mock_get):
        self.assertIsNone(sp.metaculus_get_question(999))


# ═══════════════════════════════════════════════════════════════════
# batch_analyze_markets (integration-level)
# ═══════════════════════════════════════════════════════════════════
class TestBatchAnalyzeMarkets(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(sp.batch_analyze_markets([]), [])

    @patch("signal_pipeline.full_market_analysis")
    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=False)
    def test_circuit_breaker_falls_back_to_individual(self, mock_cb, mock_full):
        mock_full.return_value = {"action": "SKIP", HYP_SLUG: "test"}
        markets = [{"question": "Q?", "price": 0.40, HYP_SLUG: "test", HYP_CLUSTERS: ["other"], "volume": 100000, "ttl_hours": 720, "end_date": ""}]
        results = sp.batch_analyze_markets(markets)
        self.assertEqual(len(results), 1)

    @patch("signal_pipeline.full_market_analysis")
    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=False)
    @patch("signal_pipeline.get_metaculus_forecast", return_value={"found": False})
    def test_high_price_skips_metaculus(self, mock_meta, mock_cb, mock_full):
        mock_full.return_value = {"action": "SKIP", HYP_SLUG: "high-price"}
        markets = [{"question": "Q?", "price": 0.50, HYP_SLUG: "high-price", HYP_CLUSTERS: ["other"], "volume": 100000, "ttl_hours": 720, "end_date": ""}]
        sp.batch_analyze_markets(markets)
        mock_meta.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# full_market_analysis — signal scoring detailed
# ═══════════════════════════════════════════════════════════════════
class TestFullMarketAnalysis(unittest.TestCase):
    def _market(self, **overrides):
        base = {
            "question": "Will AI be regulated by 2026?",
            HYP_SLUG: "test-slug",
            "price": 0.10,
            "volume": 500000,
            "liquidity": 5000,
            "end_date": (datetime.now(UTC) + timedelta(days=90)).isoformat(),
            "ttl_hours": 2160,
            HYP_CLUSTERS: ["ai_tech"],
            "oracle_type": "uma",
            "condition_token_id": "test_token",
        }
        base.update(overrides)
        return base

    def _mock_llm_response(self, prob, conf, factors):
        return {
            "estimated_probability": prob,
            HYP_CONFIDENCE: conf,
            HYP_FACTORS: factors,
        }

    @patch("signal_scorer._cluster_score_adjustment", return_value=0)
    @patch("signal_scorer.check_manifold_gap", return_value=None)
    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_scorer.requests.post")
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_buy_signal(self, mock_count, mock_get_cal, mock_post, mock_cb, mock_gap, mock_cluster):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "estimated_probability": 0.50,
            HYP_CONFIDENCE: 0.80,
            HYP_FACTORS: [
                {"factor": "strong", "direction": "supports", "weight": "high", "source": "news"},
                {"factor": "momentum", "direction": "supports", "weight": "medium", "source": "data"},
            ],
            "reasoning": "test",
        })}}]}
        mock_post.return_value = mock_resp
        with _mock_settings(), patch("signal_scorer.parse_llm_json") as mock_parse:
            mock_parse.return_value = self._mock_llm_response(0.50, 0.80, [
                {"factor": "strong", "direction": "supports", "weight": "high", "source": "news"},
                {"factor": "momentum", "direction": "supports", "weight": "medium", "source": "data"},
            ])
            with patch("order_manager.get_best_ask", return_value=None):
                result = sp.full_market_analysis(self._market(price=0.07, ttl_hours=500))
        self.assertEqual(result["action"], "BUY")
        self.assertGreater(result["signal_score"], 50)

    @patch("signal_scorer._cluster_score_adjustment", return_value=0)
    @patch("signal_scorer.check_manifold_gap", return_value=None)
    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=False)
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_circuit_breaker_fallback(self, mock_count, mock_get_cal, mock_cb, mock_gap, mock_cluster):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        with _mock_settings(), patch("order_manager.get_best_ask", return_value=None):
            result = sp.full_market_analysis(self._market())
        self.assertEqual(result["action"], "SKIP")

    @patch("signal_scorer._cluster_score_adjustment", return_value=0)
    @patch("signal_scorer.check_manifold_gap", return_value=None)
    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_scorer.requests.post")
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_low_p_model_skip(self, mock_count, mock_get_cal, mock_post, mock_cb, mock_gap, mock_cluster):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        mock_post.return_value = mock_resp
        with _mock_settings(), patch("signal_scorer.parse_llm_json") as mock_parse:
            mock_parse.return_value = self._mock_llm_response(0.02, 0.40, [])
            with patch("order_manager.get_best_ask", return_value=None):
                result = sp.full_market_analysis(self._market())
        self.assertEqual(result["action"], "SKIP")

    @patch("signal_scorer._cluster_score_adjustment", return_value=0)
    @patch("signal_scorer.check_manifold_gap")
    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_scorer.requests.post")
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def test_metaculus_override_boosts_confidence(self, mock_count, mock_get_cal, mock_post, mock_cb, mock_gap, mock_cluster):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        mock_gap.return_value = {
            "found": True,
            "probability": 0.50,
            "polymarket_prob": 0.08,
            "signal_strength": 0.80,
            "source": "manifold",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        mock_post.return_value = mock_resp
        with _mock_settings(), patch("signal_scorer.parse_llm_json") as mock_parse:
            mock_parse.return_value = self._mock_llm_response(0.25, 0.70, [
                {"factor": "test", "direction": "supports", "weight": "high", "source": "test"},
            ])
            with patch("order_manager.get_best_ask", return_value=0.08):
                result = sp.full_market_analysis(self._market(price=0.08))
        self.assertEqual(result["source_signal"], "metaculus_override")


# ═══════════════════════════════════════════════════════════════════
# advisor_pre_check
# ═══════════════════════════════════════════════════════════════════
class TestAdvisorPreCheck(unittest.TestCase):
    def _market(self):
        return {"question": "Will X happen?", HYP_SLUG: "test-slug", "price": 0.10}

    def _analysis(self):
        return {
            HYP_P_MODEL: 0.30,
            HYP_CONFIDENCE: 0.80,
            HYP_FACTORS: [{"factor": "evidence", "direction": "supports", "weight": "high"}],
            "signal_score": 70,
            "reasoning": "Strong evidence supports this trade.",
        }

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=False)
    def test_circuit_breaker_blocks_large_position(self, mock_cb):
        approved, _verdict, _conf, reason = sp.advisor_pre_check(
            self._market(), self._analysis(), estimated_size=50, balance=100
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "advisor_circuit_breaker")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=False)
    def test_circuit_breaker_allows_micro(self, mock_cb):
        approved, _verdict, _conf, reason = sp.advisor_pre_check(
            self._market(), self._analysis(), estimated_size=1, balance=100
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "advisor_cb_micro_override")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_confirm_approved(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {
            "content": json.dumps({"p_estimate": 0.35, "confidence": 0.85, "factors": ["f1"], "verdict": "CONFIRM"}),
            "reasoning": "",
        }}]}
        mock_post.return_value = mock_resp
        mock_parser = MagicMock(return_value=({"p_estimate": 0.35, HYP_CONFIDENCE: 0.85, HYP_FACTORS: ["f1"], "verdict": "CONFIRM"}, None))
        with patch.dict("sys.modules", {"advisor_script": MagicMock(parse_llm_advisor_response=mock_parser)}):
            approved, verdict, _conf, _reason = sp.advisor_pre_check(
                self._market(), self._analysis(), estimated_size=5, balance=100
            )
        self.assertTrue(approved)
        self.assertEqual(verdict, "CONFIRM")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_diverge_blocks(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {
            "content": json.dumps({"p_estimate": 0.05, "confidence": 0.70, "factors": ["f1"], "verdict": "DIVERGE"}),
            "reasoning": "",
        }}]}
        mock_post.return_value = mock_resp
        mock_parser = MagicMock(return_value=({"p_estimate": 0.05, HYP_CONFIDENCE: 0.70, HYP_FACTORS: ["f1"], "verdict": "DIVERGE"}, None))
        with patch.dict("sys.modules", {"advisor_script": MagicMock(parse_llm_advisor_response=mock_parser)}):
            approved, verdict, _conf, _reason = sp.advisor_pre_check(
                self._market(), self._analysis(), estimated_size=10, balance=100
            )
        self.assertFalse(approved)
        self.assertEqual(verdict, "DIVERGE")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_diverge_micro_allowed(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {
            "content": json.dumps({"p_estimate": 0.05, "confidence": 0.70, "factors": ["f1"], "verdict": "DIVERGE"}),
            "reasoning": "",
        }}]}
        mock_post.return_value = mock_resp
        mock_parser = MagicMock(return_value=({"p_estimate": 0.05, HYP_CONFIDENCE: 0.70, HYP_FACTORS: ["f1"], "verdict": "DIVERGE"}, None))
        with patch.dict("sys.modules", {"advisor_script": MagicMock(parse_llm_advisor_response=mock_parser)}):
            approved, _verdict, _conf, reason = sp.advisor_pre_check(
                self._market(), self._analysis(), estimated_size=1, balance=100
            )
        self.assertTrue(approved)
        self.assertEqual(reason, "diverge_micro_override")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post", side_effect=requests.exceptions.Timeout("timed out"))
    def test_timeout_blocks(self, mock_post, mock_cb):
        approved, _verdict, _conf, reason = sp.advisor_pre_check(
            self._market(), self._analysis(), estimated_size=10, balance=100
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "advisor_timeout")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_unknown_verdict_blocks(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {
            "content": json.dumps({"p_estimate": 0.10, "confidence": 0.50, "factors": [], "verdict": "UNKNOWN"}),
            "reasoning": "",
        }}]}
        mock_post.return_value = mock_resp
        mock_parser = MagicMock(return_value=({"p_estimate": 0.10, HYP_CONFIDENCE: 0.50, HYP_FACTORS: [], "verdict": "UNKNOWN"}, None))
        with patch.dict("sys.modules", {"advisor_script": MagicMock(parse_llm_advisor_response=mock_parser)}):
            approved, _verdict, _conf, _reason = sp.advisor_pre_check(
                self._market(), self._analysis(), estimated_size=10, balance=100
            )
        self.assertFalse(approved)

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_warning_with_low_conf_allows(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {
            "content": json.dumps({"p_estimate": 0.15, "confidence": 0.50, "factors": ["risk"], "verdict": "WARNING"}),
            "reasoning": "",
        }}]}
        mock_post.return_value = mock_resp
        mock_parser = MagicMock(return_value=({"p_estimate": 0.15, HYP_CONFIDENCE: 0.50, HYP_FACTORS: ["risk"], "verdict": "WARNING"}, None))
        with patch.dict("sys.modules", {"advisor_script": MagicMock(parse_llm_advisor_response=mock_parser)}):
            approved, _verdict, _conf, reason = sp.advisor_pre_check(
                self._market(), self._analysis(), estimated_size=10, balance=100
            )
        self.assertTrue(approved)
        self.assertEqual(reason, "advisor_warning_allowed")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_parse_failure_blocks(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "NOT JSON", "reasoning": ""}}]}
        mock_post.return_value = mock_resp
        mock_parser = MagicMock(return_value=(None, "json decode error"))
        with patch.dict("sys.modules", {"advisor_script": MagicMock(parse_llm_advisor_response=mock_parser)}):
            approved, _verdict, _conf, _reason = sp.advisor_pre_check(
                self._market(), self._analysis(), estimated_size=10, balance=100
            )
        self.assertFalse(approved)

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_empty_response_blocks_large(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "", "reasoning": ""}}]}
        mock_post.return_value = mock_resp
        approved, _verdict, _conf, reason = sp.advisor_pre_check(
            self._market(), self._analysis(), estimated_size=50, balance=100
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "advisor_empty_response")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_empty_response_allows_micro(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "", "reasoning": ""}}]}
        mock_post.return_value = mock_resp
        approved, _verdict, _conf, reason = sp.advisor_pre_check(
            self._market(), self._analysis(), estimated_size=1, balance=100
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "advisor_empty_micro_override")

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post", side_effect=Exception("unexpected"))
    def test_generic_exception_blocks(self, mock_post, mock_cb):
        approved, _verdict, _conf, reason = sp.advisor_pre_check(
            self._market(), self._analysis(), estimated_size=10, balance=100
        )
        self.assertFalse(approved)
        self.assertIn("advisor_error", reason)

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_diverge_direction_agrees_overrides(self, mock_post, mock_cb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {
            "content": json.dumps({"p_estimate": 0.15, "confidence": 0.70, "factors": [], "verdict": "DIVERGE"}),
            "reasoning": "",
        }}]}
        mock_post.return_value = mock_resp
        mock_parser = MagicMock(return_value=({"p_estimate": 0.15, HYP_CONFIDENCE: 0.70, HYP_FACTORS: [], "verdict": "DIVERGE"}, None))
        with patch.dict("sys.modules", {"advisor_script": MagicMock(parse_llm_advisor_response=mock_parser)}):
            approved, _verdict, _conf, reason = sp.advisor_pre_check(
                self._market(), self._analysis(), estimated_size=10, balance=100
            )
        self.assertTrue(approved)
        self.assertEqual(reason, "diverge_direction_agrees")


# ═══════════════════════════════════════════════════════════════════
# Signal scoring — time_score branches
# ═══════════════════════════════════════════════════════════════════
class TestTimeScoreBranches(unittest.TestCase):
    @patch("signal_scorer._cluster_score_adjustment", return_value=0)
    @patch("signal_scorer.check_manifold_gap", return_value=None)
    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_scorer.requests.post")
    @patch("calibration.get_calibrator")
    @patch("signal_scorer._count_resolved_hypotheses", return_value=0)
    def _run_with_ttl(self, ttl_hours, mock_count, mock_get_cal, mock_post, mock_cb, mock_gap, mock_cluster):
        mock_cal = MagicMock()
        mock_cal.is_fitted = False
        mock_get_cal.return_value = mock_cal
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        mock_post.return_value = mock_resp
        market = {
            "question": "Q?", HYP_SLUG: "test", "price": 0.10,
            "volume": 500000, "liquidity": 5000,
            "end_date": (datetime.now(UTC) + timedelta(hours=ttl_hours)).isoformat(),
            "ttl_hours": ttl_hours, HYP_CLUSTERS: ["ai_tech"], "oracle_type": "uma",
        }
        with _mock_settings(), patch("signal_scorer.parse_llm_json") as mock_parse:
            mock_parse.return_value = {
                "estimated_probability": 0.30,
                HYP_CONFIDENCE: 0.80,
                HYP_FACTORS: [{"factor": "test", "direction": "supports", "weight": "high", "source": "test"}],
            }
            with patch("order_manager.get_best_ask", return_value=None):
                return sp.full_market_analysis(market)

    def test_ttl_over_180_days_high_time_score(self):
        result = self._run_with_ttl(181 * 24)
        self.assertGreater(result.get("signal_score", 0), 55)

    def test_ttl_90_to_180_days(self):
        result = self._run_with_ttl(120 * 24)
        self.assertIsNotNone(result)

    def test_ttl_30_to_90_days(self):
        result = self._run_with_ttl(60 * 24)
        self.assertIsNotNone(result)

    def test_ttl_14_to_30_days(self):
        result = self._run_with_ttl(20 * 24)
        self.assertIsNotNone(result)

    def test_ttl_2_to_14_days(self):
        result = self._run_with_ttl(5 * 24)
        self.assertIsNotNone(result)

    def test_ttl_under_2_days_zero_time_score(self):
        result = self._run_with_ttl(24)
        self.assertIsNotNone(result)


# ═══════════════════════════════════════════════════════════════════
# Signal thresholds / constants
# ═══════════════════════════════════════════════════════════════════
class TestConstants(unittest.TestCase):
    def test_key_constants_exist(self):
        self.assertEqual(sp.MIN_PROB_RATIO, 1.5)
        self.assertEqual(sp.MIN_P_MODEL, 0.03)
        self.assertEqual(sp.MAX_P_MODEL_RATIO, 2.0)
        self.assertEqual(sp.MIN_CONFIDENCE, 0.65)
        self.assertEqual(sp.MIN_VOLUME, 25000)
        self.assertEqual(sp.MIN_TTL_HOURS, 48)
        self.assertEqual(sp.MAX_PRICE, 0.40)
        self.assertIn("crypto", sp.BANNED_CLUSTERS)
        self.assertIn("ai_tech", sp.ALLOWED_CLUSTERS)
        self.assertEqual(sp.BATCH_SIZE, 6)
        self.assertEqual(sp.METACULUS_GAP_THRESHOLD, 0.08)
        self.assertEqual(sp.ADVISOR_MIN_CONFIDENCE, 0.70)

    def test_cluster_score_adjustments(self):
        self.assertEqual(sp.CLUSTER_SCORE_ADJUSTMENTS["other"], 15)
        self.assertEqual(sp.CLUSTER_SCORE_ADJUSTMENTS["crypto"], -25)
        self.assertEqual(sp.CLUSTER_SCORE_ADJUSTMENTS["sports_nba"], -15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
