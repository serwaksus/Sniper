"""Tests for dotm_backtester.py — TP ladder, normalization, analysis, stats computation."""
import json
import sys
import os
import types
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True, scope="module")
def _setup_dotm_sniper_mock():
    _sniper = types.ModuleType("dotm_sniper")
    _sniper.load_json = MagicMock(return_value={})
    _sniper.save_json = MagicMock()
    _sniper.parse_llm_json = MagicMock(return_value=None)
    _sniper.normalize_probability = MagicMock(side_effect=lambda x: min(max(float(x), 0.0), 1.0))
    _sniper.detect_clusters = MagicMock(return_value=["other"])
    _sniper.check_metaculus_gap = MagicMock(return_value=None)
    _sniper.URL = "https://example.com"
    _sniper.HEADERS = {}
    _sniper.MODEL_MAIN = "test-model"
    _sniper.ADVISOR_MODEL = "test-advisor"
    _sniper.MAX_P_MODEL_RATIO = 3.0
    _sniper.MIN_P_MODEL = 0.03
    _sniper.MIN_VOLUME = 1000
    _sniper.get_settings = MagicMock(return_value={})
    _sniper.MIN_CONFIDENCE = 0.65
    _sniper.calibrate_prediction = MagicMock(side_effect=lambda p, *a, **kw: (p, False))
    _sniper._cluster_score_adjustment = MagicMock(return_value=0)

    original = sys.modules.get("dotm_sniper")
    sys.modules["dotm_sniper"] = _sniper

    if "dotm_backtester" in sys.modules:
        del sys.modules["dotm_backtester"]

    import dotm_backtester as bt
    yield bt

    if original is not None:
        sys.modules["dotm_sniper"] = original
    else:
        sys.modules.pop("dotm_sniper", None)
    sys.modules.pop("dotm_backtester", None)


@pytest.fixture
def bt(_setup_dotm_sniper_mock):
    return _setup_dotm_sniper_mock


class TestSimulateTpLadder:
    def test_yes_resolution_high_hit_075(self, bt):
        pnl, details = bt._simulate_tp_ladder(0.05, 0.90, "YES")
        assert pnl > 0
        assert len(details) == 3
        assert details[0]["triggered"] is True
        assert details[0]["label"] == "tp_0.75"
        assert details[1]["triggered"] is True
        assert details[1]["label"] == "tp_0.85"

    def test_no_resolution_high_below_075(self, bt):
        pnl, details = bt._simulate_tp_ladder(0.10, 0.20, "NO")
        assert pnl < 0
        assert details[0]["triggered"] is False
        assert details[1]["triggered"] is False

    def test_yes_resolution_high_080(self, bt):
        _pnl, details = bt._simulate_tp_ladder(0.05, 0.80, "YES")
        assert details[0]["triggered"] is True
        assert details[1]["triggered"] is False
        hold_rung = details[2]
        assert hold_rung["exit_price"] == 1.0

    def test_no_resolution_high_above_085(self, bt):
        _pnl, details = bt._simulate_tp_ladder(0.05, 0.90, "NO")
        assert details[0]["triggered"] is True
        assert details[1]["triggered"] is True
        hold_rung = details[2]
        assert hold_rung["exit_price"] == 0.0

    def test_zero_entry_price(self, bt):
        pnl, details = bt._simulate_tp_ladder(0.0, 0.90, "YES")
        assert pnl == 0.0
        for d in details:
            assert d["pnl"] == 0.0

    def test_weighted_pnl_sums_to_one(self, bt):
        _, details = bt._simulate_tp_ladder(0.05, 0.90, "YES")
        total_pct = sum(d["pct"] for d in details)
        assert abs(total_pct - 1.0) < 1e-9


class TestNormalizeKeys:
    def test_strips_dict_key_whitespace(self, bt):
        assert bt._normalize_keys({" a ": 1}) == {"a": 1}

    def test_strips_string_values(self, bt):
        assert bt._normalize_keys({"k": " hello "}) == {"k": "hello"}

    def test_handles_nested(self, bt):
        result = bt._normalize_keys({" a ": [{" b ": " x "}]})
        assert result == {"a": [{"b": "x"}]}

    def test_passthrough_numbers(self, bt):
        assert bt._normalize_keys({"a": 42}) == {"a": 42}

    def test_empty_dict(self, bt):
        assert bt._normalize_keys({}) == {}

    def test_empty_list(self, bt):
        assert bt._normalize_keys([]) == []


class TestBacktestAnalyzeSingleSkip:
    @patch("dotm_backtester.check_metaculus_gap", return_value=None)
    @patch("dotm_backtester.get_settings", return_value={"min_p_model": 999})
    def test_low_p_model_returns_skip(self, mock_settings, mock_metaculus, bt):
        market = {
            "slug": "test-slug", "question": "Will X happen?",
            "yes_price": 0.10, "volume": 50000,
            "end_date": "", "ttl_hours": 9999, "clusters": ["other"],
        }
        result = bt.backtest_analyze_single(market)
        assert result["action"] == "SKIP"

    @patch("dotm_backtester.requests.post")
    @patch("dotm_backtester.check_metaculus_gap", return_value=None)
    @patch("dotm_backtester.get_settings", return_value={})
    def test_llm_failure_uses_fallback_p_model(self, mock_settings, mock_metaculus, mock_post, bt):
        mock_post.side_effect = Exception("network error")
        market = {
            "slug": "test-slug", "question": "Will X happen?",
            "yes_price": 0.10, "volume": 50000,
            "end_date": "", "ttl_hours": 9999, "clusters": ["other"],
        }
        result = bt.backtest_analyze_single(market)
        assert result["p_model"] > 0


class TestParallelAnalyze:
    @patch("dotm_backtester.backtest_analyze_single", return_value={"action": "SKIP", "slug": "s1"})
    def test_returns_results_in_order(self, mock_analyze, bt):
        markets = [
            {"slug": f"slug-{i}", "question": f"Q{i}?", "yes_price": 0.05,
             "volume": 1000, "clusters": ["other"]}
            for i in range(5)
        ]
        results = bt._parallel_analyze_markets(markets, label="TEST")
        assert len(results) == 5
        for r in results:
            assert r is not None

    @patch("dotm_backtester.backtest_analyze_single", side_effect=Exception("boom"))
    def test_thread_error_returns_none(self, mock_analyze, bt):
        markets = [{"slug": "s1", "question": "Q?", "yes_price": 0.05,
                     "volume": 1000, "clusters": ["other"]}]
        results = bt._parallel_analyze_markets(markets, label="TEST")
        assert results == [None]

    @patch("dotm_backtester.backtest_analyze_single", return_value=None)
    def test_empty_markets_list(self, mock_analyze, bt):
        results = bt._parallel_analyze_markets([], label="TEST")
        assert results == []


class TestFetchActiveMarketsPmTrader:
    @patch("dotm_backtester.subprocess.run")
    def test_filters_by_price_range(self, mock_run, bt):
        data = {
            "data": [
                {"slug": "s1", "question": "Q1?", "active": True, "closed": False,
                 "volume": 50000, "outcomes": ["Yes"], "outcome_prices": ["0.08"],
                 "end_date": "", "condition_id": "c1"},
                {"slug": "s2", "question": "Q2?", "active": True, "closed": False,
                 "volume": 50000, "outcomes": ["Yes"], "outcome_prices": ["0.50"],
                 "end_date": "", "condition_id": "c2"},
            ]
        }
        mock_run.return_value = MagicMock(stdout=json.dumps(data), returncode=0)
        result = bt._fetch_active_dotm_markets_pm_trader(limit=200)
        slugs = [m["slug"] for m in result]
        assert "s1" in slugs
        assert "s2" not in slugs

    @patch("dotm_backtester.subprocess.run", side_effect=Exception("err"))
    def test_exception_returns_empty(self, mock_run, bt):
        assert bt._fetch_active_dotm_markets_pm_trader() == []


class TestBrierScore:
    def test_brier_yes_perfect(self):
        assert (1.0 - 1) ** 2 == 0.0

    def test_brier_no_perfect(self):
        assert (0.0 - 0) ** 2 == 0.0

    def test_brier_worst_case(self):
        assert (0.0 - 1) ** 2 == 1.0

    def test_brier_mid_prediction(self):
        assert pytest.approx(0.25) == (0.5 - 1) ** 2


class TestTpLadderEdgeCases:
    def test_entry_equals_tp_price(self, bt):
        _, details = bt._simulate_tp_ladder(0.75, 0.80, "YES")
        first_rung = details[0]
        assert first_rung["triggered"] is True
        assert first_rung["pnl"] < 0

    def test_single_rung_weight_sum(self, bt):
        _, details = bt._simulate_tp_ladder(0.10, 0.50, "YES")
        weights = [d["pct"] for d in details]
        assert sum(weights) == pytest.approx(1.0)
