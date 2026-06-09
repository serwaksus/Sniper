"""Tests for bayesian_updater.py — posterior init, update, exit logic, logodds, LLM classification."""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bayesian_updater as bu


class TestProbLogoddsConversion:
    def test_roundtrip_0_5(self):
        lo = bu._prob_to_logodds(0.5)
        p = bu._logodds_to_prob(lo)
        assert abs(p - 0.5) < 1e-6

    def test_roundtrip_0_1(self):
        lo = bu._prob_to_logodds(0.1)
        p = bu._logodds_to_prob(lo)
        assert abs(p - 0.1) < 1e-4

    def test_roundtrip_0_9(self):
        lo = bu._prob_to_logodds(0.9)
        p = bu._logodds_to_prob(lo)
        assert abs(p - 0.9) < 1e-4

    def test_zero_clamped(self):
        lo = bu._prob_to_logodds(0.0)
        p = bu._logodds_to_prob(lo)
        assert p < 1e-6

    def test_one_clamped(self):
        lo = bu._prob_to_logodds(1.0)
        p = bu._logodds_to_prob(lo)
        assert p > 1 - 1e-6

    def test_extreme_logodds_positive(self):
        assert bu._logodds_to_prob(600) == 1.0

    def test_extreme_logodds_negative(self):
        assert bu._logodds_to_prob(-600) == 0.0

    def test_symmetry(self):
        lo_pos = bu._prob_to_logodds(0.8)
        lo_neg = bu._prob_to_logodds(0.2)
        assert abs(abs(lo_pos) - abs(lo_neg)) < 1e-4


class TestInitPosterior:
    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json", return_value={"positions": {}})
    def test_creates_state(self, mock_load, mock_save):
        bu.init_posterior("test-slug", 0.15, 0.10)
        assert mock_save.called
        saved_data = mock_save.call_args[0][1]
        pos = saved_data["positions"]["test-slug"]
        assert abs(pos["p_model_entry"] - 0.15) < 1e-6
        assert abs(pos["p_market_entry"] - 0.10) < 1e-6
        assert pos["updates"] == 0

    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json", return_value="not a dict")
    def test_handles_corrupt_state(self, mock_load, mock_save):
        bu.init_posterior("test-slug", 0.15, 0.10)
        assert mock_save.called


class TestUpdatePosterior:
    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json")
    def test_strongly_supports_increases_posterior(self, mock_load, mock_save):
        mock_load.return_value = {"positions": {
            "test-slug": {
                "posterior_logodds": bu._prob_to_logodds(0.15),
                "p_model_entry": 0.15,
                "updates": 0,
                "history": [],
            }
        }}
        result = bu.update_posterior("test-slug", "strongly_supports")
        assert result is not None
        assert result > 0.15

    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json")
    def test_confirms_impossible_decreases_posterior(self, mock_load, mock_save):
        mock_load.return_value = {"positions": {
            "test-slug": {
                "posterior_logodds": bu._prob_to_logodds(0.15),
                "p_model_entry": 0.15,
                "updates": 0,
                "history": [],
            }
        }}
        result = bu.update_posterior("test-slug", "confirms_impossible")
        assert result is not None
        assert result < 0.15

    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json")
    def test_neutral_stays_similar(self, mock_load, mock_save):
        mock_load.return_value = {"positions": {
            "test-slug": {
                "posterior_logodds": bu._prob_to_logodds(0.15),
                "p_model_entry": 0.15,
                "updates": 0,
                "history": [],
            }
        }}
        result = bu.update_posterior("test-slug", "neutral")
        assert result is not None
        assert abs(result - 0.15) < 0.01

    @patch("bayesian_updater.load_json", return_value={"positions": {}})
    def test_missing_slug_returns_none(self, mock_load):
        result = bu.update_posterior("nonexistent", "neutral")
        assert result is None

    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json")
    def test_all_categories_move_correctly(self, mock_load, mock_save):
        positive_cats = ["moderately_supports", "strongly_supports", "confirms_inevitable"]
        negative_cats = ["confirms_impossible", "strongly_contradicts", "moderately_contradicts"]

        for cat in positive_cats:
            mock_load.return_value = {"positions": {
                "s": {"posterior_logodds": bu._prob_to_logodds(0.15), "p_model_entry": 0.15, "updates": 0, "history": []}
            }}
            result = bu.update_posterior("s", cat)
            assert result > 0.15, f"{cat} should increase posterior"

        for cat in negative_cats:
            mock_load.return_value = {"positions": {
                "s": {"posterior_logodds": bu._prob_to_logodds(0.15), "p_model_entry": 0.15, "updates": 0, "history": []}
            }}
            result = bu.update_posterior("s", cat)
            assert result < 0.15, f"{cat} should decrease posterior"

    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json")
    def test_unknown_category_uses_neutral(self, mock_load, mock_save):
        mock_load.return_value = {"positions": {
            "s": {"posterior_logodds": bu._prob_to_logodds(0.15), "p_model_entry": 0.15, "updates": 0, "history": []}
        }}
        result = bu.update_posterior("s", "totally_unknown_category")
        assert result is not None


class TestShouldExit:
    @patch("bayesian_updater.load_json")
    def test_triggers_at_40pct_drop_ratio(self, mock_load):
        entry = 0.15
        current = entry * 0.35
        mock_load.return_value = {"positions": {
            "s": {"posterior_prob": current, "p_model_entry": entry}
        }}
        should, _reason = bu.should_exit("s")
        assert should is True

    @patch("bayesian_updater.load_json")
    def test_no_exit_when_ok(self, mock_load):
        entry = 0.15
        current = entry * 0.80
        mock_load.return_value = {"positions": {
            "s": {"posterior_prob": current, "p_model_entry": entry}
        }}
        should, _reason = bu.should_exit("s")
        assert should is False

    @patch("bayesian_updater.load_json")
    def test_near_zero_triggers(self, mock_load):
        mock_load.return_value = {"positions": {
            "s": {"posterior_prob": 0.01, "p_model_entry": 0.15}
        }}
        should, _reason = bu.should_exit("s")
        assert should is True

    @patch("bayesian_updater.load_json", return_value={"positions": {}})
    def test_missing_slug_no_exit(self, mock_load):
        should, _reason = bu.should_exit("nonexistent")
        assert should is False

    @patch("bayesian_updater.load_json")
    def test_zero_entry_no_exit(self, mock_load):
        mock_load.return_value = {"positions": {
            "s": {"posterior_prob": 0.05, "p_model_entry": 0}
        }}
        should, _reason = bu.should_exit("s")
        assert should is False

    @patch("bayesian_updater.load_json", return_value="not a dict")
    def test_corrupt_state_no_exit(self, mock_load):
        should, _reason = bu.should_exit("s")
        assert should is False


class TestCleanupSlug:
    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json")
    def test_removes_slug(self, mock_load, mock_save):
        mock_load.return_value = {"positions": {"slug-a": {}, "slug-b": {}}}
        bu.cleanup_slug("slug-a")
        saved = mock_save.call_args[0][1]
        assert "slug-a" not in saved["positions"]
        assert "slug-b" in saved["positions"]

    @patch("bayesian_updater.save_json")
    @patch("bayesian_updater.load_json", return_value="not a dict")
    def test_corrupt_state_no_error(self, mock_load, mock_save):
        bu.cleanup_slug("slug-a")


class TestClassifyNewsWithLLM:
    @patch("requests.post")
    def test_llm_returns_category(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "strongly_supports"}}]}
        mock_post.return_value = mock_resp
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        result = bu.classify_news_with_llm("Will X happen?", ["News headline 1"])
        assert result == "strongly_supports"

    def test_empty_headlines_returns_neutral(self):
        assert bu.classify_news_with_llm("Q?", []) == "neutral"

    @patch("requests.post", side_effect=Exception("fail"))
    def test_exception_returns_neutral(self, mock_post):
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        assert bu.classify_news_with_llm("Q?", ["headline"]) == "neutral"
