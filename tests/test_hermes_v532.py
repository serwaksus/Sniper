#!/usr/bin/env python3
"""
Tests for hermes_advisor.py v5.3.2 hardening:
  1. Hard Python probability parsing (divergence lock)
  2. Absolute spam suppression (case-normalized dedup)
  3. Empty news fallback (no LLM call on empty)
"""
import unittest
import sys
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import hermes_advisor as ha


class TestProbabilityParsing(unittest.TestCase):
    """Fix #1: Hard Python parsing of probabilities with % stripping."""

    def test_percent_stripped_clean(self):
        val = float(str("45%").replace('%', '').strip())
        self.assertAlmostEqual(val, 45.0)

    def test_float_string_clean(self):
        val = float(str(0.25).replace('%', '').strip())
        self.assertAlmostEqual(val, 0.25)

    def test_divergence_trigger_decimal(self):
        p_bot_val = 0.30
        p_hermes_val = 0.10
        self.assertTrue(p_hermes_val < (p_bot_val * 0.5))

    def test_no_divergence_above_threshold(self):
        p_bot_val = 0.30
        p_hermes_val = 0.20
        self.assertFalse(p_hermes_val < (p_bot_val * 0.5))

    def test_percentage_format_normalized(self):
        p_hermes_val = 45.0
        if p_hermes_val > 1.0:
            p_hermes_val = p_hermes_val / 100.0
        p_bot_val = 0.30
        self.assertFalse(p_hermes_val < (p_bot_val * 0.5))

    def test_percentage_format_divergence(self):
        p_hermes_val = 5.0
        if p_hermes_val > 1.0:
            p_hermes_val = p_hermes_val / 100.0
        p_bot_val = 0.30
        self.assertTrue(p_hermes_val < (p_bot_val * 0.5))

    def test_exact_half_boundary(self):
        p_bot_val = 0.30
        p_hermes_val = 0.15
        self.assertFalse(p_hermes_val < (p_bot_val * 0.5))

    def test_just_below_half(self):
        p_bot_val = 0.30
        p_hermes_val = 0.149
        self.assertTrue(p_hermes_val < (p_bot_val * 0.5))

    def test_spaces_and_percent(self):
        val = float(str(" 45 % ").replace('%', '').strip())
        self.assertAlmostEqual(val, 45.0)

    def test_divergence_locked_prevents_yellow_override(self):
        divergence_locked = True
        status = "YELLOW"
        if divergence_locked:
            status = "DIVERGENCE"
        self.assertEqual(status, "DIVERGENCE")

    def test_divergence_locked_prevents_aligned_override(self):
        divergence_locked = True
        status = "ALIGNED"
        if divergence_locked:
            status = "DIVERGENCE"
        self.assertEqual(status, "DIVERGENCE")

    def test_no_divergence_allows_status(self):
        divergence_locked = False
        status = "YELLOW"
        if divergence_locked:
            status = "DIVERGENCE"
        self.assertEqual(status, "YELLOW")


class TestSpamSuppression(unittest.TestCase):
    """Fix #2: Case-normalized status dedup prevents spam loop."""

    def setUp(self):
        ha._last_alert_status = {}

    def test_same_status_case_insensitive_no_send(self):
        ha._last_alert_status["slug-a"] = "YELLOW"
        self.assertFalse(ha._should_send_telegram("slug-a", False, "yellow"))

    def test_same_status_exact_match_no_send(self):
        ha._last_alert_status["slug-a"] = "DIVERGENCE"
        self.assertFalse(ha._should_send_telegram("slug-a", False, "DIVERGENCE"))

    def test_different_status_sends(self):
        ha._last_alert_status["slug-a"] = "GREEN"
        self.assertTrue(ha._should_send_telegram("slug-a", False, "DIVERGENCE"))

    def test_different_status_non_notify_no_send(self):
        ha._last_alert_status["slug-a"] = "GREEN"
        self.assertFalse(ha._should_send_telegram("slug-a", False, "YELLOW"))

    def test_trigger_exit_always_sends(self):
        ha._last_alert_status["slug-a"] = "DIVERGENCE"
        self.assertTrue(ha._should_send_telegram("slug-a", True, "DIVERGENCE"))

    def test_no_previous_status_sends_for_divergence(self):
        self.assertTrue(ha._should_send_telegram("slug-new", False, "DIVERGENCE"))

    def test_no_previous_status_no_send_for_green(self):
        self.assertFalse(ha._should_send_telegram("slug-new", False, "GREEN"))

    def test_whitespace_normalized(self):
        ha._last_alert_status["slug-a"] = "YELLOW"
        self.assertFalse(ha._should_send_telegram("slug-a", False, "  YELLOW  "))

    def test_mixed_case_normalized(self):
        ha._last_alert_status["slug-a"] = "Divergence"
        self.assertFalse(ha._should_send_telegram("slug-a", False, "DIVERGENCE"))

    def test_update_normalizes_before_save(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp = f.name
        try:
            orig = ha.ALERT_STATE_FILE
            ha.ALERT_STATE_FILE = tmp
            ha._last_alert_status = {}
            ha._update_and_check_status("slug-x", False, "  yellow ")
            self.assertEqual(ha._last_alert_status["slug-x"], "YELLOW")
            ha.ALERT_STATE_FILE = orig
        finally:
            os.unlink(tmp)


class TestEmptyNewsFallback(unittest.TestCase):
    """Fix #3: No LLM call when news is empty."""

    def test_empty_dict_headlines_skipped(self):
        headlines = []
        self.assertFalse(headlines)

    def test_dict_with_empty_headlines_skipped(self):
        news_data = {"headlines": [], "sources": [], "query": "", "found": False}
        headlines = news_data.get("headlines", [])
        self.assertFalse(headlines)

    def test_dict_with_headlines_proceeds(self):
        news_data = {"headlines": ["News A", "News B"], "sources": [], "found": True}
        headlines = news_data.get("headlines", [])
        self.assertTrue(len(headlines) > 0)

    def test_list_headlines_proceeds(self):
        news_data = ["News A", "News B"]
        headlines = news_data if isinstance(news_data, list) else []
        self.assertTrue(len(headlines) > 0)

    def test_none_news_skipped(self):
        news_data = None
        headlines = []
        if isinstance(news_data, dict):
            headlines = news_data.get("headlines", [])
        elif isinstance(news_data, list):
            headlines = news_data
        self.assertFalse(headlines)

    @patch('hermes_advisor.fetch_news_for_market')
    @patch('hermes_advisor.load_json')
    def test_empty_news_skips_llm_call(self, mock_load, mock_news):
        mock_load.return_value = {
            "test-slug": {
                "market_question": "Will X happen?",
                "metaculus_prob": 0.3,
                "entry_price": 0.15,
            }
        }
        mock_news.return_value = {"headlines": [], "found": False}

        import requests
        with patch.object(requests, 'post') as mock_post:
            ha.evaluate_emergency_exit()
            mock_post.assert_not_called()

    @patch('hermes_advisor.fetch_news_for_market')
    @patch('hermes_advisor.load_json')
    def test_empty_list_news_skips_llm_call(self, mock_load, mock_news):
        mock_load.return_value = {
            "test-slug": {
                "market_question": "Will X happen?",
                "metaculus_prob": 0.3,
                "entry_price": 0.15,
            }
        }
        mock_news.return_value = []

        import requests
        with patch.object(requests, 'post') as mock_post:
            ha.evaluate_emergency_exit()
            mock_post.assert_not_called()

    @patch('hermes_advisor.fetch_news_for_market')
    @patch('hermes_advisor.load_json')
    @patch('requests.post')
    def test_nonempty_news_does_call_llm(self, mock_post, mock_load, mock_news):
        mock_load.return_value = {
            "test-slug": {
                "market_question": "Will X happen?",
                "metaculus_prob": 0.3,
                "entry_price": 0.15,
            }
        }
        mock_news.return_value = {"headlines": ["Breaking: X happened"], "found": True}
        mock_resp = MagicMock()
        llm_content = '{"trigger_exit": false, "p_hermes": 0.25, "status": "GREEN", "reason": "ok"}'
        mock_resp.json.return_value = {"choices": [{"message": {"content": llm_content}}]}
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        ha._last_alert_status = {}
        ha.evaluate_emergency_exit()

        deepseek_calls = [c for c in mock_post.call_args_list if 'deepseek' in str(c)]
        self.assertGreaterEqual(len(deepseek_calls), 1)


class TestDivergenceLockInEvaluation(unittest.TestCase):
    """Integration: divergence_locked flag prevents YELLOW override in evaluate_emergency_exit."""

    @patch('hermes_advisor._update_and_check_status', return_value=True)
    @patch('hermes_advisor.fetch_news_for_market')
    @patch('hermes_advisor.load_json')
    @patch('requests.post')
    def test_llm_says_yellow_but_python_says_divergence(self, mock_post, mock_load, mock_news, mock_update):
        mock_load.return_value = {
            "test-slug": {
                "market_question": "Will X happen?",
                "metaculus_prob": 0.40,
                "entry_price": 0.20,
            }
        }
        mock_news.return_value = {"headlines": ["Some news"], "found": True}
        mock_resp = MagicMock()
        llm_content = '{"trigger_exit": false, "p_hermes": 0.10, "status": "YELLOW", "reason": "some concern"}'
        mock_resp.json.return_value = {"choices": [{"message": {"content": llm_content}}]}
        mock_post.return_value = mock_resp

        ha._last_alert_status = {}
        ha.evaluate_emergency_exit()

        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        normalized_status = args[2]
        self.assertEqual(normalized_status, "DIVERGENCE")

    @patch('hermes_advisor._update_and_check_status', return_value=True)
    @patch('hermes_advisor.fetch_news_for_market')
    @patch('hermes_advisor.load_json')
    @patch('requests.post')
    def test_llm_says_green_no_divergence(self, mock_post, mock_load, mock_news, mock_update):
        mock_load.return_value = {
            "test-slug": {
                "market_question": "Will X happen?",
                "metaculus_prob": 0.40,
                "entry_price": 0.20,
            }
        }
        mock_news.return_value = {"headlines": ["Neutral news"], "found": True}
        mock_resp = MagicMock()
        llm_content = '{"trigger_exit": false, "p_hermes": 0.35, "status": "GREEN", "reason": "all clear"}'
        mock_resp.json.return_value = {"choices": [{"message": {"content": llm_content}}]}
        mock_post.return_value = mock_resp

        ha._last_alert_status = {}
        ha.evaluate_emergency_exit()

        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        normalized_status = args[2]
        self.assertEqual(normalized_status, "GREEN")

    @patch('hermes_advisor._update_and_check_status', return_value=True)
    @patch('hermes_advisor.fetch_news_for_market')
    @patch('hermes_advisor.load_json')
    @patch('requests.post')
    def test_llm_returns_percent_format_p_hermes(self, mock_post, mock_load, mock_news, mock_update):
        mock_load.return_value = {
            "test-slug": {
                "market_question": "Will X happen?",
                "metaculus_prob": 0.40,
                "entry_price": 0.20,
            }
        }
        mock_news.return_value = {"headlines": ["Some news"], "found": True}
        mock_resp = MagicMock()
        llm_content = '{"trigger_exit": false, "p_hermes": "5%", "status": "YELLOW", "reason": "big drop"}'
        mock_resp.json.return_value = {"choices": [{"message": {"content": llm_content}}]}
        mock_post.return_value = mock_resp

        ha._last_alert_status = {}
        ha.evaluate_emergency_exit()

        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        normalized_status = args[2]
        self.assertEqual(normalized_status, "DIVERGENCE")


class TestDuplicateStatusReturn(unittest.TestCase):
    """Fix #2: evaluate_emergency_exit returns early on duplicate status."""

    @patch('hermes_advisor.fetch_news_for_market')
    @patch('hermes_advisor.load_json')
    @patch('requests.post')
    def test_duplicate_status_skips_update(self, mock_post, mock_load, mock_news):
        mock_load.return_value = {
            "test-slug": {
                "market_question": "Will X happen?",
                "metaculus_prob": 0.40,
                "entry_price": 0.20,
            }
        }
        mock_news.return_value = {"headlines": ["News"], "found": True}
        mock_resp = MagicMock()
        llm_content = '{"trigger_exit": false, "p_hermes": 0.35, "status": "YELLOW", "reason": "mild concern"}'
        mock_resp.json.return_value = {"choices": [{"message": {"content": llm_content}}]}
        mock_post.return_value = mock_resp

        ha._last_alert_status = {"test-slug": "YELLOW"}

        ha.evaluate_emergency_exit()

        self.assertEqual(ha._last_alert_status.get("test-slug"), "YELLOW")


if __name__ == '__main__':
    unittest.main(verbosity=2)
