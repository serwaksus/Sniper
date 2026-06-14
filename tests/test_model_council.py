"""Tests for model_council module — council + judge architecture."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import model_council


class TestParseJsonArray:
    def test_simple_array(self):
        content = '[{"slug": "a", "estimated_probability": 0.15}]'
        result = model_council._parse_json_array(content)
        assert result is not None and len(result) == 1

    def test_markdown_fenced(self):
        content = '```json\n[{"slug": "x", "estimated_probability": 0.3}]\n```'
        result = model_council._parse_json_array(content)
        assert result is not None and result[0]["slug"] == "x"

    def test_multiple(self):
        content = '[{"slug": "a", "estimated_probability": 0.1}, {"slug": "b", "estimated_probability": 0.2}]'
        result = model_council._parse_json_array(content)
        assert result is not None and len(result) == 2

    def test_empty(self):
        assert model_council._parse_json_array("") is None
        assert model_council._parse_json_array(None) is None

    def test_invalid(self):
        assert model_council._parse_json_array("not json") is None

    def test_leading_colon(self):
        content = ':[{"slug": "a", "estimated_probability": 0.1}]'
        assert model_council._parse_json_array(content) is not None

    def test_individual_objects_fallback(self):
        content = 'text {"slug": "a", "estimated_probability": 0.1} more'
        result = model_council._parse_json_array(content)
        assert result is not None and len(result) == 1


class TestParseJsonObject:
    def test_simple(self):
        content = '{"estimated_probability": 0.25, "confidence": 0.7}'
        result = model_council._parse_json_object(content)
        assert result is not None and result["estimated_probability"] == 0.25

    def test_markdown(self):
        content = '```json\n{"estimated_probability": 0.3}\n```'
        assert model_council._parse_json_object(content)["estimated_probability"] == 0.3

    def test_empty(self):
        assert model_council._parse_json_object("") is None
        assert model_council._parse_json_object(None) is None

    def test_no_json(self):
        assert model_council._parse_json_object("text") is None


class TestSafeFloat:
    def test_int(self):
        assert model_council._safe_float(42) == 42.0

    def test_float(self):
        assert model_council._safe_float(0.15) == 0.15

    def test_string(self):
        assert model_council._safe_float("0.25") == 0.25

    def test_percent(self):
        assert model_council._safe_float("25%") == 0.25

    def test_invalid(self):
        assert model_council._safe_float("abc", 0.5) == 0.5

    def test_none(self):
        assert model_council._safe_float(None, 0.3) == 0.3


class TestExtractJsonFromReasoning:
    def test_extract_array(self):
        reasoning = 'Thinking...\n[{"slug": "a", "estimated_probability": 0.2}]'
        result = model_council._extract_json_from_reasoning(reasoning)
        assert result and json.loads(result) is not None

    def test_extract_object(self):
        reasoning = 'Analysis...\n{"estimated_probability": 0.3}'
        result = model_council._extract_json_from_reasoning(reasoning)
        assert result and json.loads(result)["estimated_probability"] == 0.3

    def test_no_json(self):
        assert model_council._extract_json_from_reasoning("just text") == ""


class TestBuildJudgePrompt:
    def test_batch_prompt_has_estimates(self):
        estimates = {
            "slug-a": [
                {"model": "deepseek-chat", "p": 0.12, "confidence": 0.7},
                {"model": "gpt-oss-120b", "p": 0.08, "confidence": 0.65},
            ]
        }
        items = [{"slug": "slug-a", "question": "Test?", "market_price": 0.05, "estimates": estimates["slug-a"]}]
        prompt = model_council._build_judge_prompt_batch(estimates, items)
        assert "deepseek-chat" in prompt
        assert "gpt-oss-120b" in prompt
        assert "mean=" in prompt
        assert "median=" in prompt
        assert "std=" in prompt

    def test_single_prompt_has_estimates(self):
        estimates = [
            {"model": "deepseek-chat", "p": 0.12, "confidence": 0.7},
            {"model": "gpt-oss-120b", "p": 0.08, "confidence": 0.65},
        ]
        prompt = model_council._build_judge_prompt_single("slug", "Test question?", 0.05, estimates)
        assert "deepseek-chat" in prompt
        assert "Test question?" in prompt
        assert "mean=" in prompt


class TestCouncilBatchConsensus:
    def test_ovh_disabled_returns_deepseek(self):
        with patch.object(model_council, "is_ovh_enabled", return_value=False):
            ds = [{"slug": "a", "estimated_probability": 0.15, "confidence": 0.7}]
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], ds)
            assert merged == ds
            assert meta["advisors_queried"] == []

    def test_judge_overrides_deepseek(self):
        """Judge model should override DeepSeek estimates."""
        # Mock OVH advisor responses
        advisor_response = MagicMock()
        advisor_response.status_code = 200
        advisor_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"slug": "a", "estimated_probability": 0.20, "confidence": 0.8}
            ])}}]
        }

        # Mock judge response
        judge_response = MagicMock()
        judge_response.status_code = 200
        judge_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"slug": "a", "estimated_probability": 0.18, "confidence": 0.85, "reasoning": "synthesis"}
            ])}}]
        }

        call_count = [0]
        def mock_post(*args, **kwargs):
            call_count[0] += 1
            # Last call is the judge
            if call_count[0] > len(model_council.OVH_ADVISORS):
                return judge_response
            return advisor_response

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", side_effect=mock_post):
            ds = [{"slug": "a", "estimated_probability": 0.10, "confidence": 0.7}]
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], ds)

            assert meta["judge_ok"] is True
            assert len(merged) == 1
            # Judge's estimate should override
            assert merged[0]["estimated_probability"] == 0.18

    def test_judge_fails_fallback_to_average(self):
        """When judge fails, should fall back to confidence-weighted average."""
        advisor_response = MagicMock()
        advisor_response.status_code = 200
        advisor_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"slug": "a", "estimated_probability": 0.20, "confidence": 0.8}
            ])}}]
        }

        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.headers = {}
        fail_response.text = ""

        call_count = [0]
        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(model_council.OVH_ADVISORS):
                return fail_response  # Judge fails
            return advisor_response

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", side_effect=mock_post):
            ds = [{"slug": "a", "estimated_probability": 0.10, "confidence": 0.7}]
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], ds)

            assert meta["judge_ok"] is False
            # Should have averaged
            assert merged[0]["estimated_probability"] != 0.10  # Changed from DeepSeek

    def test_all_ovh_fail_returns_deepseek(self):
        """When all OVH models fail, return DeepSeek unchanged."""
        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.headers = {}
        fail_response.text = ""

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", return_value=fail_response):
            ds = [{"slug": "a", "estimated_probability": 0.15, "confidence": 0.7}]
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], ds)

            assert merged == ds
            assert meta["judge_ok"] is False


class TestCouncilSingleConsensus:
    def test_ovh_disabled(self):
        with patch.object(model_council, "is_ovh_enabled", return_value=False):
            p, meta = model_council.council_single_consensus("prompt", "slug", 0.15, 0.7)
            assert p == 0.15
            assert meta["consensus_applied"] is False

    def test_judge_makes_decision(self):
        advisor_response = MagicMock()
        advisor_response.status_code = 200
        advisor_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "estimated_probability": 0.25, "confidence": 0.8
            })}}]
        }

        judge_response = MagicMock()
        judge_response.status_code = 200
        judge_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "estimated_probability": 0.22, "confidence": 0.85
            })}}]
        }

        call_count = [0]
        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(model_council.OVH_ADVISORS):
                return judge_response
            return advisor_response

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", side_effect=mock_post):
            p, meta = model_council.council_single_consensus(
                "prompt", "slug", 0.15, 0.7, question="Test?", price=0.05
            )

            assert meta["judge_ok"] is True
            assert meta["consensus_applied"] is True
            assert p == 0.22  # Judge's verdict

    def test_judge_fails_fallback(self):
        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.headers = {}
        fail_response.text = ""

        advisor_response = MagicMock()
        advisor_response.status_code = 200
        advisor_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "estimated_probability": 0.25, "confidence": 0.8
            })}}]
        }

        call_count = [0]
        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(model_council.OVH_ADVISORS):
                return fail_response
            return advisor_response

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", side_effect=mock_post):
            p, meta = model_council.council_single_consensus(
                "prompt", "slug", 0.15, 0.7, question="Test?", price=0.05
            )

            assert meta["judge_ok"] is False
            assert meta["consensus_applied"] is True
            # Should be weighted average between 0.15 and 0.25
            assert 0.15 < p < 0.25


class TestRateLimiter:
    def test_waits_when_recent(self):
        import time
        model_council._OVH_LAST_CALL = time.time()
        waited = model_council._ovh_rate_limit_wait()
        assert waited > 0

    def test_no_wait_when_old(self):
        import time
        model_council._OVH_LAST_CALL = time.time() - 100
        waited = model_council._ovh_rate_limit_wait()
        assert waited == 0.0
