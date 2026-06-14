"""Tests for model_council module."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import model_council


class TestParseOvhBatch:
    """Test _parse_ovh_batch response parsing."""

    def test_parse_json_array(self):
        content = '[{"slug": "a", "estimated_probability": 0.15, "confidence": 0.7}]'
        result = model_council._parse_ovh_batch(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["slug"] == "a"
        assert result[0]["estimated_probability"] == 0.15

    def test_parse_markdown_fenced(self):
        content = '```json\n[{"slug": "x", "estimated_probability": 0.3, "confidence": 0.8}]\n```'
        result = model_council._parse_ovh_batch(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["slug"] == "x"

    def test_parse_multiple_objects(self):
        content = '[{"slug": "a", "estimated_probability": 0.1}, {"slug": "b", "estimated_probability": 0.2}]'
        result = model_council._parse_ovh_batch(content)
        assert result is not None
        assert len(result) == 2

    def test_parse_empty(self):
        assert model_council._parse_ovh_batch("") is None
        assert model_council._parse_ovh_batch(None) is None

    def test_parse_invalid_json(self):
        assert model_council._parse_ovh_batch("not json at all") is None

    def test_parse_with_leading_colon(self):
        content = ':[{"slug": "a", "estimated_probability": 0.1}]'
        result = model_council._parse_ovh_batch(content)
        assert result is not None
        assert len(result) == 1

    def test_parse_individual_objects_fallback(self):
        content = 'Some text\n{"slug": "a", "estimated_probability": 0.1}\nmore text\n{"slug": "b", "estimated_probability": 0.2}'
        result = model_council._parse_ovh_batch(content)
        assert result is not None
        assert len(result) == 2


class TestParseSingleOvh:
    """Test _parse_single_ovh response parsing."""

    def test_parse_json_object(self):
        content = '{"estimated_probability": 0.25, "confidence": 0.7, "reasoning": "test"}'
        result = model_council._parse_single_ovh(content)
        assert result is not None
        assert result["estimated_probability"] == 0.25
        assert result["confidence"] == 0.7

    def test_parse_markdown_fenced(self):
        content = '```json\n{"estimated_probability": 0.3}\n```'
        result = model_council._parse_single_ovh(content)
        assert result is not None
        assert result["estimated_probability"] == 0.3

    def test_parse_empty(self):
        assert model_council._parse_single_ovh("") is None
        assert model_council._parse_single_ovh(None) is None

    def test_parse_no_json(self):
        assert model_council._parse_single_ovh("just text") is None


class TestSafeFloat:
    """Test _safe_float utility."""

    def test_int(self):
        assert model_council._safe_float(42) == 42.0

    def test_float(self):
        assert model_council._safe_float(0.15) == 0.15

    def test_string_number(self):
        assert model_council._safe_float("0.25") == 0.25

    def test_string_percent(self):
        assert model_council._safe_float("25%") == 0.25

    def test_invalid(self):
        assert model_council._safe_float("abc", 0.5) == 0.5

    def test_none(self):
        assert model_council._safe_float(None, 0.3) == 0.3


class TestExtractJsonFromReasoning:
    """Test _extract_json_from_reasoning for thinking models."""

    def test_extract_array(self):
        reasoning = "Thinking...\nFinal answer: [{\"slug\": \"a\", \"estimated_probability\": 0.2}]"
        result = model_council._extract_json_from_reasoning(reasoning)
        assert result is not None
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_extract_object(self):
        reasoning = "Analysis...\n{\"estimated_probability\": 0.3}"
        result = model_council._extract_json_from_reasoning(reasoning)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["estimated_probability"] == 0.3

    def test_no_json(self):
        assert model_council._extract_json_from_reasoning("just thinking") == ""


class TestCouncilBatchConsensus:
    """Test council_batch_consensus aggregation logic."""

    def test_deepseek_only_when_ovh_disabled(self):
        """When OVH is disabled, should return DeepSeek results unchanged."""
        with patch.object(model_council, "is_ovh_enabled", return_value=False):
            deepseek = [{"slug": "a", "estimated_probability": 0.15, "confidence": 0.7}]
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], deepseek)
            assert merged == deepseek
            assert meta["models_queried"] == []

    def test_consensus_two_models(self):
        """Test weighted consensus between DeepSeek and one OVH model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"slug": "a", "estimated_probability": 0.20, "confidence": 0.8}
            ])}}]
        }

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", return_value=mock_response):
            deepseek = [{"slug": "a", "estimated_probability": 0.10, "confidence": 0.7}]
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], deepseek)

            assert meta["consensus_applied"] is True
            assert len(merged) == 1
            # Consensus should be between 0.10 and 0.20
            p = merged[0]["estimated_probability"]
            assert 0.10 < p < 0.20

    def test_ovh_rate_limited_graceful_degradation(self):
        """When OVH returns 429, should fall back to DeepSeek only."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}
        mock_response.text = '{"message": "rate limited"}'

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", return_value=mock_response):
            deepseek = [{"slug": "a", "estimated_probability": 0.15, "confidence": 0.7}]
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], deepseek)

            # Should return DeepSeek results unchanged
            assert merged == deepseek
            assert all(m in meta["models_failed"] for m in meta["models_queried"])

    def test_deepseek_none_ovh_provides_results(self):
        """When DeepSeek fails, OVH models should provide results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"slug": "a", "estimated_probability": 0.18, "confidence": 0.75}
            ])}}]
        }

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", return_value=mock_response):
            merged, meta = model_council.council_batch_consensus("prompt", ["a"], None)

            assert merged is not None
            assert len(merged) == 1
            assert merged[0]["estimated_probability"] == 0.18


class TestCouncilSingleConsensus:
    """Test council_single_consensus aggregation."""

    def test_deepseek_only_when_ovh_disabled(self):
        with patch.object(model_council, "is_ovh_enabled", return_value=False):
            p, meta = model_council.council_single_consensus("prompt", "slug", 0.15, 0.7)
            assert p == 0.15
            assert meta["consensus_applied"] is False

    def test_consensus_two_models(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "estimated_probability": 0.25, "confidence": 0.8
            })}}]
        }

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", return_value=mock_response):
            p, meta = model_council.council_single_consensus("prompt", "slug", 0.15, 0.7)

            assert meta["consensus_applied"] is True
            # Consensus should be between 0.15 and 0.25
            assert 0.15 < p < 0.25

    def test_ovh_failure_returns_deepseek(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.text = ""

        with patch.object(model_council, "is_ovh_enabled", return_value=True), \
             patch.object(model_council, "_ovh_rate_limit_wait", return_value=0), \
             patch.object(model_council.requests, "post", return_value=mock_response):
            p, meta = model_council.council_single_consensus("prompt", "slug", 0.15, 0.7)
            assert p == 0.15
            assert meta["consensus_applied"] is False


class TestRateLimiter:
    """Test OVH rate limiter."""

    def test_rate_limit_skips_when_recent_call(self):
        """Rate limiter should wait when called too soon after previous call."""
        import time
        model_council._OVH_LAST_CALL = time.time()  # Just called
        waited = model_council._ovh_rate_limit_wait()
        assert waited > 0  # Should have waited

    def test_rate_limit_no_wait_when_old_call(self):
        """Rate limiter should not wait when enough time has passed."""
        import time
        model_council._OVH_LAST_CALL = time.time() - 100  # 100s ago
        waited = model_council._ovh_rate_limit_wait()
        assert waited == 0.0
