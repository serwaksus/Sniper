"""Tests for settings validation."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestSettingsValidation:
    def test_valid_settings_pass(self):
        from dotm_sniper import validate_settings
        s = {
            "min_p_model": 0.03,
            "max_concurrent_trades": 15,
            "signal_threshold": 55,
            "min_confidence": 0.6,
        }
        result = validate_settings(s)
        assert result is s

    def test_zero_max_concurrent_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="max_concurrent"):
            validate_settings({"max_concurrent_trades": 0, "min_p_model": 0.03})

    def test_negative_max_concurrent_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="max_concurrent"):
            validate_settings({"max_concurrent_trades": -5, "min_p_model": 0.03})

    def test_zero_min_p_model_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="min_p_model"):
            validate_settings({"min_p_model": 0})

    def test_negative_min_p_model_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="min_p_model"):
            validate_settings({"min_p_model": -1})

    def test_zero_signal_threshold_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="signal_threshold"):
            validate_settings({"signal_threshold": 0, "min_p_model": 0.03})

    def test_signal_threshold_above_100_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="signal_threshold"):
            validate_settings({"signal_threshold": 101, "min_p_model": 0.03})

    def test_confidence_above_1_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="min_confidence"):
            validate_settings({"min_confidence": 1.5, "min_p_model": 0.03})

    def test_confidence_below_0_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(ValueError, match="min_confidence"):
            validate_settings({"min_confidence": -0.1, "min_p_model": 0.03})

    def test_non_dict_fails(self):
        from dotm_sniper import validate_settings
        with pytest.raises(AttributeError):
            validate_settings("not a dict")

    def test_boundary_values_pass(self):
        from dotm_sniper import validate_settings
        validate_settings({
            "min_p_model": 0.001,
            "max_concurrent_trades": 1,
            "signal_threshold": 1,
            "min_confidence": 0.0,
        })

    def test_confidence_exactly_1_passes(self):
        from dotm_sniper import validate_settings
        validate_settings({"min_confidence": 1.0, "min_p_model": 0.03})

    def test_signal_threshold_exactly_100_passes(self):
        from dotm_sniper import validate_settings
        validate_settings({"signal_threshold": 100, "min_p_model": 0.03})
