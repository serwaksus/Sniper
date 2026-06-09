#!/usr/bin/env python3
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import config


class TestProjectRoot(unittest.TestCase):
    def test_project_root_is_parent_of_src(self):
        self.assertTrue(os.path.isdir(config.PROJECT_ROOT))
        src_dir = os.path.join(config.PROJECT_ROOT, "src")
        self.assertTrue(os.path.isdir(src_dir))

    def test_project_root_contains_tests(self):
        tests_dir = os.path.join(config.PROJECT_ROOT, "tests")
        self.assertTrue(os.path.isdir(tests_dir))


class TestPathConstants(unittest.TestCase):
    def test_all_paths_are_strings(self):
        path_attrs = [
            "DB_PATH", "LOG_DIR", "BACKUP_DIR", "PID_FILE", "POSITIONS_FILE",
            "HYPOTHESIS_DB_FILE", "SETTINGS_FILE", "PRICE_TRACKING_FILE",
            "CALIBRATION_LOG_FILE", "PLATT_MODEL_FILE", "CORRELATION_FILE",
        ]
        for attr in path_attrs:
            val = getattr(config, attr)
            self.assertIsInstance(val, str, f"{attr} should be str")

    def test_db_path_ends_with_sniper_db(self):
        self.assertTrue(config.DB_PATH.endswith("sniper.db"))

    def test_log_dir_ends_with_logs(self):
        self.assertTrue(config.LOG_DIR.endswith("logs"))

    def test_positions_file_is_json(self):
        self.assertTrue(config.POSITIONS_FILE.endswith("positions.json"))

    def test_all_paths_under_project_root(self):
        path_attrs = [
            "DB_PATH", "LOG_DIR", "POSITIONS_FILE", "HYPOTHESIS_DB_FILE",
            "CALIBRATION_LOG_FILE", "PLATT_MODEL_FILE",
        ]
        for attr in path_attrs:
            val = getattr(config, attr)
            self.assertTrue(val.startswith(config.PROJECT_ROOT), f"{attr} not under PROJECT_ROOT")


class TestSharedConstants(unittest.TestCase):
    def test_min_p_model(self):
        self.assertEqual(config.MIN_P_MODEL, 0.03)

    def test_max_p_model_ratio(self):
        self.assertEqual(config.MAX_P_MODEL_RATIO, 3.0)

    def test_min_confidence(self):
        self.assertEqual(config.MIN_CONFIDENCE, 0.65)

    def test_burn_in_trades(self):
        self.assertEqual(config.BURN_IN_TRADES, 50)

    def test_max_concurrent_trades(self):
        self.assertEqual(config.MAX_CONCURRENT_TRADES, 15)

    def test_portfolio_drawdown_stop(self):
        self.assertEqual(config.PORTFOLIO_DRAWDOWN_STOP, 0.10)

    def test_per_position_max_loss(self):
        self.assertEqual(config.PER_POSITION_MAX_LOSS, 0.50)


class TestSanitize(unittest.TestCase):
    def test_masks_bot_token(self):
        text = "token=bot123456:abcdefgHIJKLmnopqrstuvwXYZ012345"
        result = config.sanitize(text)
        self.assertIn("bot***:***REDACTED***", result)
        self.assertNotIn("bot123456", result)

    def test_no_token_unchanged(self):
        text = "hello world no tokens here"
        self.assertEqual(config.sanitize(text), text)

    def test_multiple_tokens_all_masked(self):
        text = "a=bot111:AAAAAAAAAAAAAAAAAAAAA b=bot222:BBBBBBBBBBBBBBBBBBBBB"
        result = config.sanitize(text)
        self.assertNotIn("bot111", result)
        self.assertNotIn("bot222", result)

    def test_partial_string_not_masked(self):
        text = "bot12:short"
        result = config.sanitize(text)
        self.assertEqual(result, text)

    def test_empty_string(self):
        self.assertEqual(config.sanitize(""), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
