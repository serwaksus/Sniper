"""
Tests for smart_money.py — wallet loading, saving, discovery, activity checking, and caching.
"""
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import smart_money as sm


class TestLoadSmartMoneyWallets(unittest.TestCase):
    def test_loads_from_real_file(self):
        wallets = sm.load_smart_money_wallets()
        self.assertIsInstance(wallets, list)

    @patch("smart_money.WALLET_FILE", "/tmp/__sm_test_missing.json")
    def test_missing_file_returns_empty(self):
        result = sm.load_smart_money_wallets()
        self.assertEqual(result, [])

    @patch("smart_money.WALLET_FILE", "/tmp/__sm_test_wallets.json")
    def test_loads_wallets_from_file(self):
        with open("/tmp/__sm_test_wallets.json", "w") as f:
            json.dump({"wallets": ["0xaaa", "0xbbb"]}, f)
        result = sm.load_smart_money_wallets()
        self.assertEqual(result, ["0xaaa", "0xbbb"])
        os.remove("/tmp/__sm_test_wallets.json")

    @patch("smart_money.WALLET_FILE", "/tmp/__sm_test_bad.json")
    def test_corrupt_file_returns_empty(self):
        with open("/tmp/__sm_test_bad.json", "w") as f:
            f.write("not json")
        result = sm.load_smart_money_wallets()
        self.assertEqual(result, [])
        os.remove("/tmp/__sm_test_bad.json")

    @patch("smart_money.WALLET_FILE", "/tmp/__sm_test_empty.json")
    def test_no_wallets_key_returns_empty(self):
        with open("/tmp/__sm_test_empty.json", "w") as f:
            json.dump({}, f)
        result = sm.load_smart_money_wallets()
        self.assertEqual(result, [])
        os.remove("/tmp/__sm_test_empty.json")


class TestSaveSmartMoneyWallets(unittest.TestCase):
    @patch("smart_money.WALLET_FILE", "/tmp/__sm_test_save.json")
    def test_roundtrip(self):
        wallets = ["0xdead", "0xbeef"]
        sm.save_smart_money_wallets(wallets)
        loaded = sm.load_smart_money_wallets()
        self.assertEqual(loaded, wallets)
        with open("/tmp/__sm_test_save.json") as f:
            data = json.load(f)
        self.assertIn("updated_at", data)
        os.remove("/tmp/__sm_test_save.json")


class TestCheckSmartMoneyActivity(unittest.TestCase):
    def setUp(self):
        sm._smart_money_cache.clear()

    def tearDown(self):
        sm._smart_money_cache.clear()

    @patch("smart_money.load_smart_money_wallets", return_value=[])
    def test_no_wallets_returns_empty(self, mock_load):
        result = sm.check_smart_money_activity("token123")
        self.assertFalse(result["detected"])
        self.assertEqual(result["signal_score"], 0)

    @patch.dict(os.environ, {}, clear=True)
    @patch("smart_money.load_smart_money_wallets", return_value=["0xabc"])
    def test_no_api_key_returns_empty(self, mock_load):
        os.environ.pop(sm.POLYSCAN_KEY_ENV, None)
        result = sm.check_smart_money_activity("token123")
        self.assertFalse(result["detected"])
        self.assertEqual(result["signal_score"], 0)

    @patch.dict(os.environ, {"POLYGONSCAN_API_KEY": "testkey"})
    @patch("smart_money.load_smart_money_wallets", return_value=["0xabc"])
    @patch("smart_money.requests.get")
    def test_api_returns_smart_money_match(self, mock_get, mock_load):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "1",
            "result": [{"from": "0xabc", "value": "1000"}],
        }
        mock_get.return_value = mock_resp
        result = sm.check_smart_money_activity("token123")
        self.assertTrue(result["detected"])
        self.assertEqual(result["signal_score"], 20)
        self.assertEqual(result["wallet_count"], 1)

    @patch.dict(os.environ, {"POLYGONSCAN_API_KEY": "testkey"})
    @patch("smart_money.load_smart_money_wallets", return_value=["0xabc"])
    @patch("smart_money.requests.get")
    def test_api_returns_no_match(self, mock_get, mock_load):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "1",
            "result": [{"from": "0xunknown", "value": "1000"}],
        }
        mock_get.return_value = mock_resp
        result = sm.check_smart_money_activity("token123")
        self.assertFalse(result["detected"])
        self.assertEqual(result["signal_score"], 0)

    @patch.dict(os.environ, {"POLYGONSCAN_API_KEY": "testkey"})
    @patch("smart_money.load_smart_money_wallets", return_value=["0xabc"])
    @patch("smart_money.requests.get")
    def test_api_error_returns_empty(self, mock_get, mock_load):
        mock_get.side_effect = Exception("timeout")
        result = sm.check_smart_money_activity("token123")
        self.assertFalse(result["detected"])
        self.assertEqual(result["signal_score"], 0)

    @patch("smart_money.load_smart_money_wallets", return_value=["0xabc"])
    def test_caches_result_within_ttl(self, mock_load):
        sm._smart_money_cache["token_cached"] = {
            "detected_at": time.time(),
            "result": {"detected": True, "signal_score": 20, "wallet_count": 1,
                       "total_volume_usd": 5.0, "wallets": ["0xabc"]},
        }
        result = sm.check_smart_money_activity("token_cached")
        self.assertTrue(result["detected"])
        self.assertEqual(result["signal_score"], 20)

    def test_cache_expires_after_ttl(self):
        sm._smart_money_cache["token_old"] = {
            "detected_at": time.time() - sm.CACHE_TTL - 100,
            "result": {"detected": True, "signal_score": 20, "wallet_count": 1,
                       "total_volume_usd": 5.0, "wallets": ["0xabc"]},
        }
        with patch("smart_money.load_smart_money_wallets", return_value=[]):
            result = sm.check_smart_money_activity("token_old")
        self.assertFalse(result["detected"])
        self.assertEqual(result["signal_score"], 0)


class TestDiscoverProfitableWallets(unittest.TestCase):
    @patch.dict(os.environ, {"POLYGONSCAN_API_KEY": "testkey"})
    @patch.dict(os.environ, {"POLYGONSCAN_API_KEY": "testkey"})
    @patch("smart_money.requests.get")
    def test_returns_wallets_from_txs(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "1",
            "result": [
                {"from": "0xaaa"},
                {"from": "0xbbb"},
                {"from": "0xaaa"},
            ],
        }
        mock_get.return_value = mock_resp
        wallets = sm.discover_profitable_wallets()
        self.assertEqual(len(wallets), 2)
        self.assertIn("0xaaa", wallets)
        self.assertIn("0xbbb", wallets)

    @patch.dict(os.environ, {"POLYGONSCAN_API_KEY": "testkey"})
    @patch("smart_money.requests.get")
    def test_api_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("network error")
        wallets = sm.discover_profitable_wallets()
        self.assertEqual(wallets, [])

    @patch.dict(os.environ, {"POLYGONSCAN_API_KEY": "testkey"})
    @patch("smart_money.requests.get")
    def test_non_success_status_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "0", "message": "No records"}
        mock_get.return_value = mock_resp
        wallets = sm.discover_profitable_wallets()
        self.assertEqual(wallets, [])


class TestInitSmartMoney(unittest.TestCase):
    @patch("smart_money.load_smart_money_wallets", return_value=["0xexisting"])
    def test_skips_discovery_when_wallets_exist(self, mock_load):
        with patch("smart_money.discover_profitable_wallets") as mock_disc:
            sm.init_smart_money()
            mock_disc.assert_not_called()

    @patch("smart_money.load_smart_money_wallets", return_value=[])
    @patch("smart_money.discover_profitable_wallets", return_value=["0xnew1", "0xnew2"])
    @patch("smart_money.save_smart_money_wallets")
    def test_discovers_and_saves_when_empty(self, mock_save, mock_disc, mock_load):
        sm.init_smart_money()
        mock_disc.assert_called_once()
        mock_save.assert_called_once_with(["0xnew1", "0xnew2"])

    @patch("smart_money.load_smart_money_wallets", return_value=[])
    @patch("smart_money.discover_profitable_wallets", return_value=[])
    def test_no_save_when_discovery_empty(self, mock_disc, mock_load):
        with patch("smart_money.save_smart_money_wallets") as mock_save:
            sm.init_smart_money()
            mock_save.assert_not_called()


class TestEmptySmResult(unittest.TestCase):
    def test_returns_correct_structure(self):
        result = sm._empty_sm_result("test_reason")
        self.assertFalse(result["detected"])
        self.assertEqual(result["wallet_count"], 0)
        self.assertEqual(result["total_volume_usd"], 0.0)
        self.assertEqual(result["signal_score"], 0)
        self.assertEqual(result["wallets"], [])
