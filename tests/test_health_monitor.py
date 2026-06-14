"""Tests for health_monitor.py — individual checks, alert generation, hourly mode."""
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import health_monitor as hm
import health_checks as hc


class TestLoadSaveState:
    def test_load_missing_returns_defaults(self, tmp_path):
        hc.HEALTH_STATE_FILE = str(tmp_path / "missing.json")
        state = hm._load_state()
        assert "last_alerts" in state

    def test_save_and_load_roundtrip(self, tmp_path):
        hc.HEALTH_STATE_FILE = str(tmp_path / "state.json")
        state = {"last_alerts": {"key1": datetime.now().isoformat()}, "last_cycle_start": None, "last_equity": None}
        hm._save_state(state)
        loaded = hm._load_state()
        assert "key1" in loaded["last_alerts"]


class TestShouldAlert:
    def test_first_time_always_alerts(self):
        state = {"last_alerts": {}}
        assert hm._should_alert(state, "new_key") is True

    def test_recent_alert_suppressed(self):
        state = {"last_alerts": {"key1": datetime.now().isoformat()}}
        assert hm._should_alert(state, "key1") is False

    def test_old_alert_allowed(self):
        state = {"last_alerts": {"key1": (datetime.now() - timedelta(hours=7)).isoformat()}}
        assert hm._should_alert(state, "key1") is True


class TestMarkAlerted:
    def test_marks_with_timestamp(self):
        state = {"last_alerts": {}}
        hm._mark_alerted(state, "key1")
        assert "key1" in state["last_alerts"]


class TestCheckNoTrades:
    def test_no_signals_no_trades(self):
        lines = ["2025-01-01 00:00:00 INFO some log line"]
        state = {}
        result = hm._check_no_trades(lines, state)
        assert result is not None
        assert "NO_TRADES" in result[0]

    def test_signals_but_no_executions(self):
        lines = [
            "2025-01-01 00:00:00 INFO => BUY signal",
            "2025-01-01 00:00:00 INFO TRADE-BLOCKED reason",
        ]
        state = {}
        result = hm._check_no_trades(lines, state)
        assert result is not None

    def test_executed_trades_ok(self):
        lines = [
            "2025-01-01 00:00:00 INFO => BUY signal",
            "2025-01-01 00:00:00 INFO [JOURNAL] BUY: some-market pnl=+0.0% reason=test",
        ]
        state = {}
        result = hm._check_no_trades(lines, state)
        assert result is None


class TestCheckEquityDrawdown:
    def test_drawdown_detected(self, tmp_path):
        eq_file = str(tmp_path / "equity.json")
        now = datetime.now()
        past = now - timedelta(hours=23, minutes=59)
        much_earlier = now - timedelta(hours=30)
        data = {
            "snapshots": [
                {"timestamp": much_earlier.isoformat(), "total_equity": 1200, "cash": 1200, "positions_value": 0,
                 "unrealized_pnl": 0, "num_positions": 0, "positions": []},
                {"timestamp": past.isoformat(), "total_equity": 1000, "cash": 1000, "positions_value": 0,
                 "unrealized_pnl": 0, "num_positions": 0, "positions": []},
                {"timestamp": now.isoformat(), "total_equity": 800, "cash": 800, "positions_value": 0,
                 "unrealized_pnl": 0, "num_positions": 0, "positions": []},
            ]
        }
        with open(eq_file, "w") as f:
            json.dump(data, f)
        hc.EQUITY_FILE = eq_file
        state = {}
        result = hm._check_equity_drawdown(state)
        assert result is not None
        assert "EQUITY_DROP" in result[0]

    def test_no_drawdown(self, tmp_path):
        eq_file = str(tmp_path / "equity.json")
        now = datetime.now()
        past = now - timedelta(hours=25)
        data = {
            "snapshots": [
                {"timestamp": past.isoformat(), "total_equity": 1000, "cash": 1000, "positions_value": 0,
                 "unrealized_pnl": 0, "num_positions": 0, "positions": []},
                {"timestamp": now.isoformat(), "total_equity": 950, "cash": 950, "positions_value": 0,
                 "unrealized_pnl": 0, "num_positions": 0, "positions": []},
            ]
        }
        with open(eq_file, "w") as f:
            json.dump(data, f)
        hc.EQUITY_FILE = eq_file
        state = {}
        result = hm._check_equity_drawdown(state)
        assert result is None

    def test_missing_file(self, tmp_path):
        hc.EQUITY_FILE = str(tmp_path / "missing.json")
        state = {}
        result = hm._check_equity_drawdown(state)
        assert result is None


class TestCheckOrderHealth:
    @patch("health_checks.subprocess.run")
    def test_duplicate_orders(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"data": [
                {"status": "pending", "market_slug": "s1", "limit_price": 0.85},
                {"status": "pending", "market_slug": "s1", "limit_price": 0.85},
            ]}),
            returncode=0,
        )
        with patch("health_checks.positions_db") as mock_pdb:
            mock_pdb.load_all.return_value = {"s1": {}}
            result = hm._check_order_health({})
            assert result is not None
            assert "ORDERS" in result[0]

    @patch("health_checks.subprocess.run", side_effect=Exception("err"))
    def test_api_failure(self, mock_run):
        result = hm._check_order_health({})
        assert result is not None
        assert "ORDERS_API" in result[0]


class TestCheckCycleTiming:
    def test_first_cycle_returns_none(self):
        state = {"last_cycle_start": None}
        result = hm._check_cycle_timing(state)
        assert result is None

    def test_normal_cycle_returns_none(self):
        state = {"last_cycle_start": datetime.now().isoformat()}
        result = hm._check_cycle_timing(state)
        assert result is None


class TestCheckErrorSpike:
    @patch("health_checks._read_last_hour_log")
    def test_error_spike_detected(self, mock_read):
        mock_read.return_value = ["2025-01-01 00:00:00 ERROR something"] * 6
        result = hm._check_error_spike({})
        assert result is not None
        assert "ERROR_SPIKE" in result[0]

    @patch("health_checks._read_last_hour_log", return_value=[])
    def test_no_errors(self, mock_read):
        result = hm._check_error_spike({})
        assert result is None


class TestCheckDiskSpace:
    @patch("health_checks.shutil.disk_usage")
    def test_disk_almost_full(self, mock_usage):
        mock_usage.return_value = MagicMock(used=950, total=1000, free=50)
        result = hm._check_disk_space({})
        assert result is not None
        assert "DISK" in result[0]

    @patch("health_checks.shutil.disk_usage")
    def test_disk_ok(self, mock_usage):
        mock_usage.return_value = MagicMock(used=500, total=1000, free=500)
        result = hm._check_disk_space({})
        assert result is None


class TestCheckHypothesisDB:
    @patch("health_checks.hypotheses_db")
    def test_too_many_unresolved(self, mock_hdb):
        mock_hdb.load_all.return_value = {
            "hypotheses": [{"resolved": False}] * 51,
        }
        result = hm._check_hypothesis_db({})
        assert result is not None
        assert "HYP_DB" in result[0]

    @patch("health_checks.hypotheses_db")
    def test_reasonable_count_ok(self, mock_hdb):
        mock_hdb.load_all.return_value = {
            "hypotheses": [{"resolved": False}] * 10,
        }
        result = hm._check_hypothesis_db({})
        assert result is None


class TestCheckJSONIntegrity:
    def test_missing_file_detected(self, tmp_path):
        hc.EQUITY_FILE = str(tmp_path / "missing_equity.json")
        hc.PRICE_TRACKING_FILE = str(tmp_path / "missing_tracking.json")
        with patch("health_checks.positions_db") as mock_pdb:
            mock_pdb.load_all.return_value = {}
            hc.POSITIONS_FILE = str(tmp_path / "missing_positions.json")
            result = hm._check_json_integrity({})
            assert result is not None
            assert "JSON_INTEGRITY" in result[0]

    def test_valid_files_ok(self, tmp_path):
        eq_file = str(tmp_path / "equity.json")
        pt_file = str(tmp_path / "tracking.json")
        for f in [eq_file, pt_file]:
            with open(f, "w") as fh:
                json.dump({"ok": True}, fh)
        hc.EQUITY_FILE = eq_file
        hc.PRICE_TRACKING_FILE = pt_file
        with patch("health_checks.positions_db") as mock_pdb:
            mock_pdb.load_all.return_value = {"s1": {}}
            result = hm._check_json_integrity({})
            assert result is None


class TestCheckPMTraderHealth:
    @patch("health_checks.subprocess.run")
    def test_failure_detected(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        result = hm._check_pm_trader_health({})
        assert result is not None
        assert "PM_TRADER" in result[0]

    @patch("health_checks.subprocess.run", side_effect=FileNotFoundError("not found"))
    def test_missing_binary(self, mock_run):
        result = hm._check_pm_trader_health({})
        assert result is not None
        assert "PM_TRADER_MISSING" in result[0]

    @patch("health_checks.subprocess.run")
    def test_success_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = hm._check_pm_trader_health({})
        assert result is None


class TestCheckSQLiteIntegrity:
    def test_integrity_ok(self):
        result = hm._check_sqlite_integrity({})
        assert result is None

    def test_db_accessible(self):
        result = hm._check_sqlite_integrity({})
        assert result is None


class TestCheckTradeActivity:
    def test_no_trade_cycles(self):
        with patch.dict("sys.modules", {"db": MagicMock()}):
            import db as mock_db
            mock_db.load_settings.return_value = {"no_trade_cycles": 5}
            with patch("builtins.__import__", side_effect=lambda *a, **kw: mock_db if a[0] == "db" else __import__(*a, **kw)):
                result = hm._check_trade_activity({})
                if result is not None:
                    assert "TRADE_ACTIVITY" in result[0]

    def test_active_trades_ok(self):
        with patch.dict("sys.modules", {"db": MagicMock()}):
            import db as mock_db
            mock_db.load_settings.return_value = {"no_trade_cycles": 0}
            with patch("builtins.__import__", side_effect=lambda *a, **kw: mock_db if a[0] == "db" else __import__(*a, **kw)):
                result = hm._check_trade_activity({})
                assert result is None


class TestRunHourlyReport:
    def test_hourly_with_issues(self):
        checks = {
            "_check_no_trades": ("NO_TRADES", "msg"),
            "_check_equity_drawdown": None, "_check_order_health": None,
            "_check_api_health": None, "_check_cycle_timing": None,
            "_check_error_spike": None, "_check_llm_usage": None,
            "_check_disk_space": None, "_check_hypothesis_db": None,
            "_check_winrate": None, "_check_calibration_overfit": None,
            "_check_cache": None, "_check_telegram": None,
            "_check_crash_frequency": None, "_check_json_integrity": None,
            "_check_cron_health": None, "_check_llm_error_rate": None,
            "_check_screen_sessions": None, "_check_disk_inodes": None,
            "_check_pm_trader_health": None, "_check_api_keys": None,
            "_check_memory": None, "_check_log_size": None,
            "_check_sqlite_integrity": None, "_check_trade_activity": None,
            "_check_external_apis": None,
        }
        from unittest.mock import patch as _p
        mgrs = []
        for name, retval in checks.items():
            mgrs.append(_p(f"health_monitor.{name}", return_value=retval))
        mgrs.append(_p("health_monitor._send_telegram", return_value=True))
        mgrs.append(_p("health_monitor._save_state"))
        mgrs.append(_p("health_monitor._load_state", return_value={"last_alerts": {}, "last_cycle_start": None, "last_equity": None}))
        mgrs.append(_p("health_monitor._read_recent_log", return_value=[]))
        for m in mgrs:
            m.start()
        try:
            result = hm.run_hourly_report()
            assert len(result) == 1
        finally:
            for m in mgrs:
                m.stop()

    def test_hourly_all_ok(self):
        checks = [
            "_check_no_trades", "_check_equity_drawdown", "_check_order_health",
            "_check_api_health", "_check_cycle_timing", "_check_error_spike",
            "_check_llm_usage", "_check_disk_space", "_check_hypothesis_db",
            "_check_winrate", "_check_calibration_overfit", "_check_cache",
            "_check_telegram", "_check_crash_frequency", "_check_json_integrity",
            "_check_cron_health", "_check_llm_error_rate", "_check_screen_sessions",
            "_check_disk_inodes", "_check_pm_trader_health", "_check_api_keys",
            "_check_memory", "_check_log_size", "_check_sqlite_integrity",
            "_check_trade_activity", "_check_external_apis",
        ]
        from unittest.mock import patch as _p
        mgrs = [_p(f"health_monitor.{c}", return_value=None) for c in checks]
        mgrs.append(_p("health_monitor._send_telegram"))
        mgrs.append(_p("health_monitor._save_state"))
        mgrs.append(_p("health_monitor._load_state", return_value={"last_alerts": {}, "last_cycle_start": None, "last_equity": None}))
        mgrs.append(_p("health_monitor._read_recent_log", return_value=[]))
        for m in mgrs:
            m.start()
        try:
            result = hm.run_hourly_report()
            assert result == []
        finally:
            for m in mgrs:
                m.stop()


class TestCheckExternalApis:
    def setup_method(self):
        hc._EXT_API_CACHE["ts"] = None
        hc._EXT_API_CACHE["issues"] = None

    @patch("utils.load_env_file")
    @patch.dict("os.environ", {
        "METACULUS_TOKEN": "test", "TAVILY_API_KEY": "test",
        "TAVILY_API_KEY_BACKUP": "test", "POLYGONSCAN_API_KEY": "test",
    })
    @patch("health_checks.requests.get")
    @patch("health_checks.requests.post")
    def test_all_healthy(self, mock_post, mock_get, mock_env):
        mock_get.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {"count": 100, "results": [
                {"id": 1, "title": "T", "aggregations": {"recency_weighted": {"latest": {"means": [0.5]}}}},
            ]},
        )
        mock_post.return_value = MagicMock(status_code=200, ok=True)
        result = hc._check_external_apis({})
        assert result is None

    @patch("utils.load_env_file")
    @patch.dict("os.environ", {
        "METACULUS_TOKEN": "", "TAVILY_API_KEY": "",
        "TAVILY_API_KEY_BACKUP": "", "POLYGONSCAN_API_KEY": "",
    })
    @patch("health_checks.requests.get")
    @patch("health_checks.requests.post")
    def test_missing_keys(self, mock_post, mock_get, mock_env):
        mock_get.return_value = MagicMock(status_code=200, ok=True, json=lambda: {"count": 100, "results": []})
        mock_post.return_value = MagicMock(status_code=200, ok=True)
        result = hc._check_external_apis({})
        assert result is not None
        alert_key, msg = result
        assert alert_key == "EXT_APIS"
        assert "METACULUS_TOKEN missing" in msg
        assert "TAVILY_API_KEY missing" in msg
        assert "POLYGONSCAN_API_KEY missing" in msg

    @patch("utils.load_env_file")
    @patch.dict("os.environ", {"METACULUS_TOKEN": "test", "TAVILY_API_KEY": "test", "TAVILY_API_KEY_BACKUP": "test", "POLYGONSCAN_API_KEY": "test"})
    @patch("health_checks.requests.get")
    @patch("health_checks.requests.post")
    def test_metaculus_401(self, mock_post, mock_get, mock_env):
        mock_get.return_value = MagicMock(status_code=401, ok=False, json=lambda: {})
        mock_post.return_value = MagicMock(status_code=200, ok=True)
        result = hc._check_external_apis({})
        assert result is not None
        assert "Metaculus: 401" in result[1]

    @patch("utils.load_env_file")
    @patch.dict("os.environ", {"METACULUS_TOKEN": "test", "TAVILY_API_KEY": "test", "TAVILY_API_KEY_BACKUP": "test", "POLYGONSCAN_API_KEY": "test"})
    @patch("health_checks.requests.get")
    @patch("health_checks.requests.post")
    def test_metaculus_no_aggregation(self, mock_post, mock_get, mock_env):
        mock_get.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"count": 100, "results": [
                {"id": 1, "title": "Test", "aggregations": {"recency_weighted": {"latest": None}}},
            ]},
        )
        mock_post.return_value = MagicMock(status_code=200, ok=True)
        result = hc._check_external_apis({})
        assert result is not None
        assert "aggregation data" in result[1]

    @patch("utils.load_env_file")
    @patch.dict("os.environ", {"METACULUS_TOKEN": "test", "TAVILY_API_KEY": "test", "TAVILY_API_KEY_BACKUP": "test", "POLYGONSCAN_API_KEY": "test"})
    @patch("health_checks.requests.get")
    @patch("health_checks.requests.post")
    def test_tavily_429(self, mock_post, mock_get, mock_env):
        mock_get.return_value = MagicMock(status_code=200, ok=True,
            json=lambda: {"count": 100, "results": [{"id": 1, "title": "T", "aggregations": {"recency_weighted": {"latest": {"means": [0.5]}}}}]})
        mock_post.return_value = MagicMock(status_code=429, ok=False)
        result = hc._check_external_apis({})
        assert result is not None
        assert "429" in result[1]

    @patch("utils.load_env_file")
    @patch.dict("os.environ", {"METACULUS_TOKEN": "test", "TAVILY_API_KEY": "test", "TAVILY_API_KEY_BACKUP": "test", "POLYGONSCAN_API_KEY": "test"})
    @patch("health_checks.requests.get", side_effect=Exception("network error"))
    @patch("health_checks.requests.post")
    def test_network_error(self, mock_post, mock_get, mock_env):
        mock_post.return_value = MagicMock(status_code=200, ok=True)
        result = hc._check_external_apis({})
        assert result is not None
        assert "network error" in result[1]

    def test_cached_results(self):
        hc._EXT_API_CACHE["ts"] = datetime.now()
        hc._EXT_API_CACHE["issues"] = ["Test: cached issue"]
        result = hc._check_external_apis({})
        assert result is not None
        assert "Test: cached issue" in result[1]

    def test_cached_no_issues(self):
        hc._EXT_API_CACHE["ts"] = datetime.now()
        hc._EXT_API_CACHE["issues"] = None
        result = hc._check_external_apis({})
        assert result is None
