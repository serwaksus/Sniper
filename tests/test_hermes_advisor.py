"""Tests for hermes_advisor.py — open orders, reconciliation, emergency exit, notifications."""
import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _subprocess_result(stdout="", returncode=0, stderr=""):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    r.stderr = stderr
    return r


class TestGetOpenOrders:
    @patch("hermes_advisor.subprocess.run")
    def test_returns_pending_orders(self, mock_run):
        mock_run.return_value = _subprocess_result(json.dumps({"data": [
            {"id": 1, "market_slug": "s1", "outcome": "yes", "side": "sell",
             "limit_price": "0.85", "amount": "100", "status": "pending"},
            {"id": 2, "market_slug": "s2", "outcome": "yes", "side": "buy",
             "limit_price": "0.10", "amount": "50", "status": "filled"},
        ]}))
        import hermes_advisor as ha
        orders = ha.get_open_orders()
        assert len(orders) == 1
        assert orders[0]["slug"] == "s1"
        assert orders[0]["price"] == 0.85

    @patch("hermes_advisor.subprocess.run", side_effect=Exception("err"))
    def test_exception_returns_empty(self, mock_run):
        import hermes_advisor as ha
        assert ha.get_open_orders() == []

    @patch("hermes_advisor.subprocess.run")
    def test_empty_orders(self, mock_run):
        mock_run.return_value = _subprocess_result(json.dumps({"data": []}))
        import hermes_advisor as ha
        assert ha.get_open_orders() == []


class TestCancelOrder:
    @patch("hermes_advisor.subprocess.run")
    @patch("hermes_advisor.get_open_orders")
    def test_cancel_success(self, mock_orders, mock_run):
        mock_orders.return_value = [
            {"id": 42, "slug": "s1", "side": "sell", "price": 0.85, "shares": 100}
        ]
        mock_run.return_value = _subprocess_result(json.dumps({"ok": True}))
        import hermes_advisor as ha
        assert ha.cancel_order("s1") is True

    @patch("hermes_advisor.get_open_orders", return_value=[])
    def test_no_matching_orders(self, mock_orders):
        import hermes_advisor as ha
        assert ha.cancel_order("s1") is False


class TestMarketSell:
    @patch("hermes_advisor.subprocess.run")
    @patch("hermes_advisor.get_portfolio")
    def test_sell_success(self, mock_portfolio, mock_run):
        mock_portfolio.return_value = [{"market_slug": "s1", "shares": 100}]
        mock_run.return_value = _subprocess_result(json.dumps({"ok": True}))
        import hermes_advisor as ha
        assert ha.market_sell("s1", shares=100) is True

    @patch("hermes_advisor.get_portfolio", return_value=None)
    def test_portfolio_failure(self, mock_portfolio):
        import hermes_advisor as ha
        assert ha.market_sell("s1") is False

    @patch("hermes_advisor.subprocess.run")
    def test_sell_zero_shares(self, mock_run):
        import hermes_advisor as ha
        assert ha.market_sell("s1", shares=0) is False


class TestReconcilePositions:
    @patch("hermes_advisor._merge_save_positions")
    @patch("hermes_advisor.get_open_orders", return_value=[])
    @patch("hermes_advisor.get_portfolio")
    @patch("hermes_advisor.positions_db")
    def test_removes_closed_position(self, mock_pdb, mock_portfolio, mock_orders, mock_merge):
        mock_pdb.load_all.return_value = {
            "s1": {"shares": 100, "entry_price": 0.10, "market_question": "Q?", "clusters": ["other"]},
        }
        mock_portfolio.return_value = []
        import hermes_advisor as ha
        ha.reconcile_positions()

    @patch("hermes_advisor.get_portfolio", return_value=None)
    @patch("hermes_advisor.positions_db")
    def test_skips_on_api_failure(self, mock_pdb, mock_portfolio):
        mock_pdb.load_all.return_value = {"s1": {"shares": 100}}
        import hermes_advisor as ha
        ha.reconcile_positions()

    @patch("hermes_advisor.get_portfolio", return_value=[])
    @patch("hermes_advisor.positions_db")
    def test_empty_portfolio_with_positions_skips(self, mock_pdb, mock_portfolio):
        mock_pdb.load_all.return_value = {"s1": {"shares": 100}}
        import hermes_advisor as ha
        ha.reconcile_positions()


class TestAlertState:
    def test_should_send_telegram_on_trigger_exit(self):
        import hermes_advisor as ha
        ha._last_notified_at.clear()
        ha._last_alert_status.clear()
        assert ha._should_send_telegram("s1", trigger_exit=True, current_status="RED") is True

    def test_should_not_send_for_green_status(self):
        import hermes_advisor as ha
        ha._last_notified_at.clear()
        ha._last_alert_status.clear()
        assert ha._should_send_telegram("s1", trigger_exit=False, current_status="GREEN") is False

    def test_should_send_for_divergence(self):
        import hermes_advisor as ha
        ha._last_notified_at.clear()
        ha._last_alert_status.clear()
        assert ha._should_send_telegram("s1", trigger_exit=False, current_status="DIVERGENCE") is True


class TestExecuteEmergencyExit:
    @patch("hermes_advisor.positions_db")
    @patch("hermes_advisor.market_sell", return_value=True)
    @patch("hermes_advisor.cancel_order", return_value=True)
    @patch("hermes_advisor.get_portfolio", return_value=[])
    @patch("hermes_advisor._log_emergency_exit")
    @patch("hermes_advisor.TELEGRAM_REPORTER", None)
    def test_successful_exit_deletes_position(self, mock_log, mock_portfolio, mock_cancel, mock_sell, mock_pdb):
        mock_pdb.load.return_value = {"shares": 100, "outcome": "yes", "entry_price": 0.10, "market_question": "Q?"}
        import hermes_advisor as ha
        ha._execute_emergency_exit("s1", {"shares": 100, "outcome": "yes", "entry_price": 0.10}, "test reason")
        mock_pdb.delete.assert_called_with("s1")

    @patch("hermes_advisor.positions_db")
    @patch("hermes_advisor.market_sell", return_value=False)
    @patch("hermes_advisor.cancel_order", return_value=True)
    @patch("hermes_advisor._log_emergency_exit")
    def test_failed_sell_marks_failed(self, mock_log, mock_cancel, mock_sell, mock_pdb):
        mock_pdb.load.return_value = {"shares": 100, "outcome": "yes", "entry_price": 0.10}
        import hermes_advisor as ha
        ha._execute_emergency_exit("s1", {"shares": 100, "outcome": "yes"}, "reason")
        mock_pdb.update.assert_called()


class TestNotificationKwargs:
    @patch("hermes_advisor.TELEGRAM_REPORTER")
    def test_position_closed_notification_uses_market_slug(self, mock_reporter):
        mock_reporter.alert_convergence = MagicMock()
        import hermes_advisor as ha
        ha._notify_position_closed("test-slug", {
            "entry_price": 0.10,
            "high_price": 0.20,
            "shares": 100,
            "market_question": "Q?",
        })
        mock_reporter.alert_convergence.assert_called_once()
        call_kwargs = mock_reporter.alert_convergence.call_args[1]
        assert "market_slug" in call_kwargs
        assert call_kwargs["market_slug"] == "test-slug"


class TestLogEmergencyExit:
    @patch("hermes_advisor.save_json")
    @patch("hermes_advisor.load_json", return_value=[])
    def test_logs_entry(self, mock_load, mock_save):
        import hermes_advisor as ha
        ha._log_emergency_exit("s1", {"market_question": "Q?", "entry_price": 0.10}, "test")
        assert mock_save.called
        saved = mock_save.call_args[0][1]
        assert len(saved) == 1
        assert saved[0]["slug"] == "s1"


class TestCheckResolvedMarkets:
    @patch("hermes_advisor.positions_db")
    @patch("hermes_advisor.subprocess.run")
    @patch("hermes_memory._load_memory", return_value={"predictions": {}})
    def test_resolves_closed_market(self, mock_load_mem, mock_run, mock_pdb):
        mock_run.return_value = _subprocess_result(json.dumps({"data": []}))
        import hermes_advisor as ha
        ha._check_resolved_markets()

    @patch("hermes_advisor.subprocess.run", side_effect=Exception("err"))
    @patch("hermes_memory._load_memory", return_value={"predictions": {}})
    def test_subprocess_failure_returns(self, mock_load_mem, mock_run):
        import hermes_advisor as ha
        ha._check_resolved_markets()
