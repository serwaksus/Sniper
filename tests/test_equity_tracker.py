"""Tests for equity_tracker.py — snapshots, trade logging, daily summary, curve capping."""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestLogEquitySnapshot:
    @patch("equity_tracker.save_json")
    @patch("equity_tracker.load_json")
    @patch("equity_tracker.get_portfolio", return_value=[
        {"market_slug": "s1", "current_value": 100, "unrealized_pnl": 10,
         "market_question": "Q?", "avg_entry_price": 0.10, "live_price": 0.15, "percent_pnl": 50},
    ])
    @patch("equity_tracker.get_balance", return_value={
        "cash": 500, "total_value": 600,
    })
    def test_records_snapshot(self, mock_bal, mock_port, mock_load, mock_save):
        mock_load.return_value = {"snapshots": []}
        import equity_tracker as et
        result = et.log_equity_snapshot()
        assert result is not None
        assert result["cash"] == 500
        assert result["num_positions"] == 1
        assert mock_save.called

    @patch("equity_tracker.get_balance", return_value=None)
    def test_returns_none_on_balance_failure(self, mock_bal):
        import equity_tracker as et
        result = et.log_equity_snapshot()
        assert result is None

    @patch("equity_tracker.save_json")
    @patch("equity_tracker.load_json")
    @patch("equity_tracker.get_portfolio", return_value=[])
    @patch("equity_tracker.get_balance", return_value={"cash": 500, "total_value": 500})
    def test_caps_at_1440(self, mock_bal, mock_port, mock_load, mock_save):
        snapshots = [{"timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                       "cash": 500, "positions_value": 0, "total_equity": 500,
                       "unrealized_pnl": 0, "num_positions": 0, "positions": []}
                      for i in range(1500)]
        mock_load.return_value = {"snapshots": snapshots}
        import equity_tracker as et
        et.log_equity_snapshot()
        saved = mock_save.call_args[0][1]
        assert len(saved["snapshots"]) <= 1440


class TestLogTrade:
    @patch("equity_tracker.save_json")
    @patch("equity_tracker.load_json", return_value={"trades": []})
    def test_records_trade_event(self, mock_load, mock_save):
        import equity_tracker as et
        et.log_trade("BUY", "test-slug", "Will X happen?", entry_price=0.10, shares=100, invested=10)
        assert mock_save.called
        saved = mock_save.call_args[0][1]
        assert len(saved["trades"]) == 1
        assert saved["trades"][0]["event"] == "BUY"
        assert saved["trades"][0]["slug"] == "test-slug"

    @patch("equity_tracker.save_json")
    @patch("equity_tracker.load_json", return_value={"trades": []})
    def test_trade_with_extra_fields(self, mock_load, mock_save):
        import equity_tracker as et
        et.log_trade("SELL", "s1", "Q?", pnl_pct=25.0, reason="take_profit",
                     extra={"cluster": "crypto"})
        saved = mock_save.call_args[0][1]
        assert saved["trades"][0]["cluster"] == "crypto"

    @patch("equity_tracker.save_json")
    @patch("equity_tracker.load_json", return_value="not a dict")
    def test_corrupt_journal_resets(self, mock_load, mock_save):
        import equity_tracker as et
        et.log_trade("BUY", "s1", "Q?")
        saved = mock_save.call_args[0][1]
        assert "trades" in saved


class TestGetDailySummary:
    @patch("equity_tracker.load_json")
    def test_returns_summary(self, mock_load):
        now = datetime.now()
        mock_load.side_effect = [
            {"snapshots": [{
                "timestamp": now.isoformat(),
                "cash": 500, "positions_value": 100, "total_equity": 600,
                "unrealized_pnl": 10, "num_positions": 2, "positions": [],
            }]},
            {"trades": []},
        ]
        import equity_tracker as et
        result = et.get_daily_summary()
        assert result is not None
        assert result["equity_now"] == 600
        assert result["num_positions"] == 2

    @patch("equity_tracker.load_json", return_value={"snapshots": []})
    def test_empty_snapshots_returns_empty(self, mock_load):
        import equity_tracker as et
        result = et.get_daily_summary()
        assert result == {}

    @patch("equity_tracker.load_json", return_value="not a dict")
    def test_corrupt_data_returns_empty(self, mock_load):
        import equity_tracker as et
        result = et.get_daily_summary()
        assert result == {}
