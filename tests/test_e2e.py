"""End-to-end test: full trade lifecycle with mock exchange."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db as db_module


def _setup_db(tmp_path):
    db_module.DB_PATH = str(tmp_path / "test.db")
    if hasattr(db_module._local, 'conn') and db_module._local.conn is not None:
        db_module._local.conn.close()
        db_module._local.conn = None
    db_module.init_db()


def _make_market(**overrides):
    m = {
        "slug": "test-market",
        "price": 0.10,
        "question": "Will X happen before July?",
        "outcome": "YES",
        "clusters": ["test"],
        "volume": 500000,
        "ttl_hours": 720,
    }
    m.update(overrides)
    return m


def _make_analysis(**overrides):
    a = {
        "p_model": 0.20,
        "prob_ratio": 2.0,
        "confidence": 0.80,
        "source_signal": "default",
        "reasoning": "Strong DOTM signal",
        "factors": [
            {"factor": "test catalyst", "direction": "supports", "weight": "high", "source": "test"}
        ],
        "signal_score": 65,
    }
    a.update(overrides)
    return a


def _mock_sniper_deps():
    mock_load = MagicMock(return_value={"hypotheses": [], "resolved": []})
    mock_save = MagicMock()
    mock_tr = MagicMock(return_value=None)
    return mock_load, mock_save, mock_tr


class TestFullTradeLifecycle:
    """End-to-end: complete buy → hold → stop-loss sell → resolve lifecycle."""

    def test_step1_signal_analysis_computes_score(self):
        """Market appears → signal pipeline analyzes → score computed."""
        from signal_pipeline import normalize_probability

        assert normalize_probability(0.15) == pytest.approx(0.15)
        assert normalize_probability(None) == 0
        assert normalize_probability(150) == pytest.approx(1.0)
        assert normalize_probability(0.0) == 0.0
        assert normalize_probability(-0.1) == 0.0
        assert normalize_probability(50) == pytest.approx(0.5)

        from signal_pipeline import get_time_decay_threshold
        from datetime import datetime, timedelta

        near = (datetime.now() + timedelta(days=2)).isoformat()
        assert get_time_decay_threshold(near) <= 0.10

        far = (datetime.now() + timedelta(days=60)).isoformat()
        assert get_time_decay_threshold(far) >= 0.15

        assert get_time_decay_threshold(None) == 0.08

    @patch("trade_executor.time.sleep")
    @patch("trade_executor.log_trade")
    @patch("trade_executor._get_sniper_deps")
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    @patch("order_manager._place_tp_ladder", return_value=[(0.75, 500, True, "limit_placed")])
    @patch("order_manager.log_slippage")
    @patch("order_manager.get_actual_fill_price", return_value={
        "shares": 500, "price": 0.105, "avg_price": 0.105, "amount_usd": 50
    })
    @patch("order_manager.buy", return_value=True)
    @patch("order_manager.get_best_ask", return_value=0.11)
    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    def test_step2_score_exceeds_threshold_triggers_buy(
        self, mock_adv, mock_ask, mock_buy, mock_fill, mock_slip, mock_tp,
        mock_pos_db, mock_hyp_db, mock_deps, mock_log, mock_sleep
    ):
        """Score exceeds threshold → trade executor buys."""
        mock_pos_db.get.return_value = None
        mock_hyp_db.get.return_value = None
        mock_deps.return_value = _mock_sniper_deps()

        from trade_executor import execute_trade
        result = execute_trade(
            _make_market(),
            100.0,
            ["test_factor"],
            _make_analysis(),
            1000.0,
        )
        assert result is True
        mock_buy.assert_called_once()
        assert mock_pos_db.update.call_count >= 2

        first_update = mock_pos_db.update.call_args_list[0]
        assert first_update[0][0] == "test-market"
        assert first_update[0][1].get("status") == "pending_fill"

        active_update = mock_pos_db.update.call_args_list[1]
        assert active_update[0][1].get("status") == "active"
        assert active_update[0][1].get("shares") == 500

    def test_step3_position_tracked_in_sqlite(self, tmp_path):
        """Position is stored and retrievable from SQLite."""
        _setup_db(tmp_path)

        db_module.update_position("test-market", {
            "status": "active",
            "entry_price": 0.105,
            "shares": 500,
            "high_price": 0.105,
            "stop_loss": 0.084,
            "outcome": "YES",
            "clusters": ["test"],
            "market_question": "Will X happen before July?",
        })

        pos = db_module.load_position("test-market")
        assert pos is not None
        assert pos["shares"] == 500
        assert pos["entry_price"] == pytest.approx(0.105)
        assert pos["stop_loss"] == pytest.approx(0.084)
        assert pos["outcome"] == "YES"
        assert db_module.count_positions() == 1

        all_pos = db_module.load_positions()
        assert "test-market" in all_pos

    @patch("sell_executor._get_om")
    @patch("sell_executor._get_sniper")
    @patch("sell_executor._get_et")
    @patch("sell_executor.positions_db")
    @patch("sell_executor._log_price_for_atr")
    def test_step4_stop_loss_triggers_sell(
        self, mock_atr, mock_pos_db, mock_et, mock_sniper, mock_om
    ):
        """Price drops → stop-loss triggers → sell executor sells."""
        from sell_executor import trailing_stop_check

        mock_om.return_value.get_portfolio.return_value = [
            {
                "market_slug": "test-market",
                "shares": 500,
                "outcome": "YES",
                "live_price": 0.05,
                "avg_entry_price": 0.105,
                "market_question": "Will X?",
            }
        ]
        mock_om.return_value.get_order_book.return_value = {
            "best_bid": 0.049,
            "best_ask": 0.051,
            "mid_price": 0.05,
        }
        mock_om.return_value._get_open_tp_orders.return_value = []
        mock_om.return_value._place_tp_ladder.return_value = [(0.75, 250, True, "limit")]
        mock_om.return_value._cancel_all_tp_orders.return_value = None

        mock_sell_result = MagicMock()
        mock_sell_result.stdout = '{"ok": true}'
        mock_sell_result.returncode = 0

        mock_sniper.return_value.load_hypothesis_db.return_value = {
            "hypotheses": [], "resolved": [],
        }
        mock_sniper.return_value.get_metaculus_forecast.return_value = {"found": False}
        mock_sniper.return_value.resolve_hypothesis_immediately.return_value = None
        mock_sniper.return_value._tr.return_value = None

        mock_et.return_value.log_trade.return_value = None

        mock_pos_db.get.return_value = {"entry_price": 0.105, "shares": 500}
        mock_pos_db.load_all.return_value = {
            "test-market": {
                "entry_price": 0.105,
                "shares": 500,
                "high_price": 0.12,
                "stop_loss": 0.07,
                "trailing_on": False,
                "selling_in_progress": False,
                "last_checked": "2020-01-01T00:00:00",
                "outcome": "YES",
                "clusters": ["test"],
                "market_question": "Will X?",
            }
        }

        with patch("subprocess.run", return_value=mock_sell_result):
            trailing_stop_check()

        mock_pos_db.delete.assert_called_with("test-market")
        mock_sniper.return_value.resolve_hypothesis_immediately.assert_called_once_with(
            "test-market", 0.05, 0.105
        )

    def test_step5_position_removed_from_sqlite(self, tmp_path):
        """After sell, position is removed from SQLite."""
        _setup_db(tmp_path)

        db_module.update_position("test-market", {
            "entry_price": 0.105, "shares": 500, "stop_loss": 0.07,
        })
        assert db_module.load_position("test-market") is not None
        assert db_module.count_positions() == 1

        db_module.delete_position("test-market")
        assert db_module.load_position("test-market") is None
        assert db_module.count_positions() == 0

    def test_step6_hypothesis_resolved_calibration_updated(self, tmp_path):
        """Hypothesis resolved → calibration data stored in SQLite."""
        _setup_db(tmp_path)

        db_module.update_hypothesis("test-market", {
            "p_model": 0.20,
            "market_price": 0.10,
            "question": "Will X happen before July?",
            "resolved": False,
            "clusters": ["test"],
            "factors": [{"factor": "catalyst", "direction": "supports", "weight": "high"}],
            "source_signal": "default",
        })
        hyp = db_module.load_hypothesis("test-market")
        assert hyp is not None
        assert hyp["resolved"] is False

        db_module.update_hypothesis("test-market", {
            "resolved": True,
            "outcome": "YES",
            "resolved_at": "2026-06-01T00:00:00",
            "pnl_at_exit": 0.50,
            "exit_price": 0.15,
            "exit_type": "stop_loss",
        })
        hyp = db_module.load_hypothesis("test-market")
        assert hyp["resolved"] is True
        assert hyp["outcome"] == "YES"
        assert hyp["pnl_at_exit"] == pytest.approx(0.50)
        assert db_module.count_resolved_hypotheses() == 1

        all_hyps = db_module.load_hypotheses()
        assert "test-market" in all_hyps["hypotheses"]

    def test_full_lifecycle_chain_sqlite(self, tmp_path):
        """Full chain: create position → update → verify → sell → remove → resolve."""
        _setup_db(tmp_path)

        db_module.update_position("chain-test", {
            "status": "pending_fill",
            "entry_price": 0.08,
            "shares": 0,
            "outcome": "YES",
        })
        pos = db_module.load_position("chain-test")
        assert pos["status"] == "pending_fill"

        db_module.update_position("chain-test", {
            "status": "active",
            "shares": 625,
            "entry_price": 0.08,
            "high_price": 0.08,
            "stop_loss": 0.064,
            "trailing_on": False,
        })
        pos = db_module.load_position("chain-test")
        assert pos["status"] == "active"
        assert pos["shares"] == 625
        assert pos["entry_price"] == pytest.approx(0.08)

        db_module.update_position("chain-test", {
            "high_price": 0.12,
            "trailing_on": True,
            "stop_loss": 0.09,
        })
        pos = db_module.load_position("chain-test")
        assert pos["high_price"] == pytest.approx(0.12)
        assert pos["trailing_on"] is True

        db_module.delete_position("chain-test")
        assert db_module.load_position("chain-test") is None

        db_module.update_hypothesis("chain-test", {
            "p_model": 0.15,
            "resolved": True,
            "outcome": "YES",
            "pnl_at_exit": 0.50,
        })
        assert db_module.count_resolved_hypotheses() == 1
