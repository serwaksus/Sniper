"""Tests for trade_executor.execute_trade() — the buy path."""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_market(slug="test-market", price=0.10, question="Test?", outcome="YES", clusters=None):
    return {
        "slug": slug,
        "price": price,
        "question": question,
        "outcome": outcome,
        "clusters": clusters or [],
    }


def _make_analysis(p_model=0.15, prob_ratio=2.0, confidence=0.75, source_signal="default", reasoning="test"):
    return {
        "p_model": p_model,
        "prob_ratio": prob_ratio,
        "confidence": confidence,
        "source_signal": source_signal,
        "reasoning": reasoning,
    }


def _mock_deps():
    mock_load = MagicMock(return_value={"hypotheses": [], "resolved": []})
    mock_save = MagicMock()
    mock_tr = MagicMock(return_value=None)
    return mock_load, mock_save, mock_tr


def _no_existing_pos(mock_pos_db, mock_hyp_db):
    mock_pos_db.get.return_value = None
    mock_hyp_db.get.return_value = None


class TestExecuteTradeIdempotency:
    """Verify execute_trade is idempotent — no double-buy."""

    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_skip_if_position_exists(self, mock_pos_db, mock_hyp_db):
        mock_pos_db.get.return_value = {"shares": 100, "entry_price": 0.10}
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), 100.0, [], _make_analysis(), 1000.0)
        assert result is False

    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_skip_if_hypothesis_exists(self, mock_pos_db, mock_hyp_db):
        mock_pos_db.get.return_value = None
        mock_hyp_db.get.return_value = {"p_model": 0.15, "resolved": False}
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), 100.0, [], _make_analysis(), 1000.0)
        assert result is False


class TestExecuteTradeValidation:
    """Verify input validation and early exits."""

    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_negative_estimated_size_skips(self, mock_pos_db, mock_hyp_db):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), -50.0, [], _make_analysis(), 1000.0)
        assert result is False

    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_zero_estimated_size_skips(self, mock_pos_db, mock_hyp_db):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), 0, [], _make_analysis(), 1000.0)
        assert result is False

    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_string_estimated_size_skips(self, mock_pos_db, mock_hyp_db):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), "bad", [], _make_analysis(), 1000.0)
        assert result is False


class TestExecuteTradeAdvisorVeto:
    """Verify advisor_pre_check blocks trades when vetoed."""

    @patch("signal_pipeline.advisor_pre_check", return_value=(False, "UNKNOWN", 0.0, "advisor_veto"))
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_advisor_veto_returns_false(self, mock_pos_db, mock_hyp_db, mock_advisor):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), 100.0, [], _make_analysis(), 1000.0)
        assert result is False

    @patch("signal_pipeline.advisor_pre_check", return_value=(False, "DIVERGE", 0.3, "advisor_veto"))
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_advisor_diverge_veto(self, mock_pos_db, mock_hyp_db, mock_advisor):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), 100.0, [], _make_analysis(), 1000.0)
        assert result is False


class TestExecuteTradeSlippageGuard:
    """Verify slippage guard aborts when ask is too far above price."""

    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    @patch("order_manager.get_best_ask", return_value=0.50)
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_high_slippage_aborts(self, mock_pos_db, mock_hyp_db, mock_ask, mock_advisor):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(price=0.05), 100.0, [], _make_analysis(), 1000.0)
        assert result is False

    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    @patch("order_manager.get_best_ask", return_value=None)
    @patch("order_manager.buy", return_value=False)
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_no_ask_passes_slippage_guard(self, mock_pos_db, mock_hyp_db, mock_buy, mock_ask, mock_advisor):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(price=0.10), 100.0, [], _make_analysis(), 1000.0)
        assert result is False


class TestExecuteTradeBuyFailure:
    """Verify buy() failure returns False."""

    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    @patch("order_manager.get_best_ask", return_value=0.11)
    @patch("order_manager.buy", return_value=False)
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_buy_failure_returns_false(self, mock_pos_db, mock_hyp_db, mock_buy, mock_ask, mock_advisor):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(), 100.0, [], _make_analysis(), 1000.0)
        assert result is False


class TestExecuteTradeZeroShares:
    """Verify zero-fill with zero price returns False."""

    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    @patch("order_manager.get_best_ask", return_value=0.001)
    @patch("order_manager.buy", return_value=True)
    @patch("order_manager.get_actual_fill_price", return_value=None)
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_no_fill_zero_price_returns_false(self, mock_pos_db, mock_hyp_db, mock_fill, mock_buy, mock_ask, mock_advisor):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        result = execute_trade(_make_market(price=0.0), 100.0, [], _make_analysis(), 1000.0)
        assert result is False


class TestExecuteTradeHappyPath:
    """Verify successful trade execution."""

    @patch("trade_executor.log_trade")
    @patch("trade_executor._get_sniper_deps")
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    @patch("order_manager._place_tp_ladder", return_value=[(0.75, 250, True, "limit_placed")])
    @patch("order_manager.log_slippage")
    @patch("order_manager.get_actual_fill_price", return_value={"shares": 500, "price": 0.105, "avg_price": 0.105, "amount_usd": 50})
    @patch("order_manager.buy", return_value=True)
    @patch("order_manager.get_best_ask", return_value=0.11)
    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    def test_successful_buy_saves_position(
        self, mock_advisor, mock_ask, mock_buy, mock_fill, mock_slip, mock_tp,
        mock_pos_db, mock_hyp_db, mock_deps, mock_log_trade
    ):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        mock_deps.return_value = _mock_deps()

        from trade_executor import execute_trade
        result = execute_trade(
            _make_market(slug="happy-slug", price=0.10),
            50.0,
            ["factor1"],
            _make_analysis(),
            1000.0,
        )
        assert result is True
        assert mock_pos_db.update.call_count >= 2

    @patch("trade_executor.log_trade")
    @patch("trade_executor._get_sniper_deps")
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    @patch("order_manager._place_tp_ladder", return_value=[])
    @patch("order_manager.log_slippage")
    @patch("order_manager.get_actual_fill_price", return_value={"shares": 500, "price": 0.105, "avg_price": 0.105, "amount_usd": 50})
    @patch("order_manager.buy", return_value=True)
    @patch("order_manager.get_best_ask", return_value=0.11)
    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    def test_successful_buy_with_no_tp_rungs(
        self, mock_advisor, mock_ask, mock_buy, mock_fill, mock_slip, mock_tp,
        mock_pos_db, mock_hyp_db, mock_deps, mock_log_trade
    ):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        mock_deps.return_value = _mock_deps()

        from trade_executor import execute_trade
        result = execute_trade(
            _make_market(slug="no-tp-slug", price=0.10),
            50.0,
            [],
            _make_analysis(),
            1000.0,
        )
        assert result is True


class TestExecuteTradeMissingFields:
    """Verify behavior when market dict has missing fields."""

    @patch("trade_executor.log_trade")
    @patch("trade_executor._get_sniper_deps")
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    @patch("order_manager._place_tp_ladder", return_value=[])
    @patch("order_manager.log_slippage")
    @patch("order_manager.get_actual_fill_price", return_value={"shares": 500, "price": 0.10, "avg_price": 0.10, "amount_usd": 50})
    @patch("order_manager.buy", return_value=True)
    @patch("order_manager.get_best_ask", return_value=0.11)
    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    def test_market_with_question_succeeds(
        self, mock_advisor, mock_ask, mock_buy, mock_fill, mock_slip, mock_tp,
        mock_pos_db, mock_hyp_db, mock_deps, mock_log_trade
    ):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        mock_deps.return_value = _mock_deps()

        market = {"slug": "has-question", "price": 0.10, "question": "Will X happen?", "outcome": "YES", "clusters": []}
        from trade_executor import execute_trade
        result = execute_trade(market, 50.0, [], _make_analysis(), 1000.0)
        assert result is True

    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    @patch("order_manager.get_best_ask", return_value=0.11)
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    def test_missing_price_key_raises(self, mock_pos_db, mock_hyp_db, mock_ask, mock_advisor):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        from trade_executor import execute_trade
        market = {"slug": "no-price", "outcome": "YES", "clusters": []}
        with pytest.raises(KeyError):
            execute_trade(market, 50.0, [], _make_analysis(), 1000.0)

    @patch("trade_executor.log_trade")
    @patch("trade_executor._get_sniper_deps")
    @patch("trade_executor.hypotheses_db")
    @patch("trade_executor.positions_db")
    @patch("order_manager._place_tp_ladder", return_value=[])
    @patch("order_manager.log_slippage")
    @patch("order_manager.get_actual_fill_price", return_value={"shares": 500, "price": 0.10, "avg_price": 0.10, "amount_usd": 50})
    @patch("order_manager.buy", return_value=True)
    @patch("order_manager.get_best_ask", return_value=0.11)
    @patch("signal_pipeline.advisor_pre_check", return_value=(True, "CONFIRM", 0.8, "approved"))
    def test_missing_question_after_buy_raises(
        self, mock_advisor, mock_ask, mock_buy, mock_fill, mock_slip, mock_tp,
        mock_pos_db, mock_hyp_db, mock_deps, mock_log_trade
    ):
        _no_existing_pos(mock_pos_db, mock_hyp_db)
        mock_deps.return_value = _mock_deps()
        from trade_executor import execute_trade
        market = {"slug": "no-question", "price": 0.10, "outcome": "YES", "clusters": []}
        with pytest.raises(KeyError):
            execute_trade(market, 50.0, [], _make_analysis(), 1000.0)
