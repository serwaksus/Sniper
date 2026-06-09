"""Chaos tests: verify system behavior under failure conditions."""
import json
import os
import subprocess
import sys
import threading
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


class TestAPIFailures:
    """Verify order_manager handles subprocess failures gracefully."""

    @patch("order_manager.subprocess.run")
    def test_buy_subprocess_fails(self, mock_run):
        """Verify buy failure is handled gracefully."""
        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "data": {
                        "asks": [{"price": "0.11"}],
                        "bids": [{"price": "0.10"}],
                    }
                }),
            ),
            MagicMock(returncode=1, stderr="Error: insufficient funds"),
        ]
        from order_manager import buy
        result = buy(
            {"slug": "test-market", "price": 0.10, "outcome": "YES"},
            100,
        )
        assert result is False

    @patch("order_manager.subprocess.run")
    def test_balance_api_timeout(self, mock_run):
        """Verify balance timeout returns None."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pm-trader", timeout=15)
        from order_manager import get_balance
        result = get_balance()
        assert result is None

    @patch("order_manager.subprocess.run")
    def test_portfolio_returns_garbage(self, mock_run):
        """Verify garbage JSON from portfolio API returns None."""
        mock_run.return_value = MagicMock(returncode=0, stdout="NOT JSON AT ALL")
        from order_manager import get_portfolio
        result = get_portfolio()
        assert result is None

    @patch("order_manager.subprocess.run")
    def test_portfolio_returns_empty(self, mock_run):
        """Verify empty portfolio returns empty list."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"data": []}',
        )
        from order_manager import get_portfolio
        result = get_portfolio()
        assert result == []

    @patch("order_manager.subprocess.run")
    def test_order_book_subprocess_fails(self, mock_run):
        """Verify order book failure returns None prices."""
        mock_run.side_effect = Exception("subprocess crashed")
        from order_manager import get_order_book
        result = get_order_book("test-market")
        assert result["best_bid"] is None
        assert result["best_ask"] is None
        assert result["mid_price"] is None

    @patch("order_manager.subprocess.run")
    def test_fill_price_api_fails(self, mock_run):
        """Verify fill price failure returns None."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pm-trader", timeout=15)
        from order_manager import get_actual_fill_price
        result = get_actual_fill_price("test-market")
        assert result is None


class TestSQLiteFailures:
    """Verify database validation catches corrupt data."""

    def test_corrupted_position_data(self, tmp_path):
        """Verify validation catches non-numeric entry_price."""
        _setup_db(tmp_path)
        with pytest.raises(ValueError, match="numeric"):
            db_module.update_position("test", {
                "entry_price": "not_a_number",
                "shares": 100,
            })

    def test_negative_shares_rejected(self, tmp_path):
        """Verify negative shares are rejected."""
        _setup_db(tmp_path)
        with pytest.raises(ValueError, match="shares"):
            db_module.update_position("test", {
                "entry_price": 0.10,
                "shares": -5,
            })

    def test_non_numeric_stop_loss_rejected(self, tmp_path):
        """Verify non-numeric stop_loss is rejected."""
        _setup_db(tmp_path)
        with pytest.raises(ValueError, match="numeric"):
            db_module.update_position("test", {
                "stop_loss": "invalid",
            })

    def test_non_numeric_high_price_rejected(self, tmp_path):
        """Verify non-numeric high_price is rejected."""
        _setup_db(tmp_path)
        with pytest.raises(ValueError, match="numeric"):
            db_module.update_position("test", {
                "high_price": "invalid",
            })

    def test_valid_data_passes_validation(self, tmp_path):
        """Verify valid position data is accepted."""
        _setup_db(tmp_path)
        db_module.update_position("valid-test", {
            "entry_price": 0.10,
            "shares": 100,
            "stop_loss": 0.07,
            "high_price": 0.12,
        })
        pos = db_module.load_position("valid-test")
        assert pos["shares"] == 100
        assert pos["entry_price"] == pytest.approx(0.10)

    def test_save_positions_validates_all(self, tmp_path):
        """Verify save_positions validates every entry and rolls back on failure."""
        _setup_db(tmp_path)
        with pytest.raises(ValueError):
            db_module.save_positions({
                "good": {"shares": 100, "entry_price": 0.10},
                "bad": {"shares": "invalid"},
            })
        assert db_module.load_position("bad") is None

    def test_merge_save_validates(self, tmp_path):
        """Verify merge_save_positions validates input."""
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100, "entry_price": 0.1}})
        with pytest.raises(ValueError):
            db_module.merge_save_positions({"s1": {"shares": -10}})
        pos = db_module.load_position("s1")
        assert pos["shares"] == 100


class TestLLMFailures:
    """Verify LLM failure handling in advisor_pre_check."""

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_llm_returns_html_error(self, mock_post, mock_cb):
        """Verify HTML error page from LLM is handled."""
        mock_post.return_value = MagicMock(
            status_code=502,
            text="<html>Bad Gateway</html>",
            json=lambda: (_ for _ in ()).throw(
                json.JSONDecodeError("", "", 0)
            ),
        )
        from signal_pipeline import advisor_pre_check
        approved, verdict, _conf, _reason = advisor_pre_check(
            {"slug": "test", "question": "Will X?", "price": 0.10},
            {
                "p_model": 0.20,
                "factors": [],
                "signal_score": 65,
                "reasoning": "test",
            },
            100,
            1000,
        )
        assert approved is False
        assert verdict == "UNKNOWN"

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_llm_timeout(self, mock_post, mock_cb):
        """Verify LLM timeout is handled."""
        import requests
        mock_post.side_effect = requests.Timeout("Connection timed out")
        from signal_pipeline import advisor_pre_check
        approved, _verdict, _conf, reason = advisor_pre_check(
            {"slug": "test", "question": "Will X?", "price": 0.10},
            {
                "p_model": 0.20,
                "factors": [],
                "signal_score": 65,
                "reasoning": "test",
            },
            100,
            1000,
        )
        assert approved is False
        assert "timeout" in reason.lower()

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_llm_returns_empty_json(self, mock_post, mock_cb):
        """Verify empty JSON response from LLM blocks trade."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {},
        )
        from signal_pipeline import advisor_pre_check
        approved, _verdict, _conf, reason = advisor_pre_check(
            {"slug": "test", "question": "Will X?", "price": 0.10},
            {
                "p_model": 0.20,
                "factors": [],
                "signal_score": 65,
                "reasoning": "test",
            },
            100,
            1000,
        )
        assert approved is False
        assert "empty" in reason.lower()

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=False)
    def test_circuit_breaker_blocks_large_trade(self, mock_cb):
        """Verify circuit breaker blocks non-micro trades."""
        from signal_pipeline import advisor_pre_check
        approved, _verdict, _conf, reason = advisor_pre_check(
            {"slug": "test", "question": "Will X?", "price": 0.10},
            {
                "p_model": 0.20,
                "factors": [],
                "signal_score": 65,
                "reasoning": "test",
            },
            100,
            1000,
        )
        assert approved is False
        assert "circuit" in reason.lower()

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=False)
    def test_circuit_breaker_allows_micro_position(self, mock_cb):
        """Verify circuit breaker allows micro trades (<=2% of balance)."""
        from signal_pipeline import advisor_pre_check
        approved, _verdict, _conf, reason = advisor_pre_check(
            {"slug": "test", "question": "Will X?", "price": 0.10},
            {
                "p_model": 0.20,
                "factors": [],
                "signal_score": 65,
                "reasoning": "test",
            },
            10,
            1000,
        )
        assert approved is True
        assert "micro" in reason.lower()

    @patch("signal_pipeline._check_llm_circuit_breaker", return_value=True)
    @patch("signal_pipeline.requests.post")
    def test_llm_connection_error(self, mock_post, mock_cb):
        """Verify connection error is handled."""
        import requests
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        from signal_pipeline import advisor_pre_check
        approved, _verdict, _conf, _reason = advisor_pre_check(
            {"slug": "test", "question": "Will X?", "price": 0.10},
            {
                "p_model": 0.20,
                "factors": [],
                "signal_score": 65,
                "reasoning": "test",
            },
            100,
            1000,
        )
        assert approved is False


class TestConcurrentAccess:
    """Verify concurrent SQLite access doesn't lose data."""

    def test_concurrent_position_updates(self, tmp_path):
        """Verify concurrent updates don't lose data."""
        _setup_db(tmp_path)

        db_module.save_positions({"s1": {"shares": 100, "entry_price": 0.1}})

        errors = []

        def updater(field, value):
            try:
                for _ in range(20):
                    db_module.update_position("s1", {field: value})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=updater, args=("shares", 50))
        t2 = threading.Thread(target=updater, args=("high_price", 0.15))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        pos = db_module.load_positions()["s1"]
        assert pos["shares"] == 50
        assert pos["high_price"] == pytest.approx(0.15)
        assert pos["entry_price"] == pytest.approx(0.1)

    def test_concurrent_inserts(self, tmp_path):
        """Verify concurrent inserts all succeed."""
        _setup_db(tmp_path)
        db_module.save_positions({})

        errors = []

        def inserter(prefix, count):
            try:
                for i in range(count):
                    db_module.update_position(
                        f"{prefix}_{i}",
                        {"shares": i, "entry_price": 0.1},
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=inserter, args=("a", 15)),
            threading.Thread(target=inserter, args=("b", 15)),
            threading.Thread(target=inserter, args=("c", 15)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert db_module.count_positions() == 45

    def test_concurrent_read_write(self, tmp_path):
        """Verify reads during writes don't crash."""
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100, "entry_price": 0.1}})

        errors = []
        read_results = []

        def writer():
            try:
                for i in range(20):
                    db_module.update_position("s1", {"shares": 100 + i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    pos = db_module.load_position("s1")
                    if pos is not None:
                        read_results.append(pos["shares"])
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert len(read_results) > 0
        final = db_module.load_position("s1")
        assert final["shares"] == 119
