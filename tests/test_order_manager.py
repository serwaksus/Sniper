"""Tests for order_manager.py — subprocess calls, buy logic, TP ladder, slippage."""
import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import order_manager as om


def _subprocess_result(stdout="", returncode=0, stderr=""):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    r.stderr = stderr
    return r


class TestGetBalance:
    @patch("order_manager.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = _subprocess_result(
            json.dumps({"data": {"cash": 500, "total": 1000}})
        )
        result = om.get_balance()
        assert result is not None
        assert result["cash"] == 500

    @patch("order_manager.subprocess.run")
    def test_nonzero_returncode_returns_none(self, mock_run):
        mock_run.return_value = _subprocess_result("error", returncode=1)
        assert om.get_balance() is None

    @patch("order_manager.subprocess.run")
    def test_malformed_json_returns_none(self, mock_run):
        mock_run.return_value = _subprocess_result("not json")
        assert om.get_balance() is None

    @patch("order_manager.subprocess.run", side_effect=Exception("timeout"))
    def test_exception_returns_none(self, mock_run):
        assert om.get_balance() is None


class TestGetPortfolio:
    @patch("order_manager.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = _subprocess_result(
            json.dumps({"data": [
                {"market_slug": "s1", "shares": 100},
                {"market_slug": "s2", "shares": 0.0001},
            ]})
        )
        result = om.get_portfolio()
        assert result is not None
        assert len(result) == 1

    @patch("order_manager.subprocess.run")
    def test_failure_returns_none(self, mock_run):
        mock_run.return_value = _subprocess_result("", returncode=1)
        assert om.get_portfolio() is None

    @patch("order_manager.subprocess.run")
    def test_empty_portfolio(self, mock_run):
        mock_run.return_value = _subprocess_result(json.dumps({"data": []}))
        result = om.get_portfolio()
        assert result == []


class TestGetOrderBook:
    @patch("order_manager.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = _subprocess_result(json.dumps({
            "data": {
                "asks": [{"price": "0.15"}],
                "bids": [{"price": "0.14"}],
            }
        }))
        book = om.get_order_book("test-slug")
        assert book["best_ask"] == 0.15
        assert book["best_bid"] == 0.14
        assert abs(book["mid_price"] - 0.145) < 1e-10

    @patch("order_manager.subprocess.run")
    def test_empty_book(self, mock_run):
        mock_run.return_value = _subprocess_result(json.dumps({"data": {"asks": [], "bids": []}}))
        book = om.get_order_book("test-slug")
        assert book["best_ask"] is None
        assert book["best_bid"] is None

    @patch("order_manager.subprocess.run", side_effect=Exception("err"))
    def test_exception_returns_nones(self, mock_run):
        book = om.get_order_book("test-slug")
        assert book["best_ask"] is None


class TestGetBestAsk:
    @patch("order_manager.get_order_book")
    def test_returns_best_ask(self, mock_book):
        mock_book.return_value = {"best_ask": 0.12, "best_bid": 0.11, "mid_price": 0.115}
        assert om.get_best_ask("slug") == 0.12

    @patch("order_manager.get_order_book")
    def test_no_ask_returns_none(self, mock_book):
        mock_book.return_value = {"best_ask": None, "best_bid": None, "mid_price": None}
        assert om.get_best_ask("slug") is None


class TestBuy:
    @patch("order_manager.subprocess.run")
    @patch("order_manager.get_order_book")
    def test_spread_too_wide_aborts(self, mock_book, mock_run):
        mock_book.return_value = {"best_ask": 0.50, "best_bid": 0.10, "mid_price": 0.30}
        market = {"slug": "test", "outcome": "yes", "price": 0.30, "question": "Q?"}
        result = om.buy(market, 10)
        assert result is False

    @patch("order_manager.subprocess.run")
    @patch("order_manager.get_order_book")
    def test_no_ask_aborts(self, mock_book, mock_run):
        mock_book.return_value = {"best_ask": None, "best_bid": None, "mid_price": None}
        market = {"slug": "test", "outcome": "yes", "price": 0.10, "question": "Q?"}
        result = om.buy(market, 10)
        assert result is False

    @patch("order_manager.subprocess.run")
    @patch("order_manager.get_order_book")
    def test_successful_buy(self, mock_book, mock_run):
        mock_book.return_value = {"best_ask": 0.12, "best_bid": 0.11, "mid_price": 0.115}
        mock_run.return_value = _subprocess_result(json.dumps({"ok": True}))
        market = {"slug": "test", "outcome": "yes", "price": 0.10, "question": "Q?"}
        result = om.buy(market, 10)
        assert result is True

    @patch("order_manager.subprocess.run")
    @patch("order_manager.get_order_book")
    def test_buy_api_failure(self, mock_book, mock_run):
        mock_book.return_value = {"best_ask": 0.12, "best_bid": 0.11, "mid_price": 0.115}
        mock_run.return_value = _subprocess_result(json.dumps({"ok": False}), returncode=1)
        market = {"slug": "test", "outcome": "yes", "price": 0.10, "question": "Q?"}
        result = om.buy(market, 10)
        assert result is False

    @patch("order_manager.subprocess.run")
    @patch("order_manager.get_order_book")
    def test_buy_exception_returns_false(self, mock_book, mock_run):
        mock_book.side_effect = Exception("fail")
        market = {"slug": "test", "outcome": "yes", "price": 0.10, "question": "Q?"}
        result = om.buy(market, 10)
        assert result is False


class TestGetOpenTPOrders:
    @patch("order_manager.subprocess.run")
    def test_returns_matching_orders(self, mock_run):
        mock_run.return_value = _subprocess_result(json.dumps({"data": [
            {"market_slug": "s1", "side": "sell", "status": "pending", "limit_price": 0.85},
            {"market_slug": "s1", "side": "buy", "status": "pending", "limit_price": 0.10},
            {"market_slug": "s2", "side": "sell", "status": "filled", "limit_price": 0.80},
        ]}))
        orders = om._get_open_tp_orders("s1")
        assert len(orders) == 1
        assert orders[0]["market_slug"] == "s1"
        assert orders[0]["side"] == "sell"

    @patch("order_manager.subprocess.run", side_effect=Exception("err"))
    def test_exception_returns_empty(self, mock_run):
        assert om._get_open_tp_orders("s1") == []


class TestCancelAllTPOrders:
    @patch("order_manager.subprocess.run")
    @patch("order_manager._get_open_tp_orders")
    def test_cancels_matching_orders(self, mock_get, mock_run):
        mock_get.return_value = [{"id": 42, "market_slug": "s1", "side": "sell", "status": "pending"}]
        mock_run.return_value = _subprocess_result("", returncode=0)
        om._cancel_all_tp_orders("s1")
        assert mock_run.called

    @patch("order_manager._get_open_tp_orders", return_value=[])
    def test_no_orders_no_calls(self, mock_get):
        om._cancel_all_tp_orders("s1")


class TestPlaceTPLadder:
    @patch("order_manager._get_open_tp_orders")
    def test_existing_orders_skip(self, mock_get):
        mock_get.return_value = [
            {"limit_price": 0.85, "amount": 50, "market_slug": "s1", "side": "sell", "status": "pending"},
        ]
        result = om._place_tp_ladder("s1", "yes", 100, entry_price=0.10)
        assert len(result) == 1

    @patch("order_manager._place_tp_limit_order_single")
    @patch("order_manager._get_open_tp_orders", return_value=[])
    def test_ladder_with_entry_price(self, mock_get, mock_place):
        mock_place.return_value = (True, "tp_limit_placed")
        result = om._place_tp_ladder("s1", "yes", 100, entry_price=0.10)
        assert len(result) == 2

    @patch("order_manager._place_tp_limit_order_single")
    @patch("order_manager._get_open_tp_orders", return_value=[])
    def test_ladder_with_zero_entry_price(self, mock_get, mock_place):
        mock_place.return_value = (True, "tp_limit_placed")
        result = om._place_tp_ladder("s1", "yes", 100, entry_price=0)
        assert len(result) == 2


class TestLogSlippage:
    @patch("order_manager.save_json")
    @patch("order_manager.load_json", return_value=[])
    @patch("order_manager.os.makedirs")
    def test_log_slippage_writes_entry(self, mock_mkdir, mock_load, mock_save):
        fill_data = {"avg_price": 0.12, "amount_usd": 100, "shares": 800, "levels_filled": 3}
        om.log_slippage("test-slug", 0.10, fill_data)
        assert mock_save.called

    @patch("order_manager.save_json")
    @patch("order_manager.load_json", return_value=[])
    @patch("order_manager.os.makedirs")
    def test_log_slippage_none_fill_data(self, mock_mkdir, mock_load, mock_save):
        om.log_slippage("test-slug", 0.10, None)
        assert not mock_save.called
