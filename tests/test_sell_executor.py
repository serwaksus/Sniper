"""Tests for sell_executor — stop-loss, sell safety, and trailing stop logic."""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, UTC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestCheckSellSafety:
    """Verify _check_sell_safety guards against empty order books and wide spreads."""

    def test_no_bids_returns_unsafe(self):
        from sell_executor import _check_sell_safety
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {"best_bid": None, "best_ask": 0.12}
            safe, reason, _price = _check_sell_safety("test-slug", 0.10, 100)
            assert safe is False
            assert "no_bids" in reason

    def test_zero_bids_returns_unsafe(self):
        from sell_executor import _check_sell_safety
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {"best_bid": 0, "best_ask": 0.12}
            safe, reason, _price = _check_sell_safety("test-slug", 0.10, 100)
            assert safe is False
            assert "no_bids" in reason

    def test_wide_spread_returns_unsafe(self):
        from sell_executor import _check_sell_safety
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {
                "best_bid": 0.03, "best_ask": 0.20,
            }
            safe, reason, _price = _check_sell_safety("test-slug", 0.10, 100)
            assert safe is False
            assert "spread" in reason.lower()

    def test_tight_spread_returns_safe(self):
        from sell_executor import _check_sell_safety
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {
                "best_bid": 0.098, "best_ask": 0.102,
            }
            safe, _reason, price = _check_sell_safety("test-slug", 0.10, 100)
            assert safe is True
            assert price == pytest.approx(0.098)

    def test_bid_far_below_price_low_price_market(self):
        from sell_executor import _check_sell_safety
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {
                "best_bid": 0.01, "best_ask": 0.015,
            }
            safe, _reason, _price = _check_sell_safety("test-slug", 0.10, 100)
            assert safe is False


class TestExecuteSell:
    """Verify _execute_sell logic for limit vs market orders."""

    def test_no_bids_returns_false(self):
        from sell_executor import _execute_sell
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {"best_bid": None, "best_ask": None}
            sold, _price, method = _execute_sell("slug", "YES", 100, 0.10, 0.10)
            assert sold is False
            assert method == "no_bids"

    def test_wide_spread_places_limit(self):
        from sell_executor import _execute_sell
        with patch("sell_executor._get_om") as mock_om, \
             patch("sell_executor.positions_db") as mock_pos_db:
            mock_om.return_value.get_order_book.return_value = {"best_bid": 0.08, "best_ask": 0.20}
            mock_om.return_value._get_open_tp_orders.return_value = []
            mock_om.return_value._place_limit_sell.return_value = (True, "limit_placed")
            mock_pos_db.load_all.return_value = {"slug": {}}
            mock_pos_db.update.return_value = None

            sold, _price, method = _execute_sell("slug", "YES", 100, 0.10, 0.10)
            assert sold is False
            assert method == "limit_pending"

    def test_force_market_sell(self):
        from sell_executor import _execute_sell
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {"best_bid": 0.09, "best_ask": 0.11}
            mock_om.return_value.get_portfolio.return_value = [
                {"market_slug": "slug", "shares": 100}
            ]
            mock_result = MagicMock()
            mock_result.stdout = '{"ok": true}'
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result):
                sold, _price, method = _execute_sell("slug", "YES", 100, 0.10, 0.10, force_market=True)
                assert sold is True
                assert method == "market"

    def test_market_sell_already_sold(self):
        from sell_executor import _execute_sell
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {"best_bid": 0.09, "best_ask": 0.11}
            mock_om.return_value.get_portfolio.return_value = []
            sold, _price, method = _execute_sell("slug", "YES", 100, 0.10, 0.10, force_market=True)
            assert sold is False
            assert method == "already_sold"

    def test_market_sell_exception(self):
        from sell_executor import _execute_sell
        with patch("sell_executor._get_om") as mock_om:
            mock_om.return_value.get_order_book.return_value = {"best_bid": 0.09, "best_ask": 0.11}
            mock_om.return_value.get_portfolio.return_value = [
                {"market_slug": "slug", "shares": 100}
            ]
            with patch("subprocess.run", side_effect=Exception("fail")):
                sold, _price, method = _execute_sell("slug", "YES", 100, 0.10, 0.10, force_market=True)
                assert sold is False
                assert method == "market_failed"


class TestATRCalculation:
    """Verify ATR-based stop calculations."""

    def test_calculate_atr_with_history(self):
        from sell_executor import _calculate_atr
        with patch("sell_executor.load_json") as mock_load:
            now = datetime.now()
            history = [
                {"t": (now - timedelta(minutes=10)).isoformat(), "p": 0.10},
                {"t": (now - timedelta(minutes=9)).isoformat(), "p": 0.12},
                {"t": (now - timedelta(minutes=8)).isoformat(), "p": 0.09},
                {"t": (now - timedelta(minutes=7)).isoformat(), "p": 0.11},
            ]
            mock_load.return_value = {"test-slug": history}
            atr = _calculate_atr("test-slug", 0.10)
            assert atr > 0
            assert isinstance(atr, float)

    def test_calculate_atr_no_history(self):
        from sell_executor import _calculate_atr
        with patch("sell_executor.load_json", return_value={}):
            atr = _calculate_atr("test-slug", 0.10)
            assert atr == pytest.approx(0.01, abs=0.005)

    def test_get_atr_stop_below_price(self):
        from sell_executor import _get_atr_stop
        with patch("sell_executor._calculate_atr", return_value=0.02):
            stop = _get_atr_stop("test-slug", 0.10, 0.10)
            assert stop < 0.10

    def test_get_atr_trailing_stop_below_high(self):
        from sell_executor import _get_atr_trailing_stop
        with patch("sell_executor._calculate_atr", return_value=0.02):
            stop = _get_atr_trailing_stop("test-slug", 0.20, 0.15)
            assert stop < 0.20


class TestTrailingStopCheck:
    """Verify trailing_stop_check handles key scenarios."""

    @patch("sell_executor._get_om")
    @patch("sell_executor._get_sniper")
    @patch("sell_executor.positions_db")
    def test_empty_portfolio_does_stale_cleanup(self, mock_pos_db, mock_sniper, mock_om):
        from sell_executor import trailing_stop_check
        mock_om.return_value.get_portfolio.return_value = []
        mock_sniper.return_value.load_hypothesis_db.return_value = {
            "hypotheses": [], "resolved": [],
        }
        mock_pos_db.load_all.return_value = {}
        trailing_stop_check()
        mock_pos_db.load_all.assert_called()

    @patch("sell_executor._get_om")
    @patch("sell_executor._get_sniper")
    @patch("sell_executor.positions_db")
    def test_resolved_slug_cleaned_up(self, mock_pos_db, mock_sniper, mock_om):
        from sell_executor import trailing_stop_check

        mock_om.return_value.get_portfolio.return_value = [
            {"market_slug": "resolved-slug", "shares": 100, "outcome": "yes",
             "live_price": 0.05, "avg_entry_price": 0.10, "market_question": "Q?"}
        ]
        mock_om.return_value.get_order_book.return_value = {"mid_price": 0.05}

        mock_sniper.return_value.load_hypothesis_db.return_value = {
            "hypotheses": [],
            "resolved": [{"slug": "resolved-slug", "resolved": True}],
        }

        mock_pos_db.get.return_value = {"entry_price": 0.10, "shares": 100}
        mock_pos_db.load_all.return_value = {}
        mock_pos_db.delete.return_value = None

        trailing_stop_check()
        mock_pos_db.delete.assert_called_with("resolved-slug")

    @patch("sell_executor._get_om")
    @patch("sell_executor._get_sniper")
    @patch("sell_executor.positions_db")
    def test_selling_in_progress_skips_position(self, mock_pos_db, mock_sniper, mock_om):
        from sell_executor import trailing_stop_check

        recent_sell_since = datetime.now(UTC).isoformat()

        mock_om.return_value.get_portfolio.return_value = [
            {"market_slug": "selling-slug", "shares": 100, "outcome": "yes",
             "live_price": 0.08, "avg_entry_price": 0.10, "market_question": "Q?"}
        ]
        mock_om.return_value.get_order_book.return_value = {"mid_price": 0.08}

        mock_sniper.return_value.load_hypothesis_db.return_value = {
            "hypotheses": [], "resolved": [],
        }

        now_iso = datetime.now().isoformat()
        mock_pos_db.get.return_value = {
            "entry_price": 0.10,
            "shares": 100,
        }
        mock_pos_db.load_all.return_value = {
            "selling-slug": {
                "selling_in_progress": True,
                "limit_sell_since": recent_sell_since,
                "entry_price": 0.10,
                "high_price": 0.12,
                "stop_loss": 0.05,
                "last_checked": now_iso,
                "outcome": "yes",
                "shares": 100,
            }
        }

        with patch("sell_executor._log_price_for_atr"):
            trailing_stop_check()
        mock_om.return_value._place_limit_sell.assert_not_called()
        mock_om.return_value._cancel_all_tp_orders.assert_not_called()


class TestStalePositionCleanup:
    """Verify stale positions (not in portfolio) are cleaned up."""

    @patch("sell_executor._get_om")
    @patch("sell_executor._get_sniper")
    @patch("sell_executor.positions_db")
    def test_stale_position_removed(self, mock_pos_db, mock_sniper, mock_om):
        from sell_executor import trailing_stop_check

        mock_om.return_value.get_portfolio.return_value = []

        mock_sniper.return_value.load_hypothesis_db.return_value = {
            "hypotheses": [], "resolved": [],
        }

        stale_pos = {"entry_price": 0.10, "shares": 100, "_miss_count": 2}
        mock_pos_db.load_all.return_value = {
            "stale-slug": stale_pos,
        }
        mock_pos_db.delete.return_value = None

        with patch("sell_executor._log_price_for_atr"):
            trailing_stop_check()

        mock_pos_db.delete.assert_any_call("stale-slug")
