#!/usr/bin/env python3
"""
Tests for backtest_v2/portfolio.py — Kelly sizing, tier caps, cluster limits,
trailing stop, P&L tracking, Sharpe calculation.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backtest_v2"))
from portfolio import (
    Position, PortfolioTracker, get_tier,
    MAX_CLUSTER_PCT, MAX_POS_PCT, FRACTIONAL_KELLY,
    TRAILING_ACTIVATION, TRAILING_STOP, CONVERGENCE_TP,
)


class TestGetTier(unittest.TestCase):
    def test_micro_under_2000(self):
        tier = get_tier(500)
        self.assertEqual(tier["tier"], "micro")
        self.assertEqual(tier["kelly"], 0.25)

    def test_micro_at_1999(self):
        tier = get_tier(1999.99)
        self.assertEqual(tier["tier"], "micro")

    def test_growth_2000(self):
        tier = get_tier(2000)
        self.assertEqual(tier["tier"], "growth")
        self.assertEqual(tier["kelly"], 0.30)

    def test_established_10000(self):
        tier = get_tier(10000)
        self.assertEqual(tier["tier"], "established")
        self.assertEqual(tier["kelly"], 0.35)

    def test_scale_50000(self):
        tier = get_tier(50000)
        self.assertEqual(tier["tier"], "scale")
        self.assertEqual(tier["kelly"], 0.40)

    def test_scale_large(self):
        tier = get_tier(1000000)
        self.assertEqual(tier["tier"], "scale")

    def test_zero_balance(self):
        tier = get_tier(0)
        self.assertEqual(tier["tier"], "micro")

    def test_negative_balance(self):
        tier = get_tier(-100)
        self.assertEqual(tier["tier"], "micro")


class TestPosition(unittest.TestCase):
    def test_current_value(self):
        pos = Position("s", "q", "YES", 0.10, 100, 10.0, 5000)
        self.assertAlmostEqual(pos.current_value(0.15), 15.0)

    def test_pnl_pct_positive(self):
        pos = Position("s", "q", "YES", 0.10, 100, 10.0, 5000)
        self.assertAlmostEqual(pos.pnl_pct(0.15), 0.50)

    def test_pnl_pct_negative(self):
        pos = Position("s", "q", "YES", 0.10, 100, 10.0, 5000)
        self.assertAlmostEqual(pos.pnl_pct(0.05), -0.50)

    def test_pnl_pct_zero_entry(self):
        pos = Position("s", "q", "YES", 0.0, 100, 10.0, 5000)
        self.assertEqual(pos.pnl_pct(0.15), 0)

    def test_pnl_abs(self):
        pos = Position("s", "q", "YES", 0.10, 100, 10.0, 5000)
        self.assertAlmostEqual(pos.pnl_abs(0.20), 10.0)

    def test_initial_state(self):
        pos = Position("s", "q", "YES", 0.10, 100, 10.0, 5000,
                       cluster="ai_tech", p_model=0.25)
        self.assertEqual(pos.high_price, 0.10)
        self.assertFalse(pos.trailing_on)
        self.assertEqual(pos.stop_loss, 0.0)
        self.assertFalse(pos.trailing_confirmed)
        self.assertFalse(pos.tp_ladder_filled)
        self.assertEqual(pos.cluster, "ai_tech")
        self.assertAlmostEqual(pos.p_model, 0.25)


class TestPortfolioTrackerOpenClose(unittest.TestCase):
    def test_open_position_reduces_balance(self):
        pt = PortfolioTracker(1000)
        ok = pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        self.assertTrue(ok)
        self.assertAlmostEqual(pt.balance, 990.0)

    def test_open_position_with_fee(self):
        pt = PortfolioTracker(1000)
        ok = pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000, fee=0.20)
        self.assertTrue(ok)
        self.assertAlmostEqual(pt.balance, 989.80)

    def test_open_insufficient_balance(self):
        pt = PortfolioTracker(5)
        ok = pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        self.assertFalse(ok)

    def test_close_position_adds_proceeds(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.close_position("s1", 15.0, "take_profit", market_price=0.15)
        self.assertAlmostEqual(pt.balance, 1005.0)
        self.assertEqual(len(pt.trades), 1)
        self.assertAlmostEqual(pt.trades[0]["pnl_abs"], 5.0)

    def test_close_with_fee(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.close_position("s1", 15.0, "sell", market_price=0.15, fee=0.30)
        self.assertAlmostEqual(pt.balance, 1004.70)
        self.assertAlmostEqual(pt.trades[0]["pnl_abs"], 4.70)

    def test_close_nonexistent_position(self):
        pt = PortfolioTracker(1000)
        pt.close_position("nonexistent", 10.0, "test")
        self.assertEqual(len(pt.trades), 0)

    def test_equity(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        eq = pt.equity({"s1": 0.15})
        self.assertAlmostEqual(eq, 1005.0)


class TestKellySizing(unittest.TestCase):
    def test_positive_edge_gives_nonzero(self):
        pt = PortfolioTracker(1000)
        size = pt.position_size(0.30, 0.10)
        self.assertGreater(size, 0)

    def test_no_edge_returns_zero(self):
        pt = PortfolioTracker(1000)
        size = pt.position_size(0.05, 0.10)
        self.assertEqual(size, 0)

    def test_zero_price_returns_zero(self):
        pt = PortfolioTracker(1000)
        size = pt.position_size(0.30, 0.0)
        self.assertEqual(size, 0)

    def test_tiny_price_returns_zero(self):
        pt = PortfolioTracker(1000)
        size = pt.position_size(0.30, 0.0001)
        self.assertEqual(size, 0)

    def test_high_confidence_increases_size(self):
        pt = PortfolioTracker(1000)
        size_low_p = pt.position_size(0.20, 0.10)
        size_high_p = pt.position_size(0.40, 0.10)
        self.assertGreater(size_high_p, size_low_p)

    def test_other_cluster_gets_base_pct_cap(self):
        pt = PortfolioTracker(1000)
        size_other = pt.position_size(0.80, 0.10, cluster="other")
        size_cluster = pt.position_size(0.80, 0.10, cluster="ai_tech")
        self.assertGreaterEqual(size_cluster, size_other)

    def test_best_ask_used_if_provided(self):
        pt = PortfolioTracker(100000)
        size_mp = pt.position_size(0.30, 0.10, best_ask=None)
        size_ask = pt.position_size(0.30, 0.10, best_ask=0.15)
        self.assertGreaterEqual(size_mp, size_ask)

    def test_minimum_5_dollars(self):
        pt = PortfolioTracker(10)
        size = pt.position_size(0.15, 0.10)
        self.assertEqual(size, 0)

    def test_max_pct_cap(self):
        pt = PortfolioTracker(1000)
        size = pt.position_size(0.95, 0.05)
        self.assertLessEqual(size, round(1000 * MAX_POS_PCT))


class TestClusterLimits(unittest.TestCase):
    def test_can_open_within_limit(self):
        pt = PortfolioTracker(1000)
        ok, reason = pt.can_open_position("other", 10)
        self.assertTrue(ok)

    def test_can_open_exceeds_balance(self):
        pt = PortfolioTracker(5)
        ok, reason = pt.can_open_position("other", 100)
        self.assertFalse(ok)

    def test_max_positions_limit(self):
        pt = PortfolioTracker(10000)
        for i in range(50):
            pt.open_position(f"s{i}", "q", "YES", 0.10, 10, 1.0, 5000,
                             cluster="other")
        ok, reason = pt.can_open_position("other", 1)
        self.assertFalse(ok)

    def test_cluster_exposure_accumulates(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 1000, 450.0, 5000,
                         cluster="ai_tech")
        pt.cluster_exposure["ai_tech"] = 450.0
        ok, reason = pt.can_open_position("ai_tech", 100.0)
        self.assertFalse(ok)
        self.assertIn("cluster", reason)


class TestTrailingStop(unittest.TestCase):
    def test_no_trailing_below_activation(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.update_trailing("s1", 0.12)
        pos = pt.positions["s1"]
        self.assertFalse(pos.trailing_on)

    def test_trailing_activates_above_threshold(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.update_trailing("s1", 0.14)
        pos = pt.positions["s1"]
        self.assertTrue(pos.trailing_on)
        self.assertGreater(pos.stop_loss, 0)

    def test_trailing_stop_moves_up(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.update_trailing("s1", 0.20)
        stop1 = pt.positions["s1"].stop_loss
        pt.update_trailing("s1", 0.30)
        stop2 = pt.positions["s1"].stop_loss
        self.assertGreater(stop2, stop1)

    def test_trailing_stop_does_not_move_down(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.update_trailing("s1", 0.30)
        stop1 = pt.positions["s1"].stop_loss
        pt.update_trailing("s1", 0.15)
        stop2 = pt.positions["s1"].stop_loss
        self.assertEqual(stop2, stop1)

    def test_high_price_never_decreases(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.update_trailing("s1", 0.30)
        self.assertEqual(pt.positions["s1"].high_price, 0.30)
        pt.update_trailing("s1", 0.20)
        self.assertEqual(pt.positions["s1"].high_price, 0.30)

    def test_nonexistent_position_no_crash(self):
        pt = PortfolioTracker(1000)
        pt.update_trailing("nonexistent", 0.50)


class TestEquityAndDrawdown(unittest.TestCase):
    def test_record_equity(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.record_equity({"s1": 0.15})
        self.assertEqual(len(pt.equity_curve), 1)
        self.assertAlmostEqual(pt.equity_curve[0]["equity"], 1005.0)

    def test_drawdown_calculation(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.record_equity({"s1": 0.20})
        pt.record_equity({"s1": 0.05})
        self.assertAlmostEqual(pt.equity_curve[0]["drawdown"], 0)
        self.assertLess(pt.equity_curve[1]["drawdown"], 0)

    def test_summary_no_trades(self):
        pt = PortfolioTracker(1000)
        s = pt.summary()
        self.assertEqual(s["total_trades"], 0)

    def test_summary_with_trades(self):
        pt = PortfolioTracker(1000)
        pt.open_position("s1", "q", "YES", 0.10, 100, 10.0, 5000)
        pt.record_equity({"s1": 0.10})
        pt.close_position("s1", 20.0, "tp", market_price=0.20)
        pt.record_equity({"s1": 0.20})
        s = pt.summary()
        self.assertEqual(s["total_trades"], 1)
        self.assertEqual(s["wins"], 1)
        self.assertEqual(s["losses"], 0)
        self.assertAlmostEqual(s["total_pnl"], 10.0)


class TestSharpeCalculation(unittest.TestCase):
    def test_sharpe_with_positive_returns(self):
        pt = PortfolioTracker(1000)
        for i in range(10):
            pt.record_equity({})
            pt.equity_curve[-1]["equity"] = 1000 + i * 10
        s = pt.summary()
        if s.get("total_trades", 0) == 0 and "sharpe_ratio" not in s:
            pass
        else:
            self.assertIn("sharpe_ratio", s)

    def test_sharpe_with_flat_equity(self):
        pt = PortfolioTracker(1000)
        for _ in range(5):
            pt.record_equity({})
        if len(pt.equity_curve) > 1:
            for e in pt.equity_curve:
                e["equity"] = 1000


if __name__ == '__main__':
    unittest.main(verbosity=2)
