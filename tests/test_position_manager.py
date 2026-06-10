"""Tests for position_manager.py — clusters, tiers, Kelly sizing, limits, conviction."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import position_manager as pm


class TestDetectClusters:
    def test_venezuela_cluster(self):
        result = pm.detect_clusters("Will Maduro win the Venezuela election?")
        assert "venezuela" in result

    def test_russia_ukraine_cluster(self):
        result = pm.detect_clusters("Will Russia withdraw from Ukraine?")
        assert "russia_ukraine" in result

    def test_usa_politics_cluster(self):
        result = pm.detect_clusters("Will Trump win the Republican primary?")
        assert "usa_politics" in result

    def test_fed_fomc_cluster(self):
        result = pm.detect_clusters("Will the Fed raise interest rates?")
        assert "fed_fomc" in result

    def test_crypto_cluster(self):
        result = pm.detect_clusters("Will Bitcoin reach $100k?")
        assert "crypto" in result

    def test_sports_nba_cluster(self):
        result = pm.detect_clusters("Will the Lakers win the NBA championship?")
        assert "sports_nba" in result

    def test_sports_ufc_cluster(self):
        result = pm.detect_clusters("Will the UFC fight end in knockout?")
        assert "sports_ufc" in result

    def test_ai_tech_cluster(self):
        result = pm.detect_clusters("Will OpenAI release GPT-5?")
        assert "ai_tech" in result

    def test_unknown_returns_other(self):
        result = pm.detect_clusters("Will it rain tomorrow in Paris?")
        assert result == ["other"]

    def test_multiple_clusters(self):
        result = pm.detect_clusters("Will Biden sanction Russia over Bitcoin?")
        assert len(result) >= 2


class TestGetTierParams:
    def test_micro_tier(self):
        params = pm.get_tier_params(1000)
        assert params["tier"] == "micro"
        assert params["kelly_mult"] == 0.40

    def test_growth_tier(self):
        params = pm.get_tier_params(5000)
        assert params["tier"] == "growth"

    def test_established_tier(self):
        params = pm.get_tier_params(25000)
        assert params["tier"] == "established"

    def test_scale_tier(self):
        params = pm.get_tier_params(100000)
        assert params["tier"] == "scale"

    def test_boundary_micro_growth(self):
        params = pm.get_tier_params(1999)
        assert params["tier"] == "micro"
        params = pm.get_tier_params(2000)
        assert params["tier"] == "growth"

    def test_boundary_growth_established(self):
        params = pm.get_tier_params(9999)
        assert params["tier"] == "growth"
        params = pm.get_tier_params(10000)
        assert params["tier"] == "established"


class TestPositionSize:
    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_basic_kelly(self, mock_settings):
        size = pm.position_size(0.30, 0.10, 10000, confidence=1.0)
        assert size > 0
        assert size <= 10000 * 0.15

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_zero_p_model(self, mock_settings):
        assert pm.position_size(0, 0.10, 10000) == 0

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_zero_market_price(self, mock_settings):
        assert pm.position_size(0.30, 0, 10000) == 0

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_zero_balance(self, mock_settings):
        assert pm.position_size(0.30, 0.10, 0) == 0

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_negative_kelly_returns_zero(self, mock_settings):
        size = pm.position_size(0.01, 0.50, 10000)
        assert size == 0

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_below_min_p_model(self, mock_settings):
        size = pm.position_size(0.02, 0.10, 10000)
        assert size == 0

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_price_near_one_returns_zero(self, mock_settings):
        size = pm.position_size(0.99, 0.999, 10000)
        assert size == 0

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_other_cluster_caps_lower(self, mock_settings):
        size_named = pm.position_size(0.50, 0.10, 10000, cluster="ai_tech")
        size_other = pm.position_size(0.50, 0.10, 10000, cluster="other")
        assert size_other <= size_named

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_confidence_reduces_size(self, mock_settings):
        size_high = pm.position_size(0.30, 0.10, 10000, confidence=1.0)
        size_low = pm.position_size(0.30, 0.10, 10000, confidence=0.5)
        assert size_low <= size_high

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_liquidity_cap(self, mock_settings):
        size = pm.position_size(0.30, 0.10, 10000, bid_liquidity=50)
        assert size <= 50 * 0.20 + 1

    @patch("dotm_sniper.get_settings", return_value={"min_p_model": 0.03})
    def test_minimum_order_5_dollars(self, mock_settings):
        size = pm.position_size(0.05, 0.10, 10, confidence=0.1)
        assert size == 0


class TestCheckClusterLimits:
    @patch("order_manager.get_balance", return_value={"cash": 500, "total": 1000})
    def test_under_limit_passes(self, mock_bal):
        ok, _reason = pm.check_cluster_limits(
            ["crypto"],
            [{"clusters": ["crypto"], "cost_usd": 100}],
            portfolio_value=1000,
        )
        assert ok is True

    @patch("order_manager.get_balance", return_value={"cash": 500, "total": 1000})
    def test_at_limit_fails(self, mock_bal):
        ok, _reason = pm.check_cluster_limits(
            ["crypto"],
            [{"clusters": ["crypto"], "cost_usd": 400}],
            portfolio_value=1000,
        )
        assert ok is False

    @patch("order_manager.get_balance", return_value={"cash": 500, "total": 1000})
    def test_no_current_positions(self, mock_bal):
        ok, _reason = pm.check_cluster_limits(["crypto"], [], portfolio_value=1000)
        assert ok is True


class TestGetCategoryExposure:
    def test_empty_portfolio(self):
        assert pm.get_category_exposure(1000, []) == {}

    @patch("order_manager.get_portfolio", return_value=None)
    def test_none_portfolio(self, mock_portfolio):
        assert pm.get_category_exposure(1000) == {}

    def test_mixed_clusters(self):
        portfolio = [
            {"market_slug": "btc-slug", "market_question": "Will Bitcoin reach $100k", "current_value": 100},
            {"market_slug": "trump-slug", "market_question": "Will Trump win election", "current_value": 200},
            {"market_slug": "random-slug", "market_question": "Will it rain", "current_value": 50},
        ]
        exposure = pm.get_category_exposure(1000, portfolio)
        assert "crypto" in exposure
        assert "usa_politics" in exposure
        assert "other" in exposure


class TestCheckCategoryLimits:
    def test_blocks_over_limit(self):
        portfolio = [
            {"market_slug": "btc-1", "market_question": "Bitcoin price", "current_value": 180},
        ]
        market = {"slug": "btc-2", "clusters": ["crypto"]}
        ok, _reason = pm.check_category_limits(market, 50, 1000, portfolio)
        assert ok is False

    def test_allows_under_limit(self):
        market = {"slug": "btc-1", "clusters": ["crypto"]}
        ok, _reason = pm.check_category_limits(market, 50, 1000, [])
        assert ok is True

    def test_no_clusters_defaults_other(self):
        market = {"slug": "rain", "clusters": []}
        ok, _reason = pm.check_category_limits(market, 10, 1000, [])
        assert ok is True


class TestConvictionAdjustedSize:
    def test_high_conviction(self):
        result = pm.conviction_adjusted_size(100, 15.0, 10.0)
        assert result == 100

    def test_medium_conviction(self):
        result = pm.conviction_adjusted_size(100, 12.0, 10.0)
        assert result == 60

    def test_low_conviction(self):
        result = pm.conviction_adjusted_size(100, 10.5, 10.0)
        assert result == 30

    def test_zero_min_signal(self):
        assert pm.conviction_adjusted_size(100, 15.0, 0) == 100

    def test_minimum_5(self):
        result = pm.conviction_adjusted_size(10, 5.0, 10.0)
        assert result >= 5


class TestBayesianKelly:
    def test_high_confidence_larger_than_low(self):
        high = pm.position_size(0.12, 0.05, 10000, confidence=0.90, cluster="test")
        low = pm.position_size(0.12, 0.05, 10000, confidence=0.55, cluster="test")
        assert high >= low

    def test_bayesian_smaller_than_classical(self):
        k_b, _ = pm.bayesian_kelly(0.05, 0.12, 0.05)
        fee = 0.01
        b = (1 - 0.05 - fee) / 0.05
        k_c = (b * 0.12 - 0.88) / b
        assert 0 < k_b < k_c

    def test_zero_std_converges_to_classical(self):
        k_b, penalty = pm.bayesian_kelly(0.05, 0.12, 0.001)
        fee = 0.01
        b = (1 - 0.05 - fee) / 0.05
        k_c = (b * 0.12 - 0.88) / b
        assert abs(k_b - k_c) < 0.02

    def test_no_edge_returns_zero(self):
        k_b, _ = pm.bayesian_kelly(0.50, 0.10, 0.02)
        assert k_b <= 0.001

    def test_uncertainty_penalty_decreases_size(self):
        _, p1 = pm.bayesian_kelly(0.05, 0.12, 0.01)
        _, p2 = pm.bayesian_kelly(0.05, 0.12, 0.08)
        assert p1 > p2

    def test_confidence_to_std(self):
        s_high = pm._confidence_to_std(0.12, 0.90)
        s_low = pm._confidence_to_std(0.12, 0.50)
        assert s_high < s_low
        assert s_high > 0
        assert s_low > 0

    def test_beta_params_valid(self):
        a, b = pm._mean_std_to_beta(0.12, 0.05)
        assert a > 0
        assert b > 0
        mean = a / (a + b)
        assert abs(mean - 0.12) < 0.05

    def test_beta_params_edge_cases(self):
        a, b = pm._mean_std_to_beta(0.5, 0.0)
        assert a > 0 and b > 0
        a, b = pm._mean_std_to_beta(0.0, 0.1)
        assert a >= 0.01
        a, b = pm._mean_std_to_beta(1.0, 0.1)
        assert b >= 0.01
