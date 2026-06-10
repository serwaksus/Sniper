from __future__ import annotations
import re
import logging
import sys
import os
from collections import defaultdict
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

MIN_P_MODEL = 0.03
MAX_EXPOSURE_PER_CATEGORY = 0.20

CLUSTER_KEYWORDS = {
    "venezuela": {"venezuela", "maduro", "caracas", "chavez", "bolivar"},
    "russia_ukraine": {"russia", "ukraine", "putin", "zelensky", "kremlin", "moscow", "kyiv", "nato", "war in ukraine", "russian invasion", "ceasefire", "peace deal", "peace talks", "territor", "donbas", "crimea", "donetsk"},
    "usa_politics": {"trump", "biden", "republican", "democratic", "congress", "senate", "house", "election", "president", "white house", "greenland", "tariff", "executive order", "governor", "primary", "nominee"},
    "fed_fomc": {"fed", "federal reserve", "fomc", "powell", "interest rate", "monetary", "s&p", "sp 500", "sp500", "recession", "inflation", "treasury", "stock market", "spy"},
    "sports_nba": {"nba", "basketball", "lakers", "warriors", "celtics"},
    "sports_ufc": {"ufc", "mma", "fight", "boxing", "fighter"},
    "crypto": {"bitcoin", "ethereum", "crypto", "btc", "eth", "blockchain", "solana", "monero"},
    "ai_tech": {"ai safety", "ai bill", "artificial intelligence", "openai", "google deepmind", "microsoft ai", "anthropic", "gpt", "llm", "bytedance", "ipo market cap"},
}


def detect_clusters(question: str) -> list[str]:
    question_lower = question.lower()
    found = set()
    for cluster, keywords in CLUSTER_KEYWORDS.items():
        for kw in keywords:
            if ' ' in kw:
                if kw in question_lower:
                    found.add(cluster)
                    break
            else:
                if re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|[^a-z])', question_lower):
                    found.add(cluster)
                    break
    return list(found) if found else ["other"]


def get_tier_params(balance: float) -> dict:
    if balance < 2000:
        return {"kelly_mult": 0.40, "base_pct": 0.05, "other_pct": 0.05,
                "max_pct": 0.10, "max_positions": 15, "max_price": 0.40,
                "max_cluster": 0.35, "tier": "micro"}
    elif balance < 10000:
        return {"kelly_mult": 0.30, "base_pct": 0.03, "other_pct": 0.045,
                "max_pct": 0.12, "max_positions": 20, "max_price": 0.40,
                "max_cluster": 0.35, "tier": "growth"}
    elif balance < 50000:
        return {"kelly_mult": 0.35, "base_pct": 0.035, "other_pct": 0.05,
                "max_pct": 0.15, "max_positions": 25, "max_price": 0.50,
                "max_cluster": 0.40, "tier": "established"}
    else:
        return {"kelly_mult": 0.40, "base_pct": 0.04, "other_pct": 0.06,
                "max_pct": 0.15, "max_positions": 30, "max_price": 0.50,
                "max_cluster": 0.45, "tier": "scale"}


def check_cluster_limits(new_clusters: list[str], current_positions: list[dict[str, Any]], portfolio_value: float | None = None) -> tuple[bool, str]:
    if portfolio_value is None:
        from order_manager import get_balance
        total_balance = get_balance()
        cash = total_balance.get("cash", 500) if total_balance else 500
        portfolio_value = total_balance.get("total", cash) if total_balance else cash
    tier = get_tier_params(portfolio_value)
    cluster_limit = tier["max_cluster"]

    cluster_exposure: dict[str, float] = defaultdict(float)
    for pos in current_positions:
        for c in pos.get("clusters", []):
            cost = pos.get("cost_usd", pos.get("size_pct", 0) * portfolio_value)
            cluster_exposure[c] += cost

    for cluster in new_clusters:
        if portfolio_value > 0 and cluster_exposure.get(cluster, 0) / portfolio_value >= cluster_limit:
            return False, f"Cluster {cluster} limit reached ({cluster_exposure[cluster]/portfolio_value:.1%})"
    return True, "OK"


# ============================================================
# PORTFOLIO EXPOSURE: Track correlated risks by category
# ============================================================
def get_category_exposure(balance: float, portfolio: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """
    Calculate current dollar exposure per category (tag) from open positions.

    Categories are derived from Polymarket market tags (e.g., "Politics", "Crypto",
    "Economics", "Sports"). This allows us to enforce MAX_EXPOSURE_PER_CATEGORY limit
    and avoid over-concentration in any single thematic area.

    Returns a dict: {category_name: dollar_exposure}
    """
    if portfolio is None:
        from order_manager import get_portfolio
        portfolio = get_portfolio()

    if not portfolio:
        return {}

    exposure: dict[str, float] = defaultdict(float)

    for pos in portfolio:
        shares_value = pos.get("current_value", 0) or pos.get("live_value", 0)
        slug = pos.get("market_slug", "")
        question = pos.get("market_question", "")

        slug_lower = slug.lower()
        question_lower = question.lower()
        detected_categories = set()
        for cluster, keywords in CLUSTER_KEYWORDS.items():
            for kw in keywords:
                if ' ' in kw:
                    if kw in slug_lower or kw in question_lower:
                        detected_categories.add(cluster)
                        break
                else:
                    if re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|[^a-z])', slug_lower) or re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|[^a-z])', question_lower):
                        detected_categories.add(cluster)
                        break

        if not detected_categories:
            detected_categories.add("other")

        for cat in detected_categories:
            exposure[cat] += shares_value

    # Log current exposure summary
    sum(exposure.values())
    for cat, val in sorted(exposure.items(), key=lambda x: -x[1]):
        pct = val / balance if balance > 0 else 0
        logger.info(f"[EXPOSURE] {cat}: ${val:.2f} ({pct:.1%} of balance)")

    return dict(exposure)


def check_category_limits(new_market: dict[str, Any], new_order_value: float, total_balance: float, portfolio: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """
    Check if placing a new order would exceed MAX_EXPOSURE_PER_CATEGORY.

    Before any order is placed, we check each detected category in the new market
    against the MAX_EXPOSURE_PER_CATEGORY limit (e.g., 12% of total balance). If adding
    the new order's value would breach the limit for ANY category, the order is
    rejected to prevent correlated risk concentration.

    Returns (allowed: bool, reason: str)
    """
    exposure = get_category_exposure(total_balance, portfolio)

    # NOTE: Inconsistent category detection — get_category_exposure (line ~93) uses CLUSTER_KEYWORDS
    # on slug/question text, while here we use pre-computed clusters from the market object.
    new_market.get("slug", "").lower()
    clusters = new_market.get("clusters", [])

    new_categories = set(clusters)
    if not new_categories:
        new_categories.add("other")

    max_dollar = total_balance * MAX_EXPOSURE_PER_CATEGORY

    for cat in new_categories:
        current = exposure.get(cat, 0)
        projected = current + new_order_value
        if projected > max_dollar:
            logger.warning(
                f"[EXPOSURE-BLOCK] category={cat} current=${current:.2f} + "
                f"new=${new_order_value:.2f} > max=${max_dollar:.2f} ({MAX_EXPOSURE_PER_CATEGORY:.0%})"
            )
            return False, f"Category '{cat}' exposure limit reached ({projected/total_balance:.1%} > {MAX_EXPOSURE_PER_CATEGORY:.0%})"

    logger.info(f"[EXPOSURE-OK] new order ${new_order_value:.2f} passes category checks")
    return True, "OK"


def _mean_std_to_beta(mean: float, std: float) -> tuple[float, float]:
    """Convert (mean, std) to Beta distribution parameters (alpha, beta)."""
    if std <= 0 or mean <= 0 or mean >= 1:
        return max(mean * 100, 1), max((1 - mean) * 100, 1)
    var = std * std
    t = mean * (1 - mean) / var - 1
    if t <= 0:
        return max(mean * 100, 1), max((1 - mean) * 100, 1)
    alpha = mean * t
    beta_param = (1 - mean) * t
    return max(alpha, 0.01), max(beta_param, 0.01)


def bayesian_kelly(price: float, p_mean: float, p_std: float,
                   risk_aversion: float = 2.0, n_samples: int = 500) -> tuple[float, float]:
    """
    Bayesian Kelly Criterion with uncertainty-aware position sizing.

    Instead of using point estimate p_model, integrates Kelly over the
    full posterior distribution Beta(alpha, beta) fitted from (p_mean, p_std).

    This automatically reduces position size when uncertainty is high,
    preventing overbetting on unreliable probability estimates.

    Args:
        price: current market price
        p_mean: our probability estimate (e.g. 0.12)
        p_std: uncertainty of estimate (e.g. 0.05)
        risk_aversion: higher = more conservative (default 2.0)
        n_samples: Monte Carlo integration points (500 = fast + accurate)

    Returns:
        (kelly_fraction, uncertainty_penalty) tuple
    """
    if p_mean <= 0 or price <= 0 or price >= 1:
        return 0.0, 0.0

    p_std = max(p_std, 0.01)
    p_std = min(p_std, p_mean * 0.95, (1 - p_mean) * 0.95)

    alpha, beta_param = _mean_std_to_beta(p_mean, p_std)

    fee = 0.01
    b = (1 - price - fee) / price

    samples = np.random.default_rng(42).beta(alpha, beta_param, n_samples)
    samples = np.clip(samples, 0.001, 0.999)

    kelly_values = (b * samples - (1 - samples)) / b
    kelly_values = np.maximum(kelly_values, 0)

    expected_kelly = float(np.mean(kelly_values))

    uncertainty_penalty = 1.0 / (1.0 + risk_aversion * (p_std / max(p_mean, 0.01)))

    return expected_kelly * uncertainty_penalty, uncertainty_penalty


def _confidence_to_std(p_model: float, confidence: float) -> float:
    """Convert (p_model, confidence) to estimated standard deviation.

    High confidence (0.9) -> low std (5% of p_model)
    Low confidence (0.5) -> high std (50% of p_model)
    """
    uncertainty = 1.0 - confidence
    return p_model * (0.05 + 0.45 * uncertainty)


def position_size(p_model: float, market_price: float, balance: float, confidence: float = 1.0, best_ask: float | None = None, cluster: str | None = None, bid_liquidity: float | None = None) -> int:
    """
    Bayesian Kelly position sizing with confidence-based uncertainty.

    Uses bayesian_kelly() instead of point-estimate Kelly to account
    for uncertainty in p_model. Higher confidence -> lower uncertainty ->
    larger position. Lower confidence -> higher uncertainty -> smaller position.
    """
    if not isinstance(p_model, (int, float)) or p_model < 0:
        return 0
    if not isinstance(market_price, (int, float)) or market_price <= 0:
        return 0
    if not isinstance(balance, (int, float)) or balance <= 0:
        return 0
    if balance <= 0:
        logger.warning(f"[KELLY] balance=${balance:.2f} <= 0, skipping")
        return 0
    if market_price <= 0:
        logger.warning("[KELLY] market_price <= 0, using minimum $5")
        return 0

    effective_price = market_price

    if effective_price <= 0.001:
        logger.warning(f"[KELLY] effective_price={effective_price:.6f} too small, minimum $5")
        return 0
    if effective_price >= 0.999:
        return 0

    prob_ratio = p_model / market_price if market_price > 0 else 0

    from dotm_sniper import get_settings
    min_p_model = get_settings().get("min_p_model", MIN_P_MODEL)
    if p_model < min_p_model:
        logger.info(f"[KELLY] p_model={p_model:.1%} < MIN_P_MODEL={min_p_model:.1%}, skipping")
        return 0

    # Bayesian Kelly with uncertainty from confidence
    p_std = _confidence_to_std(p_model, confidence)
    bayesian_frac, uncertainty_penalty = bayesian_kelly(
        effective_price, p_model, p_std, risk_aversion=2.0
    )

    if bayesian_frac <= 0:
        logger.info(f"[KELLY] bayesian_kelly={bayesian_frac:.4f} <= 0, no edge - skipping")
        return 0

    tier = get_tier_params(balance)
    kelly_mult = tier["kelly_mult"]
    kelly_fraction = bayesian_frac * kelly_mult

    effective_cap = tier["base_pct"] if cluster == "other" else tier["max_pct"]
    size_pct = min(kelly_fraction, effective_cap)

    kelly_dollars = round(balance * size_pct)
    if kelly_dollars < 5:
        logger.info(f"[KELLY] kelly_dollars=${kelly_dollars} < $5 minimum, skipping trade")
        return 0

    MIN_TRADE_USD = 20
    if kelly_dollars < MIN_TRADE_USD:
        if prob_ratio >= 2.0 and kelly_dollars >= 5:
            kelly_dollars = MIN_TRADE_USD
            logger.info(f"[KELLY] Rounded up to ${kelly_dollars} minimum for DOTM trade (ratio={prob_ratio:.1f}x)")
        else:
            logger.info(f"[KELLY] kelly_dollars=${kelly_dollars} < ${MIN_TRADE_USD} minimum, skipping trade")
            return 0
    kelly_dollars = min(kelly_dollars, round(balance * tier["max_pct"]))

    if bid_liquidity is not None and bid_liquidity > 0:
        liquidity_cap = round(bid_liquidity * 0.20)
        if kelly_dollars > liquidity_cap:
            logger.info(f"[KELLY] Liquidity cap: ${kelly_dollars} -> ${liquidity_cap} (bid_liq=${bid_liquidity:.0f} * 0.20)")
            kelly_dollars = liquidity_cap

    # Log classical Kelly for comparison
    fee = 0.01
    b_classical = (1 - effective_price - fee) / effective_price
    classical_kelly = max(0, (b_classical * p_model - (1 - p_model)) / b_classical)

    logger.info(
        f"[KELLY] tier={tier['tier']} bayesian={bayesian_frac:.4f} "
        f"(classical={classical_kelly:.4f}, penalty={uncertainty_penalty:.2f}, "
        f"p_std={p_std:.3f}) * frac={kelly_mult:.2f} "
        f"=> ${kelly_dollars} ({size_pct:.2%} of ${balance:.2f}) "
        f"[cap={effective_cap:.1%}, cluster={cluster}]"
    )

    return kelly_dollars


def conviction_adjusted_size(base_size: int, signal_score: float, min_signal: float) -> int:
    """Adjust position size based on signal conviction relative to threshold.
    Top signals (>1.5x threshold) get 5%, medium (1.2-1.5x) get 3%, rest get 1.5%."""
    if min_signal <= 0:
        return base_size
    conviction_ratio = signal_score / min_signal
    if conviction_ratio >= 1.5:
        multiplier = 1.0
    elif conviction_ratio >= 1.2:
        multiplier = 0.6
    else:
        multiplier = 0.3
    adjusted = max(5, round(base_size * multiplier))
    return adjusted
