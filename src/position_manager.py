from __future__ import annotations
import re
import logging
import sys
import os
from collections import defaultdict
from typing import Any

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
        return {"kelly_mult": 0.28, "base_pct": 0.05, "other_pct": 0.05,
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


def position_size(p_model: float, market_price: float, balance: float, confidence: float = 1.0, best_ask: float | None = None, cluster: str | None = None, bid_liquidity: float | None = None) -> int:
    """
    Fractional Kelly position sizing with confidence weighting.

    Kelly Criterion formula: f = (p * b - q) / b
      p = p_model (our estimated true probability)
      q = 1 - p
      b = payout coefficient = (1 - price) / price
           (net odds received on winning YES bet)

    The Kelly fraction is reduced by two factors:
      1. FRACTIONAL_KELLY_MULTIPLIER (default 0.25 = "quarter Kelly")
         Quarter Kelly is a conservative approach that reduces volatility
         while still capturing most of the edge
      2. confidence score acts as an additional multiplier since high
         confidence in our probability estimate justifies a larger bet

    Hard limits enforced:
      - Minimum order: $5 (exchange fee protection)
      - Cluster-aware max cap:
          * "other" cluster: OTHER_BOOST_POS_PCT (3.5% of balance)
          * all others: BASE_POS_PCT (2% of balance)
      - Absolute ceiling: MAX_POS_PCT (10%)

    Args:
        p_model: our estimated probability (0.0 to 1.0)
        market_price: current Polymarket price (used if best_ask not provided)
        balance: current account balance in dollars
        confidence: our confidence in p_model estimate (0.0 to 1.0)
        best_ask: best ask price from order book (more accurate than midpoint)
        cluster: primary cluster name for position sizing boost

    Returns:
        Dollar amount to bet
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

    # Prefer best_ask for Kelly calculation (actual executable price)
    # vs market_price which might be midpoint with poor liquidity
    effective_price = market_price

    if effective_price <= 0.001:
        logger.warning(f"[KELLY] effective_price={effective_price:.6f} too small, minimum $5")
        return 0
    if effective_price >= 0.999:
        return 0
    fee = 0.01  # 1% Polymarket fee
    b = (1 - effective_price - fee) / effective_price

    p = p_model
    q = 1 - p
    kelly_full = (b * p - q) / b

    logger.info(f"[KELLY] p={p:.3f}, b={b:.2f}, q={q:.3f}, kelly_full={kelly_full:.4f}")

    # Reject negative Kelly (no edge case)
    if kelly_full <= 0:
        logger.info(f"[KELLY] kelly_full={kelly_full:.4f} <= 0, no edge - skipping")
        return 0

    # Reject if our probability estimate is too low
    from dotm_sniper import get_settings
    min_p_model = get_settings().get("min_p_model", MIN_P_MODEL)
    if p < min_p_model:
        logger.info(f"[KELLY] p_model={p:.1%} < MIN_P_MODEL={min_p_model:.1%}, skipping")
        return 0

    # Step 1: Fractional Kelly reduction (adaptive by balance tier)
    tier = get_tier_params(balance)
    kelly_mult = tier["kelly_mult"]
    kelly_fraction = kelly_full * kelly_mult

    # Step 2: Confidence weighting (high confidence = bigger bet)
    kelly_with_confidence = kelly_fraction * confidence

    # Cluster-aware position cap
    # Named clusters: Kelly decides up to max_pct (10%+ depending on tier)
    # "other" cluster: conservative base_pct cap
    effective_cap = tier["base_pct"] if cluster == "other" else tier["max_pct"]
    size_pct = min(kelly_with_confidence, effective_cap)

    kelly_dollars = round(balance * size_pct)
    if kelly_dollars < 5:
        logger.info(f"[KELLY] kelly_dollars=${kelly_dollars} < $5 minimum, skipping trade")
        return 0
    kelly_dollars = min(kelly_dollars, round(balance * tier["max_pct"]))

    if bid_liquidity is not None and bid_liquidity > 0:
        liquidity_cap = round(bid_liquidity * 0.20)
        if kelly_dollars > liquidity_cap:
            logger.info(f"[KELLY] Liquidity cap: ${kelly_dollars} -> ${liquidity_cap} (bid_liq=${bid_liquidity:.0f} * 0.20)")
            kelly_dollars = liquidity_cap

    logger.info(
        f"[KELLY] tier={tier['tier']} kelly_full={kelly_full:.4f} * frac={kelly_mult:.2f} "
        f"* conf={confidence:.2f} = {kelly_with_confidence:.4f} "
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
