"""
external_oracles.py — Free, no-auth external data sources for signal score enrichment.

Three independent oracle sources, each providing bonus points to the signal_score:

1. **Alternative.me Fear & Greed Index** (crypto/tech sentiment)
   - Endpoint: GET https://api.alternative.me/fng/
   - Cache: 12 hours
   - Score: +5 if cluster ∈ {crypto, ai_tech, tech} AND index < 30 (Extreme Fear)

2. **Manifold Markets Arbitrage** (cross-platform probability gap)
   - Endpoint: GET https://api.manifold.markets/v0/search-markets?term={query}&limit=3
   - Cache: per market slug, 1 hour
   - Score: +15 if Manifold prob ≥ Polymarket price + 15% (strict)

3. **DBnomics** (macroeconomic data — Fed Funds Rate, CPI, Unemployment, GDP Growth)
   - Endpoint: GET https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}
   - Cache: 24 hours
   - Score: +10 if cluster ∈ {fed_fomc, us_economic, usa_politics} AND macro trend aligns with contract
   - Trend analysis: compares latest vs previous value (rising/falling/stable)

All calls are wrapped in try/except — a failure in any oracle must NEVER
crash the signal scoring pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# ─── Cache Infrastructure ────────────────────────────────────────────

_CACHE_LOCK = threading.RLock()
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)
os.makedirs(_CACHE_DIR, exist_ok=True)


def _load_cache(filename: str, ttl_seconds: float) -> Any | None:
    """Load from file cache if fresh (< ttl_seconds old). Returns None if stale/missing."""
    path = os.path.join(_CACHE_DIR, filename)
    try:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        age = time.time() - data.get("_cached_at", 0)
        if age < ttl_seconds:
            logger.debug(f"[ORACLE-CACHE] hit {filename} (age={age/3600:.1f}h)")
            return data.get("value")
        return None
    except Exception:
        return None


def _save_cache(filename: str, value: Any) -> None:
    """Atomically save value to file cache with timestamp."""
    path = os.path.join(_CACHE_DIR, filename)
    try:
        payload = json.dumps({"_cached_at": time.time(), "value": value})
        fd, tmp_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        logger.debug(f"[ORACLE-CACHE] save error {filename}: {e}")


# ─── 1. Alternative.me Fear & Greed Index ────────────────────────────

FNG_API = "https://api.alternative.me/fng/"
FNG_CACHE_FILE = "oracle_fng.json"
FNG_CACHE_TTL = 43200  # 12 hours
FNG_CLUSTERS = {"crypto", "ai_tech", "tech"}
FNG_BONUS = 5
FNG_EXTREME_FEAR_THRESHOLD = 30


def get_fear_greed_index() -> int | None:
    """Fetch the current Fear & Greed Index (0-100).

    Returns None on any failure. Cached for 12 hours.
    """
    # Check cache
    cached = _load_cache(FNG_CACHE_FILE, FNG_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        resp = requests.get(FNG_API, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"[FNG] HTTP {resp.status_code}")
            return None
        data = resp.json()
        value = int(data["data"][0]["value"])
        classification = data["data"][0].get("value_classification", "?")
        logger.info(f"[FNG] Index={value} ({classification})")
        _save_cache(FNG_CACHE_FILE, value)
        return value
    except Exception as e:
        logger.warning(f"[FNG] Error: {type(e).__name__}: {e}")
        return None


def fear_greed_bonus(cluster: str) -> int:
    """Return +5 if cluster is tech/crypto AND index < 30 (Extreme Fear).

    Logic: crowd is overly pessimistic → DOTM Yes contracts are cheaper
    than they should be → contrarian buy signal.
    """
    if cluster not in FNG_CLUSTERS:
        return 0
    index = get_fear_greed_index()
    if index is None:
        return 0
    if index < FNG_EXTREME_FEAR_THRESHOLD:
        logger.info(
            f"[FNG-BONUS] cluster={cluster}, FNG={index} < {FNG_EXTREME_FEAR_THRESHOLD} "
            f"(Extreme Fear) → +{FNG_BONUS}"
        )
        return FNG_BONUS
    return 0


# ─── 2. Manifold Markets Arbitrage ───────────────────────────────────

MANIFOLD_SEARCH_API = "https://api.manifold.markets/v0/search-markets"
MANIFOLD_TIMEOUT = 12
MANIFOLD_CACHE_TTL = 3600  # 1 hour per market slug
MANIFOLD_GAP_THRESHOLD = 0.15  # 15% strict gap
MANIFOLD_BONUS = 15

# In-memory cache: slug → (timestamp, bonus)
_manifold_cache: dict[str, tuple[float, int]] = {}


def _extract_keywords(question: str, max_keywords: int = 3) -> str:
    """Extract 2-3 meaningful keywords from a Polymarket question.

    Strips stop words, punctuation, and short tokens.
    """
    # Remove common prediction-market boilerplate
    question = re.sub(
        r"\b(will|before|after|by|the|a|an|in|of|on|at|to|is|are|be|this|that|"
        r"year|month|day|first|last|end|january|february|march|april|may|june|"
        r"july|august|september|october|november|december|"
        r"2024|2025|2026|2027|2028|2029|2030)\b",
        "",
        question,
        flags=re.IGNORECASE,
    )
    # Extract word tokens
    tokens = re.findall(r"[A-Za-z]{3,}", question.lower())
    # Deduplicate while preserving order
    seen: set[str] = set()
    keywords: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            keywords.append(t)
    result = " ".join(keywords[:max_keywords])
    return result if result else question[:50]


def check_manifold_arbitrage(question: str, polymarket_price: float, slug: str = "") -> int:
    """Check if Manifold Markets shows a significantly higher probability.

    Returns +15 if a highly relevant Manifold market exists and its probability
    is at least 15% strictly higher than the Polymarket price.

    Args:
        question: Polymarket question text
        polymarket_price: Current Polymarket Yes price (0.0-1.0)
        slug: Market slug for caching (optional)

    Returns:
        MANIFOLD_BONUS (15) if arbitrage found, 0 otherwise.
    """
    # Check in-memory cache
    if slug:
        with _CACHE_LOCK:
            cached = _manifold_cache.get(slug)
            if cached and (time.time() - cached[0]) < MANIFOLD_CACHE_TTL:
                return cached[1]

    bonus = 0
    try:
        query = _extract_keywords(question)
        if not query:
            return 0

        resp = requests.get(
            MANIFOLD_SEARCH_API,
            params={"term": query, "limit": 3},
            timeout=MANIFOLD_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"[MANIFOLD-ARB] HTTP {resp.status_code}")
            return 0

        markets = resp.json()
        for m in markets:
            # Only binary, unresolved markets
            if m.get("outcomeType") != "BINARY":
                continue
            if m.get("isResolved"):
                continue
            prob = m.get("probability")
            if prob is None:
                continue

            gap = prob - polymarket_price
            if gap >= MANIFOLD_GAP_THRESHOLD:
                bonus = MANIFOLD_BONUS
                logger.info(
                    f"[MANIFOLD-ARB] '{m.get('question', '')[:50]}' prob={prob:.0%} "
                    f"vs PM={polymarket_price:.0%} gap={gap:+.0%} ≥ {MANIFOLD_GAP_THRESHOLD:.0%} "
                    f"→ +{MANIFOLD_BONUS}"
                )
                break

    except requests.exceptions.Timeout:
        logger.warning(f"[MANIFOLD-ARB] Timeout ({MANIFOLD_TIMEOUT}s)")
    except Exception as e:
        logger.warning(f"[MANIFOLD-ARB] Error: {type(e).__name__}: {e}")

    # Cache result
    if slug:
        with _CACHE_LOCK:
            _manifold_cache[slug] = (time.time(), bonus)

    return bonus


# ─── 3. DBnomics Macroeconomic Data ──────────────────────────────────

DBNOMICS_API = "https://api.db.nomics.world/v22/series"
DBNOMICS_TIMEOUT = 15
DBNOMICS_CACHE_TTL = 86400  # 24 hours
DBNOMICS_CLUSTERS = {"fed_fomc", "us_economic", "usa_politics"}
DBNOMICS_BONUS = 10

# Fed Funds Rate: FED, dataset H15, series RIFSPFF_N.M (monthly, %)
FED_FUNDS_PROVIDER = "FED"
FED_FUNDS_DATASET = "H15"
FED_FUNDS_SERIES = "RIFSPFF_N.M"

# US CPI: BLS, dataset cu, series CUSR0000SA0 (monthly, All Urban Consumers All Items)
CPI_PROVIDER = "BLS"
CPI_DATASET = "cu"
CPI_SERIES = "CUSR0000SA0"

# US Unemployment Rate: BLS, dataset ln, series LNS14000000 (monthly, %)
UNEMP_PROVIDER = "BLS"
UNEMP_DATASET = "ln"
UNEMP_SERIES = "LNS14000000"

# US GDP Growth: BEA, NIPA Table 1.1.1, series A191RL-Q (quarterly, % annualized)
GDP_PROVIDER = "BEA"
GDP_DATASET = "NIPA-T10101"
GDP_SERIES = "A191RL-Q"


def _fetch_dbnomics_series(provider: str, dataset: str, series: str, cache_file: str) -> dict[str, Any] | None:
    """Fetch the latest value with trend info from a DBnomics time series.

    Returns a dict with keys:
        - latest: float — most recent value
        - previous: float | None — second-most-recent value
        - delta: float | None — latest - previous
        - trend: str — "rising" | "falling" | "stable" | "unknown"
        - period: str — latest period (e.g. "2024-06")

    Returns None on failure. Cached for 24 hours.
    """
    cached = _load_cache(cache_file, DBNOMICS_CACHE_TTL)
    if cached is not None and isinstance(cached, dict):
        return cached
        # Old float format from previous version — falls through to refresh

    series_path = f"{provider}/{dataset}/{series}"
    try:
        resp = requests.get(
            f"{DBNOMICS_API}/{series_path}",
            params={"observations": "1", "format": "json"},
            timeout=DBNOMICS_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"[DBNOMICS] HTTP {resp.status_code} for {series_path}")
            return None

        data = resp.json()
        # Navigate DBnomics response structure
        series_data = data.get("series", {})
        docs = series_data.get("docs", [])
        if not docs:
            logger.warning(f"[DBNOMICS] No docs for {series_path}")
            return None

        # Observations are in docs[0]["period"] and docs[0]["value"]
        values = docs[0].get("value", [])
        periods = docs[0].get("period", [])

        # Find the last N non-"NA" values (need at least 2 for trend)
        valid: list[tuple[float, str]] = []
        for i in range(len(values) - 1, -1, -1):
            v = values[i]
            if v != "NA" and v is not None:
                try:
                    val = float(v)
                    period = periods[i] if i < len(periods) else ""
                    valid.append((val, period))
                    if len(valid) >= 2:
                        break
                except (ValueError, TypeError):
                    continue

        if not valid:
            logger.warning(f"[DBNOMICS] No valid values for {series_path}")
            return None

        latest_val, latest_period = valid[0]
        previous_val = valid[1][0] if len(valid) >= 2 else None
        delta = round(latest_val - previous_val, 4) if previous_val is not None else None

        if delta is None:
            trend = "unknown"
        elif abs(delta) < 0.001:
            trend = "stable"
        elif delta > 0:
            trend = "rising"
        else:
            trend = "falling"

        result: dict[str, Any] = {
            "latest": latest_val,
            "previous": previous_val,
            "delta": delta,
            "trend": trend,
            "period": latest_period,
        }
        logger.info(
            f"[DBNOMICS] {series}: {latest_val} "
            f"(delta={delta}, trend={trend}, period={latest_period})"
        )
        _save_cache(cache_file, result)
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"[DBNOMICS] Timeout ({DBNOMICS_TIMEOUT}s) for {series_path}")
    except Exception as e:
        logger.warning(f"[DBNOMICS] Error: {type(e).__name__}: {e}")
    return None


def get_fed_funds_rate() -> dict[str, Any] | None:
    """Get the latest US Federal Funds Rate (%) with trend. Cached 24h."""
    return _fetch_dbnomics_series(
        FED_FUNDS_PROVIDER, FED_FUNDS_DATASET, FED_FUNDS_SERIES,
        "oracle_fed_funds.json",
    )


def get_cpi_inflation() -> dict[str, Any] | None:
    """Get the latest US CPI index value with trend. Cached 24h."""
    return _fetch_dbnomics_series(
        CPI_PROVIDER, CPI_DATASET, CPI_SERIES,
        "oracle_cpi.json",
    )


def get_unemployment_rate() -> dict[str, Any] | None:
    """Get the latest US Unemployment Rate (%) with trend. Cached 24h."""
    return _fetch_dbnomics_series(
        UNEMP_PROVIDER, UNEMP_DATASET, UNEMP_SERIES,
        "oracle_unemployment.json",
    )


def get_gdp_growth() -> dict[str, Any] | None:
    """Get the latest US GDP Growth Rate (% annualized) with trend. Cached 24h."""
    return _fetch_dbnomics_series(
        GDP_PROVIDER, GDP_DATASET, GDP_SERIES,
        "oracle_gdp.json",
    )


def _check_macro_alignment(
    question: str,
    fed_rate: dict[str, Any] | None,
    cpi: dict[str, Any] | None,
    unemployment: dict[str, Any] | None = None,
    gdp: dict[str, Any] | None = None,
) -> bool:
    """Check if the contract question aligns with current macro trends.

    Trend-based alignment rules (v2):
    - Rate cut questions: aligned when Fed rate is falling OR high (≥4%) and not rising.
    - Rate hike questions: aligned when Fed rate is rising OR low (≤2%) and not falling.
    - Inflation questions: aligned when CPI is rising (delta > 0), not just absolute level.
    - Recession questions: aligned when multiple negative signals (high+rising rates,
      negative GDP, or rising unemployment ≥4%).
    - Unemployment questions: aligned when unemployment is rising.
    - GDP/economy questions: aligned when GDP growth is negative or slowing.
    """
    q_lower = question.lower()

    # Extract trend info safely
    fed_trend = fed_rate.get("trend", "unknown") if fed_rate else "unknown"
    fed_latest = fed_rate.get("latest") if fed_rate else None
    fed_delta = fed_rate.get("delta") if fed_rate else None

    cpi_trend = cpi.get("trend", "unknown") if cpi else "unknown"
    cpi_delta = cpi.get("delta") if cpi else None

    unemp_trend = unemployment.get("trend", "unknown") if unemployment else "unknown"
    unemp_latest = unemployment.get("latest") if unemployment else None

    gdp_latest = gdp.get("latest") if gdp else None
    gdp_trend = gdp.get("trend", "unknown") if gdp else "unknown"

    # ── Rate cut alignment: rates falling OR high and stable ──────────
    if any(kw in q_lower for kw in ("rate cut", "rate decrease", "lower rate", "cut rate", "reduce rate")):
        if fed_trend == "falling" and fed_delta is not None and fed_delta < 0:
            logger.info(f"[DBNOMICS] Rate-cut aligned: Fed falling (delta={fed_delta})")
            return True
        if fed_latest is not None and fed_latest >= 4.0 and fed_trend != "rising":
            logger.info(f"[DBNOMICS] Rate-cut aligned: Fed high={fed_latest}% (trend={fed_trend})")
            return True

    # ── Rate hike alignment: rates rising OR low and stable ───────────
    if any(kw in q_lower for kw in ("rate hike", "rate increase", "raise rate", "hike rate")):
        if fed_trend == "rising" and fed_delta is not None and fed_delta > 0:
            logger.info(f"[DBNOMICS] Rate-hike aligned: Fed rising (delta={fed_delta})")
            return True
        if fed_latest is not None and fed_latest <= 2.0 and fed_trend != "falling":
            logger.info(f"[DBNOMICS] Rate-hike aligned: Fed low={fed_latest}% (trend={fed_trend})")
            return True

    # ── Inflation alignment: CPI actually rising (not just absolute level) ─
    if any(kw in q_lower for kw in ("inflation", "cpi", "price rise", "prices rise", "prices increase")) and cpi_trend == "rising" and cpi_delta is not None and cpi_delta > 0:
        logger.info(f"[DBNOMICS] Inflation aligned: CPI rising (delta={cpi_delta})")
        return True

    # ── Recession alignment: multiple negative macro signals ──────────
    if any(kw in q_lower for kw in ("recession", "economic downturn", "contraction")):
        if fed_latest is not None and fed_latest >= 4.0 and fed_trend == "rising":
            logger.info(f"[DBNOMICS] Recession aligned: Fed high={fed_latest}% AND rising")
            return True
        if gdp_latest is not None and gdp_latest < 0:
            logger.info(f"[DBNOMICS] Recession aligned: GDP negative={gdp_latest}%")
            return True
        if unemp_latest is not None and unemp_trend == "rising" and unemp_latest >= 4.0:
            logger.info(f"[DBNOMICS] Recession aligned: unemployment rising={unemp_latest}%")
            return True

    # ── Unemployment / jobless alignment ──────────────────────────────
    if any(kw in q_lower for kw in ("unemployment", "jobless", "layoff", "fired")) and unemp_trend == "rising":
        logger.info(f"[DBNOMICS] Unemployment aligned: rate rising={unemp_latest}%")
        return True

    # ── GDP / economic growth alignment ───────────────────────────────
    if any(kw in q_lower for kw in ("gdp", "economic growth", "economic slowdown", "stagnation")):
        if gdp_latest is not None and gdp_latest < 1.0:
            logger.info(f"[DBNOMICS] GDP aligned: weak growth={gdp_latest}%")
            return True
        if gdp_trend == "falling":
            logger.info(f"[DBNOMICS] GDP aligned: growth decelerating (trend={gdp_trend})")
            return True

    return False


def dbnomics_macro_bonus(cluster: str, question: str) -> int:
    """Return +10 if cluster is macro AND question aligns with macro trend.

    Uses trend-based alignment (rising/falling/stable) across 4 series:
    Fed Funds Rate, CPI, Unemployment Rate, GDP Growth.

    Args:
        cluster: Market cluster name (e.g. 'fed_fomc', 'us_economic', 'usa_politics')
        question: Polymarket question text

    Returns:
        DBNOMICS_BONUS (10) if aligned, 0 otherwise.
    """
    if cluster not in DBNOMICS_CLUSTERS:
        return 0

    fed_rate = get_fed_funds_rate()
    cpi = get_cpi_inflation()
    unemployment = get_unemployment_rate()
    gdp = get_gdp_growth()

    if _check_macro_alignment(question, fed_rate, cpi, unemployment, gdp):
        return DBNOMICS_BONUS

    return 0


# ─── 4. Yahoo Finance (yfinance) ─────────────────────────────────────

YFINANCE_BONUS = 8
YFINANCE_CACHE_TTL = 3600  # 1 hour
YFINANCE_PROXIMITY_THRESHOLD = 0.10  # Within 10% of target = "in play"

# Map keywords in Polymarket questions to Yahoo Finance tickers
TICKER_MAP: dict[str, str] = {
    "s&p 500": "SPY",
    "s&p": "SPY",
    "sp500": "SPY",
    "sp 500": "SPY",
    "nasdaq": "QQQ",
    "dow jones": "DIA",
    "dow": "DIA",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "meta ": "META",
    "facebook": "META",
    "amazon": "AMZN",
    "gold": "GLD",
    "oil": "CL=F",
    "crude": "CL=F",
    "vix": "^VIX",
    "russell": "IWM",
    "goldman": "GS",
    "jpmorgan": "JPM",
    "jpm": "JPM",
}


def _detect_ticker(question: str) -> str | None:
    """Detect the most relevant Yahoo Finance ticker from a Polymarket question."""
    q_lower = question.lower()
    for keyword, ticker in TICKER_MAP.items():
        if keyword in q_lower:
            return ticker
    return None


def _extract_price_target(question: str) -> float | None:
    """Extract a price target from a Polymarket question.

    Matches patterns like:
      - "below 5000", "under $200", "drop to 4,000"
      - "above 5000", "over $200", "reach 4000"
      - "5000 level", "$200 mark"
    """
    q_lower = question.lower()
    # Remove commas from numbers (e.g., "5,000" → "5000")
    q_clean = q_lower.replace(",", "")

    patterns = [
        # "below/under/drop to/fall to X" or "above/over/reach X"
        r"(?:below|under|drop\s+to|fall\s+to|decline\s+to|dip\s+below|sink\s+below)\s+\$?([\d]+\.?\d*)",
        r"(?:above|over|reach|hit|rally\s+to|surge\s+to|climb\s+above|break\s+above)\s+\$?([\d]+\.?\d*)",
        # "$X level/mark/level"
        r"\$([\d]+\.?\d*)\s*(?:level|mark|threshold|barrier)",
        # Bare "$XXXX" in the question
        r"\$([\d]+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q_clean)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    return None


# In-memory cache: ticker → (timestamp, current_price)
_yf_cache: dict[str, tuple[float, float]] = {}


def _get_current_price(ticker: str) -> float | None:
    """Fetch the current/latest price for a ticker via yfinance.

    Cached for 1 hour per ticker.
    """
    with _CACHE_LOCK:
        cached = _yf_cache.get(ticker)
        if cached and (time.time() - cached[0]) < YFINANCE_CACHE_TTL:
            return cached[1]

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        price = info.get("last_price") or info.get("lastPrice") or info.get("previous_close")
        if price and price > 0:
            with _CACHE_LOCK:
                _yf_cache[ticker] = (time.time(), float(price))
            logger.info(f"[YFINANCE] {ticker} = ${float(price):.2f}")
            return float(price)
    except Exception as e:
        logger.warning(f"[YFINANCE] {ticker} error: {type(e).__name__}: {e}")
    return None


def yfinance_bonus(cluster: str, question: str) -> int:
    """Return +8 if the question mentions a stock/index AND current price
    is within 10% of the contract's price target.

    Logic: If SPY is at $498 and the contract asks "Will S&P 500 drop below
    5000 by Friday?", the contract is "live" — the target is only 0.4% away.
    This means the contract is realistically achievable → boost the signal.

    Returns 0 if no ticker or price target detected, or if price is far from target.
    """
    ticker = _detect_ticker(question)
    if not ticker:
        return 0

    target = _extract_price_target(question)
    if not target or target <= 0:
        return 0

    current = _get_current_price(ticker)
    if not current or current <= 0:
        return 0

    # Compute proximity: how close is current price to target?
    proximity = abs(current - target) / target if target > 0 else 1.0

    if proximity <= YFINANCE_PROXIMITY_THRESHOLD:
        logger.info(
            f"[YFINANCE-BONUS] {ticker}=${current:.2f} vs target=${target:.2f} "
            f"(proximity={proximity:.1%} ≤ {YFINANCE_PROXIMITY_THRESHOLD:.0%}) → +{YFINANCE_BONUS}"
        )
        return YFINANCE_BONUS

    logger.debug(
        f"[YFINANCE] {ticker}=${current:.2f} vs target=${target:.2f} "
        f"(proximity={proximity:.1%} > {YFINANCE_PROXIMITY_THRESHOLD:.0%}), no bonus"
    )
    return 0


# ─── 5. Wikipedia Pageviews Spike ────────────────────────────────────

WIKI_PAGEVIEWS_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
WIKI_TIMEOUT = 10
WIKI_CACHE_TTL = 21600  # 6 hours
WIKI_BONUS = 7
WIKI_SPIKE_MULTIPLIER = 2.0  # Recent views > 2x baseline = spike
WIKI_BASELINE_DAYS = 20
WIKI_RECENT_DAYS = 3

# In-memory cache: entity → (timestamp, spike_detected)
_wiki_cache: dict[str, tuple[float, bool]] = {}


def _extract_entities(question: str) -> list[str]:
    """Extract potential Wikipedia article titles from a Polymarket question.

    Looks for capitalized multi-word phrases (e.g., "Donald Trump", "Federal Reserve").
    Returns up to 3 candidates.
    """
    # Remove leading "Will " and trailing "?"
    clean = re.sub(r"^(Will|Is|Are|Does|Did|Has|Have|Can|Could|Would|Should|May|Might)\s+", "", question.strip("? "))

    # Find capitalized sequences (1-3 words)
    entities = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", clean)

    # Filter out common false positives
    stop_entities = {
        "The", "This", "That", "These", "Those", "Will", "January", "February",
        "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December", "Yes", "No",
    }
    filtered = [e for e in entities if e.split()[0] not in stop_entities]

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for e in filtered:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique[:3]


def _search_wikipedia(entity: str) -> str | None:
    """Search Wikipedia for the best matching article title for an entity."""
    try:
        resp = requests.get(
            WIKI_SEARCH_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": entity,
                "srlimit": 1,
                "format": "json",
            },
            timeout=WIKI_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if results:
            title = results[0]["title"]
            logger.debug(f"[WIKI] Search '{entity}' → '{title}'")
            return title
    except Exception as e:
        logger.debug(f"[WIKI] Search error: {type(e).__name__}: {e}")
    return None


def _fetch_pageviews(article: str, days: int) -> list[int]:
    """Fetch daily pageview counts for a Wikipedia article.

    Returns a list of daily view counts (most recent last).
    """
    from datetime import datetime, timedelta, UTC

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    # URL-encode the article title (spaces → underscores)
    article_encoded = article.replace(" ", "_")

    try:
        resp = requests.get(
            f"{WIKI_PAGEVIEWS_API}/{article_encoded}/daily/{start_str}00/{end_str}00",
            timeout=WIKI_TIMEOUT,
            headers={"User-Agent": "DOTM-Sniper-Bot/1.0 (research)"},
        )
        if resp.status_code != 200:
            logger.debug(f"[WIKI] Pageviews HTTP {resp.status_code} for '{article}'")
            return []
        data = resp.json()
        items = data.get("items", [])
        views = [item.get("views", 0) for item in items]
        return views
    except Exception as e:
        logger.debug(f"[WIKI] Pageviews error: {type(e).__name__}: {e}")
    return []


def _detect_wiki_spike(article: str) -> bool:
    """Detect if a Wikipedia article has a recent pageviews spike.

    Compares the last WIKI_RECENT_DAYS to the WIKI_BASELINE_DAYS median.
    Spike = recent median > WIKI_SPIKE_MULTIPLIER x baseline median.
    """
    # Check cache
    with _CACHE_LOCK:
        cached = _wiki_cache.get(article)
        if cached and (time.time() - cached[0]) < WIKI_CACHE_TTL:
            return cached[1]

    total_days = WIKI_BASELINE_DAYS + WIKI_RECENT_DAYS
    views = _fetch_pageviews(article, total_days)
    if len(views) < WIKI_BASELINE_DAYS + 1:
        # Not enough data
        with _CACHE_LOCK:
            _wiki_cache[article] = (time.time(), False)
        return False

    # Split into baseline and recent
    baseline = views[:WIKI_BASELINE_DAYS]
    recent = views[WIKI_BASELINE_DAYS:]

    # Compute medians (robust to outliers)
    baseline_sorted = sorted(baseline)
    baseline_median = baseline_sorted[len(baseline_sorted) // 2] if baseline_sorted else 0

    recent_sorted = sorted(recent)
    recent_median = recent_sorted[len(recent_sorted) // 2] if recent_sorted else 0

    if baseline_median < 10:
        # Very low baseline — not enough interest to be meaningful
        with _CACHE_LOCK:
            _wiki_cache[article] = (time.time(), False)
        return False

    spike = recent_median > (baseline_median * WIKI_SPIKE_MULTIPLIER)

    if spike:
        logger.info(
            f"[WIKI-SPIKE] '{article}' baseline={baseline_median}/d → recent={recent_median}/d "
            f"({recent_median / baseline_median:.1f}x > {WIKI_SPIKE_MULTIPLIER}x) → spike!"
        )

    with _CACHE_LOCK:
        _wiki_cache[article] = (time.time(), spike)
    return spike


def wikipedia_bonus(question: str) -> int:
    """Return +7 if any entity in the question has a Wikipedia pageviews spike.

    Extracts entity names, searches Wikipedia, and checks for pageview spikes
    indicating breaking news / heightened public interest.
    """
    entities = _extract_entities(question)
    if not entities:
        return 0

    for entity in entities:
        article = _search_wikipedia(entity)
        if not article:
            continue
        if _detect_wiki_spike(article):
            logger.info(
                f"[WIKI-BONUS] '{article}' spike detected for question "
                f"'{question[:50]}...' → +{WIKI_BONUS}"
            )
            return WIKI_BONUS

    return 0


# ─── Unified Entry Point (updated) ────────────────────────────────────

def compute_oracle_bonus(
    cluster: str,
    question: str,
    polymarket_price: float,
    slug: str = "",
) -> tuple[int, dict[str, int]]:
    """Compute total bonus from all five external oracles.

    Sources:
    1. Fear & Greed Index (+5)  — crypto/tech sentiment
    2. Manifold Arbitrage (+15) — cross-platform probability gap
    3. DBnomics Macro (+10)     — Fed rate / CPI / Unemployment / GDP trend alignment
    4. Yahoo Finance (+8)       — stock/index price proximity to target
    5. Wikipedia Spike (+7)     — pageviews spike = breaking news

    Args:
        cluster: Market cluster name
        question: Polymarket question text
        polymarket_price: Current Yes price (0.0-1.0)
        slug: Market slug for caching

    Returns:
        (total_bonus, breakdown) where breakdown maps source name → points
    """
    # Test/CI guard: skip all HTTP calls when disabled
    if os.environ.get("ORACLES_DISABLED") == "1":
        return 0, {"fng": 0, "manifold_arb": 0, "dbnomics": 0, "yfinance": 0, "wiki": 0}

    breakdown: dict[str, int] = {"fng": 0, "manifold_arb": 0, "dbnomics": 0, "yfinance": 0, "wiki": 0}

    try:
        breakdown["fng"] = fear_greed_bonus(cluster)
    except Exception as e:
        logger.warning(f"[ORACLE] FNG failed: {type(e).__name__}: {e}")

    try:
        breakdown["manifold_arb"] = check_manifold_arbitrage(question, polymarket_price, slug)
    except Exception as e:
        logger.warning(f"[ORACLE] Manifold arb failed: {type(e).__name__}: {e}")

    try:
        breakdown["dbnomics"] = dbnomics_macro_bonus(cluster, question)
    except Exception as e:
        logger.warning(f"[ORACLE] DBnomics failed: {type(e).__name__}: {e}")

    try:
        breakdown["yfinance"] = yfinance_bonus(cluster, question)
    except Exception as e:
        logger.warning(f"[ORACLE] Yahoo Finance failed: {type(e).__name__}: {e}")

    try:
        breakdown["wiki"] = wikipedia_bonus(question)
    except Exception as e:
        logger.warning(f"[ORACLE] Wikipedia failed: {type(e).__name__}: {e}")

    total = sum(breakdown.values())
    if total > 0:
        logger.info(
            f"[ORACLE-BONUS] {slug[:30]}... fng={breakdown['fng']} "
            f"manifold_arb={breakdown['manifold_arb']} "
            f"dbnomics={breakdown['dbnomics']} "
            f"yfinance={breakdown['yfinance']} "
            f"wiki={breakdown['wiki']} → total=+{total}"
        )

    return total, breakdown
