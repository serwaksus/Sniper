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

3. **DBnomics** (macroeconomic data — Fed Funds Rate, CPI)
   - Endpoint: GET https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}
   - Cache: 24 hours
   - Score: +10 if cluster ∈ {fed_fomc, us_economic} AND macro trend aligns with contract

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
DBNOMICS_CLUSTERS = {"fed_fomc", "us_economic"}
DBNOMICS_BONUS = 10

# Fed Funds Rate: FRED, series FEDFUNDS
FED_FUNDS_PROVIDER = "FRED"
FED_FUNDS_DATASET = "FEDFUNDS"
FED_FUNDS_SERIES = "FEDFUNDS"

# US CPI: FRED, series CPIAUCSL (Consumer Price Index for All Urban Consumers)
CPI_PROVIDER = "FRED"
CPI_DATASET = "CPIAUCSL"
CPI_SERIES = "CPIAUCSL"


def _fetch_dbnomics_series(provider: str, dataset: str, series: str, cache_file: str) -> float | None:
    """Fetch the latest value from a DBnomics time series.

    Returns the latest period value, or None on failure.
    Cached for 24 hours.
    """
    cached = _load_cache(cache_file, DBNOMICS_CACHE_TTL)
    if cached is not None:
        return cached

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
        # Format: {"period": ["2024-01", "2024-02", ...], "value": [5.25, 5.5, ...]}
        values = docs[0].get("value", [])
        periods = docs[0].get("period", [])

        # Find the latest non-"NA" value
        latest_val: float | None = None
        latest_period = ""
        for i in range(len(values) - 1, -1, -1):
            v = values[i]
            if v != "NA" and v is not None:
                try:
                    latest_val = float(v)
                    latest_period = periods[i] if i < len(periods) else ""
                    break
                except (ValueError, TypeError):
                    continue

        if latest_val is not None:
            logger.info(f"[DBNOMICS] {series}: {latest_val} (period={latest_period})")
            _save_cache(cache_file, latest_val)
            return latest_val

        logger.warning(f"[DBNOMICS] No valid values for {series_path}")
        return None

    except requests.exceptions.Timeout:
        logger.warning(f"[DBNOMICS] Timeout ({DBNOMICS_TIMEOUT}s) for {series_path}")
    except Exception as e:
        logger.warning(f"[DBNOMICS] Error: {type(e).__name__}: {e}")
    return None


def get_fed_funds_rate() -> float | None:
    """Get the latest US Federal Funds Rate (%). Cached 24h."""
    return _fetch_dbnomics_series(
        FED_FUNDS_PROVIDER, FED_FUNDS_DATASET, FED_FUNDS_SERIES,
        "oracle_fed_funds.json",
    )


def get_cpi_inflation() -> float | None:
    """Get the latest US CPI index value. Cached 24h."""
    return _fetch_dbnomics_series(
        CPI_PROVIDER, CPI_DATASET, CPI_SERIES,
        "oracle_cpi.json",
    )


def _check_macro_alignment(question: str, fed_rate: float | None, cpi: float | None) -> bool:
    """Check if the contract question aligns with the current macro trend.

    Heuristic alignment rules:
    - If question mentions "rate cut", "rate decrease", "lower" → aligned when
      Fed rate is high (≥ 4.0%) — rate cuts are likely.
    - If question mentions "rate hike", "rate increase", "raise" → aligned when
      Fed rate is low (≤ 2.0%) — rate hikes are possible.
    - If question mentions "inflation", "CPI", "prices rise" → aligned when
      CPI is trending high (we use absolute CPI > 300 as a proxy for high inflation).
    - Default: if any macro data is available, consider it a soft alignment.
    """
    q_lower = question.lower()

    # Rate cut alignment
    if any(kw in q_lower for kw in ("rate cut", "rate decrease", "lower rate", "cut rate", "reduce rate")) and fed_rate is not None and fed_rate >= 4.0:
        logger.info(f"[DBNOMICS] Rate-cut question aligned with high Fed rate={fed_rate}%")
        return True

    # Rate hike alignment
    if any(kw in q_lower for kw in ("rate hike", "rate increase", "raise rate", "hike rate")) and fed_rate is not None and fed_rate <= 2.0:
        logger.info(f"[DBNOMICS] Rate-hike question aligned with low Fed rate={fed_rate}%")
        return True

    # Inflation alignment
    if any(kw in q_lower for kw in ("inflation", "cpi", "price rise", "prices rise")) and cpi is not None and cpi > 300:
        logger.info(f"[DBNOMICS] Inflation question aligned with high CPI={cpi}")
        return True

    # Recession alignment — high rates may cause recession
    if any(kw in q_lower for kw in ("recession", "economic downturn", "contraction")) and fed_rate is not None and fed_rate >= 4.5:
        logger.info(f"[DBNOMICS] Recession question aligned with high Fed rate={fed_rate}%")
        return True

    return False


def dbnomics_macro_bonus(cluster: str, question: str) -> int:
    """Return +10 if cluster is macro AND question aligns with macro trend.

    Args:
        cluster: Market cluster name (e.g. 'fed_fomc', 'us_economic')
        question: Polymarket question text

    Returns:
        DBNOMICS_BONUS (10) if aligned, 0 otherwise.
    """
    if cluster not in DBNOMICS_CLUSTERS:
        return 0

    fed_rate = get_fed_funds_rate()
    cpi = get_cpi_inflation()

    if _check_macro_alignment(question, fed_rate, cpi):
        return DBNOMICS_BONUS

    return 0


# ─── Unified Entry Point ─────────────────────────────────────────────

def compute_oracle_bonus(
    cluster: str,
    question: str,
    polymarket_price: float,
    slug: str = "",
) -> tuple[int, dict[str, int]]:
    """Compute total bonus from all three external oracles.

    This is the main entry point called by signal_scorer._compute_signal_score.

    Args:
        cluster: Market cluster name
        question: Polymarket question text
        polymarket_price: Current Yes price (0.0-1.0)
        slug: Market slug for caching

    Returns:
        (total_bonus, breakdown) where breakdown is {"fng": int, "manifold_arb": int, "dbnomics": int}
    """
    # Test/CI guard: skip all HTTP calls when disabled
    if os.environ.get("ORACLES_DISABLED") == "1":
        return 0, {"fng": 0, "manifold_arb": 0, "dbnomics": 0}

    breakdown: dict[str, int] = {"fng": 0, "manifold_arb": 0, "dbnomics": 0}

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

    total = sum(breakdown.values())
    if total > 0:
        logger.info(
            f"[ORACLE-BONUS] {slug[:30]}... fng={breakdown['fng']} "
            f"manifold_arb={breakdown['manifold_arb']} "
            f"dbnomics={breakdown['dbnomics']} → total=+{total}"
        )

    return total, breakdown
