"""Typed contracts for cross-module data flow.

All forecast source functions (manifold, metaculus, metaforecast) MUST
return GapCheckResult. mypy --strict enforces this at compile time,
catching missing keys and wrong field names BEFORE runtime.

Usage in source modules:
    from contracts import GapCheckResult, ForecastResult

    def check_manifold_gap(...) -> GapCheckResult | None:
        return {"found": True, "probability": 0.5, ...}

Usage in consumers:
    gap: GapCheckResult | None = check_manifold_gap(market)
    if gap and gap["found"]:
        prob: float = gap["probability"]  # mypy knows this is float
"""
from __future__ import annotations

from typing import TypedDict


class ForecastResult(TypedDict, total=False):
    """Return type for get_*_forecast functions (batch path).

    Used by: manifold.get_manifold_forecast, metaculus.get_metaculus_forecast,
    metaforecast.get_metaforecast_forecast.
    """
    found: bool
    probability: float | None
    question_title: str
    url: str
    forecaster_count: int
    stars: int
    match_score: float
    reason: str
    timestamp: str


class GapCheckResult(TypedDict, total=False):
    """Return type for check_*_gap functions (single/signal path).

    Used by: manifold.check_manifold_gap, metaculus.check_metaculus_gap,
    metaforecast.check_metaforecast_gap.

    CRITICAL: All sources MUST include 'found' and 'probability' keys
    in their success return dicts. Consumers check gap.get("found")
    and access gap["probability"] directly.

    Aliases (metaculus_prob, manifold_prob) exist for backward compat
    but should NOT be relied upon by new code.
    """
    # ── Required for consumer logic ──
    found: bool               # MUST be present in ALL returns
    probability: float | None  # MUST be present when found=True

    # ── Standard fields ──
    polymarket_prob: float
    signal_strength: float
    source: str               # "manifold" | "metaculus" | "metaforecast"
    url: str

    # ── Optional fields ──
    gap: float
    required_gap: float
    dispersion_penalty: float
    forecaster_count: int
    match_score: float
    num_platforms: int
    dispersion: float

    # ── Legacy aliases (do not use in new code) ──
    metaculus_prob: float     # Alias for probability
    manifold_prob: float      # Alias for probability

    reasoning: str
