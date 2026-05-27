#!/usr/bin/env python3
"""
Correlation matrix for DOTM Sniper positions.
Tracks pairwise price correlations and limits exposure to correlated clusters.
"""
import json
import os
import sys
import math
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json

PRICE_HISTORY_FILE = "/root/dotm-sniper/price_history.json"
CORRELATION_FILE = "/root/dotm-sniper/correlation_matrix.json"
POSITIONS_FILE = "/root/dotm-sniper/positions.json"

logger = logging.getLogger(__name__)

CORRELATED_GROUPS = {
    "trump_admin_politics": [
        "usa_politics", "russia_ukraine", "geopolitics", "venezuela",
    ],
    "us_economic": [
        "fed_fomc", "usa_politics",
    ],
    "sports": [
        "sports_nba", "sports_ufc",
    ],
    "tech_ai": [
        "ai_tech", "tech",
    ],
}

MAX_CORRELATED_GROUP_PCT = 0.25


def _get_price_series(slug: str, min_points: int = 5) -> List[float]:
    try:
        history = load_json(PRICE_HISTORY_FILE, {})
        if not isinstance(history, dict):
            return []
        slug_data = history.get(slug, [])
        if len(slug_data) < min_points:
            return []
        return [e["p"] for e in slug_data if "p" in e]
    except Exception:
        return []


def compute_pairwise_correlation(slug_a: str, slug_b: str) -> Optional[float]:
    series_a = _get_price_series(slug_a)
    series_b = _get_price_series(slug_b)

    min_len = min(len(series_a), len(series_b))
    if min_len < 5:
        return None

    a = series_a[-min_len:]
    b = series_b[-min_len:]

    ret_a = [(a[i] - a[i-1]) / max(abs(a[i-1]), 1e-6) for i in range(1, len(a))]
    ret_b = [(b[i] - b[i-1]) / max(abs(b[i-1]), 1e-6) for i in range(1, len(b))]

    if not ret_a or not ret_b:
        return None

    n = min(len(ret_a), len(ret_b))
    ret_a = ret_a[:n]
    ret_b = ret_b[:n]

    mean_a = sum(ret_a) / n
    mean_b = sum(ret_b) / n

    cov = sum((ret_a[i] - mean_a) * (ret_b[i] - mean_b) for i in range(n)) / n
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in ret_a) / n) if n > 1 else 0
    std_b = math.sqrt(sum((x - mean_b) ** 2 for x in ret_b) / n) if n > 1 else 0

    if std_a < 1e-8 or std_b < 1e-8:
        return None

    corr = cov / (std_a * std_b)
    return max(-1.0, min(1.0, corr))


def get_correlated_exposure(positions: Dict, balance: float) -> Dict[str, float]:
    if not positions or balance <= 0:
        return {}

    cluster_investment = defaultdict(float)
    for slug, pos in positions.items():
        cluster = pos.get("clusters", ["other"])[0] if isinstance(pos.get("clusters"), list) else "other"
        invested = pos.get("entry_price", 0) * pos.get("shares", 0)
        cluster_investment[cluster] += invested

    group_exposure = {}
    for group_name, clusters in CORRELATED_GROUPS.items():
        total = sum(cluster_investment.get(c, 0) for c in clusters)
        if total > 0:
            pct = total / balance
            group_exposure[group_name] = round(pct, 3)

    return group_exposure


def check_correlation_limit(new_cluster: str, positions: Dict, balance: float,
                            new_investment: float = 0) -> Tuple[bool, str]:
    if not positions or balance <= 0:
        return True, "ok"

    new_group = None
    for group_name, clusters in CORRELATED_GROUPS.items():
        if new_cluster in clusters:
            new_group = group_name
            break

    if not new_group:
        return True, "ok"

    group_clusters = CORRELATED_GROUPS[new_group]
    cluster_investment = defaultdict(float)
    for slug, pos in positions.items():
        cluster = pos.get("clusters", ["other"])[0] if isinstance(pos.get("clusters"), list) else "other"
        if cluster in group_clusters:
            invested = pos.get("entry_price", 0) * pos.get("shares", 0)
            cluster_investment[cluster] += invested

    current_group_total = sum(cluster_investment.values()) + new_investment
    group_pct = current_group_total / balance

    if group_pct > MAX_CORRELATED_GROUP_PCT:
        return False, (f"correlated_group={new_group} exposure={group_pct:.1%} > "
                       f"{MAX_CORRELATED_GROUP_PCT:.0%} (clusters: {group_clusters})")

    return True, "ok"


def update_correlation_matrix():
    positions = load_json(POSITIONS_FILE, {})
    if not isinstance(positions, dict) or len(positions) < 2:
        return

    slugs = list(positions.keys())
    matrix = {}

    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            corr = compute_pairwise_correlation(slugs[i], slugs[j])
            if corr is not None:
                key = f"{slugs[i]}|{slugs[j]}"
                matrix[key] = round(corr, 3)

    if matrix:
        save_json(CORRELATION_FILE, {
            "matrix": matrix,
            "updated_at": datetime.now().isoformat(),
        })
        logger.info(f"[CORR] Updated {len(matrix)} pairwise correlations")
