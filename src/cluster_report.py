"""Per-cluster PnL tracking and reporting."""
from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import positions_db
from position_manager import detect_clusters
from db import load_settings
from utils import load_json
from config import EQUITY_CURVE_FILE

logger = logging.getLogger(__name__)


def compute_cluster_stats() -> dict:
    positions = positions_db.load_all()
    settings = load_settings()
    equity = load_json(EQUITY_CURVE_FILE, {})

    total_resolved = settings.get("total_resolved", 0)
    total_pnl = settings.get("total_pnl", 0)

    clusters: dict[str, dict] = {}

    for slug, pos in positions.items():
        question = pos.get("market_question", "")
        entry = pos.get("entry_price", 0)
        shares = pos.get("shares", 0)
        investment = entry * shares if entry and shares else 0

        cluster_list = pos.get("clusters") or detect_clusters(question)
        primary = cluster_list[0] if cluster_list else "other"

        if primary not in clusters:
            clusters[primary] = {
                "positions": 0,
                "investment": 0.0,
                "slugs": [],
            }
        clusters[primary]["positions"] += 1
        clusters[primary]["investment"] += investment
        clusters[primary]["slugs"].append(slug)

    snapshots = equity.get("snapshots", [])
    current_equity = snapshots[-1] if snapshots else {}

    return {
        "total_resolved": total_resolved,
        "total_pnl": total_pnl,
        "total_equity": current_equity.get("total_equity", 0),
        "cash": current_equity.get("cash", 0),
        "open_positions": len(positions),
        "clusters": clusters,
    }


def format_cluster_report() -> str:
    stats = compute_cluster_stats()
    lines = ["📊 Cluster PnL Report"]
    lines.append(f"Equity: ${stats['total_equity']:.2f} | Cash: ${stats['cash']:.2f}")
    lines.append(f"Resolved: {stats['total_resolved']} | PnL: ${stats['total_pnl']:.2f}")
    lines.append(f"Open positions: {stats['open_positions']}")
    lines.append("")

    total_invested = 0.0
    for cluster, data in sorted(stats["clusters"].items(), key=lambda x: -x[1]["investment"]):
        total_invested += data["investment"]
        lines.append(
            f"  {cluster}: {data['positions']} pos, ${data['investment']:.2f} invested"
        )

    lines.append(f"\n  Total invested: ${total_invested:.2f}")
    return "\n".join(lines)
