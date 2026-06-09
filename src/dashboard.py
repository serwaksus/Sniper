"""Lightweight HTML dashboard for DOTM Sniper."""
from __future__ import annotations

import html
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cluster_report import compute_cluster_stats
import positions_db
from utils import load_json
from config import EQUITY_CURVE_FILE, HEALTH_STATE_FILE


def build_dashboard_html() -> str:
    stats = compute_cluster_stats()
    positions = positions_db.load_all()
    equity = load_json(EQUITY_CURVE_FILE, {})
    health = load_json(HEALTH_STATE_FILE, {})

    snapshots = equity.get("snapshots", [])
    latest = snapshots[-1] if snapshots else {}

    total_equity = float(stats.get("total_equity") or 0)
    cash = float(stats.get("cash") or 0)
    total_pnl = float(stats.get("total_pnl") or 0)
    open_positions = stats.get("open_positions", 0)
    total_resolved = stats.get("total_resolved", 0)

    positions_value = float(latest.get("positions_value") or 0)
    unrealized_pnl = float(latest.get("unrealized_pnl") or 0)

    pnl_class = "positive" if total_pnl >= 0 else "negative"
    pnl_sign = "+" if total_pnl >= 0 else ""

    cluster_rows = ""
    total_invested = 0.0
    for cluster, data in sorted(stats["clusters"].items(), key=lambda x: -x[1]["investment"]):
        total_invested += data["investment"]
        cluster_rows += (
            f"<tr><td>{html.escape(cluster)}</td>"
            f"<td>{data['positions']}</td>"
            f"<td>${data['investment']:.2f}</td></tr>\n"
        )
    cluster_rows += (
        f"<tr><td><b>Total</b></td><td></td><td><b>${total_invested:.2f}</b></td></tr>\n"
    )

    equity_positions = latest.get("positions", [])
    price_map: dict[str, dict] = {}
    for ep in equity_positions:
        slug = ep.get("slug", "")
        price_map[slug] = ep

    position_rows = ""
    for slug, pos in sorted(positions.items()):
        entry = float(pos.get("entry_price", 0) or 0)
        shares = int(pos.get("shares", 0) or 0)
        question = pos.get("market_question", slug)
        cluster_list = pos.get("clusters") or ["other"]
        primary_cluster = cluster_list[0] if cluster_list else "other"

        ep = price_map.get(slug, {})
        current = float(ep.get("current", entry))
        pnl_pct = float(ep.get("pnl_pct", 0))

        pnl_color = "positive" if pnl_pct >= 0 else "negative"
        pnl_sign_pos = "+" if pnl_pct >= 0 else ""

        position_rows += (
            f"<tr>"
            f"<td title='{html.escape(slug)}'>{html.escape(question[:50])}</td>"
            f"<td>${entry:.4f}</td>"
            f"<td>${current:.4f}</td>"
            f"<td>{shares}</td>"
            f"<td class='{pnl_color}'>{pnl_sign_pos}{pnl_pct:.1f}%</td>"
            f"<td>{html.escape(primary_cluster)}</td>"
            f"</tr>\n"
        )

    if not position_rows:
        position_rows = "<tr><td colspan='6'>No open positions</td></tr>\n"

    health_status = "ok" if not health.get("alerts") else "warning"
    health_color = "#00ff88" if health_status == "ok" else "#ff4444"

    return f"""<!DOCTYPE html>
<html><head>
<title>DOTM Sniper Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 20px; }}
h1 {{ color: #00d4ff; }}
h2 {{ color: #7b68ee; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
th {{ color: #00d4ff; }}
.positive {{ color: #00ff88; }}
.negative {{ color: #ff4444; }}
.card {{ background: #16213e; padding: 15px; border-radius: 8px; display: inline-block; margin: 5px; min-width: 150px; }}
.card .value {{ font-size: 24px; font-weight: bold; }}
.card .label {{ color: #888; font-size: 12px; }}
</style>
</head><body>
<h1>DOTM Sniper Dashboard</h1>

<div>
  <div class="card"><div class="label">Equity</div><div class="value">${total_equity:.2f}</div></div>
  <div class="card"><div class="label">Cash</div><div class="value">${cash:.2f}</div></div>
  <div class="card"><div class="label">Positions Value</div><div class="value">${positions_value:.2f}</div></div>
  <div class="card"><div class="label">PnL</div><div class="value {pnl_class}">{pnl_sign}${total_pnl:.2f}</div></div>
  <div class="card"><div class="label">Unrealized PnL</div><div class="value {'positive' if unrealized_pnl >= 0 else 'negative'}">${unrealized_pnl:.2f}</div></div>
  <div class="card"><div class="label">Positions</div><div class="value">{open_positions}</div></div>
  <div class="card"><div class="label">Resolved</div><div class="value">{total_resolved}</div></div>
  <div class="card"><div class="label">Health</div><div class="value" style="color:{health_color}">{health_status}</div></div>
</div>

<h2>By Cluster</h2>
<table>
<tr><th>Cluster</th><th>Positions</th><th>Invested</th></tr>
{cluster_rows}
</table>

<h2>Active Positions</h2>
<table>
<tr><th>Market</th><th>Entry</th><th>Current</th><th>Shares</th><th>PnL%</th><th>Cluster</th></tr>
{position_rows}
</table>

</body></html>"""
