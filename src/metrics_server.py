#!/usr/bin/env python3
"""Lightweight metrics server for DOTM Sniper monitoring.

Start standalone:
    python3 src/metrics_server.py [port]

Cron suggestion (health probe every 5 min):
    */5 * * * * curl -sf http://localhost:8765/health >/dev/null || echo "DOTM metrics server down" | mail -s "ALERT" admin
"""
from __future__ import annotations
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json
import positions_db
from db import load_settings
from config import EQUITY_CURVE_FILE, HEALTH_STATE_FILE


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/metrics/prometheus":
            self._serve_prometheus()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_metrics(self) -> None:
        data = {}
        positions = positions_db.load_all()
        settings = load_settings()
        equity = load_json(EQUITY_CURVE_FILE, {})
        health = load_json(HEALTH_STATE_FILE, {})

        data["positions_count"] = len(positions)
        data["total_resolved"] = settings.get("total_resolved", 0)

        snapshots = equity.get("snapshots", [])
        if snapshots:
            latest = snapshots[-1]
            data["total_equity"] = latest.get("total_equity")
            data["cash"] = latest.get("cash")
            data["positions_value"] = latest.get("positions_value")
            data["unrealized_pnl"] = latest.get("unrealized_pnl")

        data["health_alerts"] = len(health.get("alerts", []))
        data["position_slugs"] = list(positions.keys())

        self._json_response(data)

    def _serve_prometheus(self):
        positions = positions_db.load_all()
        settings = load_settings()
        equity = load_json(EQUITY_CURVE_FILE, {})
        health = load_json(HEALTH_STATE_FILE, {})

        positions_count = len(positions)
        total_resolved = settings.get("total_resolved", 0)
        total_equity = 0.0
        cash = 0.0
        unrealized_pnl = 0.0

        snapshots = equity.get("snapshots", [])
        if snapshots:
            latest = snapshots[-1]
            total_equity = float(latest.get("total_equity") or 0)
            cash = float(latest.get("cash") or 0)
            unrealized_pnl = float(latest.get("unrealized_pnl") or 0)

        health_alerts = len(health.get("alerts", []))

        lines = [
            "# HELP dotm_positions_open Number of open positions",
            "# TYPE dotm_positions_open gauge",
            f"dotm_positions_open {positions_count}",
            "# HELP dotm_equity_total Total equity in dollars",
            "# TYPE dotm_equity_total gauge",
            f"dotm_equity_total {total_equity:.2f}",
            "# HELP dotm_cash Available cash",
            "# TYPE dotm_cash gauge",
            f"dotm_cash {cash:.2f}",
            "# HELP dotm_pnl_total Total PnL in dollars",
            "# TYPE dotm_pnl_total gauge",
            f"dotm_pnl_total {unrealized_pnl:.2f}",
            "# HELP dotm_trades_resolved Total resolved trades",
            "# TYPE dotm_trades_resolved gauge",
            f"dotm_trades_resolved {total_resolved}",
            "# HELP dotm_health_alerts Active health alerts",
            "# TYPE dotm_health_alerts gauge",
            f"dotm_health_alerts {health_alerts}",
        ]
        body = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def _serve_health(self) -> None:
        health = load_json(HEALTH_STATE_FILE, {})
        positions = positions_db.load_all()
        self._json_response({
            "status": "ok" if not health.get("alerts") else "warning",
            "alerts": health.get("alerts", []),
            "positions": len(positions),
        })

    def _json_response(self, data: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def log_message(self, format: str, *args) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(("127.0.0.1", port), MetricsHandler)
    print(f"Metrics server on :{port}")
    server.serve_forever()
