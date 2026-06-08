#!/usr/bin/env python3
"""Lightweight metrics server for DOTM Sniper monitoring.

Start standalone:
    python3 src/metrics_server.py [port]

Cron suggestion (health probe every 5 min):
    */5 * * * * curl -sf http://localhost:8765/health >/dev/null || echo "DOTM metrics server down" | mail -s "ALERT" admin
"""
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json
import positions_db
from db import load_settings


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_metrics(self):
        data = {}
        positions = positions_db.load_all()
        settings = load_settings()
        equity = load_json("/root/dotm-sniper/equity_curve.json", {})
        health = load_json("/root/dotm-sniper/health_state.json", {})

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

    def _serve_health(self):
        health = load_json("/root/dotm-sniper/health_state.json", {})
        positions = positions_db.load_all()
        self._json_response({
            "status": "ok" if not health.get("alerts") else "warning",
            "alerts": health.get("alerts", []),
            "positions": len(positions),
        })

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"Metrics server on :{port}")
    server.serve_forever()
