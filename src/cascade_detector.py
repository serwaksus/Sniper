"""
Cascade Detector — early detection of information cascades across related markets.

When a significant event occurs, information spreads through correlated markets
in a cascade. The first markets to react create arbitrage opportunities on
markets that haven't moved yet.

Uses the market graph for correlation data + price history for movement detection.

Signal: +10 points for markets in an active cascade that haven't moved yet.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

CASCADE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml_models")
CASCADE_STATE_FILE = os.path.join(CASCADE_DIR, "cascade_state.json")
MOVE_THRESHOLD_PCT = 0.15
MIN_MOVERS = 2
CASCADE_WINDOW_MINUTES = 60
CASCADE_DECAY_HOURS = 2.0
CASCADE_SIGNAL_SCORE = 10


class CascadeDetector:
    def __init__(self) -> None:
        self._price_snapshots: dict[str, list[dict[str, Any]]] = {}
        self._active_cascades: list[dict[str, Any]] = []
        self._cascade_markets: set[str] = set()
        self._last_check = 0.0

    def record_prices(self, markets: list[dict[str, Any]]) -> None:
        now = datetime.now()
        for m in markets:
            slug = m.get("slug", "")
            price = m.get("price", 0)
            if not slug or price <= 0:
                continue
            if slug not in self._price_snapshots:
                self._price_snapshots[slug] = []
            self._price_snapshots[slug].append({
                "ts": now.isoformat(),
                "price": float(price),
                "volume": float(m.get("volume", 0) or 0),
            })
            history = self._price_snapshots[slug]
            if len(history) > 20:
                self._price_snapshots[slug] = history[-20:]

    def detect_cascades(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = datetime.now()
        cutoff = now - timedelta(minutes=CASCADE_WINDOW_MINUTES)

        movers = []
        for m in markets:
            slug = m.get("slug", "")
            history = self._price_snapshots.get(slug, [])
            if len(history) < 2:
                continue

            recent = [h for h in history if self._parse_ts(h["ts"]) >= cutoff]
            if len(recent) < 2:
                continue

            first_price = recent[0]["price"]
            last_price = recent[-1]["price"]
            if first_price <= 0:
                continue

            pct_change = (last_price - first_price) / first_price
            if abs(pct_change) >= MOVE_THRESHOLD_PCT:
                movers.append({
                    "slug": slug,
                    "question": m.get("question", ""),
                    "change": round(pct_change, 3),
                    "first_price": first_price,
                    "last_price": last_price,
                    "cluster": m.get("clusters", ["other"])[0] if m.get("clusters") else "other",
                    "ts": recent[-1]["ts"],
                })

        movers.sort(key=lambda x: x["ts"])

        cascades = []
        if len(movers) >= MIN_MOVERS:
            clusters = [m["cluster"] for m in movers]
            from collections import Counter
            cluster_counts = Counter(clusters)
            dominant = cluster_counts.most_common(1)[0][0] if cluster_counts else "other"
            same_cluster_movers = [m for m in movers if m["cluster"] == dominant]

            if len(same_cluster_movers) >= MIN_MOVERS:
                leader = same_cluster_movers[0]
                followers = same_cluster_movers[1:]

                cascades.append({
                    "cascade_detected": True,
                    "leader": leader,
                    "followers": followers,
                    "cluster": dominant,
                    "n_movers": len(same_cluster_movers),
                    "detected_at": now.isoformat(),
                    "direction": "up" if leader["change"] > 0 else "down",
                })

        if cascades:
            self._active_cascades = cascades
            self._cascade_markets = set()
            for c in cascades:
                self._cascade_markets.add(c["leader"]["slug"])
                for f in c["followers"]:
                    self._cascade_markets.add(f["slug"])
            self._expire_old_cascades(now)
            logger.info(
                f"[CASCADE] Detected {len(cascades)} cascade(s): "
                f"leader={cascades[0]['leader']['slug'][:30]}... "
                f"({cascades[0]['leader']['change']:+.0%}), "
                f"{len(cascades[0]['followers'])} followers, "
                f"cluster={cascades[0]['cluster']}"
            )

        return cascades

    def find_laggard_opportunities(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._active_cascades:
            return []

        opportunities = []
        cascade = self._active_cascades[0]
        cascade_cluster = cascade["cluster"]
        leader_change = cascade["leader"]["change"]
        leader_direction = cascade["direction"]
        moved_slugs = self._cascade_markets

        try:
            from market_graph import get_analyzer
            analyzer = get_analyzer()
        except Exception:
            analyzer = None

        for m in markets:
            slug = m.get("slug", "")
            if slug in moved_slugs:
                continue

            m_cluster = m.get("clusters", ["other"])[0] if m.get("clusters") else "other"
            connected = False
            if analyzer and analyzer.graph.has_node(slug):
                for moved_slug in moved_slugs:
                    if analyzer.graph.has_node(moved_slug) and analyzer.graph.has_edge(slug, moved_slug):
                        connected = True
                        break

            if m_cluster == cascade_cluster or connected:
                history = self._price_snapshots.get(slug, [])
                recent_change = 0.0
                if len(history) >= 2:
                    recent_change = (history[-1]["price"] - history[0]["price"]) / max(history[0]["price"], 0.001)

                expected_abs = abs(leader_change) * 0.6
                if leader_direction == "up" and recent_change < MOVE_THRESHOLD_PCT * 0.5:
                    opportunities.append({
                        "slug": slug,
                        "question": m.get("question", ""),
                        "price": m.get("price", 0),
                        "cluster": m_cluster,
                        "expected_move": round(expected_abs, 3),
                        "current_change": round(recent_change, 3),
                        "direction": leader_direction,
                        "signal_score": CASCADE_SIGNAL_SCORE,
                        "reason": f"cascade_laggard: {cascade['leader']['slug'][:25]} moved {leader_change:+.0%}",
                    })

        if opportunities:
            logger.info(f"[CASCADE] Found {len(opportunities)} laggard opportunities in cluster '{cascade_cluster}'")

        return opportunities[:10]

    def get_cascade_signal(self, slug: str) -> int:
        if not self._active_cascades:
            return 0
        if slug in self._cascade_markets:
            return 0
        try:
            from market_graph import get_analyzer
            analyzer = get_analyzer()
            if analyzer.graph.has_node(slug):
                for moved_slug in self._cascade_markets:
                    if analyzer.graph.has_node(moved_slug) and analyzer.graph.has_edge(slug, moved_slug):
                        return CASCADE_SIGNAL_SCORE
        except Exception:
            pass
        return 0

    def _expire_old_cascades(self, now: datetime) -> None:
        keep = []
        for c in self._active_cascades:
            try:
                detected = self._parse_ts(c["detected_at"])
                age_hours = (now - detected).total_seconds() / 3600
                if age_hours < CASCADE_DECAY_HOURS:
                    keep.append(c)
            except Exception:
                continue
        if len(keep) < len(self._active_cascades):
            expired = len(self._active_cascades) - len(keep)
            logger.info(f"[CASCADE] Expired {expired} old cascade(s)")
            self._active_cascades = keep
            if not keep:
                self._cascade_markets = set()

    def _parse_ts(self, ts_str: str) -> datetime:
        try:
            return datetime.fromisoformat(ts_str)
        except Exception:
            return datetime.min

    def save(self) -> None:
        os.makedirs(CASCADE_DIR, exist_ok=True)
        state = {
            "active_cascades": self._active_cascades,
            "cascade_markets": list(self._cascade_markets),
            "last_check": self._last_check,
            "saved_at": datetime.now().isoformat(),
        }
        with open(CASCADE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load(self) -> bool:
        if not os.path.exists(CASCADE_STATE_FILE):
            return False
        try:
            with open(CASCADE_STATE_FILE) as f:
                state = json.load(f)
            self._active_cascades = state.get("active_cascades", [])
            self._cascade_markets = set(state.get("cascade_markets", []))
            self._last_check = state.get("last_check", 0)
            n = len(self._active_cascades)
            if n:
                logger.info(f"[CASCADE] Loaded {n} active cascade(s)")
            return n > 0
        except Exception as e:
            logger.debug(f"[CASCADE] Load failed: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        return {
            "active_cascades": len(self._active_cascades),
            "cascade_markets": list(self._cascade_markets)[:10],
            "price_snapshots": len(self._price_snapshots),
        }


_detector: CascadeDetector | None = None


def get_detector() -> CascadeDetector:
    global _detector
    if _detector is None:
        _detector = CascadeDetector()
        _detector.load()
    return _detector


def record_prices(markets: list[dict[str, Any]]) -> None:
    get_detector().record_prices(markets)


def detect_and_find(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detector = get_detector()
    detector.detect_cascades(markets)
    return detector.find_laggard_opportunities(markets)


def get_cascade_signal(slug: str) -> int:
    return get_detector().get_cascade_signal(slug)
