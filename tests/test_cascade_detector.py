from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def _tmp_state(tmp_path, monkeypatch):
    import cascade_detector as cd
    state_file = str(tmp_path / "cascade_state.json")
    monkeypatch.setattr(cd, "CASCADE_STATE_FILE", state_file)
    monkeypatch.setattr(cd, "CASCADE_DIR", str(tmp_path))
    cd._detector = None
    yield
    cd._detector = None


def _make_market(slug, price, volume=1000, clusters=None):
    return {
        "slug": slug,
        "question": f"Will {slug} happen?",
        "price": price,
        "volume": volume,
        "clusters": clusters or ["other"],
    }


class TestPriceRecording:
    def test_records_prices(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        d.record_prices([_make_market("m1", 0.10), _make_market("m2", 0.20)])
        assert "m1" in d._price_snapshots
        assert len(d._price_snapshots["m1"]) == 1

    def test_limits_history(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        for i in range(25):
            d.record_prices([_make_market("m1", 0.10 + i * 0.001)])
        assert len(d._price_snapshots["m1"]) <= 20


class TestCascadeDetection:
    def test_no_cascade_with_few_movers(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        now = datetime.now()
        for i in range(5):
            d._price_snapshots[f"m{i}"] = [
                {"ts": (now - timedelta(minutes=30)).isoformat(), "price": 0.05, "volume": 1000},
                {"ts": now.isoformat(), "price": 0.05, "volume": 1000},
            ]
        markets = [_make_market(f"m{i}", 0.05) for i in range(5)]
        result = d.detect_cascades(markets)
        assert len(result) == 0

    def test_detects_cascade(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        now = datetime.now()
        early = (now - timedelta(minutes=30)).isoformat()
        d._price_snapshots["leader"] = [
            {"ts": early, "price": 0.05, "volume": 1000},
            {"ts": now.isoformat(), "price": 0.08, "volume": 5000},
        ]
        d._price_snapshots["follower"] = [
            {"ts": early, "price": 0.10, "volume": 1000},
            {"ts": now.isoformat(), "price": 0.13, "volume": 3000},
        ]
        markets = [
            _make_market("leader", 0.08, clusters=["usa_politics"]),
            _make_market("follower", 0.13, clusters=["usa_politics"]),
        ]
        result = d.detect_cascades(markets)
        assert len(result) == 1
        assert result[0]["cascade_detected"] is True
        assert result[0]["leader"]["slug"] == "leader"

    def test_different_clusters_no_cascade(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        now = datetime.now()
        early = (now - timedelta(minutes=30)).isoformat()
        d._price_snapshots["m1"] = [
            {"ts": early, "price": 0.05, "volume": 1000},
            {"ts": now.isoformat(), "price": 0.08, "volume": 5000},
        ]
        d._price_snapshots["m2"] = [
            {"ts": early, "price": 0.05, "volume": 1000},
            {"ts": now.isoformat(), "price": 0.08, "volume": 3000},
        ]
        markets = [
            _make_market("m1", 0.08, clusters=["usa_politics"]),
            _make_market("m2", 0.08, clusters=["sports_nba"]),
        ]
        result = d.detect_cascades(markets)
        assert len(result) == 0

    def test_insufficient_price_history(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        d._price_snapshots["m1"] = [{"ts": datetime.now().isoformat(), "price": 0.05, "volume": 1000}]
        markets = [_make_market("m1", 0.08)]
        result = d.detect_cascades(markets)
        assert len(result) == 0


class TestLaggardOpportunities:
    def test_finds_laggards(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        now = datetime.now()
        early = (now - timedelta(minutes=30)).isoformat()
        d._price_snapshots["leader"] = [
            {"ts": early, "price": 0.05, "volume": 1000},
            {"ts": now.isoformat(), "price": 0.08, "volume": 5000},
        ]
        d._price_snapshots["follower"] = [
            {"ts": early, "price": 0.10, "volume": 1000},
            {"ts": now.isoformat(), "price": 0.13, "volume": 3000},
        ]
        d._price_snapshots["laggard"] = [
            {"ts": early, "price": 0.05, "volume": 1000},
            {"ts": now.isoformat(), "price": 0.051, "volume": 1000},
        ]
        markets = [
            _make_market("leader", 0.08, clusters=["usa_politics"]),
            _make_market("follower", 0.13, clusters=["usa_politics"]),
            _make_market("laggard", 0.051, clusters=["usa_politics"]),
        ]
        d.detect_cascades(markets)
        opps = d.find_laggard_opportunities(markets)
        assert len(opps) == 1
        assert opps[0]["slug"] == "laggard"
        assert opps[0]["signal_score"] == 10

    def test_no_laggards_without_cascade(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        markets = [_make_market("m1", 0.10)]
        opps = d.find_laggard_opportunities(markets)
        assert len(opps) == 0


class TestCascadeSignal:
    def test_signal_for_connected_market(self):
        from cascade_detector import CascadeDetector
        import networkx as nx
        d = CascadeDetector()
        d._active_cascades = [{
            "leader": {"slug": "leader", "change": 0.6, "cluster": "usa_politics"},
            "followers": [],
            "cluster": "usa_politics",
            "detected_at": datetime.now().isoformat(),
            "direction": "up",
        }]
        d._cascade_markets = {"leader"}

        mock_analyzer = type("A", (), {"graph": nx.Graph()})()
        mock_analyzer.graph.add_node("leader")
        mock_analyzer.graph.add_node("connected")
        mock_analyzer.graph.add_edge("leader", "connected", weight=0.5)
        with patch("market_graph.get_analyzer", return_value=mock_analyzer):
            sig = d.get_cascade_signal("connected")
        assert sig == 10

    def test_no_signal_for_moved_market(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        d._active_cascades = [{"detected_at": datetime.now().isoformat()}]
        d._cascade_markets = {"leader"}
        assert d.get_cascade_signal("leader") == 0


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        d._active_cascades = [{"detected_at": datetime.now().isoformat(), "leader": {"slug": "test"}}]
        d._cascade_markets = {"test"}
        d.save()

        d2 = CascadeDetector()
        loaded = d2.load()
        assert loaded
        assert len(d2._active_cascades) == 1
        assert "test" in d2._cascade_markets

    def test_load_nonexistent(self):
        from cascade_detector import CascadeDetector
        d = CascadeDetector()
        assert not d.load()


class TestModuleFunctions:
    def test_get_detector_singleton(self):
        import cascade_detector as cd
        d1 = cd.get_detector()
        d2 = cd.get_detector()
        assert d1 is d2

    def test_record_and_detect(self):
        import cascade_detector as cd
        cd.record_prices([_make_market("m1", 0.05)])
        assert "m1" in cd.get_detector()._price_snapshots

    def test_get_status(self):
        import cascade_detector as cd
        status = cd.get_detector().get_status()
        assert "active_cascades" in status
        assert "price_snapshots" in status
