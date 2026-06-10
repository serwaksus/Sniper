"""Tests for market_graph.py — graph construction, cascades, diversification, correlation, hedges, persistence."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import market_graph as mg
from market_graph import MarketGraphAnalyzer


def _make_markets():
    return [
        {"slug": "trump-tariffs", "question": "Will Trump impose new tariffs on China?",
         "price": 0.15, "clusters": ["usa_politics", "fed_fomc"]},
        {"slug": "fed-rate-cut", "question": "Will the Fed cut interest rates in 2026?",
         "price": 0.25, "clusters": ["fed_fomc"]},
        {"slug": "russia-ceasefire", "question": "Will Russia agree to a ceasefire with Ukraine?",
         "price": 0.10, "clusters": ["russia_ukraine"]},
        {"slug": "putin-nato", "question": "Will Putin attack a NATO country?",
         "price": 0.05, "clusters": ["russia_ukraine"]},
        {"slug": "nba-champs", "question": "Will the Lakers win the NBA championship?",
         "price": 0.08, "clusters": ["sports_nba"]},
        {"slug": "btc-100k", "question": "Will Bitcoin reach $100k?",
         "price": 0.30, "clusters": ["crypto"]},
        {"slug": "biden-election", "question": "Will Biden win the election?",
         "price": 0.12, "clusters": ["usa_politics"]},
        {"slug": "oil-prices", "question": "Will oil prices spike due to recession fears?",
         "price": 0.20, "clusters": ["fed_fomc", "russia_ukraine"]},
    ]


class TestBuildGraph:
    def test_builds_nodes(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        assert a.graph.number_of_nodes() == 8

    def test_builds_edges(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        assert a.graph.number_of_edges() > 0

    def test_shared_cluster_creates_edge(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        assert a.graph.has_edge("trump-tariffs", "fed-rate-cut")

    def test_no_edge_across_unrelated_clusters(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        assert not a.graph.has_edge("nba-champs", "btc-100k")

    def test_cache_prevents_rebuild(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        first_built = a._last_built
        a.build_graph(_make_markets(), force=False)
        assert a._last_built == first_built

    def test_force_rebuilds(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        first_built = a._last_built
        a.build_graph(_make_markets(), force=True)
        assert a._last_built >= first_built

    def test_empty_slug_skipped(self):
        a = MarketGraphAnalyzer()
        markets = _make_markets() + [{"slug": "", "question": "No slug"}]
        a.build_graph(markets, force=True)
        assert a.graph.number_of_nodes() == 8


class TestComputeEdgeWeight:
    def test_shared_non_other_cluster(self):
        a = MarketGraphAnalyzer()
        n1 = {"clusters": ["usa_politics", "fed_fomc"], "question": "Will Trump win?"}
        n2 = {"clusters": ["usa_politics"], "question": "Will Biden win?"}
        w = a._compute_edge_weight(n1, n2)
        assert w >= 0.5

    def test_only_other_cluster(self):
        a = MarketGraphAnalyzer()
        n1 = {"clusters": ["other"], "question": "Something random"}
        n2 = {"clusters": ["other"], "question": "Something else"}
        w = a._compute_edge_weight(n1, n2)
        assert w <= 0.1

    def test_entity_overlap(self):
        a = MarketGraphAnalyzer()
        n1 = {"clusters": ["russia_ukraine"], "question": "Will Russia withdraw troops from Ukraine?"}
        n2 = {"clusters": ["russia_ukraine"], "question": "Will Russia agree to peace deal with Ukraine?"}
        w = a._compute_edge_weight(n1, n2)
        assert w >= 0.5

    def test_entity_relationship(self):
        a = MarketGraphAnalyzer()
        n1 = {"clusters": ["fed_fomc"], "question": "Will the Fed raise rates?"}
        n2 = {"clusters": ["fed_fomc"], "question": "Will Powell signal rate cuts?"}
        w = a._compute_edge_weight(n1, n2)
        assert w >= 0.4


class TestInformationCascade:
    def test_cascade_returns_neighbors(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        cascade = a.find_information_cascade("trump-tariffs")
        assert len(cascade) > 0

    def test_cascade_depth_limited(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        cascade = a.find_information_cascade("trump-tariffs")
        for info in cascade.values():
            assert info["depth"] <= 3

    def test_cascade_probability_decays(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        cascade = a.find_information_cascade("trump-tariffs")
        for info in cascade.values():
            assert 0 < info["probability"] <= 1.0

    def test_cascade_missing_slug_returns_empty(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        assert a.find_information_cascade("nonexistent") == {}


class TestDiversificationScore:
    def test_single_position_ok(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        result = a.portfolio_diversification_score(["trump-tariffs"])
        assert result["diversification_score"] == 1.0

    def test_same_cluster_low_score(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        result = a.portfolio_diversification_score(["trump-tariffs", "fed-rate-cut", "oil-prices"])
        assert result["diversification_score"] < 1.0

    def test_uncorrelated_high_score(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        result = a.portfolio_diversification_score(["nba-champs", "btc-100k"])
        assert result["diversification_score"] == 1.0

    def test_empty_list_ok(self):
        a = MarketGraphAnalyzer()
        result = a.portfolio_diversification_score([])
        assert result["diversification_score"] == 1.0


class TestCheckCorrelationBeforeTrade:
    def test_detects_correlated_pair(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        result = a.check_correlation_before_trade("fed-rate-cut", ["trump-tariffs"])
        assert result["correlated"] is True
        assert len(result["warnings"]) > 0

    def test_uncorrelated_ok(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        result = a.check_correlation_before_trade("nba-champs", ["btc-100k"])
        assert result["correlated"] is False

    def test_missing_slug_ok(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        result = a.check_correlation_before_trade("nonexistent", ["nba-champs"])
        assert result["correlated"] is False

    def test_empty_positions_ok(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        result = a.check_correlation_before_trade("fed-rate-cut", [])
        assert result["correlated"] is False


class TestSuggestHedges:
    def test_returns_different_cluster(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        hedges = a.suggest_hedges("trump-tariffs")
        for h in hedges:
            assert h["cluster"] != "usa_politics"

    def test_missing_slug_empty(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        assert a.suggest_hedges("nonexistent") == []

    def test_hedge_has_required_fields(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        hedges = a.suggest_hedges("trump-tariffs")
        for h in hedges:
            assert "slug" in h
            assert "question" in h
            assert "price" in h
            assert "cluster" in h


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        a1 = MarketGraphAnalyzer()
        orig_file = mg.GRAPH_STATE_FILE
        tmpdir = tempfile.mkdtemp()
        mg.GRAPH_STATE_FILE = os.path.join(tmpdir, "graph.json")
        try:
            a1.build_graph(_make_markets(), force=True)
            a1.save()
            a2 = MarketGraphAnalyzer()
            loaded = a2.load()
            assert loaded is True
            assert a2.graph.number_of_nodes() == a1.graph.number_of_nodes()
            assert a2.graph.number_of_edges() == a1.graph.number_of_edges()
            assert len(a2._communities) == len(a1._communities)
        finally:
            mg.GRAPH_STATE_FILE = orig_file

    def test_load_missing_file(self):
        a = MarketGraphAnalyzer()
        orig_file = mg.GRAPH_STATE_FILE
        mg.GRAPH_STATE_FILE = "/tmp/nonexistent_graph_test.json"
        try:
            assert a.load() is False
        finally:
            mg.GRAPH_STATE_FILE = orig_file


class TestCommunityDetection:
    def test_communities_detected(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        comms = a.get_communities()
        assert len(comms) >= 1

    def test_all_nodes_in_community(self):
        a = MarketGraphAnalyzer()
        a.build_graph(_make_markets(), force=True)
        comms = a.get_communities()
        all_nodes = set()
        for c in comms:
            all_nodes |= c
        assert all_nodes == set(a.graph.nodes)


class TestModuleFunctions:
    def test_get_analyzer_returns_instance(self):
        mg._analyzer = None
        a = mg.get_analyzer()
        assert isinstance(a, MarketGraphAnalyzer)

    def test_build_graph_if_stale(self):
        mg._analyzer = None
        mg.build_graph_if_stale(_make_markets())
        a = mg.get_analyzer()
        assert a.graph.number_of_nodes() == 8

    def test_check_correlation_module_func(self):
        mg._analyzer = None
        mg.build_graph_if_stale(_make_markets())
        result = mg.check_correlation("fed-rate-cut", ["trump-tariffs"])
        assert "correlated" in result

    def test_portfolio_diversification_module_func(self):
        mg._analyzer = None
        mg.build_graph_if_stale(_make_markets())
        result = mg.portfolio_diversification(["nba-champs", "btc-100k"])
        assert "diversification_score" in result
