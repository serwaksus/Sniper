"""
Cross-Market Graph Analysis — detect hidden correlations, information cascades,
and portfolio concentration risk via graph-based analysis.

Graph construction strategy:
  - Explicit edges: shared clusters (weight=0.5), shared entities extracted from
    market questions via keyword overlap (weight=0.3)
  - Price correlation: if we have price history, compute Pearson correlation
    between co-held positions (weight=correlation_coefficient)
  - NO LLM calls for graph building (too expensive O(n²))
  - LLM only for explicit pair analysis when adding highly correlated positions

Uses: networkx for graph ops, community detection (Louvain)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "market_graph.json")
CACHE_TTL = 3600

ENTITY_RELATIONSHIPS = {
    "trump": {"usa_politics", "russia_ukraine", "fed_fomc"},
    "biden": {"usa_politics"},
    "putin": {"russia_ukraine"},
    "zelensky": {"russia_ukraine"},
    "fed": {"fed_fomc"},
    "powell": {"fed_fomc"},
    "nato": {"russia_ukraine", "usa_politics"},
    "iran": {"russia_ukraine", "usa_politics"},
    "china": {"usa_politics", "fed_fomc"},
    "oil": {"fed_fomc", "russia_ukraine"},
    "tariff": {"usa_politics", "fed_fomc"},
    "recession": {"fed_fomc"},
    "s&p": {"fed_fomc"},
}

STOP_WORDS = frozenset({
    "the", "a", "an", "in", "of", "to", "by", "will", "is", "be",
    "for", "on", "or", "and", "it", "at",
})


class MarketGraphAnalyzer:
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self._last_built = 0.0
        self._communities: list[set[str]] = []

    def build_graph(self, markets: list[dict[str, Any]], force: bool = False) -> None:
        """Build market correlation graph from market data.

        Edges based on:
        1. Shared clusters (weight=0.5)
        2. Entity overlap from questions (weight=0.3)
        3. Same entity relationships (weight=0.4)
        """
        now = time.time()
        if not force and self._last_built and now - self._last_built < CACHE_TTL:
            return

        self.graph.clear()

        for m in markets:
            slug = m.get("slug", "")
            if not slug:
                continue
            self.graph.add_node(
                slug,
                question=m.get("question", ""),
                price=m.get("price", 0),
                clusters=m.get("clusters", []),
                cluster=m.get("clusters", ["other"])[0] if m.get("clusters") else "other",
            )

        slugs = list(self.graph.nodes)
        for i, s1 in enumerate(slugs):
            n1 = self.graph.nodes[s1]
            for s2 in slugs[i + 1 :]:
                n2 = self.graph.nodes[s2]
                weight = self._compute_edge_weight(n1, n2)
                if weight > 0.15:
                    self.graph.add_edge(s1, s2, weight=round(weight, 3))

        self._last_built = now

        try:
            import community as community_louvain

            partition = community_louvain.best_partition(self.graph)
            communities: dict[int, set[str]] = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            self._communities = list(communities.values())
        except Exception:
            self._communities = []

        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        n_comm = len(self._communities)
        logger.info(f"[GRAPH] Built: {n_nodes} nodes, {n_edges} edges, {n_comm} communities")

    def _compute_edge_weight(self, n1: dict, n2: dict) -> float:
        """Compute correlation weight between two market nodes."""
        weight = 0.0

        c1 = set(n1.get("clusters", []))
        c2 = set(n2.get("clusters", []))
        shared = c1 & c2
        if shared and shared != {"other"}:
            weight += 0.5
        elif shared:
            weight += 0.1

        q1_words = set(n1.get("question", "").lower().split())
        q2_words = set(n2.get("question", "").lower().split())
        overlap = q1_words & q2_words
        meaningful = overlap - STOP_WORDS
        if len(meaningful) >= 2:
            weight += 0.3

        for entity, _related_clusters in ENTITY_RELATIONSHIPS.items():
            e_in_1 = entity in n1.get("question", "").lower()
            e_in_2 = entity in n2.get("question", "").lower()
            if e_in_1 and e_in_2:
                weight += 0.4
                break

        return min(weight, 1.0)

    def find_information_cascade(self, triggered_slug: str) -> dict[str, dict[str, Any]]:
        """When one market moves, which markets are likely to follow?"""
        if triggered_slug not in self.graph:
            return {}

        cascade: dict[str, dict[str, Any]] = {}
        visited = {triggered_slug}
        frontier = [(triggered_slug, 0)]

        while frontier:
            current, depth = frontier.pop(0)
            if depth >= 3:
                continue
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                edge_data = self.graph[current][neighbor]
                base_weight = edge_data.get("weight", 0.5)
                decay = 0.7 ** (depth + 1)
                cascade[neighbor] = {
                    "probability": round(base_weight * decay, 3),
                    "depth": depth + 1,
                    "weight": base_weight,
                    "question": self.graph.nodes[neighbor].get("question", ""),
                }
                frontier.append((neighbor, depth + 1))

        return dict(sorted(cascade.items(), key=lambda x: -x[1]["probability"]))

    def portfolio_diversification_score(self, position_slugs: list[str]) -> dict[str, Any]:
        """Assess how diversified the portfolio is based on graph structure."""
        if len(position_slugs) < 2:
            return {"diversification_score": 1.0, "concentration_risk": 0.0, "recommendation": "OK"}

        valid = [s for s in position_slugs if s in self.graph]
        if len(valid) < 2:
            return {"diversification_score": 1.0, "concentration_risk": 0.0, "recommendation": "OK"}

        subgraph = self.graph.subgraph(valid)
        density = nx.density(subgraph)

        try:
            centrality = nx.pagerank(subgraph, weight="weight")
            max_centrality = max(centrality.values()) if centrality else 0
        except Exception:
            max_centrality = 0

        score = max(0, 1.0 - density)

        recommendation = "OK"
        if density > 0.5:
            recommendation = "HIGHLY_CORRELATED — reduce exposure"
        elif density > 0.3:
            recommendation = "MODERATE_CORRELATION — consider hedging"

        return {
            "diversification_score": round(score, 3),
            "concentration_risk": round(max_centrality, 3),
            "density": round(density, 3),
            "n_edges": subgraph.number_of_edges(),
            "recommendation": recommendation,
        }

    def check_correlation_before_trade(self, new_slug: str, position_slugs: list[str],
                                       threshold: float = 0.4) -> dict[str, Any]:
        """Check if adding new_slug would create unwanted correlation with existing positions."""
        if new_slug not in self.graph or not position_slugs:
            return {"correlated": False, "max_correlation": 0, "warnings": []}

        warnings = []
        max_corr = 0.0
        for existing in position_slugs:
            if existing not in self.graph:
                continue
            if self.graph.has_edge(new_slug, existing):
                weight = self.graph[new_slug][existing].get("weight", 0)
                if weight > threshold:
                    q = self.graph.nodes[existing].get("question", existing)[:40]
                    warnings.append(f"{q}... (r={weight:.2f})")
                    max_corr = max(max_corr, weight)

        return {
            "correlated": max_corr > threshold,
            "max_correlation": round(max_corr, 3),
            "warnings": warnings,
        }

    def suggest_hedges(self, position_slug: str) -> list[dict[str, Any]]:
        """Suggest hedge positions for a given market."""
        if position_slug not in self.graph:
            return []

        cluster = self.graph.nodes[position_slug].get("cluster", "other")
        hedges = []
        for node in self.graph.nodes:
            if node == position_slug:
                continue
            node_cluster = self.graph.nodes[node].get("cluster", "other")
            if node_cluster != cluster and not self.graph.has_edge(position_slug, node):
                hedges.append({
                    "slug": node,
                    "question": self.graph.nodes[node].get("question", ""),
                    "price": self.graph.nodes[node].get("price", 0),
                    "cluster": node_cluster,
                })

        return hedges[:5]

    def get_communities(self) -> list[set[str]]:
        return self._communities

    def save(self) -> None:
        data = nx.node_link_data(self.graph)
        data["last_built"] = self._last_built
        data["communities"] = [list(c) for c in self._communities]
        with open(GRAPH_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        if not os.path.exists(GRAPH_STATE_FILE):
            return False
        try:
            with open(GRAPH_STATE_FILE) as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, directed=False)
            self._last_built = data.get("last_built", 0)
            self._communities = [set(c) for c in data.get("communities", [])]
            logger.info(f"[GRAPH] Loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return True
        except Exception as e:
            logger.debug(f"[GRAPH] Load failed: {e}")
            return False


_analyzer: MarketGraphAnalyzer | None = None


def get_analyzer() -> MarketGraphAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = MarketGraphAnalyzer()
        _analyzer.load()
    return _analyzer


def build_graph_if_stale(markets: list[dict[str, Any]]) -> None:
    get_analyzer().build_graph(markets)


def check_correlation(new_slug: str, position_slugs: list[str]) -> dict[str, Any]:
    return get_analyzer().check_correlation_before_trade(new_slug, position_slugs)


def portfolio_diversification(position_slugs: list[str]) -> dict[str, Any]:
    return get_analyzer().portfolio_diversification_score(position_slugs)
