"""
analysis/centrality.py
-------------------------
Graph-wide centrality metrics used to find structurally important assets
(e.g. an unremarkable jump-box that sits on every path to the domain
controller). Prefers Neo4j Graph Data Science (GDS) if the plugin is
installed; otherwise falls back to computing the same metrics locally
with networkx (useful for local dev without the GDS plugin).
"""

import logging
from typing import List, Dict, Any

from graph.connection import get_connection
from graph import queries as q

logger = logging.getLogger(__name__)

GRAPH_NAME = "attack-graph"


def _gds_available(conn) -> bool:
    try:
        conn.run_query("CALL gds.version() YIELD gitVersion RETURN gitVersion")
        return True
    except Exception:
        return False


def _project_graph(conn) -> None:
    conn.run_write(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
    conn.run_write(
        f"""
        CALL gds.graph.project(
            '{GRAPH_NAME}',
            'Asset',
            'CONNECTS_TO'
        )
        """
    )


def compute_pagerank_gds() -> List[Dict[str, Any]]:
    conn = get_connection()
    _project_graph(conn)
    rows = conn.run_query(
        f"""
        CALL gds.pageRank.stream('{GRAPH_NAME}')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).asset_id AS asset_id,
               gds.util.asNode(nodeId).hostname AS hostname,
               score
        ORDER BY score DESC
        """
    )
    conn.run_write(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
    return rows


def compute_betweenness_gds() -> List[Dict[str, Any]]:
    conn = get_connection()
    _project_graph(conn)
    rows = conn.run_query(
        f"""
        CALL gds.betweenness.stream('{GRAPH_NAME}')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).asset_id AS asset_id,
               gds.util.asNode(nodeId).hostname AS hostname,
               score
        ORDER BY score DESC
        """
    )
    conn.run_write(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
    return rows


def _to_networkx():
    import networkx as nx

    conn = get_connection()
    edges = conn.run_query(
        "MATCH (a:Asset)-[:CONNECTS_TO]->(b:Asset) "
        "RETURN a.asset_id AS src, b.asset_id AS dst"
    )
    nodes = conn.run_query("MATCH (a:Asset) RETURN a.asset_id AS id, a.hostname AS hostname")

    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], hostname=n["hostname"])
    for e in edges:
        g.add_edge(e["src"], e["dst"])
    return g


def compute_pagerank_networkx() -> List[Dict[str, Any]]:
    import networkx as nx

    g = _to_networkx()
    scores = nx.pagerank(g) if g.number_of_nodes() else {}
    return sorted(
        [
            {"asset_id": n, "hostname": g.nodes[n].get("hostname"), "score": s}
            for n, s in scores.items()
        ],
        key=lambda r: r["score"],
        reverse=True,
    )


def compute_betweenness_networkx() -> List[Dict[str, Any]]:
    import networkx as nx

    g = _to_networkx()
    scores = nx.betweenness_centrality(g) if g.number_of_nodes() else {}
    return sorted(
        [
            {"asset_id": n, "hostname": g.nodes[n].get("hostname"), "score": s}
            for n, s in scores.items()
        ],
        key=lambda r: r["score"],
        reverse=True,
    )


def pagerank() -> List[Dict[str, Any]]:
    conn = get_connection()
    if _gds_available(conn):
        try:
            return compute_pagerank_gds()
        except Exception as e:
            logger.warning("GDS pagerank failed (%s), falling back to networkx", e)
    return compute_pagerank_networkx()


def betweenness() -> List[Dict[str, Any]]:
    conn = get_connection()
    if _gds_available(conn):
        try:
            return compute_betweenness_gds()
        except Exception as e:
            logger.warning("GDS betweenness failed (%s), falling back to networkx", e)
    return compute_betweenness_networkx()


def degree() -> List[Dict[str, Any]]:
    """Simple degree centrality — always available, no GDS/networkx needed."""
    return q.run(q.Q_DEGREE_RAW)


if __name__ == "__main__":
    import json

    print("PageRank:", json.dumps(pagerank()[:5], indent=2))
    print("Betweenness:", json.dumps(betweenness()[:5], indent=2))
