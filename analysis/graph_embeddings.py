"""
analysis/graph_embeddings.py
--------------------------------
Structural graph embeddings: a dense vector per Asset node that encodes its
position/role in the network topology, learned from graph structure alone
(no vulnerability/risk features involved) -- distinct from the text
embeddings in ml/embeddings.py (which embed CVE *descriptions*, not graph
structure) and from the hand-engineered features in ml/features.py (degree,
PageRank as individual scalar columns).

These embeddings are used as a baseline comparison in ml/gnn_models.py, and
are generally useful for: similarity search ("find assets structurally
similar to our known-compromised host"), clustering, and as an additional
feature source for the RiskDNN.

Primary method: Neo4j GDS FastRP (fast, scalable, designed for exactly this).
Fallback (no GDS plugin): truncated SVD on the adjacency matrix -- a
legitimate, well-established spectral graph embedding method, just less
scalable than FastRP. This mirrors the GDS/networkx fallback pattern
already used in analysis/centrality.py.
"""

import logging
from typing import List, Dict, Any

import numpy as np

from graph.connection import get_connection

logger = logging.getLogger(__name__)

GRAPH_NAME = "attack-graph-embed"
EMBEDDING_DIM = 32


def _gds_available(conn) -> bool:
    try:
        conn.run_query("CALL gds.version() YIELD gitVersion RETURN gitVersion")
        return True
    except Exception:
        return False


def compute_fastrp_embeddings(dim: int = EMBEDDING_DIM) -> List[Dict[str, Any]]:
    """FastRP (Fast Random Projection) via Neo4j GDS -- a scalable structural
    embedding method well-suited to this graph size and larger."""
    conn = get_connection()
    conn.run_write(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
    conn.run_write(
        f"CALL gds.graph.project('{GRAPH_NAME}', 'Asset', "
        f"{{CONNECTS_TO: {{orientation: 'UNDIRECTED'}}}})"
    )
    rows = conn.run_query(
        f"""
        CALL gds.fastRP.stream('{GRAPH_NAME}', {{
            embeddingDimension: {dim},
            randomSeed: 42
        }})
        YIELD nodeId, embedding
        RETURN gds.util.asNode(nodeId).asset_id AS asset_id,
               gds.util.asNode(nodeId).hostname AS hostname,
               embedding
        """
    )
    conn.run_write(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
    return rows


def _build_adjacency(asset_ids: List[str]) -> np.ndarray:
    conn = get_connection()
    edges = conn.run_query(
        "MATCH (a:Asset)-[:CONNECTS_TO]-(b:Asset) RETURN a.asset_id AS src, b.asset_id AS dst"
    )
    idx = {aid: i for i, aid in enumerate(asset_ids)}
    n = len(asset_ids)
    adj = np.zeros((n, n), dtype=np.float32)
    for e in edges:
        if e["src"] in idx and e["dst"] in idx:
            adj[idx[e["src"]], idx[e["dst"]]] = 1.0
    return adj


def compute_svd_embeddings(dim: int = EMBEDDING_DIM) -> List[Dict[str, Any]]:
    """Fallback when the GDS plugin isn't installed: truncated SVD on the
    (symmetrized) adjacency matrix. This is a real spectral embedding
    technique -- related to Laplacian eigenmaps -- not a placeholder; it's
    just less scalable than FastRP for very large graphs."""
    from sklearn.decomposition import TruncatedSVD

    conn = get_connection()
    nodes = conn.run_query("MATCH (a:Asset) RETURN a.asset_id AS asset_id, a.hostname AS hostname")
    asset_ids = [n["asset_id"] for n in nodes]
    if len(asset_ids) < 2:
        return []

    adj = _build_adjacency(asset_ids)
    n_components = min(dim, len(asset_ids) - 1, adj.shape[1] - 1)
    if n_components < 1:
        return [{"asset_id": aid, "hostname": n["hostname"], "embedding": [0.0]} for aid, n in zip(asset_ids, nodes)]

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(adj)

    return [
        {"asset_id": aid, "hostname": n["hostname"], "embedding": reduced[i].tolist()}
        for i, (aid, n) in enumerate(zip(asset_ids, nodes))
    ]


def compute_graph_embeddings(dim: int = EMBEDDING_DIM) -> List[Dict[str, Any]]:
    """Primary entry point -- tries FastRP via GDS, falls back to SVD."""
    conn = get_connection()
    if _gds_available(conn):
        try:
            return compute_fastrp_embeddings(dim=dim)
        except Exception as e:
            logger.warning("FastRP failed (%s), falling back to SVD embeddings", e)
    return compute_svd_embeddings(dim=dim)


def store_embeddings_on_graph(dim: int = EMBEDDING_DIM) -> int:
    """Computes embeddings and writes them back onto Asset nodes as
    `graph_embedding`, so they can be reused as GNN baseline features or for
    similarity queries without recomputing each time."""
    embeddings = compute_graph_embeddings(dim=dim)
    if not embeddings:
        return 0

    rows = [{"asset_id": e["asset_id"], "embedding": e["embedding"]} for e in embeddings]
    query = """
    UNWIND $rows AS row
    MATCH (a:Asset {asset_id: row.asset_id})
    SET a.graph_embedding = row.embedding
    """
    get_connection().run_batch_write(query, rows, batch_size=len(rows))
    return len(rows)


if __name__ == "__main__":
    import json

    embeddings = compute_graph_embeddings()
    print(f"Computed {len(embeddings)} graph embeddings (dim={EMBEDDING_DIM})")
    if embeddings:
        print("Sample:", json.dumps(embeddings[0], indent=2)[:300])
