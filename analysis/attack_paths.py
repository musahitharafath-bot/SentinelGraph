"""
analysis/attack_paths.py
--------------------------
Discovers attacker-relevant paths through the network graph:
  - shortest path between two arbitrary assets
  - all simple paths (bounded depth) between two assets
  - all paths from internet-exposed assets to critical assets
  - "weighted" attack paths where each hop is scored by the exploitability
    of vulnerabilities on the traversed asset, so shorter-but-safer paths
    can be distinguished from shorter-but-riskier ones.
"""

from typing import List, Dict, Any, Optional
from graph.connection import get_connection
from graph import queries as q

Q_WEIGHTED_PATHS = """
MATCH (start:Asset {asset_id: $start_id}), (end:Asset {asset_id: $end_id})
MATCH p = (start)-[:CONNECTS_TO*..%d]->(end)
WITH p, nodes(p) AS ns
UNWIND ns AS n
OPTIONAL MATCH (n)-[:HAS_VULNERABILITY]->(v:Vulnerability)
WITH p, ns, n, coalesce(max(v.cvss_score), 0.0) AS node_max_cvss
WITH p, ns, collect(node_max_cvss) AS cvss_per_node
RETURN [x IN ns | x.asset_id] AS path_assets,
       [x IN ns | x.hostname] AS path_hostnames,
       length(p) AS hops,
       reduce(s = 0.0, c IN cvss_per_node | s + c) AS path_risk_score
ORDER BY path_risk_score DESC, hops ASC
LIMIT 25
"""


def shortest_path(start_id: str, end_id: str) -> Optional[Dict[str, Any]]:
    """Single shortest hop-count path between two assets."""
    return q.get_shortest_attack_path(start_id, end_id)


def all_paths(start_id: str, end_id: str, max_hops: int = 6) -> List[Dict[str, Any]]:
    """All simple paths up to max_hops, ranked by cumulative vulnerability risk."""
    query = Q_WEIGHTED_PATHS % max_hops
    return get_connection().run_query(
        query, {"start_id": start_id, "end_id": end_id}
    )


def internet_to_critical_paths() -> List[Dict[str, Any]]:
    """All shortest paths from any internet-exposed asset to any critical asset —
    these represent the primary attack surface an adversary would use."""
    return q.get_paths_from_internet_to_critical()


def most_dangerous_path(start_id: str, end_id: str, max_hops: int = 6) -> Optional[Dict[str, Any]]:
    """The single riskiest path (highest cumulative CVSS along the route)."""
    paths = all_paths(start_id, end_id, max_hops)
    return paths[0] if paths else None


def summarize_attack_surface() -> Dict[str, Any]:
    """High-level attack-surface summary: exposed assets, critical assets,
    and the riskiest internet -> critical routes."""
    exposed = q.get_exposed_critical_assets()
    paths = internet_to_critical_paths()
    return {
        "exposed_critical_assets": exposed,
        "internet_to_critical_paths": paths,
        "num_attack_routes": len(paths),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(summarize_attack_surface(), indent=2, default=str))
