"""
graph/queries.py
-----------------
Reusable, parameterized Cypher queries used across the analysis, ML and
API layers. Keeping them in one place makes the Cypher easy to audit.
"""

from typing import List, Dict, Any, Optional
from graph.connection import get_connection

# ---------------------------------------------------------------- lookups

Q_ALL_ASSETS = """
MATCH (a:Asset)
RETURN a.asset_id AS asset_id, a.hostname AS hostname, a.asset_type AS asset_type,
       a.criticality AS criticality, a.department AS department,
       a.exposed_to_internet AS exposed_to_internet
ORDER BY a.asset_id
"""

Q_ASSET_DETAIL = """
MATCH (a:Asset {asset_id: $asset_id})
OPTIONAL MATCH (a)-[:HAS_VULNERABILITY]->(v:Vulnerability)
OPTIONAL MATCH (a)-[:RUNS]->(s:Software)
OPTIONAL MATCH (u:User)-[:OWNS]->(a)
RETURN a AS asset,
       collect(DISTINCT v) AS vulnerabilities,
       collect(DISTINCT s) AS software,
       collect(DISTINCT u) AS owners
"""

Q_VULNERABLE_ASSETS = """
MATCH (a:Asset)-[:HAS_VULNERABILITY]->(v:Vulnerability)
WHERE v.cvss_score >= $min_cvss
RETURN a.asset_id AS asset_id, a.hostname AS hostname, a.criticality AS criticality,
       v.cve_id AS cve_id, v.cvss_score AS cvss_score, v.severity AS severity,
       v.exploit_available AS exploit_available
ORDER BY v.cvss_score DESC
"""

Q_EXPOSED_CRITICAL_ASSETS = """
MATCH (a:Asset)
WHERE a.exposed_to_internet = true AND a.criticality IN ['critical', 'high']
OPTIONAL MATCH (a)-[:HAS_VULNERABILITY]->(v:Vulnerability)
RETURN a.asset_id AS asset_id, a.hostname AS hostname, a.criticality AS criticality,
       count(v) AS vuln_count, max(v.cvss_score) AS max_cvss
ORDER BY max_cvss DESC
"""

# ------------------------------------------------------------ attack paths

Q_SHORTEST_ATTACK_PATH = """
MATCH (start:Asset {asset_id: $start_id}), (end:Asset {asset_id: $end_id})
WHERE start <> end
MATCH p = shortestPath((start)-[:CONNECTS_TO*..10]->(end))
RETURN [n IN nodes(p) | n.asset_id] AS path_assets,
       [n IN nodes(p) | n.hostname] AS path_hostnames,
       length(p) AS hops
"""

Q_ALL_ATTACK_PATHS = """
MATCH (start:Asset {asset_id: $start_id}), (end:Asset {asset_id: $end_id})
MATCH p = (start)-[:CONNECTS_TO*..$max_hops]->(end)
RETURN [n IN nodes(p) | n.asset_id] AS path_assets,
       [n IN nodes(p) | n.hostname] AS path_hostnames,
       length(p) AS hops
ORDER BY hops ASC
LIMIT 25
"""

Q_PATHS_FROM_INTERNET_TO_CRITICAL = """
MATCH (start:Asset {exposed_to_internet: true})
MATCH (end:Asset {criticality: 'critical'})
WHERE start.asset_id <> end.asset_id
MATCH p = shortestPath((start)-[:CONNECTS_TO*..8]->(end))
RETURN start.asset_id AS from_asset, start.hostname AS from_hostname,
       end.asset_id AS to_asset, end.hostname AS to_hostname,
       [n IN nodes(p) | n.hostname] AS path, length(p) AS hops
ORDER BY hops ASC
"""

# --------------------------------------------------------------- centrality

Q_DEGREE_RAW = """
MATCH (a:Asset)
OPTIONAL MATCH (a)-[r:CONNECTS_TO]-()
RETURN a.asset_id AS asset_id, a.hostname AS hostname, count(r) AS degree
ORDER BY degree DESC
"""

# ------------------------------------------------------------ ML features

Q_ASSET_FEATURES = """
MATCH (a:Asset)
OPTIONAL MATCH (a)-[:HAS_VULNERABILITY]->(v:Vulnerability)
OPTIONAL MATCH (a)-[out:CONNECTS_TO]->()
OPTIONAL MATCH (a)<-[inn:CONNECTS_TO]-()
OPTIONAL MATCH (u:User)-[:OWNS]->(a)
WITH a,
     count(DISTINCT v) AS vuln_count,
     coalesce(max(v.cvss_score), 0.0) AS max_cvss,
     coalesce(avg(v.cvss_score), 0.0) AS avg_cvss,
     sum(CASE WHEN v.exploit_available THEN 1 ELSE 0 END) AS exploitable_count,
     sum(CASE WHEN v.patch_available = false THEN 1 ELSE 0 END) AS unpatched_count,
     count(DISTINCT out) AS out_degree,
     count(DISTINCT inn) AS in_degree,
     sum(CASE WHEN u.is_admin THEN 1 ELSE 0 END) AS admin_owner_count
RETURN a.asset_id AS asset_id,
       a.criticality AS criticality,
       a.asset_type AS asset_type,
       a.exposed_to_internet AS exposed_to_internet,
       vuln_count, max_cvss, avg_cvss, exploitable_count, unpatched_count,
       out_degree, in_degree, admin_owner_count
ORDER BY a.asset_id
"""

# ---------------------------------------------------------------- utility


def run(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return get_connection().run_query(query, params or {})


def get_all_assets() -> List[Dict[str, Any]]:
    return run(Q_ALL_ASSETS)


def get_asset_detail(asset_id: str) -> Dict[str, Any]:
    rows = run(Q_ASSET_DETAIL, {"asset_id": asset_id})
    return rows[0] if rows else {}


def get_vulnerable_assets(min_cvss: float = 7.0) -> List[Dict[str, Any]]:
    return run(Q_VULNERABLE_ASSETS, {"min_cvss": min_cvss})


def get_exposed_critical_assets() -> List[Dict[str, Any]]:
    return run(Q_EXPOSED_CRITICAL_ASSETS)


def get_shortest_attack_path(start_id: str, end_id: str) -> Optional[Dict[str, Any]]:
    if start_id == end_id:
        detail = get_asset_detail(start_id)
        asset = detail.get("asset") if detail else None
        if asset:
            hostname = asset.get("hostname", start_id)
            return {
                "path_assets": [start_id],
                "path_hostnames": [hostname],
                "hops": 0,
            }
        return None
    rows = run(Q_SHORTEST_ATTACK_PATH, {"start_id": start_id, "end_id": end_id})
    return rows[0] if rows else None


def get_paths_from_internet_to_critical() -> List[Dict[str, Any]]:
    return run(Q_PATHS_FROM_INTERNET_TO_CRITICAL)


def get_asset_features() -> List[Dict[str, Any]]:
    return run(Q_ASSET_FEATURES)


# ------------------------------------------------------ NVD CVE catalog

Q_CVE_CATALOG_STATS = """
MATCH (v:CVECatalog)
RETURN count(v) AS total_cves,
       avg(v.cvss_score) AS avg_cvss,
       max(v.cvss_score) AS max_cvss,
       sum(CASE WHEN v.severity = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_count,
       sum(CASE WHEN v.severity = 'HIGH' THEN 1 ELSE 0 END) AS high_count
"""

Q_CVE_CATALOG_SEARCH = """
MATCH (v:CVECatalog)
WHERE v.cve_id = $cve_id
RETURN v
"""

Q_TOP_CWE_CATEGORIES = """
MATCH (v:CVECatalog)
WHERE v.cwe_id IS NOT NULL
RETURN v.cwe_id AS cwe_id, count(*) AS cve_count, avg(v.cvss_score) AS avg_cvss
ORDER BY cve_count DESC
LIMIT 20
"""


def get_cve_catalog_stats() -> Dict[str, Any]:
    rows = run(Q_CVE_CATALOG_STATS)
    return rows[0] if rows else {}


def get_cve_by_id(cve_id: str) -> Dict[str, Any]:
    rows = run(Q_CVE_CATALOG_SEARCH, {"cve_id": cve_id})
    return rows[0]["v"] if rows else {}


def get_top_cwe_categories() -> List[Dict[str, Any]]:
    return run(Q_TOP_CWE_CATEGORIES)


Q_SEMANTIC_SEARCH = """
CALL db.index.vector.queryNodes('cve_catalog_embedding_index', $top_k, $embedding)
YIELD node, score
RETURN node.cve_id AS cve_id, node.description AS description,
       node.cvss_score AS cvss_score, node.severity AS severity, score
ORDER BY score DESC
"""


def semantic_search_cves(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Vector similarity search over CVE descriptions. `embedding` must be a
    384-dim vector from ml.embeddings.embed_query() (all-MiniLM-L6-v2), and
    the catalog must have been embedded first (see
    ingestion.nvd_loader.embed_catalog / --embed flag)."""
    return run(Q_SEMANTIC_SEARCH, {"embedding": embedding, "top_k": top_k})
