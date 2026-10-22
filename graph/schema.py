"""
graph/schema.py
----------------
Defines the graph data model and sets up Neo4j constraints/indexes.

Node labels:
    Asset(asset_id, hostname, ip_address, asset_type, os, criticality,
          department, exposed_to_internet)
    Vulnerability(vuln_id, cve_id, description, cvss_score, severity,
                   exploit_available, patch_available, discovered_date)
    User(user_id, username, full_name, department, role, is_admin)
    Software(software_id, name, version, vendor)

Relationships:
    (Asset)-[:HAS_VULNERABILITY]->(Vulnerability)
    (Asset)-[:RUNS]->(Software)
    (Software)-[:HAS_VULNERABILITY]->(Vulnerability)
    (User)-[:OWNS]->(Asset)
    (User)-[:HAS_ACCESS_TO]->(Asset)
    (Asset)-[:CONNECTS_TO {port, protocol, connection_type}]->(Asset)
    (Asset)-[:EXPOSED_TO]->(Internet)   # synthetic node for exposed assets
"""

from graph.connection import get_connection

CONSTRAINTS = [
    "CREATE CONSTRAINT asset_id_unique IF NOT EXISTS "
    "FOR (a:Asset) REQUIRE a.asset_id IS UNIQUE",
    "CREATE CONSTRAINT vuln_id_unique IF NOT EXISTS "
    "FOR (v:Vulnerability) REQUIRE v.vuln_id IS UNIQUE",
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS "
    "FOR (u:User) REQUIRE u.user_id IS UNIQUE",
    "CREATE CONSTRAINT software_id_unique IF NOT EXISTS "
    "FOR (s:Software) REQUIRE s.software_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX asset_type_idx IF NOT EXISTS FOR (a:Asset) ON (a.asset_type)",
    "CREATE INDEX asset_criticality_idx IF NOT EXISTS FOR (a:Asset) ON (a.criticality)",
    "CREATE INDEX vuln_cvss_idx IF NOT EXISTS FOR (v:Vulnerability) ON (v.cvss_score)",
    "CREATE INDEX vuln_cve_idx IF NOT EXISTS FOR (v:Vulnerability) ON (v.cve_id)",
]


def initialize_schema() -> None:
    """Create all constraints and indexes. Idempotent — safe to re-run."""
    conn = get_connection()
    for stmt in CONSTRAINTS + INDEXES:
        conn.run_write(stmt)
    print(f"Initialized {len(CONSTRAINTS)} constraints and {len(INDEXES)} indexes.")


def reset_graph(confirm: bool = False) -> None:
    """Delete all nodes/relationships. Requires explicit confirm=True."""
    if not confirm:
        raise ValueError("Pass confirm=True to actually wipe the graph.")
    conn = get_connection()
    conn.run_write("MATCH (n) DETACH DELETE n")
    print("Graph wiped.")


if __name__ == "__main__":
    initialize_schema()
