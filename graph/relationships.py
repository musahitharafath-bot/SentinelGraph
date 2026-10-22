"""
graph/relationships.py
-----------------------
MERGE-based helper functions to wire up relationships between nodes.
"""

from typing import List, Dict, Any
from graph.connection import get_connection

LINK_ASSET_VULNERABILITY = """
UNWIND $rows AS row
MATCH (a:Asset {asset_id: row.asset_id})
MATCH (v:Vulnerability {vuln_id: row.vuln_id})
MERGE (a)-[:HAS_VULNERABILITY]->(v)
"""

LINK_ASSET_SOFTWARE = """
UNWIND $rows AS row
MATCH (a:Asset {asset_id: row.asset_id})
MATCH (s:Software {software_id: row.software_id})
MERGE (a)-[:RUNS]->(s)
"""

LINK_USER_ASSET = """
UNWIND $rows AS row
MATCH (u:User {user_id: row.user_id})
MATCH (a:Asset {asset_id: row.asset_id})
MERGE (u)-[:OWNS]->(a)
"""

LINK_NETWORK = """
UNWIND $rows AS row
MATCH (src:Asset {asset_id: row.source_asset_id})
MATCH (tgt:Asset {asset_id: row.target_asset_id})
MERGE (src)-[r:CONNECTS_TO]->(tgt)
SET r.connection_type = row.connection_type,
    r.port = toInteger(row.port),
    r.protocol = row.protocol
"""

LINK_EXPOSED_TO_INTERNET = """
MATCH (a:Asset {exposed_to_internet: true})
MERGE (i:Internet {name: 'internet'})
MERGE (a)-[:EXPOSED_TO]->(i)
"""


def link_asset_vulnerabilities(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(LINK_ASSET_VULNERABILITY, rows)


def link_asset_software(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(LINK_ASSET_SOFTWARE, rows)


def link_user_assets(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(LINK_USER_ASSET, rows)


def link_network(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(LINK_NETWORK, rows)


def link_exposed_assets() -> None:
    get_connection().run_write(LINK_EXPOSED_TO_INTERNET)
