"""
graph/nodes.py
---------------
MERGE-based helper functions to create/update each node type in bulk.
All functions accept a list of dicts (rows) matching the CSV schema and
use UNWIND for efficient batched writes.
"""

from typing import List, Dict, Any
from graph.connection import get_connection

MERGE_ASSET = """
UNWIND $rows AS row
MERGE (a:Asset {asset_id: row.asset_id})
SET a.hostname = row.hostname,
    a.ip_address = row.ip_address,
    a.asset_type = row.asset_type,
    a.os = row.os,
    a.criticality = row.criticality,
    a.department = row.department,
    a.exposed_to_internet = (row.exposed_to_internet = 'true')
"""

MERGE_VULNERABILITY = """
UNWIND $rows AS row
MERGE (v:Vulnerability {vuln_id: row.vuln_id})
SET v.cve_id = row.cve_id,
    v.description = row.description,
    v.cvss_score = toFloat(row.cvss_score),
    v.severity = row.severity,
    v.exploit_available = (row.exploit_available = 'true'),
    v.patch_available = (row.patch_available = 'true'),
    v.discovered_date = row.discovered_date
"""

MERGE_USER = """
UNWIND $rows AS row
MERGE (u:User {user_id: row.user_id})
SET u.username = row.username,
    u.full_name = row.full_name,
    u.department = row.department,
    u.role = row.role,
    u.is_admin = (row.is_admin = 'true')
"""

MERGE_SOFTWARE = """
UNWIND $rows AS row
MERGE (s:Software {software_id: row.software_id})
SET s.name = row.name,
    s.version = row.version,
    s.vendor = row.vendor
"""


def create_assets(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(MERGE_ASSET, rows)


def create_vulnerabilities(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(MERGE_VULNERABILITY, rows)


def create_users(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(MERGE_USER, rows)


def create_software(rows: List[Dict[str, Any]]) -> None:
    get_connection().run_batch_write(MERGE_SOFTWARE, rows)
