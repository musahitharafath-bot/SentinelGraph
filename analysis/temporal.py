"""
analysis/temporal.py
------------------------
Temporal attack graph support: reconstructs what the vulnerability
landscape (and therefore risk) looked like at a past point in time, and
computes risk trends over time.

Scope, stated plainly: this is temporal *analysis over a versioned
property graph* -- point-in-time reconstruction from timestamped
properties (discovered_date, patched_date) plus an explicit change-event
log for edits timestamps alone can't capture (patches, topology changes).
It is NOT a bitemporal graph database and NOT a trained temporal model
(TGN/EvolveGCN-style temporal GNNs are a materially larger undertaking --
see analysis/dynamic_paths.py for the heuristic forecasting piece, which
is honestly labeled as a heuristic rather than a trained temporal model).
This module covers the real, common use case: "what did our risk look
like on date X" and "how has it trended since."

Two ingredients:
  1. Timestamp properties already on Vulnerability/CVECatalog nodes
     (discovered_date) -- this module adds `patched_date` support so
     "active at time T" can be reconstructed: discovered <= T AND
     (not patched OR patched_date > T).
  2. An explicit :ChangeEvent log for anything a bare timestamp can't
     capture on its own (e.g. *why* something changed), queryable by
     entity and time range.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd

from graph.connection import get_connection
from analysis.risk_score import CRITICALITY_WEIGHTS, W_VULN, W_EXPOSURE, W_CRITICALITY

logger = logging.getLogger(__name__)


def init_temporal_schema() -> None:
    get_connection().run_write(
        "CREATE CONSTRAINT change_event_id_unique IF NOT EXISTS "
        "FOR (e:ChangeEvent) REQUIRE e.event_id IS UNIQUE"
    )


def record_event(
    event_type: str,
    asset_id: Optional[str] = None,
    vuln_id: Optional[str] = None,
    details: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Logs a discrete change (vulnerability discovered/patched, asset
    added, connection changed, ...) as a :ChangeEvent node linked to the
    entities it concerns. Returns the generated event_id."""
    import uuid

    event_id = str(uuid.uuid4())
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    conn.run_write(
        """
        CREATE (e:ChangeEvent {
            event_id: $event_id, event_type: $event_type,
            timestamp: $timestamp, details: $details
        })
        """,
        {"event_id": event_id, "event_type": event_type, "timestamp": ts, "details": details},
    )
    if asset_id:
        conn.run_write(
            "MATCH (e:ChangeEvent {event_id: $event_id}), (a:Asset {asset_id: $asset_id}) "
            "MERGE (e)-[:CONCERNS]->(a)",
            {"event_id": event_id, "asset_id": asset_id},
        )
    if vuln_id:
        conn.run_write(
            "MATCH (e:ChangeEvent {event_id: $event_id}), (v:Vulnerability {vuln_id: $vuln_id}) "
            "MERGE (e)-[:CONCERNS]->(v)",
            {"event_id": event_id, "vuln_id": vuln_id},
        )
    return event_id


def mark_vulnerability_patched(vuln_id: str, patched_date: Optional[str] = None, label: str = "Vulnerability") -> None:
    """Sets patched_date on a vulnerability node and logs the event, so
    later point-in-time queries correctly treat it as inactive after this
    date. `label` lets this work for either the demo :Vulnerability nodes
    or the bulk :CVECatalog nodes (which share the base :Vulnerability label)."""
    ts = patched_date or datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    conn.run_write(
        f"MATCH (v:{label} {{vuln_id: $vuln_id}}) SET v.patched_date = $ts",
        {"vuln_id": vuln_id, "ts": ts},
    )
    record_event("vulnerability_patched", vuln_id=vuln_id, timestamp=ts, details=f"{vuln_id} marked patched")


def get_event_log(
    asset_id: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    conditions = []
    params: Dict[str, Any] = {"limit": limit}
    if asset_id:
        conditions.append("a.asset_id = $asset_id")
        params["asset_id"] = asset_id
    if start:
        conditions.append("e.timestamp >= $start")
        params["start"] = start
    if end:
        conditions.append("e.timestamp <= $end")
        params["end"] = end

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
    MATCH (e:ChangeEvent)-[:CONCERNS]->(target)
    OPTIONAL MATCH (a:Asset {{asset_id: target.asset_id}})
    {where_clause}
    RETURN e.event_id AS event_id, e.event_type AS event_type,
           e.timestamp AS timestamp, e.details AS details,
           labels(target) AS target_labels
    ORDER BY e.timestamp DESC
    LIMIT $limit
    """
    return get_connection().run_query(query, params)


Q_VULNS_ACTIVE_AT = """
MATCH (a:Asset)-[:HAS_VULNERABILITY]->(v)
WHERE v.discovered_date IS NOT NULL AND v.discovered_date <= $timestamp
  AND (v.patched_date IS NULL OR v.patched_date > $timestamp)
RETURN a.asset_id AS asset_id, a.criticality AS criticality,
       a.exposed_to_internet AS exposed_to_internet,
       v.cve_id AS cve_id, v.cvss_score AS cvss_score
"""


def vulnerabilities_active_at(timestamp: str) -> List[Dict[str, Any]]:
    """Reconstructs which asset-vulnerability links were active as of a
    given ISO timestamp, based on discovered_date/patched_date."""
    return get_connection().run_query(Q_VULNS_ACTIVE_AT, {"timestamp": timestamp})


def _risk_score_from_active_vulns(active_rows_for_asset: List[Dict[str, Any]]) -> float:
    """Recomputes the same composite formula as analysis/risk_score.py, but
    restricted to whichever vulnerabilities were active at a given point in
    time, instead of all currently-known ones."""
    if not active_rows_for_asset:
        return 0.0

    max_cvss = max((r["cvss_score"] or 0.0) for r in active_rows_for_asset)
    criticality = active_rows_for_asset[0]["criticality"]
    exposed = active_rows_for_asset[0]["exposed_to_internet"]

    vuln_component = max_cvss / 10.0
    exposure_component = 1.0 if exposed else 0.0
    criticality_component = CRITICALITY_WEIGHTS.get(criticality, 0.25)
    # connectivity omitted here -- it's structural, not time-varying, in this schema

    composite = (
        W_VULN * vuln_component
        + W_EXPOSURE * exposure_component
        + W_CRITICALITY * criticality_component
    ) / (W_VULN + W_EXPOSURE + W_CRITICALITY) * 100
    return round(composite, 2)


def asset_risk_at(asset_id: str, timestamp: str) -> Dict[str, Any]:
    """Point-in-time risk score for a single asset."""
    active = [r for r in vulnerabilities_active_at(timestamp) if r["asset_id"] == asset_id]
    return {
        "asset_id": asset_id,
        "timestamp": timestamp,
        "risk_score": _risk_score_from_active_vulns(active),
        "active_vuln_count": len(active),
    }


def risk_trend_over_time(
    asset_id: str, start_date: str, end_date: str, interval: str = "ME"
) -> List[Dict[str, Any]]:
    """Samples risk score at regular intervals between two dates.
    `interval` follows pandas offset aliases: 'D' daily, 'W' weekly,
    'ME' month-end (default), 'YE' year-end."""
    dates = pd.date_range(start=start_date, end=end_date, freq=interval)
    if len(dates) == 0 or dates[-1] < pd.Timestamp(end_date):
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))

    trend = []
    for d in dates:
        ts = d.strftime("%Y-%m-%d")
        point = asset_risk_at(asset_id, ts)
        trend.append(point)
    return trend


def fleet_risk_trend(start_date: str, end_date: str, interval: str = "ME") -> List[Dict[str, Any]]:
    """Same as risk_trend_over_time but aggregated (mean) across all assets
    that have at least one dated vulnerability -- gives an overall
    "is our risk posture improving or worsening" signal."""
    all_active_ever = vulnerabilities_active_at(end_date)  # upper bound of relevant assets
    asset_ids = sorted({r["asset_id"] for r in all_active_ever})
    if not asset_ids:
        return []

    dates = pd.date_range(start=start_date, end=end_date, freq=interval)
    trend = []
    for d in dates:
        ts = d.strftime("%Y-%m-%d")
        active_at_ts = vulnerabilities_active_at(ts)
        by_asset: Dict[str, List[Dict[str, Any]]] = {}
        for r in active_at_ts:
            by_asset.setdefault(r["asset_id"], []).append(r)

        scores = [_risk_score_from_active_vulns(rows) for rows in by_asset.values()]
        trend.append({
            "timestamp": ts,
            "mean_risk_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "assets_with_active_vulns": len(by_asset),
        })
    return trend


if __name__ == "__main__":
    import json

    init_temporal_schema()
    print(json.dumps(fleet_risk_trend("2018-01-01", "2024-01-01", interval="YE"), indent=2))
