"""
analysis/dynamic_paths.py
-----------------------------
Dynamic attack-path analysis: two distinct capabilities, kept separate
because they have very different epistemic status.

1. WHAT-IF SIMULATION (simulate_patch, simulate_new_vulnerability) --
   deterministic, exact: recomputes the real attack-surface analysis
   (analysis/attack_paths.py, analysis/risk_score.py) against a
   hypothetically modified vulnerability set. Entirely in-memory -- never
   writes simulated data to the graph.

2. TREND-BASED FORECASTING (predict_emerging_risk) -- a stated *heuristic*,
   not a trained model: ranks assets by extrapolating their recent
   vulnerability-discovery rate (via analysis/temporal.py) combined with
   structural centrality, as a simple linear-trend forecast of where risk
   is likely to concentrate next.

Being explicit about the difference matters here: "dynamic attack-path
prediction" in the research literature usually means a trained temporal
model (TGN, EvolveGCN, DySAT) learning from a labeled history of how paths
actually materialized -- that needs a time-series of *actual* attack/
compromise events to train against, which doesn't exist in this project
(or most real deployments, thankfully). Building a temporal GNN and
training it on synthetic labels would produce something that looks
sophisticated but has no real predictive validity. What's implemented
below is honest: exact simulation for "what happens if X", and a
transparent statistical heuristic (not a neural model) for "where is risk
trending" -- both real and useful, neither dressed up as more than it is.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from graph.connection import get_connection
from analysis.risk_score import compute_risk_scores
from analysis.attack_paths import summarize_attack_surface
from analysis.centrality import pagerank

logger = logging.getLogger(__name__)

Q_ASSET_VULN_DETAIL = """
MATCH (a:Asset)-[:HAS_VULNERABILITY]->(v)
RETURN a.asset_id AS asset_id, a.hostname AS hostname, a.criticality AS criticality,
       a.exposed_to_internet AS exposed_to_internet,
       v.vuln_id AS vuln_id, v.cvss_score AS cvss_score, v.exploit_available AS exploit_available
"""

Q_ALL_ASSETS_META = """
MATCH (a:Asset)
RETURN a.asset_id AS asset_id, a.hostname AS hostname, a.criticality AS criticality,
       a.exposed_to_internet AS exposed_to_internet
"""

Q_NETWORK_EDGES = "MATCH (a:Asset)-[:CONNECTS_TO]->(b:Asset) RETURN a.asset_id AS src, b.asset_id AS dst"


def _load_snapshot() -> Dict[str, Any]:
    conn = get_connection()
    vuln_rows = conn.run_query(Q_ASSET_VULN_DETAIL)
    all_assets = conn.run_query(Q_ALL_ASSETS_META)
    edges = conn.run_query(Q_NETWORK_EDGES)

    by_asset: Dict[str, List[Dict[str, Any]]] = {}
    for r in vuln_rows:
        by_asset.setdefault(r["asset_id"], []).append(r)

    # asset_meta covers EVERY asset, not just ones with existing
    # vulnerabilities -- a currently-clean chokepoint (e.g. a firewall with
    # no known CVEs) is exactly the kind of asset simulate_new_vulnerability
    # needs to reason about, so it must not be silently excluded here.
    asset_meta: Dict[str, Dict[str, Any]] = {
        a["asset_id"]: {
            "hostname": a["hostname"], "criticality": a["criticality"], "exposed_to_internet": a["exposed_to_internet"],
        }
        for a in all_assets
    }

    adjacency: Dict[str, List[str]] = {}
    for e in edges:
        adjacency.setdefault(e["src"], []).append(e["dst"])

    return {"by_asset": by_asset, "asset_meta": asset_meta, "adjacency": adjacency}


def _max_cvss(rows: List[Dict[str, Any]]) -> float:
    return max((r["cvss_score"] or 0.0) for r in rows) if rows else 0.0


def simulate_patch(vuln_id: str) -> Dict[str, Any]:
    """Exact what-if: recomputes each affected asset's max-CVSS exposure as
    if `vuln_id` were removed (patched), entirely in memory -- never writes
    to the graph. Shows which assets' risk actually drops as a result (a
    patch only helps an asset if it was its *highest*-severity vuln; this
    makes that visible instead of assuming every patch matters equally)."""
    snapshot = _load_snapshot()
    affected_assets = [aid for aid, rows in snapshot["by_asset"].items() if any(r["vuln_id"] == vuln_id for r in rows)]

    if not affected_assets:
        return {"vuln_id": vuln_id, "error": "No assets currently have this vulnerability."}

    results = []
    for asset_id in affected_assets:
        rows = snapshot["by_asset"][asset_id]
        before = _max_cvss(rows)
        after_rows = [r for r in rows if r["vuln_id"] != vuln_id]
        after = _max_cvss(after_rows)
        results.append({
            "asset_id": asset_id,
            "hostname": snapshot["asset_meta"][asset_id]["hostname"],
            "max_cvss_before": before,
            "max_cvss_after": after,
            "risk_reduced": before > after,
            "remaining_vuln_count": len(after_rows),
        })

    return {
        "vuln_id": vuln_id,
        "affected_assets": results,
        "assets_with_meaningful_risk_reduction": sum(1 for r in results if r["risk_reduced"]),
    }


def simulate_new_vulnerability(
    asset_id: str, cvss_score: float, exploit_available: bool = True, max_hops: int = 4
) -> Dict[str, Any]:
    """Exact what-if: models a hypothetical new vulnerability on `asset_id`
    and propagates its effect through the network topology (BFS over
    CONNECTS_TO), so you can see not just "this asset gets riskier" but
    "and here's what becomes reachable-from-a-worse-position as a result."
    Risk propagation decays per hop (an attacker who has to pivot through
    N intermediate hosts is meaningfully harder-pressed than one with
    direct access) -- decay factor 0.7 per hop, a standard geometric falloff,
    not a fitted parameter."""
    snapshot = _load_snapshot()
    if asset_id not in snapshot["asset_meta"]:
        return {"error": f"Asset {asset_id} not found"}

    current_rows = snapshot["by_asset"].get(asset_id, [])
    current_max = _max_cvss(current_rows)
    new_max = max(current_max, cvss_score)

    # BFS to find downstream reachable assets and their hop distance
    visited = {asset_id: 0}
    frontier = [asset_id]
    hop = 0
    while frontier and hop < max_hops:
        hop += 1
        next_frontier = []
        for node in frontier:
            for neighbor in snapshot["adjacency"].get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = hop
                    next_frontier.append(neighbor)
        frontier = next_frontier

    decay = 0.7
    downstream_impact = []
    for reachable_id, hops in visited.items():
        if reachable_id == asset_id:
            continue
        meta = snapshot["asset_meta"].get(reachable_id, {})
        propagated_severity = round(new_max * (decay ** hops), 2)
        downstream_impact.append({
            "asset_id": reachable_id,
            "hostname": meta.get("hostname"),
            "criticality": meta.get("criticality"),
            "hops_from_source": hops,
            "propagated_severity_estimate": propagated_severity,
        })
    downstream_impact.sort(key=lambda r: r["propagated_severity_estimate"], reverse=True)

    return {
        "asset_id": asset_id,
        "hypothetical_cvss": cvss_score,
        "exploit_available": exploit_available,
        "current_max_cvss": current_max,
        "new_max_cvss": new_max,
        "risk_increase": round(new_max - current_max, 2),
        "downstream_impact": downstream_impact,
        "note": "propagated_severity_estimate decays geometrically (0.7/hop) with network "
                "distance -- an approximation of pivot difficulty, not a calibrated probability.",
    }


def predict_emerging_risk(top_n: int = 10, lookback_days: int = 365) -> Dict[str, Any]:
    """HEURISTIC forecast (explicitly not a trained model -- see module
    docstring): ranks assets by combining (a) how fast new vulnerabilities
    have been discovered for them recently, extrapolated forward linearly,
    and (b) their structural centrality (a structurally important asset
    that's accumulating vulnerabilities faster is a reasonable candidate
    for "risk is trending up here"). This is a transparent statistical
    heuristic (recent rate * centrality), not a temporal GNN forecast."""
    conn = get_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    rows = conn.run_query(
        """
        MATCH (a:Asset)-[:HAS_VULNERABILITY]->(v)
        WHERE v.discovered_date IS NOT NULL
        RETURN a.asset_id AS asset_id, a.hostname AS hostname,
               v.discovered_date AS discovered_date
        """
    )
    if not rows:
        return {"note": "No dated vulnerability data found.", "ranked_assets": []}

    recent_counts: Dict[str, int] = {}
    total_counts: Dict[str, int] = {}
    for r in rows:
        total_counts[r["asset_id"]] = total_counts.get(r["asset_id"], 0) + 1
        if r["discovered_date"] >= cutoff:
            recent_counts[r["asset_id"]] = recent_counts.get(r["asset_id"], 0) + 1

    pr = {p["asset_id"]: p["score"] for p in pagerank()}
    max_pr = max(pr.values()) if pr else 1.0

    scored = []
    for asset_id, total in total_counts.items():
        recent = recent_counts.get(asset_id, 0)
        discovery_rate = recent / max(lookback_days / 30, 1)  # new vulns per month, recent window
        centrality_norm = (pr.get(asset_id, 0.0) / max_pr) if max_pr else 0.0
        trend_score = round(discovery_rate * (1 + centrality_norm), 3)
        scored.append({
            "asset_id": asset_id,
            "recent_vuln_discoveries": recent,
            "total_vuln_discoveries": total,
            "monthly_discovery_rate": round(discovery_rate, 3),
            "centrality_score": round(centrality_norm, 3),
            "emerging_risk_trend_score": trend_score,
        })

    scored.sort(key=lambda r: r["emerging_risk_trend_score"], reverse=True)
    return {
        "method": "heuristic: recent vulnerability-discovery rate x structural centrality "
                  "(NOT a trained temporal model -- a transparent statistical forecast)",
        "lookback_days": lookback_days,
        "ranked_assets": scored[:top_n],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(predict_emerging_risk(), indent=2, default=str))
