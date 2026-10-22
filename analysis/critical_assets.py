"""
analysis/critical_assets.py
------------------------------
Combines risk_score.py (vulnerability/exposure/business-criticality) with
centrality.py (structural importance in the network) to produce a single
ranked list of "crown jewel" / most-critical-to-defend assets.
"""

from typing import List, Dict, Any
from analysis.risk_score import compute_risk_scores
from analysis.centrality import pagerank, betweenness

W_RISK = 0.6
W_PAGERANK = 0.2
W_BETWEENNESS = 0.2


def _normalize(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def rank_critical_assets(top_n: int = 10) -> List[Dict[str, Any]]:
    risk_scores = compute_risk_scores()
    pr = {r["asset_id"]: r["score"] for r in pagerank()}
    bw = {r["asset_id"]: r["score"] for r in betweenness()}

    pr_norm = _normalize(pr)
    bw_norm = _normalize(bw)

    ranked = []
    for r in risk_scores:
        asset_id = r["asset_id"]
        risk_norm = r["risk_score"] / 100.0
        pagerank_score = pr_norm.get(asset_id, 0.0)
        betweenness_score = bw_norm.get(asset_id, 0.0)

        composite = (
            W_RISK * risk_norm
            + W_PAGERANK * pagerank_score
            + W_BETWEENNESS * betweenness_score
        ) * 100

        ranked.append(
            {
                "asset_id": asset_id,
                "criticality_label": r["criticality"],
                "asset_type": r["asset_type"],
                "critical_asset_score": round(composite, 2),
                "risk_score": r["risk_score"],
                "pagerank_score": round(pagerank_score, 4),
                "betweenness_score": round(betweenness_score, 4),
                "vuln_count": r["vuln_count"],
            }
        )

    ranked.sort(key=lambda r: r["critical_asset_score"], reverse=True)
    return ranked[:top_n]


if __name__ == "__main__":
    import json

    print(json.dumps(rank_critical_assets(), indent=2))
