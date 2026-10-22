"""
analysis/risk_score.py
-------------------------
Computes a composite risk score (0-100) per asset combining:
  - vulnerability severity (CVSS, exploit availability, patch status)
  - exposure (internet-facing or not)
  - business criticality (declared asset criticality)
  - blast radius (how many other assets it can reach — filled in by
    centrality.py, optional)

This is a transparent, explainable scoring model (not the DNN — see
ml/train.py for the learned risk-prediction model). It's useful as a
baseline / sanity check and as a label-generation heuristic for training
the neural network.
"""

from typing import Dict, Any, List
from graph import queries as q

CRITICALITY_WEIGHTS = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}

# Scoring weights (sum to 1.0)
W_VULN = 0.45
W_EXPOSURE = 0.20
W_CRITICALITY = 0.20
W_CONNECTIVITY = 0.15


def _vuln_component(features: Dict[str, Any]) -> float:
    """0-1 score from vulnerability posture on the asset."""
    max_cvss = float(features.get("max_cvss") or 0.0)  # 0-10
    exploitable = int(features.get("exploitable_count") or 0)
    unpatched = int(features.get("unpatched_count") or 0)

    base = max_cvss / 10.0
    exploit_bonus = min(exploitable * 0.1, 0.3)
    unpatched_bonus = min(unpatched * 0.05, 0.2)
    return min(base + exploit_bonus + unpatched_bonus, 1.0)


def _exposure_component(features: Dict[str, Any]) -> float:
    return 1.0 if features.get("exposed_to_internet") else 0.0


def _criticality_component(features: Dict[str, Any]) -> float:
    return CRITICALITY_WEIGHTS.get(features.get("criticality"), 0.25)


def _connectivity_component(features: Dict[str, Any], max_degree: int) -> float:
    """Normalized out-degree — proxy for lateral-movement blast radius."""
    if max_degree <= 0:
        return 0.0
    out_degree = int(features.get("out_degree") or 0)
    return min(out_degree / max_degree, 1.0)


def compute_risk_scores() -> List[Dict[str, Any]]:
    features = q.get_asset_features()
    if not features:
        return []

    max_degree = max((f.get("out_degree") or 0) for f in features) or 1

    results = []
    for f in features:
        vuln = _vuln_component(f)
        exposure = _exposure_component(f)
        criticality = _criticality_component(f)
        connectivity = _connectivity_component(f, max_degree)

        composite = (
            W_VULN * vuln
            + W_EXPOSURE * exposure
            + W_CRITICALITY * criticality
            + W_CONNECTIVITY * connectivity
        ) * 100

        results.append(
            {
                "asset_id": f["asset_id"],
                "criticality": f["criticality"],
                "asset_type": f["asset_type"],
                "risk_score": round(composite, 2),
                "components": {
                    "vulnerability": round(vuln * 100, 2),
                    "exposure": round(exposure * 100, 2),
                    "criticality": round(criticality * 100, 2),
                    "connectivity": round(connectivity * 100, 2),
                },
                "vuln_count": f["vuln_count"],
                "max_cvss": f["max_cvss"],
            }
        )

    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return results


def risk_score_for_asset(asset_id: str) -> Dict[str, Any]:
    scores = compute_risk_scores()
    for s in scores:
        if s["asset_id"] == asset_id:
            return s
    return {}


if __name__ == "__main__":
    import json

    print(json.dumps(compute_risk_scores(), indent=2))
