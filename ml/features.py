"""
ml/features.py
-----------------
Pulls graph-derived features per asset (vulnerability posture, exposure,
connectivity, centrality) into a pandas DataFrame, ready for training or
inference with the DNN in ml/train.py / ml/predict.py.

Label: `compromise_risk_label` — a synthetic-but-principled 0/1 label used
for supervised training, derived from the same heuristic in
analysis/risk_score.py (>= 60 composite risk => "likely target"). In a
real deployment this would instead be replaced with actual incident /
red-team data.
"""

from typing import List, Dict, Any
import pandas as pd

from graph import queries as q
from analysis.centrality import pagerank, betweenness
from analysis.risk_score import compute_risk_scores

CATEGORICAL_COLS = ["criticality", "asset_type"]
NUMERIC_COLS = [
    "vuln_count",
    "max_cvss",
    "avg_cvss",
    "exploitable_count",
    "unpatched_count",
    "out_degree",
    "in_degree",
    "admin_owner_count",
    "pagerank_score",
    "betweenness_score",
]
BOOLEAN_COLS = ["exposed_to_internet"]


def build_feature_frame() -> pd.DataFrame:
    raw_features = q.get_asset_features()
    if not raw_features:
        return pd.DataFrame()

    df = pd.DataFrame(raw_features)

    pr = {r["asset_id"]: r["score"] for r in pagerank()}
    bw = {r["asset_id"]: r["score"] for r in betweenness()}
    df["pagerank_score"] = df["asset_id"].map(pr).fillna(0.0)
    df["betweenness_score"] = df["asset_id"].map(bw).fillna(0.0)

    df["exposed_to_internet"] = df["exposed_to_internet"].fillna(False).astype(int)

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # one-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dummy_na=False)

    return df


def build_labels(df: pd.DataFrame, threshold: float = 60.0) -> pd.Series:
    """Synthetic label generation from the explainable risk-score heuristic.
    Replace with real historical compromise/incident data when available."""
    risk_scores = {r["asset_id"]: r["risk_score"] for r in compute_risk_scores()}
    labels = df["asset_id"].map(risk_scores).fillna(0.0)
    return (labels >= threshold).astype(int)


def get_training_data():
    """Returns (X: DataFrame of features, y: Series of 0/1 labels, feature_cols: list)."""
    df = build_feature_frame()
    if df.empty:
        raise RuntimeError("No asset features found — has the graph been ingested?")

    y = build_labels(df)
    feature_cols = [c for c in df.columns if c not in ("asset_id",)]
    X = df[feature_cols].astype(float)
    return X, y, feature_cols, df["asset_id"]


if __name__ == "__main__":
    X, y, cols, ids = get_training_data()
    print(X.head())
    print("Label distribution:\n", y.value_counts())
