"""
ml/cve_severity_model.py
----------------------------
A second, independent DNN: trained directly on the NVD CVE catalog
(ingestion/nvd_loader.py) rather than on graph-structural features. This
is the task the Kaggle dataset was actually built for -- predicting the
continuous CVSS base score from a CVE's characteristics (attack vector,
complexity, privileges required, impact metrics, CWE category, etc).

This is deliberately kept separate from ml/train.py (RiskDNN), which
predicts per-*asset* compromise risk from *graph* features. This model
predicts per-*CVE* severity from *NVD* features and is useful for:
  - Estimating a CVSS score for a newly disclosed CVE before NVD has
    scored it (a real, common industry need -- NVD scoring often lags
    disclosure by days to weeks).
  - Sanity-checking/imputing missing CVSS scores in the catalog.

Architecture: feed-forward regression DNN (128 -> 64 -> 16 -> 1, ReLU,
batch norm, dropout), trained with MSE loss + Adam, evaluated with
RMSE / MAE / R^2. Categorical CVSS metrics (attack vector, complexity,
privileges required, user interaction, scope, impact metrics) are
one-hot encoded; CWE ID is frequency-encoded (too high-cardinality to
one-hot at 337k rows).

NLP features (optional, on by default when available): if the catalog has
been embedded (see ingestion/nvd_loader.py::embed_catalog / --embed flag),
sentence-transformer embeddings of each CVE's description are PCA-reduced
and concatenated onto the structured features. `train_severity_model_comparison()`
trains the model both with and without these NLP features and reports the
delta, so the improvement is measured rather than assumed.

Usage:
    python -m ml.cve_severity_model                # train (NLP features if available)
    python -m ml.cve_severity_model --compare       # train both variants, print comparison
    python -m ml.cve_severity_model --no-nlp        # structured features only
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from graph.connection import get_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = [
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "scope",
    "confidentiality_impact",
    "integrity_impact",
    "availability_impact",
]

PCA_COMPONENTS = 50  # NLP embedding dims kept after PCA reduction

Q_CVE_CATALOG_FEATURES = """
MATCH (v:CVECatalog)
WHERE v.cvss_score IS NOT NULL
RETURN v.cve_id AS cve_id, v.cvss_score AS cvss_score,
       v.attack_vector AS attack_vector, v.attack_complexity AS attack_complexity,
       v.privileges_required AS privileges_required, v.user_interaction AS user_interaction,
       v.scope AS scope, v.confidentiality_impact AS confidentiality_impact,
       v.integrity_impact AS integrity_impact, v.availability_impact AS availability_impact,
       v.cwe_id AS cwe_id, v.exploitability_score AS exploitability_score,
       v.impact_score AS impact_score, v.embedding AS embedding
"""


class SeverityDNN(nn.Module):
    """Feed-forward regression DNN predicting CVSS base score (0-10)."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = (128, 64, 16), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_cve_dataframe(limit: int = None) -> pd.DataFrame:
    """Pulls scored CVE records (plus embeddings, if present) from the Neo4j
    catalog. For very large catalogs (300k+), consider `limit` for faster
    iteration during development."""
    query = Q_CVE_CATALOG_FEATURES
    if limit:
        query += f"\nLIMIT {int(limit)}"
    rows = get_connection().run_query(query)
    return pd.DataFrame(rows)


def _engineer_structured_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()

    if "cwe_id" in df.columns:
        freq = df["cwe_id"].value_counts(normalize=True)
        df["cwe_frequency"] = df["cwe_id"].map(freq).fillna(0.0)
    else:
        df["cwe_frequency"] = 0.0

    for col in ["exploitability_score", "impact_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df[cat_cols_present] = df[cat_cols_present].fillna("unknown")
    df = pd.get_dummies(df, columns=cat_cols_present, dummy_na=False)

    y = pd.to_numeric(df["cvss_score"], errors="coerce")
    feature_cols = [
        c for c in df.columns
        if c not in ("cve_id", "cvss_score", "cwe_id", "embedding") and df[c].dtype != object
    ]
    X = df[feature_cols].astype(float)

    mask = y.notnull()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), feature_cols


def _add_nlp_features(
    X: pd.DataFrame, df: pd.DataFrame, mask: pd.Series, pca: Optional[PCA] = None
) -> Tuple[pd.DataFrame, List[str], PCA]:
    """PCA-reduces sentence-transformer embeddings and concatenates them onto
    the structured feature matrix. Raises if the catalog wasn't embedded --
    caller should fall back to structured-only in that case."""
    embeddings = df.loc[mask, "embedding"]
    if embeddings.isnull().any():
        raise ValueError(
            "Some CVE catalog rows are missing embeddings. Run "
            "`python -m ingestion.nvd_loader --embed` (or POST /cve-catalog/embed) first."
        )
    embedding_matrix = np.vstack(embeddings.values)

    if pca is None:
        n_components = min(PCA_COMPONENTS, embedding_matrix.shape[0], embedding_matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embedding_matrix)
    else:
        reduced = pca.transform(embedding_matrix)

    embedding_cols = [f"emb_{i}" for i in range(reduced.shape[1])]
    embedding_df = pd.DataFrame(reduced, columns=embedding_cols, index=X.index)
    X_combined = pd.concat([X.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
    return X_combined, embedding_cols, pca


def _train_core(X: pd.DataFrame, y: pd.Series, epochs: int, lr: float, batch_size: int, val_split: float, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = SeverityDNN(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss, best_state = float("inf"), None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("Epoch %3d | train_mse=%.4f val_mse=%.4f", epoch, epoch_loss, val_loss)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_preds = model(X_val_t).numpy()

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_val, final_preds))),
        "mae": float(mean_absolute_error(y_val, final_preds)),
        "r2": float(r2_score(y_val, final_preds)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    return model, scaler, metrics, history


def train_severity_model(
    limit: int = None,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_split: float = 0.2,
    seed: int = 42,
    use_nlp: bool = True,
) -> dict:
    """Trains and persists the severity model. If use_nlp=True and the
    catalog has been embedded, NLP features are included; otherwise falls
    back to structured-only automatically (with a warning) rather than
    failing the whole training run."""
    df = load_cve_dataframe(limit=limit)
    if df.empty:
        raise RuntimeError(
            "No scored CVE records found. Run "
            "`python -m ingestion.nvd_loader --path data/nvd_cves.csv` first."
        )

    X_structured, y, structured_cols = _engineer_structured_features(df)
    mask = pd.to_numeric(df["cvss_score"], errors="coerce").notnull()

    pca = None
    feature_cols = structured_cols
    X = X_structured
    used_nlp = False

    if use_nlp:
        try:
            X, embedding_cols, pca = _add_nlp_features(X_structured, df, mask)
            feature_cols = structured_cols + embedding_cols
            used_nlp = True
        except ValueError as e:
            logger.warning("%s Falling back to structured-only features.", e)

    logger.info("Training on %d CVE records, %d features (nlp=%s)", X.shape[0], X.shape[1], used_nlp)
    model, scaler, metrics, history = _train_core(X, y, epochs, lr, batch_size, val_split, seed)
    metrics["used_nlp_features"] = used_nlp

    torch.save(model.state_dict(), MODEL_DIR / "cve_severity_dnn.pt")
    joblib.dump(scaler, MODEL_DIR / "cve_severity_scaler.pkl")
    if pca is not None:
        joblib.dump(pca, MODEL_DIR / "cve_severity_pca.pkl")
    with open(MODEL_DIR / "cve_severity_feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "input_dim": X.shape[1], "used_nlp_features": used_nlp}, f, indent=2)
    with open(MODEL_DIR / "cve_severity_training_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "history": history}, f, indent=2)

    logger.info("CVE severity model training complete. Metrics: %s", metrics)
    return {"metrics": metrics, "feature_cols": feature_cols}


def train_severity_model_comparison(limit: int = None, epochs: int = 60, seed: int = 42) -> dict:
    """Trains structured-only and structured+NLP variants back to back and
    returns both metric sets plus the RMSE improvement -- mirrors the
    comparison run in the exploration notebook, so the NLP contribution is
    measured on your actual data rather than assumed. Persists whichever
    model performs better as the active model."""
    df = load_cve_dataframe(limit=limit)
    if df.empty:
        raise RuntimeError("No scored CVE records found -- run ingestion first.")

    X_structured, y, structured_cols = _engineer_structured_features(df)
    mask = pd.to_numeric(df["cvss_score"], errors="coerce").notnull()

    logger.info("--- Training Model A (structured-only) ---")
    model_a, scaler_a, metrics_a, _ = _train_core(X_structured, y, epochs, 1e-3, 256, 0.2, seed)

    try:
        X_combined, embedding_cols, pca = _add_nlp_features(X_structured, df, mask)
    except ValueError as e:
        return {
            "error": str(e),
            "structured_only_metrics": metrics_a,
        }

    logger.info("--- Training Model B (structured + NLP embeddings) ---")
    model_b, scaler_b, metrics_b, _ = _train_core(X_combined, y, epochs, 1e-3, 256, 0.2, seed)

    rmse_improvement_pct = round(100 * (metrics_a["rmse"] - metrics_b["rmse"]) / metrics_a["rmse"], 2)

    if metrics_b["r2"] >= metrics_a["r2"]:
        best_model, best_scaler, best_pca = model_b, scaler_b, pca
        best_feature_cols, best_metrics, used_nlp = structured_cols + embedding_cols, metrics_b, True
    else:
        best_model, best_scaler, best_pca = model_a, scaler_a, None
        best_feature_cols, best_metrics, used_nlp = structured_cols, metrics_a, False

    torch.save(best_model.state_dict(), MODEL_DIR / "cve_severity_dnn.pt")
    joblib.dump(best_scaler, MODEL_DIR / "cve_severity_scaler.pkl")
    if best_pca is not None:
        joblib.dump(best_pca, MODEL_DIR / "cve_severity_pca.pkl")
    with open(MODEL_DIR / "cve_severity_feature_cols.json", "w") as f:
        json.dump(
            {"feature_cols": best_feature_cols, "input_dim": len(best_feature_cols), "used_nlp_features": used_nlp},
            f, indent=2,
        )

    result = {
        "structured_only_metrics": metrics_a,
        "structured_plus_nlp_metrics": metrics_b,
        "rmse_improvement_pct": rmse_improvement_pct,
        "persisted_model": "structured_plus_nlp" if used_nlp else "structured_only",
    }
    logger.info("Comparison result: %s", result)
    return result


class SeverityPredictor:
    """Loads the trained model for scoring new/unscored CVE feature rows."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        with open(model_dir / "cve_severity_feature_cols.json") as f:
            schema = json.load(f)
        self.feature_cols = schema["feature_cols"]
        self.used_nlp_features = schema.get("used_nlp_features", False)
        self.model = SeverityDNN(input_dim=schema["input_dim"])
        self.model.load_state_dict(
            torch.load(model_dir / "cve_severity_dnn.pt", map_location="cpu")
        )
        self.model.eval()
        self.scaler = joblib.load(model_dir / "cve_severity_scaler.pkl")

        pca_path = model_dir / "cve_severity_pca.pkl"
        self.pca = joblib.load(pca_path) if (self.used_nlp_features and pca_path.exists()) else None

    def predict(self, feature_row: dict, embedding: Optional[List[float]] = None) -> float:
        """feature_row must already be one-hot/frequency-encoded structured
        features. If this model was trained with NLP features, pass the raw
        384-dim `embedding` (e.g. from ml.embeddings.embed_query on the CVE
        description) and it will be PCA-reduced automatically."""
        row = dict(feature_row)
        if self.used_nlp_features:
            if embedding is None or self.pca is None:
                raise ValueError("This model was trained with NLP features -- pass `embedding`.")
            reduced = self.pca.transform(np.array([embedding]))[0]
            for i, val in enumerate(reduced):
                row[f"emb_{i}"] = float(val)

        x = np.array([[row.get(c, 0.0) for c in self.feature_cols]])
        x_scaled = self.scaler.transform(x)
        with torch.no_grad():
            pred = self.model(torch.tensor(x_scaled, dtype=torch.float32))
        return float(pred.item())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the CVE severity regression DNN.")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows for a quick dev run")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no-nlp", action="store_true", help="Structured features only, skip embeddings")
    parser.add_argument("--compare", action="store_true", help="Train both variants and print the comparison")
    args = parser.parse_args()

    if args.compare:
        result = train_severity_model_comparison(limit=args.limit, epochs=args.epochs)
        print(json.dumps(result, indent=2))
    else:
        result = train_severity_model(limit=args.limit, epochs=args.epochs, use_nlp=not args.no_nlp)
        print(json.dumps(result["metrics"], indent=2))
