"""
ml/predict.py
----------------
Loads the trained RiskDNN model + scaler + feature schema and produces
compromise-risk predictions for current graph assets. Also exposes a
single-asset prediction helper for the API layer.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import joblib

from ml.train import RiskDNN
from ml.features import build_feature_frame

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class RiskPredictor:
    """Wraps the trained model so it's loaded once and reused across calls."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols: List[str] = []
        self._load()

    def _load(self) -> None:
        schema_path = self.model_dir / "feature_cols.json"
        model_path = self.model_dir / "risk_dnn.pt"
        scaler_path = self.model_dir / "scaler.pkl"

        if not (schema_path.exists() and model_path.exists() and scaler_path.exists()):
            raise FileNotFoundError(
                "Trained model artifacts not found. Run `python -m ml.train` first."
            )

        with open(schema_path) as f:
            schema = json.load(f)
        self.feature_cols = schema["feature_cols"]
        input_dim = schema["input_dim"]

        self.model = RiskDNN(input_dim=input_dim)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.scaler = joblib.load(scaler_path)

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure inference-time one-hot columns match training-time schema
        (missing dummy columns -> 0, extra/unseen columns dropped)."""
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.feature_cols].astype(float)

    def predict_all(self) -> List[Dict[str, Any]]:
        df = build_feature_frame()
        if df.empty:
            return []

        asset_ids = df["asset_id"].tolist()
        X = self._align_columns(df.drop(columns=["asset_id"]))
        X_scaled = self.scaler.transform(X.values)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).numpy()

        results = [
            {
                "asset_id": aid,
                "predicted_compromise_probability": round(float(p), 4),
                "predicted_label": "high_risk" if p >= 0.5 else "low_risk",
            }
            for aid, p in zip(asset_ids, probs)
        ]
        results.sort(key=lambda r: r["predicted_compromise_probability"], reverse=True)
        return results

    def predict_one(self, asset_id: str) -> Dict[str, Any]:
        all_preds = self.predict_all()
        for r in all_preds:
            if r["asset_id"] == asset_id:
                return r
        return {"asset_id": asset_id, "error": "asset not found in feature set"}


_predictor_singleton: RiskPredictor = None


def get_predictor() -> RiskPredictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = RiskPredictor()
    return _predictor_singleton


if __name__ == "__main__":
    predictor = RiskPredictor()
    for row in predictor.predict_all():
        print(row)
