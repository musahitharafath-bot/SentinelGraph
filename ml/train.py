"""
ml/train.py
--------------
Trains a deep neural network (PyTorch) to predict per-asset compromise
risk from graph-derived features (vulnerability posture, exposure,
centrality, connectivity).

Architecture: a fully-connected feed-forward DNN with batch norm + dropout
regularization, trained as a binary classifier (sigmoid output = predicted
probability an asset is a high-value attack target).

Usage:
    python -m ml.train
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import joblib

from ml.features import get_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class RiskDNN(nn.Module):
    """Feed-forward DNN for asset compromise-risk classification.

    input -> [Linear -> BatchNorm -> ReLU -> Dropout] x3 -> Linear -> Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = (128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits


def _prepare_tensors(
    X: pd.DataFrame, y: pd.Series, scaler: StandardScaler = None, fit_scaler: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values) if fit_scaler else scaler.transform(X.values)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    return X_t, y_t, scaler


def train_model(
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 8,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, feature_cols, asset_ids = get_training_data()

    # small synthetic dataset -> stratify only if both classes have enough samples
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=stratify
    )

    X_train_t, y_train_t, scaler = _prepare_tensors(X_train, y_train, fit_scaler=True)
    X_val_t, y_val_t, _ = _prepare_tensors(X_val, y_val, scaler=scaler, fit_scaler=False)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    model = RiskDNN(input_dim=X.shape[1])
    pos_weight_val = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t) if len(X_val_t) else torch.tensor([])
            val_loss = criterion(val_logits, y_val_t).item() if len(X_val_t) else epoch_loss

        scheduler.step(val_loss)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info("Epoch %3d | train_loss=%.4f val_loss=%.4f", epoch, epoch_loss, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- evaluation ----
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t)).numpy() if len(X_val_t) else np.array([])
    metrics = {}
    if len(val_probs):
        val_preds = (val_probs >= 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(y_val, val_preds))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, val_probs))
        except ValueError:
            metrics["roc_auc"] = None
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="binary", zero_division=0
        )
        metrics.update({"precision": float(precision), "recall": float(recall), "f1": float(f1)})

    # ---- persist artifacts ----
    torch.save(model.state_dict(), MODEL_DIR / "risk_dnn.pt")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "input_dim": X.shape[1]}, f, indent=2)
    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "history": history}, f, indent=2)

    logger.info("Training complete. Metrics: %s", metrics)
    logger.info("Artifacts saved to %s", MODEL_DIR)
    return {"metrics": metrics, "feature_cols": feature_cols}


if __name__ == "__main__":
    train_model()
