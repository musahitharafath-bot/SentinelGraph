"""
ml/gnn_models.py
--------------------
GNN-based risk prediction: unlike ml/train.py's RiskDNN (a plain
feed-forward network fed hand-engineered graph features -- degree,
PageRank, betweenness as individual scalar columns), the models here are
true message-passing GNNs. They learn directly from the graph's structure
via convolution over neighborhoods, so an asset's predicted risk is
influenced by its neighbors' *learned representations*, not just a
precomputed centrality score.

Two architectures, both via PyTorch Geometric:
  - GraphSAGE (SAGEConv): aggregates and samples neighbor features --
    scales well, a strong general-purpose default.
  - GAT (GATConv): attention-weighted aggregation -- learns *which*
    neighbors matter most for a given prediction, and those attention
    weights are directly inspectable (see `explain_prediction`), which
    GraphSAGE and the plain RiskDNN can't offer.

Node features: the same structured features as ml/features.py (vulnerability
posture, exposure, criticality, connectivity), so the only thing that
changes vs. RiskDNN is *how* the model uses graph structure -- explicit
message passing instead of precomputed centrality scalars.

Task: binary node classification (same "likely high-value target" label as
ml/features.py::build_labels), trained transductively on the whole asset
graph at once (standard for single-graph node classification -- there's
no train/test graph split, only a train/test node mask).

Usage:
    python -m ml.gnn_models --model sage
    python -m ml.gnn_models --model gat --compare-to-dnn
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import joblib

from graph.connection import get_connection
from ml.features import build_feature_frame, build_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

Q_EDGES = "MATCH (a:Asset)-[:CONNECTS_TO]->(b:Asset) RETURN a.asset_id AS src, b.asset_id AS dst"


class GraphSAGENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


class GATNet(nn.Module):
    """GAT: attention-weighted neighbor aggregation. `heads` attention heads
    on the first layer, concatenated then averaged down in the second --
    the standard GAT pattern. Attention weights from the final layer are
    exposed via `return_attention_weights` for explainability."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention: bool = False):
        if return_attention:
            x, (edge_index_1, alpha_1) = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, (edge_index_2, alpha_2) = self.conv2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            logits = self.classifier(x).squeeze(-1)
            return logits, (edge_index_1, alpha_1), (edge_index_2, alpha_2)

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


def build_pyg_graph() -> Tuple[Data, List[str], List[str], StandardScaler]:
    """Assembles a torch_geometric Data object from the current Neo4j graph:
    node features from ml/features.py, edges from CONNECTS_TO, labels from
    the same heuristic ml/features.py uses for RiskDNN (so GNN vs. RiskDNN
    comparisons are apples-to-apples on identical features/labels -- the
    only difference is architecture)."""
    df = build_feature_frame()
    if df.empty:
        raise RuntimeError("No asset features found -- has the graph been ingested?")

    y = build_labels(df)
    asset_ids = df["asset_id"].tolist()
    id_to_idx = {aid: i for i, aid in enumerate(asset_ids)}

    feature_cols = [c for c in df.columns if c != "asset_id"]
    X = df[feature_cols].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    edges = get_connection().run_query(Q_EDGES)
    edge_list = [
        (id_to_idx[e["src"]], id_to_idx[e["dst"]])
        for e in edges
        if e["src"] in id_to_idx and e["dst"] in id_to_idx
    ]
    # make undirected for message passing (attackers/analysis don't care
    # about the CONNECTS_TO direction for "who is this asset near")
    edge_list += [(b, a) for a, b in edge_list]

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y.values, dtype=torch.float32),
    )
    return data, asset_ids, feature_cols, scaler


def _train_gnn(model: nn.Module, data: Data, epochs: int, lr: float, seed: int = 42) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = data.num_nodes
    idx = np.arange(n)
    # transductive node classification: split *nodes*, not the graph itself
    stratify = data.y.numpy() if len(np.unique(data.y.numpy())) > 1 and n >= 10 else None
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=stratify)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_idx] = True

    pos_weight = float((data.y[train_mask] == 0).sum()) / max(float((data.y[train_mask] == 1).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_loss, best_state = float("inf"), None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_loss = criterion(val_logits[val_mask], data.y[val_mask]).item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info("Epoch %3d | train_loss=%.4f val_loss=%.4f", epoch, loss.item(), val_loss)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(data.x, data.edge_index))[val_mask].numpy()
    val_labels = data.y[val_mask].numpy()

    val_preds = (val_probs >= 0.5).astype(int)
    metrics = {"accuracy": float(accuracy_score(val_labels, val_preds))}
    try:
        metrics["roc_auc"] = float(roc_auc_score(val_labels, val_probs))
    except ValueError:
        metrics["roc_auc"] = None
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="binary", zero_division=0)
    metrics.update({"precision": float(precision), "recall": float(recall), "f1": float(f1)})

    return {"model": model, "metrics": metrics, "history": history}


def train_gnn_model(model_type: str = "sage", epochs: int = 150, lr: float = 0.01, hidden_dim: int = 64) -> Dict[str, Any]:
    """Trains a GraphSAGE or GAT node classifier and persists it."""
    data, asset_ids, feature_cols, scaler = build_pyg_graph()
    input_dim = data.num_node_features

    if model_type == "sage":
        model = GraphSAGENet(input_dim=input_dim, hidden_dim=hidden_dim)
    elif model_type == "gat":
        model = GATNet(input_dim=input_dim, hidden_dim=max(hidden_dim // 2, 8))
    else:
        raise ValueError(f"Unknown model_type: {model_type} (use 'sage' or 'gat')")

    result = _train_gnn(model, data, epochs=epochs, lr=lr)

    torch.save(result["model"].state_dict(), MODEL_DIR / f"gnn_{model_type}.pt")
    joblib.dump(scaler, MODEL_DIR / f"gnn_{model_type}_scaler.pkl")
    with open(MODEL_DIR / f"gnn_{model_type}_schema.json", "w") as f:
        json.dump(
            {"feature_cols": feature_cols, "input_dim": input_dim, "hidden_dim": hidden_dim, "model_type": model_type},
            f, indent=2,
        )
    with open(MODEL_DIR / f"gnn_{model_type}_metrics.json", "w") as f:
        json.dump({"metrics": result["metrics"], "history": result["history"]}, f, indent=2)

    logger.info("%s training complete. Metrics: %s", model_type.upper(), result["metrics"])
    return {"model_type": model_type, "metrics": result["metrics"], "n_nodes": data.num_nodes, "n_edges": data.num_edges}


def compare_gnn_vs_dnn(epochs: int = 150) -> Dict[str, Any]:
    """Trains GraphSAGE, GAT, and the existing plain-feature RiskDNN on the
    *same* underlying features/labels and compares them -- demonstrates
    whether message passing actually earns its complexity on this graph's
    size and connectivity (small/sparse graphs often show little or even
    negative GNN benefit over a plain DNN with centrality features already
    baked in as columns -- that's a real, expected possibility, not a bug,
    and this comparison is what would tell you which case you're in)."""
    from ml.train import train_model as train_dnn

    sage_result = train_gnn_model(model_type="sage", epochs=epochs)
    gat_result = train_gnn_model(model_type="gat", epochs=epochs)
    dnn_result = train_dnn(epochs=epochs)

    return {
        "graphsage": sage_result["metrics"],
        "gat": gat_result["metrics"],
        "plain_dnn_precomputed_features": dnn_result["metrics"],
    }


class GNNRiskPredictor:
    """Loads a trained GNN for inference over the current graph."""

    def __init__(self, model_type: str = "sage", model_dir: Path = MODEL_DIR):
        self.model_type = model_type
        with open(model_dir / f"gnn_{model_type}_schema.json") as f:
            schema = json.load(f)
        self.feature_cols = schema["feature_cols"]

        if model_type == "sage":
            self.model = GraphSAGENet(input_dim=schema["input_dim"], hidden_dim=schema["hidden_dim"])
        else:
            self.model = GATNet(input_dim=schema["input_dim"], hidden_dim=max(schema["hidden_dim"] // 2, 8))
        self.model.load_state_dict(torch.load(model_dir / f"gnn_{model_type}.pt", map_location="cpu"))
        self.model.eval()
        self.scaler = joblib.load(model_dir / f"gnn_{model_type}_scaler.pkl")

    def predict_all(self) -> List[Dict[str, Any]]:
        data, asset_ids, feature_cols, _ = build_pyg_graph()
        # re-scale with the *trained* scaler, not a freshly-fit one
        x_raw = data.x.numpy()
        # data.x was already scaled by build_pyg_graph's own scaler; undo isn't
        # trivial, so for inference-time consistency we rebuild features raw:
        from ml.features import build_feature_frame

        df = build_feature_frame()
        X = df[self.feature_cols].astype(float)
        x_scaled = self.scaler.transform(X.values)
        x_t = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(x_t, data.edge_index)).numpy()

        results = [
            {
                "asset_id": aid,
                "predicted_compromise_probability": round(float(p), 4),
                "predicted_label": "high_risk" if p >= 0.5 else "low_risk",
                "model": f"gnn_{self.model_type}",
            }
            for aid, p in zip(df["asset_id"].tolist(), probs)
        ]
        results.sort(key=lambda r: r["predicted_compromise_probability"], reverse=True)
        return results

    def explain_prediction(self, asset_id: str) -> Dict[str, Any]:
        """GAT-only: returns which neighbors received the most attention
        weight for this asset's prediction -- a genuine, model-native
        explainability signal that GraphSAGE and the plain RiskDNN can't
        provide, since attention weights are learned per-edge."""
        if self.model_type != "gat":
            raise ValueError("explain_prediction is only available for GAT models")

        data, asset_ids, feature_cols, _ = build_pyg_graph()
        from ml.features import build_feature_frame

        df = build_feature_frame()
        X = df[self.feature_cols].astype(float)
        x_scaled = self.scaler.transform(X.values)
        x_t = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            _, (edge_index_1, alpha_1), _ = self.model(x_t, data.edge_index, return_attention=True)

        if asset_id not in asset_ids:
            return {"asset_id": asset_id, "error": "not found"}
        target_idx = asset_ids.index(asset_id)

        alpha_mean = alpha_1.mean(dim=1)  # average over attention heads
        neighbor_weights = []
        edge_src, edge_dst = edge_index_1
        for i in range(edge_index_1.shape[1]):
            if edge_dst[i].item() == target_idx:
                src_idx = edge_src[i].item()
                neighbor_weights.append({"neighbor_asset_id": asset_ids[src_idx], "attention_weight": float(alpha_mean[i])})

        neighbor_weights.sort(key=lambda x: x["attention_weight"], reverse=True)
        return {"asset_id": asset_id, "top_influencing_neighbors": neighbor_weights}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GraphSAGE/GAT GNN risk models.")
    parser.add_argument("--model", choices=["sage", "gat"], default="sage")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--compare-to-dnn", action="store_true")
    args = parser.parse_args()

    if args.compare_to_dnn:
        print(json.dumps(compare_gnn_vs_dnn(epochs=args.epochs), indent=2))
    else:
        print(json.dumps(train_gnn_model(model_type=args.model, epochs=args.epochs), indent=2))
