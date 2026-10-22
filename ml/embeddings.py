"""
ml/embeddings.py
-------------------
Thin, singleton wrapper around a sentence-transformer model, shared by:
  - ingestion/nvd_loader.py (embeds CVE descriptions when populating the catalog)
  - ml/cve_severity_model.py (uses embeddings as DNN input features)
  - api/main.py (embeds a user's free-text query for semantic search)

Kept as one module so the model is loaded once per process instead of once
per call site.
"""

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
EMBEDDING_DIM = 384

_model_singleton = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model_singleton
    if _model_singleton is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        _model_singleton = SentenceTransformer(model_name)
    return _model_singleton


def embed_texts(texts: List[str], batch_size: int = 128) -> np.ndarray:
    """Batch-embeds a list of strings. Returns an (N, 384) float32 array."""
    model = get_embedding_model()
    cleaned = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    return model.encode(cleaned, batch_size=batch_size, show_progress_bar=len(cleaned) > 500, convert_to_numpy=True)


def embed_query(text: str) -> List[float]:
    """Embeds a single free-text query, returns a plain list (JSON/Cypher-safe)."""
    model = get_embedding_model()
    return model.encode(text).tolist()
