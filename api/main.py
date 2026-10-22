"""
api/main.py
--------------
FastAPI backend for the cyber-attack-graph project. Exposes:
  - /assets                 graph data access
  - /analysis/*             attack paths, risk scores, centrality, critical assets
  - /ml/*                   DNN-based compromise-risk predictions
  - /llm/*                  natural-language querying + RAG risk narratives

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import queries as q
from graph.schema import initialize_schema
from ingestion.network_loader import run_full_ingestion
from analysis.attack_paths import (
    shortest_path,
    all_paths,
    internet_to_critical_paths,
    summarize_attack_surface,
)
from analysis.risk_score import compute_risk_scores, risk_score_for_asset
from analysis.centrality import pagerank, betweenness, degree
from analysis.critical_assets import rank_critical_assets

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cyber Attack Graph API",
    description="Neo4j-backed cybersecurity attack graph analysis, DNN risk prediction, and LLM querying.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------- schemas


class PathRequest(BaseModel):
    start_id: str
    end_id: str
    max_hops: int = 6


class CypherQuestion(BaseModel):
    question: str


class AttackPathExplainRequest(BaseModel):
    start_id: str
    end_id: str


# ------------------------------------------------------------------ meta


@app.get("/")
def root():
    return {
        "service": "Cyber Attack Graph API",
        "status": "running",
        "docs": "/docs",
    }


@app.post("/admin/init-schema")
def init_schema():
    initialize_schema()
    return {"status": "schema initialized"}


@app.post("/admin/ingest")
def ingest_data():
    """Runs the full CSV -> Neo4j ingestion pipeline."""
    try:
        counts = run_full_ingestion()
        return {"status": "ingested", "counts": counts}
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------- assets


@app.get("/assets")
def list_assets():
    return q.get_all_assets()


@app.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    detail = q.get_asset_detail(asset_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
    return detail


@app.get("/assets/vulnerable/list")
def vulnerable_assets(min_cvss: float = Query(7.0, ge=0, le=10)):
    return q.get_vulnerable_assets(min_cvss=min_cvss)


@app.get("/assets/exposed/critical")
def exposed_critical_assets():
    return q.get_exposed_critical_assets()


# ------------------------------------------------------ NVD CVE catalog


@app.get("/cve-catalog/stats")
def cve_catalog_stats():
    return q.get_cve_catalog_stats()


@app.get("/cve-catalog/{cve_id}")
def cve_catalog_lookup(cve_id: str):
    result = q.get_cve_by_id(cve_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"{cve_id} not found in catalog")
    return result


@app.get("/cve-catalog/top-cwe")
def cve_catalog_top_cwe():
    return q.get_top_cwe_categories()


@app.post("/cve-catalog/embed")
def cve_catalog_embed(limit: Optional[int] = Query(None)):
    """Computes sentence-transformer embeddings for any catalog CVE that
    doesn't have one yet, and ensures the vector index exists. Safe to
    call repeatedly / incrementally as new CVEs are ingested."""
    try:
        from ingestion.nvd_loader import create_vector_index, embed_catalog

        create_vector_index()
        n = embed_catalog(limit=limit)
        return {"embedded": n}
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cve-catalog/semantic-search")
def cve_catalog_semantic_search(q_text: str = Query(..., alias="q"), top_k: int = Query(5, ge=1, le=50)):
    """Free-text semantic search over CVE descriptions, e.g.
    ?q=remote code execution in internet-facing web servers"""
    try:
        from ml.embeddings import embed_query

        embedding = embed_query(q_text)
        return q.semantic_search_cves(embedding, top_k=top_k)
    except Exception as e:
        logger.exception("Semantic search failed")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------- analysis


@app.get("/analysis/attack-surface")
def attack_surface():
    return summarize_attack_surface()


@app.post("/analysis/shortest-path")
def api_shortest_path(req: PathRequest):
    result = shortest_path(req.start_id, req.end_id)
    if not result:
        return {
            "found": False,
            "path_assets": [],
            "path_hostnames": [],
            "hops": 0,
            "message": f"No path found between {req.start_id} and {req.end_id}",
        }
    return {"found": True, **result}


@app.post("/analysis/all-paths")
def api_all_paths(req: PathRequest):
    return all_paths(req.start_id, req.end_id, req.max_hops)


@app.get("/analysis/internet-to-critical")
def api_internet_to_critical():
    return internet_to_critical_paths()


@app.get("/analysis/risk-scores")
def api_risk_scores():
    return compute_risk_scores()


@app.get("/analysis/risk-scores/{asset_id}")
def api_risk_score_for_asset(asset_id: str):
    result = risk_score_for_asset(asset_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No risk score for asset {asset_id}")
    return result


@app.get("/analysis/centrality/pagerank")
def api_pagerank():
    return pagerank()


@app.get("/analysis/centrality/betweenness")
def api_betweenness():
    return betweenness()


@app.get("/analysis/centrality/degree")
def api_degree():
    return degree()


@app.get("/analysis/critical-assets")
def api_critical_assets(top_n: int = Query(10, ge=1, le=100)):
    return rank_critical_assets(top_n=top_n)


# --------------------------------------------------------------------- ml


@app.get("/ml/predict")
def ml_predict_all():
    try:
        from ml.predict import get_predictor

        return get_predictor().predict_all()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/ml/predict/{asset_id}")
def ml_predict_one(asset_id: str):
    try:
        from ml.predict import get_predictor

        return get_predictor().predict_one(asset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ml/train")
def ml_train():
    """Kicks off (synchronous) DNN training. For a real deployment this
    should be a background job / queued task rather than inline."""
    try:
        from ml.train import train_model

        return train_model()
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------- ml: CVE severity DNN


class CveIngestRequest(BaseModel):
    path: str
    chunksize: int = 5000
    limit_rows: Optional[int] = None


@app.post("/ml/cve-catalog/ingest")
def ingest_cve_catalog(req: CveIngestRequest):
    """Loads the NVD Kaggle CSV (337k+ CVEs) into Neo4j as a :CVECatalog.
    Point `path` at wherever you've downloaded the dataset, e.g.
    'data/nvd_cves.csv'. This can take a while for the full file."""
    try:
        from ingestion.nvd_loader import load_nvd_catalog, link_catalog_to_assets, create_index
        from pathlib import Path

        csv_path = Path(req.path)
        if not csv_path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {csv_path}")

        create_index()
        n = load_nvd_catalog(csv_path, chunksize=req.chunksize, limit_rows=req.limit_rows)
        links = link_catalog_to_assets()
        return {"cves_loaded": n, "asset_links_created": links}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CVE catalog ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/cve-severity/train")
def train_cve_severity_model(
    limit: Optional[int] = Query(None),
    epochs: int = Query(60, ge=1, le=1000),
    use_nlp: bool = Query(True, description="Include sentence-transformer embedding features if the catalog is embedded"),
    compare: bool = Query(False, description="Train structured-only vs structured+NLP and return both, instead of just one model"),
):
    """Trains the CVSS-score regression DNN on the ingested CVE catalog.
    Set compare=true to train both variants and see the measured RMSE/R²
    improvement from adding NLP embedding features (requires the catalog
    to have been embedded first via POST /cve-catalog/embed)."""
    try:
        if compare:
            from ml.cve_severity_model import train_severity_model_comparison

            return train_severity_model_comparison(limit=limit, epochs=epochs)
        from ml.cve_severity_model import train_severity_model

        return train_severity_model(limit=limit, epochs=epochs, use_nlp=use_nlp)
    except Exception as e:
        logger.exception("CVE severity model training failed")
        raise HTTPException(status_code=500, detail=str(e))


class CveFeaturesRequest(BaseModel):
    features: dict
    description: Optional[str] = None  # only needed if the trained model used NLP features


@app.post("/ml/cve-severity/predict")
def predict_cve_severity(req: CveFeaturesRequest):
    """Predicts a CVSS base score for a CVE given its (already encoded)
    structured feature dict -- see ml/cve_severity_model.py::_engineer_structured_features
    for the expected encoding. If the active model was trained with NLP
    features (POST /ml/cve-severity/train?compare=true), also pass
    `description` and it will be embedded and PCA-reduced automatically."""
    try:
        from ml.cve_severity_model import SeverityPredictor

        predictor = SeverityPredictor()
        embedding = None
        if predictor.used_nlp_features:
            if not req.description:
                raise HTTPException(
                    status_code=400,
                    detail="This model was trained with NLP features -- pass `description` too.",
                )
            from ml.embeddings import embed_query

            embedding = embed_query(req.description)

        return {"predicted_cvss_score": predictor.predict(req.features, embedding=embedding)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# -------------------------------------------------------------------- llm


@app.post("/llm/ask")
def llm_ask(req: CypherQuestion):
    try:
        from llm.cypher_chain import ask

        return ask(req.question)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("LLM query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/explain-path")
def llm_explain_path(req: AttackPathExplainRequest):
    try:
        from llm.rag import explain_attack_path

        return explain_attack_path(req.start_id, req.end_id)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Path explanation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/explain-top-routes")
def llm_explain_top_routes(limit: int = Query(3, ge=1, le=10)):
    try:
        from llm.rag import explain_top_attack_routes

        return explain_top_attack_routes(limit=limit)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Top routes explanation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/graph-rag")
def llm_graph_rag(req: CypherQuestion, top_k_cves: int = Query(5, ge=1, le=20)):
    """Hybrid GraphRAG: semantically retrieves relevant CVEs by meaning
    (vector search over the embedded CVE catalog), expands each through the
    graph to find what it actually affects here, and generates an answer
    grounded in that combined context. Requires the catalog to have been
    embedded first (POST /cve-catalog/embed)."""
    try:
        from llm.graph_rag import graph_rag_query

        return graph_rag_query(req.question, top_k_cves=top_k_cves)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("GraphRAG query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/executive-summary")
def llm_executive_summary(top_n: int = Query(5, ge=1, le=20)):
    try:
        from llm.rag import executive_risk_summary

        return executive_risk_summary(top_n=top_n)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Executive summary generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------ reports


@app.post("/reports/generate")
def generate_report(top_n: int = Query(10, ge=1, le=50), include_llm_narrative: bool = Query(True)):
    """Generates a security risk report (.docx) and returns its server path.
    Fetch the file separately, e.g. by serving reports/output/ as static
    files, or adapt this to return a FileResponse/StreamingResponse."""
    try:
        from reports.report_generator import generate_security_report

        path = generate_security_report(top_n=top_n, include_llm_narrative=include_llm_narrative)
        return {"report_path": str(path), "filename": path.name}
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/download/{filename}")
def download_report(filename: str):
    from fastapi.responses import FileResponse
    from reports.report_generator import REPORT_DIR

    path = REPORT_DIR / filename
    if not path.exists() or path.parent != REPORT_DIR:
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


# ============================================================ Level 4: research

# ------------------------------------------------------------- graph embeddings


@app.get("/research/graph-embeddings")
def get_graph_embeddings(dim: int = Query(32, ge=2, le=256), persist: bool = Query(False)):
    """Structural graph embeddings (FastRP via Neo4j GDS, SVD fallback).
    Set persist=true to also write them onto Asset.graph_embedding."""
    try:
        from analysis.graph_embeddings import compute_graph_embeddings, store_embeddings_on_graph

        if persist:
            n = store_embeddings_on_graph(dim=dim)
            return {"embedded_assets": n}
        return compute_graph_embeddings(dim=dim)
    except Exception as e:
        logger.exception("Graph embedding computation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------ GNN


class GnnTrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_type: str = "sage"  # "sage" or "gat"
    epochs: int = 150


@app.post("/research/gnn/train")
def train_gnn(req: GnnTrainRequest):
    """Trains a GraphSAGE or GAT node classifier (true message-passing GNN,
    as opposed to /ml/train's plain DNN over precomputed centrality
    features)."""
    try:
        from ml.gnn_models import train_gnn_model

        return train_gnn_model(model_type=req.model_type, epochs=req.epochs)
    except Exception as e:
        logger.exception("GNN training failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/gnn/compare")
def compare_gnn(epochs: int = Query(150, ge=10, le=1000)):
    """Trains GraphSAGE, GAT, and the plain RiskDNN on identical
    features/labels and returns all three metric sets side by side."""
    try:
        from ml.gnn_models import compare_gnn_vs_dnn

        return compare_gnn_vs_dnn(epochs=epochs)
    except Exception as e:
        logger.exception("GNN comparison failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research/gnn/predict")
def gnn_predict(model_type: str = Query("sage", pattern="^(sage|gat)$")):
    try:
        from ml.gnn_models import GNNRiskPredictor

        return GNNRiskPredictor(model_type=model_type).predict_all()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/research/gnn/explain/{asset_id}")
def gnn_explain(asset_id: str):
    """GAT-only: which neighboring assets' features most influenced this
    asset's risk prediction, via learned attention weights."""
    try:
        from ml.gnn_models import GNNRiskPredictor

        return GNNRiskPredictor(model_type="gat").explain_prediction(asset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------------- temporal


@app.get("/research/temporal/vulnerabilities-at")
def temporal_vulns_at(timestamp: str = Query(..., description="ISO date, e.g. 2023-06-01")):
    from analysis.temporal import vulnerabilities_active_at

    return vulnerabilities_active_at(timestamp)


@app.get("/research/temporal/asset-risk-trend/{asset_id}")
def temporal_asset_trend(
    asset_id: str,
    start_date: str = Query(...),
    end_date: str = Query(...),
    interval: str = Query("ME", description="pandas offset alias: D, W, ME, YE"),
):
    from analysis.temporal import risk_trend_over_time

    return risk_trend_over_time(asset_id, start_date, end_date, interval=interval)


@app.get("/research/temporal/fleet-risk-trend")
def temporal_fleet_trend(
    start_date: str = Query(...),
    end_date: str = Query(...),
    interval: str = Query("ME"),
):
    from analysis.temporal import fleet_risk_trend

    return fleet_risk_trend(start_date, end_date, interval=interval)


class MarkPatchedRequest(BaseModel):
    vuln_id: str
    patched_date: Optional[str] = None


@app.post("/research/temporal/mark-patched")
def temporal_mark_patched(req: MarkPatchedRequest):
    from analysis.temporal import mark_vulnerability_patched

    mark_vulnerability_patched(req.vuln_id, patched_date=req.patched_date)
    return {"status": "patched", "vuln_id": req.vuln_id}


@app.get("/research/temporal/event-log")
def temporal_event_log(
    asset_id: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    from analysis.temporal import get_event_log

    return get_event_log(asset_id=asset_id, start=start, end=end, limit=limit)


# ---------------------------------------------------------------- dynamic paths


@app.get("/research/dynamic/simulate-patch/{vuln_id}")
def dynamic_simulate_patch(vuln_id: str):
    from analysis.dynamic_paths import simulate_patch

    return simulate_patch(vuln_id)


class SimulateNewVulnRequest(BaseModel):
    asset_id: str
    cvss_score: float
    exploit_available: bool = True
    max_hops: int = 4


@app.post("/research/dynamic/simulate-new-vulnerability")
def dynamic_simulate_new_vuln(req: SimulateNewVulnRequest):
    from analysis.dynamic_paths import simulate_new_vulnerability

    return simulate_new_vulnerability(
        req.asset_id, req.cvss_score, exploit_available=req.exploit_available, max_hops=req.max_hops
    )


@app.get("/research/dynamic/emerging-risk")
def dynamic_emerging_risk(top_n: int = Query(10, ge=1, le=50), lookback_days: int = Query(365, ge=30, le=3650)):
    """Heuristic forecast (not a trained model -- see analysis/dynamic_paths.py
    docstring) ranking assets by recent vulnerability-discovery rate x
    structural centrality."""
    from analysis.dynamic_paths import predict_emerging_risk

    return predict_emerging_risk(top_n=top_n, lookback_days=lookback_days)
