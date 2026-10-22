"""
llm/rag.py
-------------
Retrieval-Augmented Generation over the attack graph: pulls structured
context (attack paths, vulnerability details, risk scores, DNN
predictions) straight from Neo4j / the analysis+ml modules ("retrieval"),
then asks a Groq-hosted LLM to turn that structured context into a
human-readable risk narrative or executive summary ("generation").

This deliberately does NOT use a vector store — the "documents" being
retrieved are graph query results, which are already precise/structured,
so classic RAG-over-embeddings would add noise rather than value here.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from analysis.attack_paths import shortest_path, internet_to_critical_paths
from analysis.critical_assets import rank_critical_assets
from graph import queries as q
from llm.prompts import ATTACK_PATH_EXPLAIN_PROMPT, RISK_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0.3) -> ChatGroq:
    model = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Get a key from https://console.groq.com and "
            "export GROQ_API_KEY=... before using the RAG explainer."
        )
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


def _gather_path_context(start_id: str, end_id: str) -> Dict[str, Any]:
    """Retrieval step: pull the path + vulnerability details of every asset
    on that path from the graph."""
    path = shortest_path(start_id, end_id)
    if not path:
        return {"path": None, "assets": []}

    asset_details = []
    for asset_id in path["path_assets"]:
        detail = q.get_asset_detail(asset_id)
        if not detail:
            continue
        asset_node = detail.get("asset", {})
        vulns = [
            {
                "cve_id": v.get("cve_id"),
                "cvss_score": v.get("cvss_score"),
                "severity": v.get("severity"),
                "exploit_available": v.get("exploit_available"),
                "patch_available": v.get("patch_available"),
            }
            for v in detail.get("vulnerabilities", [])
            if v
        ]
        asset_details.append(
            {
                "asset_id": asset_node.get("asset_id"),
                "hostname": asset_node.get("hostname"),
                "criticality": asset_node.get("criticality"),
                "exposed_to_internet": asset_node.get("exposed_to_internet"),
                "vulnerabilities": vulns,
            }
        )

    return {"path": path, "assets": asset_details}


def explain_attack_path(start_id: str, end_id: str) -> Dict[str, Any]:
    """Retrieve a specific attack path + generate an executive-readable
    risk narrative with remediation recommendations."""
    context = _gather_path_context(start_id, end_id)
    if not context["path"]:
        return {
            "start_id": start_id,
            "end_id": end_id,
            "narrative": f"No connected path was found between {start_id} and {end_id}.",
            "context": context,
        }

    try:
        llm = _get_llm()
        chain = ATTACK_PATH_EXPLAIN_PROMPT | llm | StrOutputParser()
        narrative = chain.invoke({"context": json.dumps(context, default=str)})
        narrative_str = narrative.strip()
    except Exception as e:
        logger.exception("LLM attack path explanation failed")
        narrative_str = f"Attack path explanation unavailable ({e})."

    return {
        "start_id": start_id,
        "end_id": end_id,
        "narrative": narrative_str,
        "context": context,
    }


def explain_top_attack_routes(limit: int = 3) -> Dict[str, Any]:
    """Retrieve the riskiest internet-exposed -> critical-asset routes and
    explain them collectively."""
    routes = internet_to_critical_paths()[:limit]
    if not routes:
        return {"narrative": "No internet-exposed paths to critical assets were found.", "routes": []}

    try:
        llm = _get_llm()
        chain = ATTACK_PATH_EXPLAIN_PROMPT | llm | StrOutputParser()
        narrative = chain.invoke({"context": json.dumps(routes, default=str)})
        narrative_str = narrative.strip()
    except Exception as e:
        logger.exception("LLM top attack routes explanation failed")
        narrative_str = f"Route explanation unavailable ({e})."

    return {"narrative": narrative_str, "routes": routes}


def executive_risk_summary(top_n: int = 5) -> Dict[str, Any]:
    """Retrieve top critical assets (risk score + centrality) and produce a
    CISO-style executive summary."""
    top_assets = rank_critical_assets(top_n=top_n)
    if not top_assets:
        return {"narrative": "No asset risk data is available yet — has the graph been ingested?", "assets": []}

    # Minimize context payload to top 5 assets and essential fields only to avoid payload limits
    context_assets = [
        {
            "asset_id": a.get("asset_id"),
            "criticality": a.get("criticality_label") or a.get("criticality"),
            "type": a.get("asset_type"),
            "risk_score": a.get("risk_score"),
            "vuln_count": a.get("vuln_count"),
        }
        for a in top_assets[:5]
    ]
    try:
        llm = _get_llm()
        chain = RISK_SUMMARY_PROMPT | llm | StrOutputParser()
        narrative = chain.invoke({"context": json.dumps(context_assets, default=str)})
        narrative_str = narrative.strip()
    except Exception as e:
        logger.exception("LLM executive summary generation failed")
        narrative_str = f"Executive summary generation unavailable ({e}). Top critical assets are listed below."

    return {"narrative": narrative_str, "assets": top_assets}


if __name__ == "__main__":
    print(json.dumps(executive_risk_summary(), indent=2, default=str))
