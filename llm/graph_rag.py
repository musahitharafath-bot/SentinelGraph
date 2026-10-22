"""
llm/graph_rag.py
--------------------
True hybrid GraphRAG: llm/rag.py already does retrieval-then-generation, but
its retrieval is pure Cypher (exact graph matches only) -- it can't answer
"what CVEs are conceptually similar to X" or handle a fuzzy natural-language
question that doesn't map to an exact asset/CVE ID. This module adds the
missing half: vector similarity search over CVE descriptions (the embedding
index built by ingestion/nvd_loader.py) as the *retrieval* step, then
expands each semantically-matched CVE outward through the graph (which
assets it affects, what else is reachable from there) to assemble grounded
structured context, then hands that to the LLM for generation.

Pipeline:
    question
      -> embed question (ml/embeddings.embed_query)
      -> vector search over :CVECatalog.embedding (semantic retrieval)
      -> for each matched CVE, graph-expand to affected assets + their
         network neighborhood (structural retrieval)
      -> assemble combined context
      -> LLM generates a grounded answer, citing only retrieved data

This is different from llm/cypher_chain.py (which generates a fresh Cypher
query per question) and llm/rag.py (which retrieves a specific, named
attack path or the top-N critical assets) -- graph_rag.py is for open-ended
questions where the relevant entry point into the graph isn't known in
advance and has to be found semantically first.
"""

import os
import json
import logging
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.connection import get_connection
from graph import queries as q

logger = logging.getLogger(__name__)

GRAPH_EXPANSION_QUERY = """
MATCH (c:CVECatalog {cve_id: $cve_id})
OPTIONAL MATCH (a:Asset)-[:HAS_VULNERABILITY]->(c)
OPTIONAL MATCH (a)-[:CONNECTS_TO]-(neighbor:Asset)
RETURN c.cve_id AS cve_id, c.description AS description, c.cvss_score AS cvss_score,
       c.severity AS severity, c.cwe_id AS cwe_id,
       collect(DISTINCT {asset_id: a.asset_id, hostname: a.hostname,
                          criticality: a.criticality, exposed: a.exposed_to_internet}) AS affected_assets,
       collect(DISTINCT neighbor.asset_id) AS reachable_neighbors
"""

HYBRID_RAG_TEMPLATE = """You are a cybersecurity analyst assistant. Answer the question using ONLY
the retrieved context below, which combines semantically-matched CVEs (found
by meaning, not exact ID) with the graph structure showing what those CVEs
actually affect in this specific environment.

If the context doesn't contain enough information to answer confidently,
say so explicitly rather than guessing.

Question: {question}

Retrieved context (semantically similar CVEs + their graph neighborhood):
{context}

Answer, citing specific CVE IDs and asset IDs from the context above:"""

HYBRID_RAG_PROMPT = PromptTemplate(input_variables=["question", "context"], template=HYBRID_RAG_TEMPLATE)


def _get_llm(temperature: float = 0.2) -> ChatGroq:
    model = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Get a key from https://console.groq.com and "
            "export GROQ_API_KEY=... before using GraphRAG."
        )
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


def semantic_retrieve(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Step 1: vector similarity search -- finds CVEs conceptually related
    to the question even if it doesn't name a specific CVE ID."""
    from ml.embeddings import embed_query

    embedding = embed_query(question)
    return q.semantic_search_cves(embedding, top_k=top_k)


def graph_expand(cve_id: str) -> Dict[str, Any]:
    """Step 2: structural retrieval -- for a semantically-matched CVE, pulls
    which real assets it affects and their immediate network neighborhood,
    grounding the abstract text match in this environment's actual graph."""
    conn = get_connection()
    rows = conn.run_query(GRAPH_EXPANSION_QUERY, {"cve_id": cve_id})
    return rows[0] if rows else {}


def hybrid_retrieve(question: str, top_k_cves: int = 5) -> Dict[str, Any]:
    """Combines both retrieval steps into one structured context bundle."""
    semantic_hits = semantic_retrieve(question, top_k=top_k_cves)
    expanded = []
    for hit in semantic_hits:
        detail = graph_expand(hit["cve_id"])
        if detail:
            # drop assets/neighbors that are entirely null (no real linkage)
            detail["affected_assets"] = [
                a for a in detail.get("affected_assets", []) if a.get("asset_id")
            ]
            detail["reachable_neighbors"] = [
                n for n in detail.get("reachable_neighbors", []) if n
            ]
            detail["similarity_score"] = hit.get("score")
            expanded.append(detail)
        else:
            expanded.append({**hit, "affected_assets": [], "reachable_neighbors": []})
    return {"question": question, "retrieved_cves": expanded}


def graph_rag_query(question: str, top_k_cves: int = 5) -> Dict[str, Any]:
    """Full hybrid GraphRAG pipeline: semantic retrieval -> graph expansion
    -> LLM generation grounded in the combined context."""
    context = hybrid_retrieve(question, top_k_cves=top_k_cves)

    if not context["retrieved_cves"]:
        return {
            "question": question,
            "answer": (
                "No semantically relevant CVEs were found. This usually means "
                "the CVE catalog hasn't been embedded yet -- run "
                "`python -m ingestion.nvd_loader --embed` or POST /cve-catalog/embed first."
            ),
            "context": context,
        }

    llm = _get_llm()
    chain = HYBRID_RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": json.dumps(context, indent=2, default=str)})

    return {"question": question, "answer": answer.strip(), "context": context}


if __name__ == "__main__":
    result = graph_rag_query("What remote code execution vulnerabilities could affect our internet-facing servers?")
    print(json.dumps(result, indent=2, default=str))
