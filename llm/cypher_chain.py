"""
llm/cypher_chain.py
----------------------
Natural-language -> Cypher -> Neo4j -> natural-language-answer pipeline,
powered by Groq's LLM API via langchain_groq.

This is a hand-rolled, READ-ONLY version of LangChain's GraphCypherQAChain:
  1. LLM turns the user's question into a Cypher query (using the schema
     described in llm/prompts.py).
  2. We sanity-check the query is read-only before executing it.
  3. Cypher runs against Neo4j.
  4. LLM turns the raw results back into a natural-language answer.

Env vars required:
    GROQ_API_KEY

Model default: "llama-3.3-70b-versatile" (fast + strong at structured
generation tasks like Cypher). Override with GROQ_MODEL env var.
"""

import os
import re
import logging
from typing import Dict, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from graph.connection import get_connection
from llm.prompts import (
    CYPHER_GENERATION_PROMPT,
    QA_PROMPT,
    GRAPH_SCHEMA_DESCRIPTION,
)

logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+apoc\.periodic|LOAD\s+CSV)\b",
    re.IGNORECASE,
)


def _get_llm(temperature: float = 0.0) -> ChatGroq:
    model = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Get a key from https://console.groq.com and "
            "export GROQ_API_KEY=... before using the LLM chain."
        )
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


def _clean_cypher(raw: str) -> str:
    """Strip markdown fences / stray commentary the LLM sometimes adds."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:cypher)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _assert_read_only(cypher: str) -> None:
    if FORBIDDEN_KEYWORDS.search(cypher):
        raise ValueError(
            f"Generated Cypher contains a write/mutation keyword and was blocked "
            f"for safety:\n{cypher}"
        )


class GraphCypherQA:
    """Read-only natural-language interface over the attack graph."""

    def __init__(self, temperature: float = 0.0):
        self.llm = _get_llm(temperature=temperature)
        self.cypher_chain = CYPHER_GENERATION_PROMPT | self.llm | StrOutputParser()
        self.qa_chain = QA_PROMPT | self.llm | StrOutputParser()
        self.conn = get_connection()

    def generate_cypher(self, question: str) -> str:
        raw = self.cypher_chain.invoke(
            {"schema": GRAPH_SCHEMA_DESCRIPTION, "question": question}
        )
        cypher = _clean_cypher(raw)
        _assert_read_only(cypher)
        return cypher

    def run(self, question: str, return_intermediate: bool = True) -> Dict[str, Any]:
        cypher = self.generate_cypher(question)
        logger.info("Generated Cypher: %s", cypher)

        try:
            results = self.conn.run_query(cypher)
        except Exception as e:
            logger.error("Cypher execution failed: %s", e)
            return {
                "question": question,
                "cypher_query": cypher,
                "results": [],
                "answer": (
                    f"I generated a query but it failed to execute against the "
                    f"database ({e}). Try rephrasing the question."
                ),
                "error": str(e),
            }

        answer = self.qa_chain.invoke(
            {
                "question": question,
                "cypher_query": cypher,
                "results": str(results[:50]),  # cap payload sent back to LLM
            }
        )

        output = {"question": question, "answer": answer.strip()}
        if return_intermediate:
            output["cypher_query"] = cypher
            output["results"] = results
        return output


_qa_singleton: Optional[GraphCypherQA] = None


def get_qa_chain() -> GraphCypherQA:
    global _qa_singleton
    if _qa_singleton is None:
        _qa_singleton = GraphCypherQA()
    return _qa_singleton


def ask(question: str) -> Dict[str, Any]:
    return get_qa_chain().run(question)


if __name__ == "__main__":
    import json

    q = "Which internet-exposed assets have critical vulnerabilities with an exploit available?"
    print(json.dumps(ask(q), indent=2, default=str))
