"""
llm/prompts.py
------------------
Prompt templates used by the Cypher-generation chain (llm/cypher_chain.py)
and the RAG explainer (llm/rag.py). Kept separate so wording can be tuned
without touching the chain logic.
"""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------- schema

GRAPH_SCHEMA_DESCRIPTION = """
Node labels and key properties:
  Asset(asset_id, hostname, ip_address, asset_type, os, criticality [critical|high|medium|low],
        department, exposed_to_internet [boolean])
  Vulnerability(vuln_id, cve_id, description, cvss_score [0-10 float], severity, exploit_available [bool],
                patch_available [bool], discovered_date)
  User(user_id, username, full_name, department, role, is_admin [bool])
  Software(software_id, name, version, vendor)
  Internet(name)  -- synthetic node representing the public internet

Relationships:
  (Asset)-[:HAS_VULNERABILITY]->(Vulnerability)
  (Asset)-[:RUNS]->(Software)
  (User)-[:OWNS]->(Asset)
  (Asset)-[:CONNECTS_TO {port, protocol, connection_type}]->(Asset)
  (Asset)-[:EXPOSED_TO]->(Internet)
"""

# --------------------------------------------------------- text-to-cypher

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query writer for a cybersecurity attack-graph database.

Graph schema:
{schema}

Rules:
- Only use the node labels, relationship types, and properties given in the schema above.
- Always return meaningful, human-readable properties (e.g. hostname, cve_id) rather than internal node ids.
- Prefer LIMIT clauses (e.g. LIMIT 25) for queries that could return many rows, unless the user asks for everything.
- Never generate queries that DELETE, DETACH DELETE, SET, MERGE, CREATE, or otherwise mutate the graph -- this chain is READ-ONLY.
- Return ONLY the raw Cypher query. No explanation, no markdown code fences, no commentary.

Question: {question}

Cypher query:"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

# ------------------------------------------------------------- QA / answer

QA_TEMPLATE = """You are a cybersecurity analyst assistant. You were given a natural language
question about an organization's attack graph, the Cypher query that was run against
the graph database, and the raw results returned.

Write a clear, concise, analyst-friendly answer to the question using ONLY the
information in the query results. If the results are empty, say so plainly and
suggest the question might need to be rephrased. Do not invent data that is not
present in the results.

Question: {question}
Cypher query used: {cypher_query}
Query results (JSON): {results}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["question", "cypher_query", "results"],
    template=QA_TEMPLATE,
)

# ---------------------------------------------------------- RAG explainer

ATTACK_PATH_EXPLAIN_TEMPLATE = """You are a senior penetration tester writing a plain-English risk narrative for
a non-technical executive audience.

Below is structured context pulled from a cybersecurity knowledge graph describing
a specific attack path, the assets it passes through, and known vulnerabilities.

Context:
{context}

Task: Explain how an attacker could realistically traverse this path, what makes it
dangerous, and give 3 concrete, prioritized remediation recommendations. Keep the
explanation under 300 words, avoid unnecessary jargon, and be specific to the
assets/CVEs shown in the context (do not invent CVEs or assets not present above).

Risk narrative:"""

ATTACK_PATH_EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=ATTACK_PATH_EXPLAIN_TEMPLATE,
)

RISK_SUMMARY_TEMPLATE = """You are a CISO-level cybersecurity advisor. Using the structured risk data below
(top critical assets, their risk scores, and predicted compromise probabilities
from a graph neural feature model), write an executive summary of the organization's
current cyber risk posture.

Data:
{context}

Task: Summarize the top risks in 4-6 sentences, then list the top 3 assets that need
immediate attention with a one-line reason each. Be specific and only use data given
above.

Executive summary:"""

RISK_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=RISK_SUMMARY_TEMPLATE,
)
