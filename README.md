# Cyber Attack Graph

A Neo4j-backed cybersecurity attack graph platform: model assets, vulnerabilities,
users, software, and network connectivity as a graph; run attack-path and
centrality analysis; predict per-asset compromise risk with a PyTorch deep
neural network; and query the graph in natural language via a Groq-hosted LLM
(LangChain).

## Architecture

```
CSV data --> ingestion/ --> Neo4j (graph/) --> analysis/ (paths, risk, centrality)
                                            \-> ml/ (DNN feature engineering, training, prediction)
                                            \-> llm/ (text-to-Cypher QA + RAG risk narratives, via langchain-groq)
                                                     |
                                              api/main.py (FastAPI)
                                                     |
                                            dashboard/app.py (Streamlit)
```

- **Database**: Neo4j (+ Neo4j Graph Data Science plugin for PageRank/betweenness,
  with an automatic `networkx` fallback if GDS isn't installed)
- **Backend**: FastAPI
- **Dashboard**: Streamlit (a working substitute for a separate React frontend —
  swap in `frontend/` + React later if you want a dedicated SPA; the API is
  already decoupled and CORS-enabled for that)
- **ML**: PyTorch feed-forward DNN for compromise-risk classification
- **LLM**: `langchain-groq` (`ChatGroq`) for text-to-Cypher querying and
  RAG-style risk narratives

## Project layout

```
cyber-attack-graph/
├── data/            Sample CSVs (assets, vulnerabilities, network, users, software)
├── ingestion/        CSV -> Neo4j loaders
├── graph/            Connection mgmt, schema/constraints, node/relationship writers, Cypher query library
├── analysis/         Attack paths, composite risk scoring, centrality, critical-asset ranking
├── ml/               Feature engineering, DNN training (PyTorch), inference
├── llm/               Text-to-Cypher QA chain + RAG risk narratives (langchain-groq)
├── api/               FastAPI app tying it all together
├── dashboard/         Streamlit UI
├── notebooks/         Exploration + risk/DNN analysis notebooks
└── models/            Saved DNN weights, scaler, feature schema (created at train time)
```

## Setup

### 1. Neo4j

Run Neo4j locally (Docker is easiest) with the GDS plugin for centrality:

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  neo4j:5.24
```

(No GDS plugin? `analysis/centrality.py` automatically falls back to `networkx`.)

### 2. Python environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in NEO4J_PASSWORD and GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com.

### 3. Ingest the sample data

```bash
python -m ingestion.network_loader
```

This initializes schema constraints/indexes and loads assets, users, software,
vulnerabilities, and network topology from `data/*.csv` into Neo4j.

### 4. Train the DNN risk model

```bash
python -m ml.train
```

Trains a feed-forward DNN (see `ml/train.py`) on graph-derived features
(vulnerability posture, exposure, connectivity, centrality) and saves
`models/risk_dnn.pt`, `models/scaler.pkl`, `models/feature_cols.json`.

### 5. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`. Key endpoints:

| Endpoint | Description |
|---|---|
| `POST /admin/ingest` | Run the full ingestion pipeline via API |
| `GET /assets` | List all assets |
| `GET /analysis/attack-surface` | Exposed critical assets + internet→critical routes |
| `POST /analysis/shortest-path` | Shortest path between two assets |
| `POST /analysis/all-paths` | All paths, ranked by cumulative vulnerability risk |
| `GET /analysis/risk-scores` | Composite (explainable) risk score per asset |
| `GET /analysis/centrality/pagerank` | PageRank centrality |
| `GET /analysis/critical-assets` | Risk + centrality blended "crown jewels" ranking |
| `POST /ml/train` | Train/retrain the DNN |
| `GET /ml/predict` | DNN compromise-probability predictions for all assets |
| `POST /llm/ask` | Natural-language question → Cypher → answer |
| `POST /llm/explain-path` | LLM risk narrative for a specific attack path |
| `GET /llm/executive-summary` | LLM-generated executive risk summary |

### 6. Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Using the NVD Kaggle dataset (337k+ CVEs)

The project also supports loading the full [NVD Cybersecurity Dataset (337k+
CVEs, 1988-2026)](https://www.kaggle.com/datasets/nisargshroff2402/nvd-cybersecurity-dataset)
as a real, large-scale CVE knowledge base — separate from the small demo
`data/vulnerabilities.csv`.

1. Download the CSV from Kaggle and save it as `data/nvd_cves.csv`.
2. Load it into Neo4j (streamed in batches, so 337k rows is fine):

   ```bash
   python -m ingestion.nvd_loader --path data/nvd_cves.csv
   # quick test run first: add --limit 10000
   ```

   This creates `(:Vulnerability:CVECatalog)` nodes and links them to any
   `Asset` whose existing CVE (from `vulnerabilities.csv`) matches by ID.
   Column names are auto-detected against common NVD export variants (see
   `COLUMN_ALIASES` in `ingestion/nvd_loader.py`) — if your download uses
   different headers, run `inspect_columns()` and adjust the aliases.

3. Train the **CVE severity regression DNN** — this is the task the
   dataset was built for: predicting a continuous CVSS base score from a
   CVE's characteristics (attack vector, complexity, privileges required,
   impact metrics, CWE category):

   ```bash
   python -m ml.cve_severity_model --epochs 60
   # or via the API: POST /ml/cve-severity/train
   ```

   Useful for estimating CVSS on newly disclosed CVEs before NVD scores
   them, or imputing missing scores. This is a separate model from
   `RiskDNN` (`ml/train.py`), which predicts per-*asset* compromise risk
   from *graph-structural* features — the two solve different problems and
   don't share weights.

4. Browse the catalog: `GET /cve-catalog/stats`, `GET /cve-catalog/{cve_id}`,
   `GET /cve-catalog/top-cwe`.

## Notes on the DNN

`ml/train.py` implements `RiskDNN`: a 3-hidden-layer feed-forward network
(128 → 64 → 32 → 1) with batch normalization, dropout, and class-weighted
`BCEWithLogitsLoss`, trained with Adam + a plateau LR scheduler. Features come
from `ml/features.py`, pulled straight out of the graph (vulnerability
counts/CVSS/exploitability, internet exposure, in/out-degree connectivity,
PageRank/betweenness centrality, admin ownership).

The training label is currently derived from the explainable heuristic in
`analysis/risk_score.py` (risk ≥ 60 → "high risk") — this makes the demo
runnable without real incident data. **In production, replace `build_labels()`
in `ml/features.py` with real historical compromise/incident/red-team data**
for the DNN to learn signal beyond what the heuristic already encodes.

## Using the fixed Colab notebook with this project

If you've been prototyping in the "CyberGraph AI" Colab notebook, here's how
it maps onto this codebase — most of what the notebook does, this project
already implements more robustly, and the two things it adds (NLP-embedding
severity features, semantic search) are now wired in below.

| Notebook piece | Project equivalent | Notes |
|---|---|---|
| Kaggle download + column-alias CSV loading | `ingestion/nvd_loader.py` | Already batched (`UNWIND`) and handles 337k rows; just run `python -m ingestion.nvd_loader --path data/nvd_cves.csv` instead of the notebook cells |
| Row-by-row `run_query` loops | `graph/connection.py::run_batch_write` | Same UNWIND-batching pattern, already used everywhere in `ingestion/` and `graph/nodes.py` |
| `is_safe_cypher` + `SECURITY_SYSTEM_PROMPT` | `llm/cypher_chain.py` | Already wired in with word-boundary-safe regex (`\bSET\b`, not substring match — the notebook's naive version would incorrectly block every query mentioning `Asset`) and `llama-3.3-70b-versatile` |
| Sentence-transformer embeddings + vector index | `ingestion/nvd_loader.py::embed_catalog` / `create_vector_index` + `ml/embeddings.py` | New — see below |
| DNN comparing structured-only vs. structured+NLP CVSS prediction | `ml/cve_severity_model.py::train_severity_model_comparison` | New — see below |

### Running the workflow end to end

```bash
# 1. Load the bulk CVE catalog (as before)
python -m ingestion.nvd_loader --path data/nvd_cves.csv --limit 20000   # drop --limit for the full 337k

# 2. Embed CVE descriptions for semantic search + NLP model features
python -m ingestion.nvd_loader --path data/nvd_cves.csv --embed --limit 20000
# (or, if step 1 already ran: POST /cve-catalog/embed)

# 3. Train the severity DNN, structured-only vs. structured+NLP, and compare
python -m ml.cve_severity_model --compare
# or via the API: POST /ml/cve-severity/train?compare=true
# returns both metric sets + rmse_improvement_pct, and persists whichever
# variant actually performs better on your data

# 4. Semantic search over the catalog
curl "http://localhost:8000/cve-catalog/semantic-search?q=remote+code+execution+in+internet-facing+web+servers&top_k=5"
```

If you still want to run the notebook itself (e.g. for its targeted,
keyword-grounded live NVD API pull that links real CVEs to the demo
assets' installed software), just point its Neo4j credentials at the same
instance as this project's `.env` — anything it writes shows up in the API
and dashboard immediately, no extra steps.

## Notes on the LLM layer

- `llm/cypher_chain.py` implements a hand-rolled, **read-only** text-to-Cypher
  QA chain: the LLM generates Cypher from the schema in `llm/prompts.py`, a
  regex guard blocks any write/mutation keywords before execution, the query
  runs against Neo4j, and the LLM turns the results back into a natural
  language answer.
- `llm/rag.py` retrieves structured context directly from the graph (attack
  paths + vulnerability details, or top critical assets) and asks the LLM to
  produce an executive-readable risk narrative — this is retrieval-augmented
  generation over graph query results rather than a vector store, since the
  underlying data is already structured and precise.
- Model defaults to `llama-3.3-70b-versatile` on Groq; override with the
  `GROQ_MODEL` env var.

## Extending to a React frontend

The FastAPI backend is already CORS-enabled and fully decoupled from the
Streamlit dashboard, so a `frontend/` React app can be dropped in and pointed
at the same `api/main.py` endpoints without any backend changes.

## Level 3 / Level 4 capabilities

Everything below sits under `/reports`, `/llm/graph-rag`, and `/research/*`
in the API. Scope is stated plainly where a technique is a heuristic rather
than a trained model -- see the honesty notes below before treating any of
this as more rigorous than it is.

### Security report generation (`reports/report_generator.py`)
`POST /reports/generate` produces an actual `.docx` (not just narrative
text in a JSON response): executive summary, top critical assets, attack
surface, internet→critical routes, and the full risk-score table. LLM
sections degrade gracefully to structured-data-only if `GROQ_API_KEY`
isn't set. `GET /reports/download/{filename}` serves the file.

### Hybrid GraphRAG (`llm/graph_rag.py`)
`llm/rag.py` already did retrieval-then-generation, but its retrieval was
pure Cypher (exact matches only). `POST /llm/graph-rag` adds real hybrid
retrieval: vector similarity search over embedded CVE descriptions finds
semantically relevant CVEs even without an exact ID match, then each match
is expanded through the graph to the assets it actually affects here.
Requires the catalog to be embedded first (`POST /cve-catalog/embed`).

### Graph embeddings (`analysis/graph_embeddings.py`)
`GET /research/graph-embeddings` -- structural embeddings of each Asset's
position in the network topology (FastRP via Neo4j GDS; falls back to
truncated SVD on the adjacency matrix, a legitimate spectral-embedding
method, if GDS isn't installed). Distinct from the CVE *text* embeddings
used for semantic search -- these encode graph structure, not language.

### GraphSAGE / GAT / GNN-based risk prediction (`ml/gnn_models.py`)
`POST /research/gnn/train` (`model_type: "sage"` or `"gat"`) trains a true
message-passing GNN on the same features/labels as `ml/train.py`'s
RiskDNN -- the difference is architecture: the GNN's prediction for an
asset is influenced by its neighbors' *learned* representations via graph
convolution, not by precomputed centrality scalars fed in as columns.
`POST /research/gnn/compare` trains GraphSAGE, GAT, and the plain DNN
side by side. `GET /research/gnn/explain/{asset_id}` (GAT only) returns
which neighbors' attention weights most drove a prediction -- a genuine,
model-native explainability signal the plain DNN can't offer.

**Honesty note:** on a small, sparse demo graph like the sample data here,
a GNN often does *not* outperform a plain DNN that already has centrality
scores as explicit features -- message passing earns its complexity on
larger, denser graphs. Use `/research/gnn/compare` to see which case your
actual graph is in rather than assuming the GNN wins.

### Temporal attack graphs (`analysis/temporal.py`)
Point-in-time reconstruction of the vulnerability landscape from
`discovered_date`/`patched_date` properties, plus an explicit `:ChangeEvent`
log for edits a bare timestamp can't capture. `GET
/research/temporal/vulnerabilities-at?timestamp=...`, `GET
/research/temporal/asset-risk-trend/{asset_id}`, `GET
/research/temporal/fleet-risk-trend`, `POST
/research/temporal/mark-patched`.

**Honesty note:** this is temporal *analysis over a versioned property
graph* (point-in-time queries + trend computation from timestamped
properties), not a bitemporal graph database and not a trained temporal
model.

### Dynamic / predictive attack-path analysis (`analysis/dynamic_paths.py`)
Two genuinely different capabilities, kept separate:
- **What-if simulation** (`GET /research/dynamic/simulate-patch/{vuln_id}`,
  `POST /research/dynamic/simulate-new-vulnerability`) -- exact,
  deterministic recomputation against a hypothetical change, entirely
  in-memory (never writes simulated data to the graph). Patching only
  reduces an asset's risk if that vulnerability was its highest-severity
  one -- the simulation makes that visible instead of assuming every patch
  helps equally. New-vulnerability impact propagates through the network
  with a geometric hop-distance decay (0.7/hop, a stated approximation,
  not a fitted parameter).
- **Emerging-risk forecasting** (`GET /research/dynamic/emerging-risk`) --
  a transparent statistical heuristic: recent vulnerability-discovery rate
  × structural centrality. **This is explicitly not a trained temporal
  model.** "Dynamic attack-path prediction" in the research literature
  usually means a temporal GNN (TGN/EvolveGCN/DySAT) trained on a labeled
  history of how attack paths actually materialized -- that requires a
  time-series of real compromise events to train against, which doesn't
  exist in this project (or in most real deployments). Building one and
  training it on synthetic labels would produce something that looks more
  sophisticated than it actually is. What's here is honest and useful:
  exact simulation for "what happens if X," and a stated heuristic for
  "where is risk trending" -- if you have real historical incident data,
  that's the actual missing ingredient for the trained-model version, not
  more model architecture.
