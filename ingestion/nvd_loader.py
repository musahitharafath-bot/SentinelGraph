"""
ingestion/nvd_loader.py
--------------------------
Loads the Kaggle "NVD Cybersecurity Dataset: 337k+ CVEs (1988-2026)" into
Neo4j as a standalone :Vulnerability catalog (:Vulnerability:CVECatalog),
separate from the small hand-built vulnerabilities.csv used for the demo
assets. This gives you a real, large-scale CVE knowledge base you can:

  1. Query/browse directly (trends, severity distribution, CWE breakdown).
  2. Link to your Asset inventory by CVE ID, wherever an asset's known
     CVEs (from vulnerabilities.csv, a scanner export, etc.) match a row
     here -- see `link_catalog_to_assets()`.
  3. Use as training data for a DNN that predicts CVSS base score from
     CVE characteristics -- see ml/cve_severity_model.py.

Because the source file is large (337k+ rows) and hasn't been inspected
directly (Kaggle isn't reachable from this environment), this loader is
defensive:
  - Column names are matched via aliases (case/whitespace-insensitive),
    covering the common variants seen across NVD-derived Kaggle exports.
  - Unmapped/unexpected columns are ignored rather than erroring.
  - Rows are streamed and written in large batches (default 5000) so this
    scales to hundreds of thousands of rows without loading the whole
    write path into memory at once (pandas read_csv chunking).

If your actual file's columns don't match the guesses below, run this
module's `inspect_columns()` first (or just open the CSV) and adjust
COLUMN_ALIASES -- everything else works unchanged.

Usage:
    python -m ingestion.nvd_loader --path data/nvd_cves.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd

from graph.connection import get_connection

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Canonical field -> list of possible source column names (checked case/space-insensitively)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "cve_id": ["cve_id", "cveid", "cve", "id"],
    "description": ["description", "desc", "summary", "cve_description"],
    "cvss_score": [
        "cvss_score", "cvss_base_score", "cvss_v3_base_score", "basescore",
        "base_score", "cvssv3_basescore", "cvss3_score", "score",
    ],
    "severity": ["severity", "cvss_severity", "base_severity", "cvss_v3_severity"],
    "attack_vector": ["attack_vector", "av", "cvss_attack_vector"],
    "attack_complexity": ["attack_complexity", "ac", "cvss_attack_complexity"],
    "privileges_required": ["privileges_required", "pr"],
    "user_interaction": ["user_interaction", "ui"],
    "scope": ["scope"],
    "confidentiality_impact": ["confidentiality_impact", "conf_impact", "c_impact"],
    "integrity_impact": ["integrity_impact", "i_impact"],
    "availability_impact": ["availability_impact", "a_impact"],
    "cwe_id": ["cwe_id", "cwe", "cwe_code"],
    "published_date": ["published_date", "publisheddate", "pub_date", "published"],
    "last_modified_date": ["last_modified_date", "lastmodifieddate", "modified_date"],
    "vendor": ["vendor", "vendor_name"],
    "product": ["product", "product_name"],
    "exploitability_score": ["exploitability_score", "exploitabilityscore"],
    "impact_score": ["impact_score", "impactscore"],
}


def inspect_columns(path: Path, n: int = 5) -> pd.DataFrame:
    """Quick peek at a large CSV's columns + a few sample rows without
    loading the whole file."""
    sample = pd.read_csv(path, nrows=n)
    print("Columns found:", list(sample.columns))
    return sample


def _build_rename_map(columns: List[str]) -> Dict[str, str]:
    normalized = {c: c.strip().lower().replace(" ", "_") for c in columns}
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        alias_set = {a.lower() for a in aliases}
        for orig_col, norm_col in normalized.items():
            if norm_col in alias_set:
                rename_map[orig_col] = canonical
                break
    return rename_map


MERGE_CVE_CATALOG = """
UNWIND $rows AS row
MERGE (v:Vulnerability:CVECatalog {cve_id: row.cve_id})
SET v.description = row.description,
    v.cvss_score = row.cvss_score,
    v.severity = row.severity,
    v.attack_vector = row.attack_vector,
    v.attack_complexity = row.attack_complexity,
    v.privileges_required = row.privileges_required,
    v.user_interaction = row.user_interaction,
    v.scope = row.scope,
    v.confidentiality_impact = row.confidentiality_impact,
    v.integrity_impact = row.integrity_impact,
    v.availability_impact = row.availability_impact,
    v.cwe_id = row.cwe_id,
    v.published_date = row.published_date,
    v.vendor = row.vendor,
    v.product = row.product,
    v.exploitability_score = row.exploitability_score,
    v.impact_score = row.impact_score
"""

LINK_CATALOG_TO_ASSETS = """
MATCH (a:Asset)-[:HAS_VULNERABILITY]->(v:Vulnerability)
WHERE v.cve_id IS NOT NULL AND NOT v:CVECatalog
MATCH (cat:CVECatalog {cve_id: v.cve_id})
MERGE (a)-[:HAS_VULNERABILITY]->(cat)
RETURN count(*) AS links_created
"""


def _clean_chunk(df: pd.DataFrame, rename_map: Dict[str, str]) -> List[dict]:
    df = df.rename(columns=rename_map)
    keep_cols = [c for c in COLUMN_ALIASES.keys() if c in df.columns]
    if "cve_id" not in keep_cols:
        raise ValueError(
            "Could not find a CVE ID column in this file. Run "
            "ingestion.nvd_loader.inspect_columns(path) and add the right "
            "alias to COLUMN_ALIASES['cve_id']."
        )
    df = df[keep_cols].copy()
    df = df.dropna(subset=["cve_id"])

    # numeric coercions -- keep as float/None, never raise on bad values
    for numeric_col in ["cvss_score", "exploitability_score", "impact_score"]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def load_nvd_catalog(
    path: Path,
    chunksize: int = 5000,
    limit_rows: Optional[int] = None,
) -> int:
    """Streams the (potentially large) NVD CSV into Neo4j in batches.

    Args:
        path: path to the downloaded Kaggle CSV.
        chunksize: rows per pandas chunk / Neo4j batch write.
        limit_rows: optional cap for a quick test run (e.g. 10000).
    """
    conn = get_connection()
    total_written = 0

    header = pd.read_csv(path, nrows=0)
    rename_map = _build_rename_map(list(header.columns))
    logger.info("Detected column mapping: %s", rename_map)

    reader = pd.read_csv(path, chunksize=chunksize)
    for i, chunk in enumerate(reader):
        rows = _clean_chunk(chunk, rename_map)
        if not rows:
            continue
        conn.run_batch_write(MERGE_CVE_CATALOG, rows, batch_size=len(rows))
        total_written += len(rows)
        logger.info("Ingested %d rows so far...", total_written)

        if limit_rows and total_written >= limit_rows:
            logger.info("Reached limit_rows=%d, stopping early.", limit_rows)
            break

    logger.info("NVD catalog ingestion complete: %d CVE records written.", total_written)
    return total_written


def link_catalog_to_assets() -> int:
    """After loading both the small demo vulnerabilities.csv (linked to
    Assets) and the big NVD catalog, this connects any Asset's existing
    CVE to the richer catalog record with the same cve_id, so asset-level
    queries can pull the full NVD fields (CWE, attack vector, etc.)."""
    result = get_connection().run_write(LINK_CATALOG_TO_ASSETS)
    count = result[0]["links_created"] if result else 0
    logger.info("Linked %d asset-vulnerability edges to the CVE catalog.", count)
    return count


def create_index() -> None:
    get_connection().run_write(
        "CREATE INDEX cve_catalog_cve_id_idx IF NOT EXISTS "
        "FOR (v:CVECatalog) ON (v.cve_id)"
    )
    get_connection().run_write(
        "CREATE INDEX cve_catalog_cvss_idx IF NOT EXISTS "
        "FOR (v:CVECatalog) ON (v.cvss_score)"
    )


# ------------------------------------------------------------ NLP embeddings

MERGE_EMBEDDING = """
UNWIND $rows AS row
MATCH (v:CVECatalog {cve_id: row.cve_id})
SET v.embedding = row.embedding
"""

Q_UNEMBEDDED = """
MATCH (v:CVECatalog)
WHERE v.description IS NOT NULL AND v.embedding IS NULL
RETURN v.cve_id AS cve_id, v.description AS description
"""


def create_vector_index() -> None:
    """Creates the Neo4j vector index used for semantic search over CVE
    descriptions (all-MiniLM-L6-v2, 384 dims, cosine similarity)."""
    from ml.embeddings import EMBEDDING_DIM

    get_connection().run_write(
        f"""
        CREATE VECTOR INDEX cve_catalog_embedding_index IF NOT EXISTS
        FOR (v:CVECatalog) ON v.embedding
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
    )


def embed_catalog(batch_size: int = 500, limit: Optional[int] = None) -> int:
    """Embeds any :CVECatalog node with a description but no embedding yet
    (so this is safe to re-run incrementally as new CVEs are ingested).
    Requires `sentence-transformers` -- see ml/embeddings.py."""
    from ml.embeddings import embed_texts

    conn = get_connection()
    rows = conn.run_query(Q_UNEMBEDDED)
    if limit:
        rows = rows[:limit]
    if not rows:
        logger.info("No un-embedded CVE catalog rows found.")
        return 0

    total = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        texts = [r["description"] for r in chunk]
        embeddings = embed_texts(texts)
        write_rows = [
            {"cve_id": r["cve_id"], "embedding": emb.tolist()}
            for r, emb in zip(chunk, embeddings)
        ]
        conn.run_batch_write(MERGE_EMBEDDING, write_rows, batch_size=len(write_rows))
        total += len(write_rows)
        logger.info("Embedded %d / %d CVE descriptions so far", total, len(rows))

    logger.info("Embedding complete: %d CVE catalog nodes updated.", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Load the NVD CVE dataset into Neo4j.")
    parser.add_argument("--path", type=str, default=str(DATA_DIR / "nvd_cves.csv"))
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for a test run")
    parser.add_argument("--inspect-only", action="store_true", help="Just print columns and exit")
    parser.add_argument(
        "--embed", action="store_true",
        help="After loading, also compute sentence-transformer embeddings for "
             "semantic search (requires sentence-transformers, see ml/embeddings.py)",
    )
    args = parser.parse_args()

    csv_path = Path(args.path)
    if not csv_path.exists():
        raise SystemExit(
            f"File not found: {csv_path}\n"
            f"Download the dataset from Kaggle and place it at this path, "
            f"or pass --path /your/download/location.csv"
        )

    if args.inspect_only:
        inspect_columns(csv_path)
        raise SystemExit(0)

    create_index()
    n = load_nvd_catalog(csv_path, chunksize=args.chunksize, limit_rows=args.limit)
    links = link_catalog_to_assets()
    print(f"Loaded {n} CVE records. Linked {links} asset-vulnerability edges to the catalog.")

    if args.embed:
        create_vector_index()
        n_embedded = embed_catalog(limit=args.limit)
        print(f"Embedded {n_embedded} CVE descriptions for semantic search.")
