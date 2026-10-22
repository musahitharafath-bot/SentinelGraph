"""
ingestion/network_loader.py
------------------------------
Loads network.csv (CONNECTS_TO edges between assets) into Neo4j, and
exposes `run_full_ingestion()` which orchestrates the entire pipeline:
assets -> vulnerabilities -> network -> derived (EXPOSED_TO) edges.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict

from graph.schema import initialize_schema
from graph.relationships import link_network, link_exposed_assets
from ingestion.asset_loader import load_assets, load_users, load_software
from ingestion.vulnerability_loader import load_vulnerabilities

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_network(path: Path = DATA_DIR / "network.csv") -> int:
    rows = _read_csv(path)
    link_network(rows)
    logger.info("Loaded %d CONNECTS_TO relationships", len(rows))
    return len(rows)


def run_full_ingestion() -> Dict[str, int]:
    """End-to-end pipeline: schema -> assets/users/software -> vulns -> network."""
    logging.basicConfig(level=logging.INFO)
    initialize_schema()

    counts = {}
    counts["assets"] = load_assets()
    counts["users"] = load_users()
    counts["software"] = load_software()
    counts["vulnerabilities"] = load_vulnerabilities()
    counts["network_edges"] = load_network()
    link_exposed_assets()

    logger.info("Full ingestion complete: %s", counts)
    return counts


from graph.connection import get_connection

if __name__ == "__main__":
    try:
        result = run_full_ingestion()
        print("Ingestion summary:", result)
    finally:
        get_connection().close()
