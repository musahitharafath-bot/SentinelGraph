"""
ingestion/asset_loader.py
--------------------------
Loads assets.csv, users.csv, and software.csv into Neo4j, including the
relationships that tie them together (OWNS, RUNS).
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict

from graph.nodes import create_assets, create_users, create_software
from graph.relationships import link_user_assets, link_asset_software, link_exposed_assets

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_assets(path: Path = DATA_DIR / "assets.csv") -> int:
    rows = _read_csv(path)
    create_assets(rows)
    logger.info("Loaded %d assets", len(rows))
    return len(rows)


def load_users(path: Path = DATA_DIR / "users.csv") -> int:
    rows = _read_csv(path)
    create_users(rows)
    link_user_assets(rows)  # each row has user_id + asset_id
    logger.info("Loaded %d users and OWNS relationships", len(rows))
    return len(rows)


def load_software(path: Path = DATA_DIR / "software.csv") -> int:
    rows = _read_csv(path)
    create_software(rows)
    link_asset_software(rows)  # each row has asset_id + software_id
    logger.info("Loaded %d software entries and RUNS relationships", len(rows))
    return len(rows)


def run_all() -> None:
    logging.basicConfig(level=logging.INFO)
    n_assets = load_assets()
    n_users = load_users()
    n_software = load_software()
    link_exposed_assets()
    print(f"Assets: {n_assets}, Users: {n_users}, Software: {n_software} — done.")


if __name__ == "__main__":
    run_all()
