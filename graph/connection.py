"""
graph/connection.py
--------------------
Central Neo4j driver management. Provides a singleton-style connection
manager so the rest of the app doesn't need to know connection details.
"""

import os
import atexit
import logging
from contextlib import contextmanager
from typing import Optional, Any, Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

load_dotenv()

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Wraps a Neo4j driver and exposes convenience methods for running
    read/write Cypher queries with automatic session management."""

    _instance: Optional["Neo4jConnection"] = None

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver: Optional[Driver] = None

    def connect(self) -> "Neo4jConnection":
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                self._driver.verify_connectivity()
                logger.info("Connected to Neo4j at %s", self.uri)
            except (ServiceUnavailable, AuthError) as e:
                logger.error("Failed to connect to Neo4j: %s", e)
                raise
        return self

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as e:
                logger.debug("Error closing driver: %s", e)
            self._driver = None
        Neo4jConnection._instance = None

    @classmethod
    def get_instance(cls) -> "Neo4jConnection":
        """Return a process-wide singleton connection (lazy connect)."""
        if cls._instance is None:
            cls._instance = cls().connect()
        return cls._instance

    @contextmanager
    def session(self):
        if self._driver is None:
            self.connect()
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def run_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a read or write query and return results as a list of dicts."""
        parameters = parameters or {}
        with self.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def run_write(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a write query inside an explicit transaction."""
        parameters = parameters or {}

        def _tx(tx):
            result = tx.run(query, parameters)
            return [record.data() for record in result]

        with self.session() as session:
            return session.execute_write(_tx)

    def run_batch_write(
        self, query: str, rows: List[Dict[str, Any]], batch_size: int = 500
    ) -> None:
        """UNWIND-based batched write, e.g. for bulk CSV ingestion."""

        def _tx(tx, batch):
            tx.run(query, rows=batch)

        with self.session() as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                session.execute_write(_tx, batch)
        logger.info("Batch-wrote %d rows", len(rows))


def get_connection() -> Neo4jConnection:
    return Neo4jConnection.get_instance()


def _close_on_exit():
    if Neo4jConnection._instance is not None:
        Neo4jConnection._instance.close()


atexit.register(_close_on_exit)
