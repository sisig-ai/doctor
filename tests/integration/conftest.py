"""Fixtures for integration tests."""

import duckdb
import pytest


@pytest.fixture
def in_memory_duckdb_connection():
    """Create an in-memory DuckDB connection for integration testing.

    This connection has the proper setup for vector search using the same
    setup logic as the main application:
    - VSS extension loaded
    - document_embeddings table created with the proper schema
    - HNSW index created
    - pages table created (for document service tests)
    """
    from src.lib.database import DatabaseOperations

    conn = duckdb.connect(":memory:")

    # Use the same setup functions as the main application
    try:
        # Create a Database instance and initialize with our in-memory connection
        db = DatabaseOperations()
        db.db.conn = conn

        # Initialize all tables and extensions
        db.db.initialize()

        yield conn
    finally:
        conn.close()
