"""Fixtures for integration tests."""

import pytest
import duckdb


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
    from src.lib.database import Database

    conn = duckdb.connect(":memory:")

    # Use the same setup functions as the main application
    try:
        # Create a Database instance and initialize with our in-memory connection
        db = Database()
        db.conn = conn

        # Create base tables first
        db.ensure_tables()

        # Set up VSS extension and embeddings tables
        db.ensure_vss_extension()

        yield conn
    finally:
        conn.close()
