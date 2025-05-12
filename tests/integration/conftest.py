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
    from src.common.db_setup import ensure_duckdb_tables, ensure_duckdb_vss_extension

    conn = duckdb.connect(":memory:")

    # Use the same setup functions as the main application
    try:
        # Create base tables first
        ensure_duckdb_tables(conn)

        # Set up VSS extension and embeddings tables
        ensure_duckdb_vss_extension(conn)

        yield conn
    finally:
        conn.close()
