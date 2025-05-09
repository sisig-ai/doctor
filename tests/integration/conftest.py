"""Fixtures for integration tests."""

import pytest
import duckdb

# Flag to track if VSS is available
VSS_AVAILABLE = True


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
    global VSS_AVAILABLE
    from src.common.db_setup import ensure_duckdb_tables, ensure_duckdb_vss_extension

    conn = duckdb.connect(":memory:")

    # Use the same setup functions as the main application
    try:
        # Create base tables first
        ensure_duckdb_tables(conn)

        # Try to set up VSS extension and embeddings tables
        try:
            ensure_duckdb_vss_extension(conn)
        except Exception as e:
            print(f"Warning: Could not set up VSS extension: {e}")
            print("These tests require DuckDB with Vector Search Support.")
            print("Some tests may be skipped or fail.")
            VSS_AVAILABLE = False

        yield conn
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def skip_if_no_vss(request):
    """Skip tests that require VSS if it's not available."""
    if request.node.get_closest_marker("requires_vss") and not VSS_AVAILABLE:
        pytest.skip("Test requires DuckDB with VSS extension")
