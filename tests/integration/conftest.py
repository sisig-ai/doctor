"""Fixtures for integration tests."""

import pytest
import duckdb
from src.common.config import VECTOR_SIZE

# Flag to track if VSS is available
VSS_AVAILABLE = True


@pytest.fixture
def in_memory_duckdb_connection():
    """Create an in-memory DuckDB connection for integration testing.

    This connection has the proper setup for vector search:
    - VSS extension loaded
    - document_embeddings table created with the proper schema
    - HNSW index created
    - pages table created (for document service tests)
    """
    global VSS_AVAILABLE
    conn = duckdb.connect(":memory:")

    # First install the VSS extension, then load it
    try:
        conn.execute("INSTALL vss;")
        conn.execute("LOAD vss;")
    except Exception as e:
        # If installation or loading fails, print a helpful message
        print(f"Warning: Could not load VSS extension: {e}")
        print("These tests require DuckDB with Vector Search Support.")
        print("Some tests may be skipped or fail.")
        VSS_AVAILABLE = False

    # Create document_embeddings table
    try:
        conn.execute(f"""
        CREATE TABLE document_embeddings (
            id VARCHAR PRIMARY KEY,
            embedding FLOAT4[{VECTOR_SIZE}] NOT NULL,
            text_chunk VARCHAR,
            page_id VARCHAR,
            url VARCHAR,
            domain VARCHAR,
            tags VARCHAR[],
            job_id VARCHAR
        );
        """)
    except Exception as e:
        print(f"Warning: Could not create document_embeddings table: {e}")
        VSS_AVAILABLE = False

    # Create HNSW index
    try:
        conn.execute("""
        CREATE INDEX hnsw_index_on_embeddings
        ON document_embeddings
        USING HNSW (embedding)
        WITH (metric = 'cosine');
        """)
    except Exception as e:
        # Ignore "already exists" errors and log other errors but continue
        if "already exists" not in str(e):
            print(f"Warning: Could not create HNSW index: {e}")
            print("Tests will continue but vector search may not work correctly")

    # Create pages table for document service tests
    try:
        conn.execute("""
        CREATE TABLE pages (
            id VARCHAR PRIMARY KEY,
            url VARCHAR,
            domain VARCHAR,
            crawl_date VARCHAR,
            tags VARCHAR,
            raw_text VARCHAR
        );
        """)
    except Exception as e:
        print(f"Warning: Could not create pages table: {e}")
        VSS_AVAILABLE = False

    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def skip_if_no_vss(request):
    """Skip tests that require VSS if it's not available."""
    if request.node.get_closest_marker("requires_vss") and not VSS_AVAILABLE:
        pytest.skip("Test requires DuckDB with VSS extension")
