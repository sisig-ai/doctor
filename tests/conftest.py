"""Common fixtures for tests."""

import pytest
import asyncio
import random
from typing import List

from src.common.config import VECTOR_SIZE


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_url():
    """Sample URL for testing."""
    return "https://example.com"


@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam auctor,
    nisl nec ultricies lacinia, nisl nisl aliquet nisl, nec ultricies nisl
    nisl nec ultricies lacinia, nisl nisl aliquet nisl, nec ultricies nisl
    nisl nec ultricies lacinia, nisl nisl aliquet nisl, nec ultricies nisl.

    Pellentesque habitant morbi tristique senectus et netus et malesuada
    fames ac turpis egestas. Sed euismod, nisl nec ultricies lacinia, nisl
    nisl aliquet nisl, nec ultricies nisl nisl nec ultricies lacinia.
    """


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing (legacy version: 384 dim)."""
    random.seed(42)  # For reproducibility
    return [random.random() for _ in range(384)]


@pytest.fixture
def sample_embedding_full_size() -> List[float]:
    """Sample embedding vector with the full VECTOR_SIZE dimension for DuckDB tests."""
    random.seed(42)  # For reproducibility
    return [random.random() for _ in range(VECTOR_SIZE)]


@pytest.fixture
def sample_crawl_result():
    """Sample crawl result for testing."""

    class MockCrawlResult:
        def __init__(self, url, markdown=None, extracted_content=None, html=None):
            self.url = url
            self._markdown = markdown
            self.extracted_content = extracted_content
            self.html = html

            # Create a mock _markdown attribute if provided
            if markdown:

                class MockMarkdown:
                    def __init__(self, raw_markdown):
                        self.raw_markdown = raw_markdown

                self._markdown = MockMarkdown(markdown)

    return MockCrawlResult(
        url="https://example.com",
        markdown="# Example Page\n\nThis is some example content.",
        extracted_content="Example Page. This is some example content.",
        html="<html><head><title>Example</title></head><body><h1>Example Page</h1><p>This is some example content.</p></body></html>",
    )


@pytest.fixture
def job_id():
    """Sample job ID for testing."""
    return "test-job-123"


@pytest.fixture
def page_id():
    """Sample page ID for testing."""
    return "test-page-456"


@pytest.fixture
def sample_tags():
    """Sample tags for testing."""
    return ["test", "example", "documentation"]


@pytest.fixture
def in_memory_duckdb_connection():
    """Create an in-memory DuckDB connection for testing.

    This connection has the proper setup for vector search:
    - VSS extension loaded
    - document_embeddings table created with the proper schema
    - HNSW index created
    - pages table created (for document service tests)

    Usage:
        def test_something(in_memory_duckdb_connection):
            # Use the connection for testing
            ...
    """
    import duckdb

    conn = duckdb.connect(":memory:")
    conn.execute("LOAD vss;")

    # Create document_embeddings table
    conn.execute(f"""
    CREATE TABLE document_embeddings (
        id BIGSERIAL PRIMARY KEY,
        embedding FLOAT4[{VECTOR_SIZE}] NOT NULL,
        text_chunk VARCHAR,
        page_id VARCHAR,
        url VARCHAR,
        domain VARCHAR,
        tags VARCHAR[],
        job_id VARCHAR
    );
    """)

    # Create HNSW index
    try:
        conn.execute("""
        CREATE INDEX hnsw_index_on_embeddings
        ON document_embeddings
        USING HNSW (embedding)
        WITH (metric = 'cosine');
        """)
    except Exception as e:
        # Ignore "already exists" errors
        if "already exists" not in str(e):
            raise

    # Create pages table for document service tests
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

    yield conn
    conn.close()
