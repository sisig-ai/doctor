"""Integration tests for the document service with real DuckDB."""

import pytest
from unittest.mock import patch

from src.common.config import VECTOR_SIZE
from src.common.indexer import VectorIndexer
from src.web_service.services.document_service import (
    search_docs,
    list_doc_pages,
    get_doc_page,
    list_tags,
)


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.requires_vss
async def test_search_docs_with_duckdb(in_memory_duckdb_connection):
    """Test searching documents with DuckDB backend."""
    # Create test data in the in-memory database
    indexer = VectorIndexer(connection=in_memory_duckdb_connection)

    # Create test vectors and payloads
    test_vector1 = [0.1] * VECTOR_SIZE
    test_payload1 = {
        "text": "This is a document about artificial intelligence",
        "page_id": "page1",
        "url": "https://example.com/ai",
        "domain": "example.com",
        "tags": ["ai", "tech"],
        "job_id": "job1",
    }

    test_vector2 = [0.2] * VECTOR_SIZE
    test_payload2 = {
        "text": "This is a document about machine learning",
        "page_id": "page2",
        "url": "https://example.com/ml",
        "domain": "example.com",
        "tags": ["ml", "tech"],
        "job_id": "job1",
    }

    # Index test data
    await indexer.index_vector(test_vector1, test_payload1)
    await indexer.index_vector(test_vector2, test_payload2)

    # Mock the embedding generation to return a vector similar to test_vector1
    with patch("src.lib.embedder.generate_embedding", return_value=[0.11] * VECTOR_SIZE):
        # Search with a query that should match better with vector1
        result = await search_docs(
            conn=in_memory_duckdb_connection,
            query="artificial intelligence",
            tags=None,
            max_results=5,
        )

        # Verify results
        assert len(result.results) == 2  # Should return both results
        assert "artificial intelligence" in result.results[0].chunk_text
        assert result.results[0].page_id == "page1"

        # Test with tag filter for "ai"
        result = await search_docs(
            conn=in_memory_duckdb_connection,
            query="artificial intelligence",
            tags=["ai"],
            max_results=5,
        )

        # Verify results
        assert len(result.results) == 1
        assert "artificial intelligence" in result.results[0].chunk_text
        assert result.results[0].page_id == "page1"
        assert "ai" in result.results[0].tags


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.requires_vss
async def test_list_doc_pages_with_duckdb(in_memory_duckdb_connection):
    """Test listing document pages with DuckDB backend."""
    # Insert test data directly into the pages table
    in_memory_duckdb_connection.execute("""
    INSERT INTO pages (id, url, domain, crawl_date, tags, raw_text)
    VALUES
        ('page1', 'https://example.com/page1', 'example.com', '2023-01-01',
         '["doc","example"]', 'This is page 1'),
        ('page2', 'https://example.com/page2', 'example.com', '2023-01-02',
         '["doc","test"]', 'This is page 2'),
        ('page3', 'https://example.org/page3', 'example.org', '2023-01-03',
         '["other","example"]', 'This is page 3')
    """)

    # Test with no filters
    result = await list_doc_pages(conn=in_memory_duckdb_connection, page=1, tags=None)

    # Verify results
    assert result.total_pages >= 1  # There is at least 1 page of results
    assert result.current_page == 1
    assert len(result.doc_pages) == 3

    # Test with tag filter
    result = await list_doc_pages(conn=in_memory_duckdb_connection, page=1, tags=["doc"])

    # Verify filtered results
    assert len(result.doc_pages) == 2
    assert all("doc" in page.tags for page in result.doc_pages)


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.requires_vss
async def test_get_doc_page_with_duckdb(in_memory_duckdb_connection):
    """Test retrieving a document page with DuckDB backend."""
    # Insert test data directly into the pages table
    in_memory_duckdb_connection.execute("""
    INSERT INTO pages (id, url, domain, crawl_date, tags, raw_text)
    VALUES ('test-page', 'https://example.com/test', 'example.com', '2023-01-01',
            '["test"]', 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5')
    """)

    # Test retrieving the entire page
    result = await get_doc_page(
        conn=in_memory_duckdb_connection, page_id="test-page", starting_line=1, ending_line=-1
    )

    # Verify results
    assert result.text == "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    assert result.total_lines == 5

    # Test retrieving specific lines
    result = await get_doc_page(
        conn=in_memory_duckdb_connection, page_id="test-page", starting_line=2, ending_line=4
    )

    # Verify partial results
    assert result.text == "Line 2\nLine 3\nLine 4"
    assert result.total_lines == 5


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.requires_vss
async def test_list_tags_with_duckdb(in_memory_duckdb_connection):
    """Test listing unique tags with DuckDB backend."""
    # Insert test data directly into the pages table with different tags
    in_memory_duckdb_connection.execute("""
    INSERT INTO pages (id, url, domain, crawl_date, tags, raw_text)
    VALUES
        ('page1', 'https://example.com/page1', 'example.com', '2023-01-01',
         '["tag1","common"]', 'Page 1'),
        ('page2', 'https://example.com/page2', 'example.com', '2023-01-02',
         '["tag2","common"]', 'Page 2'),
        ('page3', 'https://example.com/page3', 'example.com', '2023-01-03',
         '["tag3","special"]', 'Page 3')
    """)

    # Test listing all tags
    result = await list_tags(conn=in_memory_duckdb_connection, search_substring=None)

    # Verify results contain all unique tags
    assert len(result.tags) == 5
    assert set(result.tags) == {"tag1", "tag2", "tag3", "common", "special"}

    # Test with search substring
    result = await list_tags(conn=in_memory_duckdb_connection, search_substring="tag")

    # Verify filtered results
    assert len(result.tags) == 3
    assert all("tag" in tag for tag in result.tags)
