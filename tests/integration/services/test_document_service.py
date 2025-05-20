"""Integration tests for the document service with real DuckDB."""

from unittest.mock import patch

import pytest

from src.common.config import VECTOR_SIZE
from src.common.indexer import VectorIndexer
from src.web_service.services.document_service import (
    get_doc_page,
    list_doc_pages,
    list_tags,
    search_docs,
)


@pytest.mark.integration
@pytest.mark.async_test
async def test_search_docs_with_duckdb(in_memory_duckdb_connection):
    """Test searching documents with DuckDB backend using hybrid search (vector and BM25)."""
    # Create test data in the in-memory database
    indexer = VectorIndexer(connection=in_memory_duckdb_connection)

    # Insert test pages with raw text for BM25 search
    in_memory_duckdb_connection.execute("""
    INSERT INTO pages (id, url, domain, crawl_date, tags, raw_text, job_id)
    VALUES
        ('page1', 'https://example.com/ai', 'example.com', '2023-01-01',
         '["ai", "tech"]', 'This is a document about artificial intelligence and machine learning.', 'job1'),
        ('page2', 'https://example.com/ml', 'example.com', '2023-01-02',
         '["ml", "tech"]', 'This is a document about machine learning algorithms and neural networks.', 'job1'),
        ('page3', 'https://example.com/nlp', 'example.com', '2023-01-03',
         '["nlp", "tech"]', 'Natural language processing is a field of artificial intelligence.', 'job1')
    """)

    # Try creating FTS index for BM25 search
    try:
        in_memory_duckdb_connection.execute("INSTALL fts; LOAD fts;")
        in_memory_duckdb_connection.execute("PRAGMA create_fts_index('pages', 'id', 'raw_text');")
    except Exception as e:
        print(f"Warning: Could not create FTS index: {e}. BM25 search may not work in tests.")

    # Create test vectors and payloads for vector search
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

    test_vector3 = [0.3] * VECTOR_SIZE
    test_payload3 = {
        "text": "Natural language processing is a field of artificial intelligence",
        "page_id": "page3",
        "url": "https://example.com/nlp",
        "domain": "example.com",
        "tags": ["nlp", "tech"],
        "job_id": "job1",
    }

    # Index test data for vector search
    await indexer.index_vector(test_vector1, test_payload1)
    await indexer.index_vector(test_vector2, test_payload2)
    await indexer.index_vector(test_vector3, test_payload3)

    # Mock the embedding generation to return a vector similar to test_vector1
    with patch("src.lib.embedder.generate_embedding", return_value=[0.11] * VECTOR_SIZE):
        # Test 1: Standard hybrid search with a query that should match via both vector and BM25
        result = await search_docs(
            conn=in_memory_duckdb_connection,
            query="artificial intelligence",
            tags=None,
            max_results=5,
        )

        # Verify results - should find page1 and page3 as they mention 'artificial intelligence'
        assert len(result.results) >= 2  # Should return at least 2 results
        # The first result should be about artificial intelligence
        assert any("artificial intelligence" in r.chunk_text for r in result.results)
        # Check page_ids - both page1 and page3 should be in results
        result_page_ids = [r.page_id for r in result.results]
        assert "page1" in result_page_ids
        assert "page3" in result_page_ids

        # Test 2: Using tag filter
        result = await search_docs(
            conn=in_memory_duckdb_connection,
            query="artificial intelligence",
            tags=["ai"],
            max_results=5,
        )

        # Verify filtered results
        assert any(r.page_id == "page1" for r in result.results)
        assert all("ai" in r.tags for r in result.results)

        # Test 3: Test with BM25-specific term not in vectors but in raw text
        result = await search_docs(
            conn=in_memory_duckdb_connection,
            query="neural networks",
            tags=None,
            max_results=5,
        )

        # Verify BM25 results - should find page2 which mentions 'neural networks'
        assert any(r.page_id == "page2" for r in result.results)
        assert any("neural networks" in r.chunk_text for r in result.results)

        # Test 4: Adjusting hybrid weights
        result = await search_docs(
            conn=in_memory_duckdb_connection,
            query="artificial intelligence",
            tags=None,
            max_results=5,
            hybrid_weight=0.3,  # More weight on BM25 (0.7) than vector (0.3)
        )

        # With more weight on BM25, we expect different result ordering
        # Check that the results are not empty
        assert len(result.results) > 0


@pytest.mark.integration
@pytest.mark.async_test
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
        conn=in_memory_duckdb_connection,
        page_id="test-page",
        starting_line=1,
        ending_line=-1,
    )

    # Verify results
    assert result.text == "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    assert result.total_lines == 5

    # Test retrieving specific lines
    result = await get_doc_page(
        conn=in_memory_duckdb_connection,
        page_id="test-page",
        starting_line=2,
        ending_line=4,
    )

    # Verify partial results
    assert result.text == "Line 2\nLine 3\nLine 4"
    assert result.total_lines == 5


@pytest.mark.integration
@pytest.mark.async_test
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
