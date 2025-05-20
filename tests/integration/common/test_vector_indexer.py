"""Integration tests for the VectorIndexer class with real DuckDB."""

import pytest

from src.common.config import VECTOR_SIZE
from src.common.indexer import VectorIndexer


@pytest.mark.integration
@pytest.mark.async_test
async def test_vectorindexer_with_real_duckdb(in_memory_duckdb_connection):
    """Test the DuckDB implementation of VectorIndexer with a real in-memory database."""
    # Create VectorIndexer with the in-memory test connection
    indexer = VectorIndexer(connection=in_memory_duckdb_connection)

    # Test vector with random values (correct dimension)
    test_vector = [0.1] * VECTOR_SIZE
    test_payload = {
        "text": "Test chunk",
        "page_id": "test-page-id",
        "url": "https://example.com",
        "domain": "example.com",
        "tags": ["test", "example"],
        "job_id": "test-job",
    }

    # Test index_vector
    point_id = await indexer.index_vector(test_vector, test_payload)
    assert point_id is not None

    # Test search - should find the vector we just indexed
    results = await indexer.search(test_vector, limit=1)
    assert len(results) == 1
    assert results[0]["id"] == point_id
    assert results[0]["payload"]["text"] == "Test chunk"
    assert results[0]["payload"]["tags"] == ["test", "example"]

    # Test tag filtering
    filter_payload = {"must": [{"key": "tags", "match": {"any": ["test"]}}]}

    results = await indexer.search(test_vector, limit=10, filter_payload=filter_payload)
    assert len(results) == 1

    # Test with non-matching tag filter
    filter_payload = {"must": [{"key": "tags", "match": {"any": ["nonexistent"]}}]}

    results = await indexer.search(test_vector, limit=10, filter_payload=filter_payload)
    assert len(results) == 0

    # Add another vector with different tags
    test_vector2 = [0.2] * VECTOR_SIZE
    test_payload2 = {
        "text": "Another test chunk",
        "page_id": "test-page-id-2",
        "url": "https://example.org",
        "domain": "example.org",
        "tags": ["different", "example"],
        "job_id": "test-job",
    }

    await indexer.index_vector(test_vector2, test_payload2)

    # Test filtering by the "example" tag which both vectors have
    filter_payload = {"must": [{"key": "tags", "match": {"any": ["example"]}}]}

    results = await indexer.search(test_vector, limit=10, filter_payload=filter_payload)
    assert len(results) == 2
