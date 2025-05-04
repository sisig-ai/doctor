"""Tests for the indexer module."""

import pytest
from unittest.mock import patch, MagicMock
import uuid

from src.lib.indexer import VectorIndexer


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    mock = MagicMock()
    # Mock the upsert method
    mock.upsert = MagicMock()
    # Mock the search method
    mock.search = MagicMock()
    return mock


@pytest.mark.unit
def test_vector_indexer_initialization():
    """Test VectorIndexer initialization."""
    # Test with default collection name
    with (
        patch("src.lib.indexer.get_qdrant_client") as mock_get_client,
        patch("src.lib.indexer.QDRANT_COLLECTION_NAME", "default_collection"),
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        indexer = VectorIndexer()

        # Check that the client was retrieved
        mock_get_client.assert_called_once()

        # Check that the indexer was initialized with the correct values
        assert indexer.client == mock_client
        assert indexer.collection_name == "default_collection"

    # Test with custom collection name
    with patch("src.lib.indexer.get_qdrant_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        indexer = VectorIndexer(collection_name="custom_collection")

        # Check that the indexer was initialized with the correct values
        assert indexer.client == mock_client
        assert indexer.collection_name == "custom_collection"


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_vector(sample_embedding, mock_qdrant_client):
    """Test indexing a single vector."""
    with (
        patch("src.lib.indexer.get_qdrant_client", return_value=mock_qdrant_client),
        patch(
            "src.lib.indexer.uuid.uuid4",
            return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        ),
    ):
        indexer = VectorIndexer(collection_name="test_collection")

        # Create a test payload
        payload = {"text": "Sample text", "metadata": {"source": "test"}}

        # Test with auto-generated ID
        point_id = await indexer.index_vector(sample_embedding, payload)

        # Check that upsert was called with the correct arguments
        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="test_collection",
            points=[
                {
                    "id": "12345678-1234-5678-1234-567812345678",
                    "vector": sample_embedding,
                    "payload": payload,
                }
            ],
        )

        # Check that we got the expected ID
        assert point_id == "12345678-1234-5678-1234-567812345678"

        # Test with provided ID
        mock_qdrant_client.upsert.reset_mock()
        provided_id = "custom-id-123"
        point_id = await indexer.index_vector(sample_embedding, payload, point_id=provided_id)

        # Check that upsert was called with the correct arguments
        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="test_collection",
            points=[{"id": provided_id, "vector": sample_embedding, "payload": payload}],
        )

        # Check that we got the expected ID
        assert point_id == provided_id


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_vector_with_empty_vector():
    """Test indexing an empty vector."""
    indexer = VectorIndexer()

    with pytest.raises(ValueError, match="Vector cannot be empty"):
        await indexer.index_vector([], {"text": "Sample text"})


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_vector_error_handling(sample_embedding, mock_qdrant_client):
    """Test error handling when indexing a vector."""
    with patch("src.lib.indexer.get_qdrant_client", return_value=mock_qdrant_client):
        # Simulate a client error
        mock_qdrant_client.upsert.side_effect = Exception("Client error")

        indexer = VectorIndexer()

        with pytest.raises(Exception, match="Client error"):
            await indexer.index_vector(sample_embedding, {"text": "Sample text"})


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_batch(sample_embedding, mock_qdrant_client):
    """Test indexing a batch of vectors."""
    with (
        patch("src.lib.indexer.get_qdrant_client", return_value=mock_qdrant_client),
        patch(
            "src.lib.indexer.uuid.uuid4",
            side_effect=[
                uuid.UUID("12345678-1234-5678-1234-567812345678"),
                uuid.UUID("87654321-8765-4321-8765-432187654321"),
                uuid.UUID("11111111-2222-3333-4444-555555555555"),
            ],
        ),
    ):
        indexer = VectorIndexer(collection_name="test_collection")

        # Create test vectors and payloads
        vectors = [
            sample_embedding,
            sample_embedding[::-1],  # Reverse the embedding for variety
            [0.5] * len(sample_embedding),
        ]

        payloads = [
            {"text": "Text 1", "metadata": {"source": "test1"}},
            {"text": "Text 2", "metadata": {"source": "test2"}},
            {"text": "Text 3", "metadata": {"source": "test3"}},
        ]

        # Test with auto-generated IDs
        point_ids = await indexer.index_batch(vectors, payloads)

        # Check that upsert was called with the correct arguments
        expected_points = [
            {
                "id": "12345678-1234-5678-1234-567812345678",
                "vector": vectors[0],
                "payload": payloads[0],
            },
            {
                "id": "87654321-8765-4321-8765-432187654321",
                "vector": vectors[1],
                "payload": payloads[1],
            },
            {
                "id": "11111111-2222-3333-4444-555555555555",
                "vector": vectors[2],
                "payload": payloads[2],
            },
        ]

        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="test_collection", points=expected_points
        )

        # Check that we got the expected IDs
        assert point_ids == [
            "12345678-1234-5678-1234-567812345678",
            "87654321-8765-4321-8765-432187654321",
            "11111111-2222-3333-4444-555555555555",
        ]

        # Test with provided IDs
        mock_qdrant_client.upsert.reset_mock()
        provided_ids = ["id1", "id2", "id3"]
        point_ids = await indexer.index_batch(vectors, payloads, point_ids=provided_ids)

        # Check that upsert was called with the correct arguments
        expected_points = [
            {"id": "id1", "vector": vectors[0], "payload": payloads[0]},
            {"id": "id2", "vector": vectors[1], "payload": payloads[1]},
            {"id": "id3", "vector": vectors[2], "payload": payloads[2]},
        ]

        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="test_collection", points=expected_points
        )

        # Check that we got the expected IDs
        assert point_ids == provided_ids


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_batch_validation():
    """Test validation when indexing a batch of vectors."""
    indexer = VectorIndexer()

    # Test with empty vectors
    with pytest.raises(ValueError, match="Vectors and payloads cannot be empty"):
        await indexer.index_batch([], [])

    # Test with mismatched vectors and payloads
    with pytest.raises(ValueError, match="Number of vectors must match number of payloads"):
        await indexer.index_batch([1, 2, 3], [{"a": 1}, {"b": 2}])

    # Test with mismatched point_ids
    with pytest.raises(ValueError, match="Number of point_ids must match number of vectors"):
        await indexer.index_batch(
            [1, 2, 3], [{"a": 1}, {"b": 2}, {"c": 3}], point_ids=["id1", "id2"]
        )


@pytest.mark.unit
@pytest.mark.async_test
async def test_search(sample_embedding, mock_qdrant_client):
    """Test searching for similar vectors."""
    with patch("src.lib.indexer.get_qdrant_client", return_value=mock_qdrant_client):
        # Mock search results
        search_results = [
            {"id": "id1", "score": 0.95, "payload": {"text": "Text 1"}},
            {"id": "id2", "score": 0.85, "payload": {"text": "Text 2"}},
            {"id": "id3", "score": 0.75, "payload": {"text": "Text 3"}},
        ]
        mock_qdrant_client.search.return_value = search_results

        indexer = VectorIndexer(collection_name="test_collection")

        # Test search with default parameters
        results = await indexer.search(sample_embedding)

        # Check that search was called with the correct arguments
        mock_qdrant_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=sample_embedding,
            limit=10,
            query_filter=None,
        )

        # Check that we got the expected results
        assert results == search_results

        # Test search with custom parameters
        mock_qdrant_client.search.reset_mock()
        filter_payload = {"metadata.source": "test"}
        results = await indexer.search(sample_embedding, limit=5, filter_payload=filter_payload)

        # Check that search was called with the correct arguments
        mock_qdrant_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=sample_embedding,
            limit=5,
            query_filter=filter_payload,
        )


@pytest.mark.unit
@pytest.mark.async_test
async def test_search_error_handling(sample_embedding, mock_qdrant_client):
    """Test error handling when searching for vectors."""
    with patch("src.lib.indexer.get_qdrant_client", return_value=mock_qdrant_client):
        # Simulate a client error
        mock_qdrant_client.search.side_effect = Exception("Search error")

        indexer = VectorIndexer()

        with pytest.raises(Exception, match="Search error"):
            await indexer.search(sample_embedding)
