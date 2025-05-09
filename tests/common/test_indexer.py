"""Tests for the indexer module."""

import pytest
from unittest.mock import patch, MagicMock
import uuid
import duckdb

from src.common.indexer import VectorIndexer
from src.common.config import VECTOR_SIZE


@pytest.fixture
def mock_duckdb_connection():
    """Mock DuckDB connection."""
    mock = MagicMock(spec=duckdb.DuckDBPyConnection)
    # Mock the execute method
    mock.execute = MagicMock()
    # Mock the execute method for search results
    mock.execute.return_value.fetchall = MagicMock(return_value=[])
    # Mock the executemany method
    mock.executemany = MagicMock()
    return mock


@pytest.mark.unit
def test_vector_indexer_initialization(mock_duckdb_connection):
    """Test VectorIndexer initialization."""
    # Test with default table name
    with patch("src.common.indexer.get_duckdb_connection", return_value=mock_duckdb_connection):
        indexer = VectorIndexer()

        # Check that the connection was retrieved
        assert indexer.conn == mock_duckdb_connection

        # Check that VSS extension was loaded
        mock_duckdb_connection.execute.assert_called_with("LOAD vss;")

    # Test with custom table name
    with patch("src.common.indexer.get_duckdb_connection", return_value=mock_duckdb_connection):
        mock_duckdb_connection.execute.reset_mock()
        indexer = VectorIndexer(table_name="custom_table")

        # Check that the indexer was initialized with the correct table name
        assert indexer.table_name == "custom_table"

        # Check that VSS extension was loaded
        mock_duckdb_connection.execute.assert_called_with("LOAD vss;")

    # Test with provided connection
    mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)
    indexer = VectorIndexer(connection=mock_conn)

    # Check that the provided connection was used
    assert indexer.conn == mock_conn
    assert indexer._own_connection is False


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_vector(mock_duckdb_connection):
    """Test indexing a single vector."""
    # Create a full-size sample embedding
    sample_embedding = [0.1] * VECTOR_SIZE

    with (
        patch("src.common.indexer.get_duckdb_connection", return_value=mock_duckdb_connection),
        patch(
            "src.common.indexer.uuid.uuid4",
            return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        ),
    ):
        indexer = VectorIndexer(table_name="test_table")

        # Create a test payload
        payload = {
            "text": "Sample text",
            "page_id": "page-123",
            "url": "https://example.com",
            "domain": "example.com",
            "tags": ["tag1", "tag2"],
            "job_id": "job-123",
        }

        # Reset mock because of the LOAD vss; call in the constructor
        mock_duckdb_connection.execute.reset_mock()

        # Test with auto-generated ID
        point_id = await indexer.index_vector(sample_embedding, payload)

        # Check that execute was called with the correct arguments
        mock_duckdb_connection.execute.assert_called_once()
        call_args = mock_duckdb_connection.execute.call_args
        assert "INSERT INTO test_table" in call_args[0][0]
        assert call_args[0][1][0] == "12345678-1234-5678-1234-567812345678"
        assert call_args[0][1][1] == sample_embedding
        assert call_args[0][1][2] == "Sample text"
        assert call_args[0][1][3] == "page-123"

        # Check that we got the expected ID
        assert point_id == "12345678-1234-5678-1234-567812345678"

        # Test with provided ID
        mock_duckdb_connection.execute.reset_mock()
        provided_id = "custom-id-123"
        point_id = await indexer.index_vector(sample_embedding, payload, point_id=provided_id)

        # Check that execute was called with the correct arguments
        mock_duckdb_connection.execute.assert_called_once()
        call_args = mock_duckdb_connection.execute.call_args
        assert "INSERT INTO test_table" in call_args[0][0]
        assert call_args[0][1][0] == provided_id
        assert call_args[0][1][1] == sample_embedding

        # Check that we got the expected ID
        assert point_id == provided_id


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_vector_with_empty_vector():
    """Test indexing an empty vector."""
    mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)

    indexer = VectorIndexer(connection=mock_conn)

    # Clear the mock calls that happened during initialization
    mock_conn.execute.reset_mock()

    with pytest.raises(ValueError, match="Vector cannot be empty"):
        await indexer.index_vector([], {"text": "Sample text"})


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_vector_with_wrong_dimension():
    """Test indexing a vector with incorrect dimensions."""
    mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)

    indexer = VectorIndexer(connection=mock_conn)

    # Clear the mock calls that happened during initialization
    mock_conn.execute.reset_mock()

    with pytest.raises(ValueError, match=f"Vector dimension must be {VECTOR_SIZE}"):
        await indexer.index_vector([0.1, 0.2, 0.3], {"text": "Sample text"})


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_vector_error_handling(mock_duckdb_connection):
    """Test error handling when indexing a vector."""
    sample_embedding = [0.1] * VECTOR_SIZE

    with patch("src.common.indexer.get_duckdb_connection", return_value=mock_duckdb_connection):
        # We need to handle the LOAD vss call first, then simulate an error on the INSERT
        mock_duckdb_connection.execute.side_effect = [None, Exception("Database error")]

        indexer = VectorIndexer()

        with pytest.raises(Exception, match="Database error"):
            await indexer.index_vector(sample_embedding, {"text": "Sample text"})


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_batch(mock_duckdb_connection):
    """Test indexing a batch of vectors."""
    # Create full-size sample embeddings
    sample_embedding = [0.1] * VECTOR_SIZE

    with (
        patch("src.common.indexer.get_duckdb_connection", return_value=mock_duckdb_connection),
        patch(
            "src.common.indexer.uuid.uuid4",
            side_effect=[
                uuid.UUID("12345678-1234-5678-1234-567812345678"),
                uuid.UUID("87654321-8765-4321-8765-432187654321"),
                uuid.UUID("11111111-2222-3333-4444-555555555555"),
            ],
        ),
    ):
        indexer = VectorIndexer(table_name="test_table")

        # Reset the mock after initialization
        mock_duckdb_connection.execute.reset_mock()
        mock_duckdb_connection.executemany.reset_mock()

        # Create test vectors and payloads
        vectors = [
            sample_embedding.copy(),
            sample_embedding.copy(),
            sample_embedding.copy(),
        ]

        payloads = [
            {
                "text": "Text 1",
                "page_id": "page-1",
                "url": "https://example.com/1",
                "domain": "example.com",
                "tags": ["tag1"],
                "job_id": "job-1",
            },
            {
                "text": "Text 2",
                "page_id": "page-2",
                "url": "https://example.com/2",
                "domain": "example.com",
                "tags": ["tag2"],
                "job_id": "job-2",
            },
            {
                "text": "Text 3",
                "page_id": "page-3",
                "url": "https://example.com/3",
                "domain": "example.com",
                "tags": ["tag3"],
                "job_id": "job-3",
            },
        ]

        # Test with auto-generated IDs
        point_ids = await indexer.index_batch(vectors, payloads)

        # Check that executemany was called with the correct arguments
        expected_data = [
            (
                "12345678-1234-5678-1234-567812345678",
                vectors[0],
                "Text 1",
                "page-1",
                "https://example.com/1",
                "example.com",
                ["tag1"],
                "job-1",
            ),
            (
                "87654321-8765-4321-8765-432187654321",
                vectors[1],
                "Text 2",
                "page-2",
                "https://example.com/2",
                "example.com",
                ["tag2"],
                "job-2",
            ),
            (
                "11111111-2222-3333-4444-555555555555",
                vectors[2],
                "Text 3",
                "page-3",
                "https://example.com/3",
                "example.com",
                ["tag3"],
                "job-3",
            ),
        ]

        mock_duckdb_connection.executemany.assert_called_once()
        call_args = mock_duckdb_connection.executemany.call_args
        assert "INSERT INTO test_table" in call_args[0][0]
        assert call_args[0][1] == expected_data

        # Check that we got the expected IDs
        assert point_ids == [
            "12345678-1234-5678-1234-567812345678",
            "87654321-8765-4321-8765-432187654321",
            "11111111-2222-3333-4444-555555555555",
        ]

        # Test with provided IDs
        mock_duckdb_connection.executemany.reset_mock()
        provided_ids = ["id1", "id2", "id3"]
        point_ids = await indexer.index_batch(vectors, payloads, point_ids=provided_ids)

        # Check that executemany was called with the correct arguments
        expected_data = [
            (
                "id1",
                vectors[0],
                "Text 1",
                "page-1",
                "https://example.com/1",
                "example.com",
                ["tag1"],
                "job-1",
            ),
            (
                "id2",
                vectors[1],
                "Text 2",
                "page-2",
                "https://example.com/2",
                "example.com",
                ["tag2"],
                "job-2",
            ),
            (
                "id3",
                vectors[2],
                "Text 3",
                "page-3",
                "https://example.com/3",
                "example.com",
                ["tag3"],
                "job-3",
            ),
        ]

        mock_duckdb_connection.executemany.assert_called_once()
        call_args = mock_duckdb_connection.executemany.call_args
        assert "INSERT INTO test_table" in call_args[0][0]
        assert call_args[0][1] == expected_data

        # Check that we got the expected IDs
        assert point_ids == provided_ids


@pytest.mark.unit
@pytest.mark.async_test
async def test_index_batch_validation():
    """Test validation when indexing a batch of vectors."""
    mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)
    indexer = VectorIndexer(connection=mock_conn)

    # Clear the mock calls that happened during initialization
    mock_conn.execute.reset_mock()
    mock_conn.executemany.reset_mock()

    # Test with empty vectors
    with pytest.raises(ValueError, match="Vectors and payloads cannot be empty"):
        await indexer.index_batch([], [])

    # Test with mismatched vectors and payloads
    with pytest.raises(ValueError, match="Number of vectors must match number of payloads"):
        await indexer.index_batch(
            [[0.1] * VECTOR_SIZE, [0.2] * VECTOR_SIZE], [{"text": "Sample text"}]
        )

    # Test with mismatched point_ids
    with pytest.raises(ValueError, match="Number of point_ids must match number of vectors"):
        await indexer.index_batch(
            [[0.1] * VECTOR_SIZE, [0.2] * VECTOR_SIZE],
            [{"text": "Text 1"}, {"text": "Text 2"}],
            point_ids=["id1"],
        )

    # Test with incorrect vector dimensions
    with pytest.raises(ValueError, match=f"Vector dimension must be {VECTOR_SIZE}"):
        await indexer.index_batch(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [{"text": "Text 1"}, {"text": "Text 2"}],
        )


@pytest.mark.unit
@pytest.mark.async_test
async def test_search(mock_duckdb_connection):
    """Test searching for similar vectors."""
    # Create a full-size sample embedding for querying
    query_vector = [0.1] * VECTOR_SIZE

    # Mock search results from DuckDB
    mock_search_results = [
        (
            "id1",
            "Text 1",
            "page-1",
            "https://example.com/1",
            "example.com",
            ["tag1", "tag2"],
            "job-1",
            0.1,
        ),
        (
            "id2",
            "Text 2",
            "page-2",
            "https://example.com/2",
            "example.com",
            ["tag2", "tag3"],
            "job-2",
            0.2,
        ),
        (
            "id3",
            "Text 3",
            "page-3",
            "https://example.com/3",
            "example.com",
            ["tag3", "tag4"],
            "job-3",
            0.3,
        ),
    ]

    with patch("src.common.indexer.get_duckdb_connection", return_value=mock_duckdb_connection):
        # Set up mock response for search query
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_search_results

        # Handle the LOAD vss; call first, then return our mock cursor
        mock_duckdb_connection.execute.side_effect = [None, mock_cursor]

        indexer = VectorIndexer(table_name="test_table")

        # Reset mocks after initialization
        mock_duckdb_connection.execute.reset_mock()
        mock_duckdb_connection.execute.side_effect = None
        mock_duckdb_connection.execute.return_value = mock_cursor

        # Test search with default parameters
        results = await indexer.search(query_vector)

        # Check that execute was called with the correct SQL query structure
        mock_duckdb_connection.execute.assert_called_once()
        call_args = mock_duckdb_connection.execute.call_args
        assert "SELECT" in call_args[0][0]
        assert "FROM test_table" in call_args[0][0]
        assert "array_cosine_distance" in call_args[0][0]
        assert "ORDER BY cosine_distance ASC LIMIT ?" in call_args[0][0]
        assert call_args[0][1][0] == query_vector  # First param is the query vector
        assert call_args[0][1][1] == 10  # Second param is the limit

        # Check that we got the expected results
        assert len(results) == 3

        # Check the structure of the results
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.9  # 1.0 - 0.1
        assert results[0]["payload"]["text"] == "Text 1"
        assert results[0]["payload"]["page_id"] == "page-1"
        assert results[0]["payload"]["tags"] == ["tag1", "tag2"]

        # Test search with custom limit
        mock_duckdb_connection.execute.reset_mock()
        mock_duckdb_connection.execute.return_value = mock_cursor
        results = await indexer.search(query_vector, limit=5)

        # Check the limit parameter
        call_args = mock_duckdb_connection.execute.call_args
        assert call_args[0][1][1] == 5

        # Test search with tag filter
        mock_duckdb_connection.execute.reset_mock()
        mock_duckdb_connection.execute.return_value = mock_cursor

        filter_payload = {"must": [{"key": "tags", "match": {"any": ["tag1", "tag2"]}}]}

        results = await indexer.search(query_vector, filter_payload=filter_payload)

        # Check that the WHERE clause was included
        call_args = mock_duckdb_connection.execute.call_args
        assert "WHERE array_has_any(tags, ?::VARCHAR[])" in call_args[0][0]
        assert call_args[0][1][1] == ["tag1", "tag2"]  # Second param is the tag list
        assert call_args[0][1][2] == 10  # Third param is the limit


@pytest.mark.unit
@pytest.mark.async_test
async def test_search_error_handling(mock_duckdb_connection):
    """Test error handling when searching for vectors."""
    query_vector = [0.1] * VECTOR_SIZE

    with patch("src.common.indexer.get_duckdb_connection", return_value=mock_duckdb_connection):
        # First handle the LOAD vss; call, then simulate an error on search
        mock_duckdb_connection.execute.side_effect = [None, Exception("Search error")]

        indexer = VectorIndexer()

        with pytest.raises(Exception, match="Search error"):
            await indexer.search(query_vector)


@pytest.mark.unit
@pytest.mark.skip(reason="Requires actual DuckDB with VSS extension")
@pytest.mark.async_test
async def test_vectorindexer_with_real_duckdb(in_memory_duckdb_connection):
    """Test the DuckDB implementation of VectorIndexer with a real in-memory database.

    This test is skipped by default as it requires the actual DuckDB with the VSS extension.
    """
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
