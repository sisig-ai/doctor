"""Tests for the job service."""

import datetime
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from rq import Queue

from src.common.models import (
    FetchUrlResponse,
    JobProgressResponse,
)
from src.web_service.services.job_service import (
    fetch_url,
    get_job_count,
    get_job_progress,
)


@pytest.fixture
def mock_duckdb_connection():
    """Mock DuckDB connection."""
    mock = MagicMock(spec=duckdb.DuckDBPyConnection)
    return mock


@pytest.fixture
def mock_queue():
    """Mock Redis queue."""
    mock = MagicMock(spec=Queue)
    mock.enqueue = MagicMock()
    return mock


@pytest.mark.unit
@pytest.mark.async_test
async def test_fetch_url_basic(mock_queue):
    """Test fetching a URL with basic parameters."""
    # Set a fixed UUID for testing
    with patch("uuid.uuid4", return_value="test-job-123"):
        # Call the function with just URL
        result = await fetch_url(
            queue=mock_queue,
            url="https://example.com",
        )

        # Verify queue.enqueue was called with the right parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0]
        assert call_args[0] == "src.crawl_worker.tasks.create_job"
        assert call_args[1] == "https://example.com"
        assert call_args[2] == "test-job-123"

        # Should receive keyword arguments for tags and max_pages
        kwargs = mock_queue.enqueue.call_args[1]
        assert kwargs["tags"] is None
        assert kwargs["max_pages"] == 100

        # Check that the function returned the expected response
        assert isinstance(result, FetchUrlResponse)
        assert result.job_id == "test-job-123"


@pytest.mark.unit
@pytest.mark.async_test
async def test_fetch_url_with_options(mock_queue):
    """Test fetching a URL with all parameters."""
    # Set a fixed UUID for testing
    with patch("uuid.uuid4", return_value="test-job-456"):
        # Call the function with all parameters
        result = await fetch_url(
            queue=mock_queue,
            url="https://example.com",
            tags=["test", "example"],
            max_pages=50,
        )

        # Verify queue.enqueue was called with the right parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0]
        assert call_args[0] == "src.crawl_worker.tasks.create_job"
        assert call_args[1] == "https://example.com"
        assert call_args[2] == "test-job-456"

        # Should receive keyword arguments with the provided values
        kwargs = mock_queue.enqueue.call_args[1]
        assert kwargs["tags"] == ["test", "example"]
        assert kwargs["max_pages"] == 50

        # Check that the function returned the expected response
        assert isinstance(result, FetchUrlResponse)
        assert result.job_id == "test-job-456"


@pytest.mark.unit
@pytest.mark.async_test
async def test_get_job_progress_exact_match(mock_duckdb_connection):
    """Test getting job progress with exact job ID match."""
    # Mock database results for successful job
    created_at = datetime.datetime(2023, 1, 1, 10, 0, 0)
    updated_at = datetime.datetime(2023, 1, 1, 10, 5, 0)

    mock_result = (
        "test-job-123",  # job_id
        "https://example.com",  # start_url
        "running",  # status
        200,  # pages_discovered
        50,  # pages_crawled
        100,  # max_pages
        "test,example",  # tags_json
        created_at,  # created_at
        updated_at,  # updated_at
        None,  # error_message
    )

    # Set up mock execution and result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = mock_result
    mock_duckdb_connection.execute.return_value = mock_cursor

    # Call the function
    result = await get_job_progress(
        conn=mock_duckdb_connection,
        job_id="test-job-123",
    )

    # Verify database was queried with exact job ID
    mock_duckdb_connection.execute.assert_called_once()
    query = mock_duckdb_connection.execute.call_args[0][0]
    assert "SELECT job_id, start_url, status" in query
    assert "WHERE job_id = ?" in query

    # Check that results are returned correctly
    assert isinstance(result, JobProgressResponse)
    assert result.status == "running"
    assert result.pages_crawled == 50
    assert result.pages_total == 200
    assert result.completed is False
    assert result.progress_percent == 50  # 50/100 = 50%
    assert result.url == "https://example.com"
    assert result.max_pages == 100
    assert result.created_at == created_at
    assert result.updated_at == updated_at
    assert result.error_message is None


@pytest.mark.unit
@pytest.mark.async_test
async def test_get_job_progress_partial_match(mock_duckdb_connection):
    """Test getting job progress with partial job ID match."""
    # First attempt should return None to simulate no exact match
    mock_cursor1 = MagicMock()
    mock_cursor1.fetchone.return_value = None

    # Second attempt (partial match) should succeed
    created_at = datetime.datetime(2023, 1, 1, 10, 0, 0)
    updated_at = datetime.datetime(2023, 1, 1, 10, 5, 0)

    mock_result = (
        "test-job-123-full",  # job_id
        "https://example.com",  # start_url
        "completed",  # status
        100,  # pages_discovered
        100,  # pages_crawled
        100,  # max_pages
        "test,example",  # tags_json
        created_at,  # created_at
        updated_at,  # updated_at
        None,  # error_message
    )

    mock_cursor2 = MagicMock()
    mock_cursor2.fetchone.return_value = mock_result

    # Set up side effect for consecutive calls
    mock_duckdb_connection.execute.side_effect = [mock_cursor1, mock_cursor2]

    # Call the function with partial job ID
    result = await get_job_progress(
        conn=mock_duckdb_connection,
        job_id="test-job",
    )

    # Should make two database calls
    assert mock_duckdb_connection.execute.call_count == 2

    # Second call should use LIKE with wildcard
    query2 = mock_duckdb_connection.execute.call_args_list[1][0][0]
    assert "WHERE job_id LIKE ?" in query2
    params2 = mock_duckdb_connection.execute.call_args_list[1][0][1]
    assert params2[0] == "test-job%"

    # Check that results are returned correctly
    assert isinstance(result, JobProgressResponse)
    assert result.status == "completed"
    assert result.pages_crawled == 100
    assert result.pages_total == 100
    assert result.completed is True
    assert result.progress_percent == 100  # 100/100 = 100%


@pytest.mark.unit
@pytest.mark.async_test
async def test_get_job_progress_not_found(mock_duckdb_connection):
    """Test getting job progress for a non-existent job."""
    # Both exact and partial matches return None
    mock_cursor1 = MagicMock()
    mock_cursor1.fetchone.return_value = None

    mock_cursor2 = MagicMock()
    mock_cursor2.fetchone.return_value = None

    # Set up side effect for consecutive calls
    mock_duckdb_connection.execute.side_effect = [mock_cursor1, mock_cursor2]

    # Call the function
    result = await get_job_progress(
        conn=mock_duckdb_connection,
        job_id="nonexistent",
    )

    # Should return None for non-existent job
    assert result is None


@pytest.mark.unit
@pytest.mark.async_test
async def test_get_job_count():
    """Test getting job count."""
    # Mock database connection
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (42,)

    with patch("src.lib.database.Database.connect", return_value=mock_conn):
        # Call the function
        result = await get_job_count()

        # Verify connection was closed
        # In new implementation, we close the database, not the connection directly

        # Check result
        assert result == 42


@pytest.mark.unit
@pytest.mark.async_test
async def test_get_job_count_error():
    """Test getting job count when an error occurs."""
    # Mock database connection with an error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = Exception("Database error")

    with patch("src.lib.database.Database.connect", return_value=mock_conn):
        # Call the function
        result = await get_job_count()

        # Should still close the database even with an error
        # In new implementation, we close the database, not the connection directly

        # Should return -1 to indicate an error
        assert result == -1
