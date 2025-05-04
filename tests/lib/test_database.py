"""Tests for the database module."""

import pytest
from unittest.mock import patch, MagicMock
import datetime
import uuid

from src.lib.database import store_page, update_job_status


@pytest.fixture
def mock_duckdb_connection():
    """Mock DuckDB connection."""
    mock = MagicMock()
    mock.execute = MagicMock()
    mock.commit = MagicMock()
    mock.rollback = MagicMock()
    mock.close = MagicMock()
    return mock


@pytest.fixture
def mock_datetime():
    """Mock datetime.datetime."""
    mock = MagicMock()
    mock.now.return_value = datetime.datetime(2023, 1, 1, 12, 0, 0)
    return mock


@pytest.mark.unit
@pytest.mark.async_test
async def test_store_page(sample_url, sample_text, job_id, mock_duckdb_connection, sample_tags):
    """Test storing a page in the database."""
    with (
        patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection),
        patch("src.lib.database.datetime.datetime") as mock_datetime,
        patch(
            "src.lib.database.uuid.uuid4",
            return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        ),
        patch("src.lib.database.serialize_tags", return_value="test,example,documentation"),
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Call the function without providing a page_id
        page_id = await store_page(
            url=sample_url, text=sample_text, job_id=job_id, tags=sample_tags
        )

        # Check that the database operations were performed correctly
        mock_duckdb_connection.execute.assert_called_once_with(
            """
            INSERT INTO pages (id, url, domain, raw_text, crawl_date, tags, job_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "12345678-1234-5678-1234-567812345678",
                sample_url,
                "example.com",
                sample_text,
                now,
                "test,example,documentation",
                job_id,
            ),
        )

        mock_duckdb_connection.commit.assert_called_once()
        mock_duckdb_connection.close.assert_called_once()

        # Check that the function returned the expected page_id
        assert page_id == "12345678-1234-5678-1234-567812345678"

        # Test with a provided page_id
        mock_duckdb_connection.reset_mock()
        provided_page_id = "custom-page-id"

        page_id = await store_page(
            url=sample_url,
            text=sample_text,
            job_id=job_id,
            tags=sample_tags,
            page_id=provided_page_id,
        )

        # Check that the database operations were performed with the provided page_id
        mock_duckdb_connection.execute.assert_called_once_with(
            """
            INSERT INTO pages (id, url, domain, raw_text, crawl_date, tags, job_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                provided_page_id,
                sample_url,
                "example.com",
                sample_text,
                now,
                "test,example,documentation",
                job_id,
            ),
        )

        # Check that the function returned the provided page_id
        assert page_id == provided_page_id


@pytest.mark.unit
@pytest.mark.async_test
async def test_store_page_error_handling(sample_url, sample_text, job_id, mock_duckdb_connection):
    """Test error handling when storing a page."""
    with patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection):
        # Simulate a database error
        mock_duckdb_connection.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await store_page(url=sample_url, text=sample_text, job_id=job_id)

        # Check that rollback and close were called
        mock_duckdb_connection.rollback.assert_called_once()
        mock_duckdb_connection.close.assert_called_once()


@pytest.mark.unit
def test_update_job_status_basic(job_id, mock_duckdb_connection):
    """Test updating a job status with basic options."""
    with (
        patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection),
        patch("src.lib.database.datetime.datetime") as mock_datetime,
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Call the function with just status
        update_job_status(job_id=job_id, status="running")

        # Check that the database operations were performed correctly
        mock_duckdb_connection.execute.assert_called_once_with(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?", ("running", now, job_id)
        )

        mock_duckdb_connection.commit.assert_called_once()
        mock_duckdb_connection.close.assert_called_once()


@pytest.mark.unit
def test_update_job_status_with_all_options(job_id, mock_duckdb_connection):
    """Test updating a job status with all options."""
    with (
        patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection),
        patch("src.lib.database.datetime.datetime") as mock_datetime,
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Call the function with all options
        update_job_status(
            job_id=job_id,
            status="running",
            pages_discovered=10,
            pages_crawled=5,
            error_message="Some error occurred",
        )

        # Check that the database operations were performed correctly with all fields
        mock_duckdb_connection.execute.assert_called_once()

        # The exact query string construction is complex due to string formatting,
        # so we'll check that the parameters are correct
        args, kwargs = mock_duckdb_connection.execute.call_args
        query = args[0]
        params = args[1]

        # Check that the query contains all the expected fields
        assert "UPDATE jobs SET" in query
        assert "status = ?" in query
        assert "pages_discovered = ?" in query
        assert "pages_crawled = ?" in query
        assert "error_message = ?" in query

        # Check the parameters
        assert params[0] == "running"  # status
        assert params[1] == now  # updated_at
        assert params[2] == 10  # pages_discovered
        assert params[3] == 5  # pages_crawled
        assert params[4] == "Some error occurred"  # error_message
        assert params[5] == job_id  # job_id

        mock_duckdb_connection.commit.assert_called_once()


@pytest.mark.unit
def test_update_job_status_checkpoint(job_id, mock_duckdb_connection):
    """Test that checkpoint is called for completed/failed status."""
    with patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection):
        # Test with 'completed' status
        update_job_status(job_id=job_id, status="completed")

        # Check that checkpoint was called
        assert mock_duckdb_connection.execute.call_count == 2
        mock_duckdb_connection.execute.assert_any_call("CHECKPOINT")

        # Reset and test with 'failed' status
        mock_duckdb_connection.reset_mock()
        update_job_status(job_id=job_id, status="failed")

        # Check that checkpoint was called
        assert mock_duckdb_connection.execute.call_count == 2
        mock_duckdb_connection.execute.assert_any_call("CHECKPOINT")

        # Reset and test with other status
        mock_duckdb_connection.reset_mock()
        update_job_status(job_id=job_id, status="running")

        # Check that checkpoint was not called
        assert mock_duckdb_connection.execute.call_count == 1
        mock_duckdb_connection.execute.assert_called_once()
        # Check that CHECKPOINT was not in the call arguments
        call_args = mock_duckdb_connection.execute.call_args[0][0]
        assert "CHECKPOINT" not in call_args


@pytest.mark.unit
def test_update_job_status_error_handling(job_id, mock_duckdb_connection):
    """Test error handling when updating a job status."""
    with patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection):
        # Simulate a database error
        mock_duckdb_connection.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            update_job_status(job_id=job_id, status="running")

        # Check that rollback and close were called
        mock_duckdb_connection.rollback.assert_called_once()
        mock_duckdb_connection.close.assert_called_once()
