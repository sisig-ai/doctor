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
        assert mock_duckdb_connection.execute.call_count == 2
        # First call should be to begin transaction
        mock_duckdb_connection.execute.assert_any_call("BEGIN TRANSACTION")

        # Second call should be the INSERT
        mock_duckdb_connection.execute.assert_any_call(
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
        assert mock_duckdb_connection.execute.call_count == 2
        # First call should be to begin transaction
        mock_duckdb_connection.execute.assert_any_call("BEGIN TRANSACTION")

        # Second call should be the INSERT with provided page_id
        mock_duckdb_connection.execute.assert_any_call(
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
        # Set up the mock to fail on the second execute call (the INSERT)
        mock_duckdb_connection.execute.side_effect = [
            "BEGIN TRANSACTION",
            Exception("Database error"),
        ]

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

        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor

        # Call the function with just status
        update_job_status(job_id=job_id, status="running")

        # Check that the database operations were performed correctly
        assert mock_duckdb_connection.execute.call_count == 2
        # First call should be to begin transaction
        mock_duckdb_connection.execute.assert_any_call("BEGIN TRANSACTION")

        # Second call should be the UPDATE
        mock_duckdb_connection.execute.assert_any_call(
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

        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor

        # Call the function with all options
        update_job_status(
            job_id=job_id,
            status="running",
            pages_discovered=10,
            pages_crawled=5,
            error_message="Some error occurred",
        )

        # Check that the execute method was called at least twice
        assert mock_duckdb_connection.execute.call_count >= 2

        # First call should be to begin transaction
        mock_duckdb_connection.execute.assert_any_call("BEGIN TRANSACTION")

        # Check that one of the calls to execute has the correct parameters
        update_call_found = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            query = call_args[0][0]
            if "UPDATE jobs SET" in query and "WHERE job_id = ?" in query:
                params = call_args[0][1]
                assert params[0] == "running"  # status
                assert params[1] == now  # updated_at
                assert params[2] == 10  # pages_discovered
                assert params[3] == 5  # pages_crawled
                assert params[4] == "Some error occurred"  # error_message
                assert params[5] == job_id  # job_id
                update_call_found = True
                break

        assert update_call_found, "Update query with correct parameters not found"
        mock_duckdb_connection.commit.assert_called_once()


@pytest.mark.unit
def test_update_job_status_checkpoint(job_id, mock_duckdb_connection):
    """Test that checkpoint is called for completed/failed status."""
    with patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection):
        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor
        mock_duckdb_connection.execute.side_effect = None  # Remove any previous side effects

        # Test with 'completed' status - expect 4 calls:
        # 1. BEGIN TRANSACTION, 2. UPDATE, 3. CHECKPOINT, 4. SELECT to verify job
        update_job_status(job_id=job_id, status="completed")

        # Check that checkpoint was called and verify expected call count
        assert mock_duckdb_connection.execute.call_count == 4
        mock_duckdb_connection.execute.assert_any_call("CHECKPOINT")
        mock_duckdb_connection.execute.assert_any_call("BEGIN TRANSACTION")

        # Select to verify job should be called
        select_call_found = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            query = call_args[0][0]
            if "SELECT status, pages_discovered, pages_crawled FROM jobs WHERE job_id = ?" in query:
                select_call_found = True
                break
        assert select_call_found, "Verification SELECT query not found"

        # Reset and test with 'failed' status - should also be 4 calls
        mock_duckdb_connection.reset_mock()
        mock_duckdb_connection.execute.return_value = mock_cursor

        update_job_status(job_id=job_id, status="failed")

        # Check that checkpoint was called
        assert mock_duckdb_connection.execute.call_count == 4
        mock_duckdb_connection.execute.assert_any_call("CHECKPOINT")

        # Reset and test with other status - should only be 2 calls (BEGIN and UPDATE)
        mock_duckdb_connection.reset_mock()
        mock_duckdb_connection.execute.return_value = mock_cursor

        update_job_status(job_id=job_id, status="running")

        # Check that checkpoint was not called
        assert mock_duckdb_connection.execute.call_count == 2
        assert "CHECKPOINT" not in [
            call_args[0][0] for call_args in mock_duckdb_connection.execute.call_args_list
        ]


@pytest.mark.unit
def test_update_job_status_error_handling(job_id, mock_duckdb_connection):
    """Test error handling when updating a job status."""
    with patch("src.lib.database.get_duckdb_connection", return_value=mock_duckdb_connection):
        # Simulate a database error after BEGIN TRANSACTION
        mock_duckdb_connection.execute.side_effect = [
            "BEGIN TRANSACTION",
            Exception("Database error"),
        ]

        with pytest.raises(Exception, match="Database error"):
            update_job_status(job_id=job_id, status="running")

        # Check that rollback and close were called
        mock_duckdb_connection.rollback.assert_called_once()
        mock_duckdb_connection.close.assert_called_once()
