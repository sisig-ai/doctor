"""Tests for the database module."""

import pytest
from unittest.mock import patch, MagicMock
import datetime
import uuid

from src.lib.database import Database


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
        patch("src.lib.database.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.datetime.datetime") as mock_datetime,
        patch(
            "src.lib.database.uuid.uuid4",
            return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        ),
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection

        # Call the method without providing a page_id
        page_id = await db.store_page(
            url=sample_url, text=sample_text, job_id=job_id, tags=sample_tags
        )

        # Check that the database operations were performed correctly
        assert mock_duckdb_connection.execute.call_count >= 1

        # One of the calls should be the INSERT
        insert_call_found = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            if "INSERT INTO pages" in call_args[0][0]:
                params = call_args[0][1]
                assert params[0] == "12345678-1234-5678-1234-567812345678"  # page_id
                assert params[1] == sample_url  # url
                assert params[2] == "example.com"  # domain
                assert params[3] == sample_text  # text
                assert params[4] == now  # crawl_date
                # Check that the tags were serialized correctly
                assert isinstance(params[5], str)  # tags (serialized JSON)
                assert params[6] == job_id  # job_id
                insert_call_found = True
                break

        assert insert_call_found, "INSERT query was not found in execute calls"
        mock_duckdb_connection.commit.assert_called_once()

        # Check that the function returned the expected page_id
        assert page_id == "12345678-1234-5678-1234-567812345678"

        # Test with a provided page_id
        mock_duckdb_connection.reset_mock()
        provided_page_id = "custom-page-id"

        page_id = await db.store_page(
            url=sample_url,
            text=sample_text,
            job_id=job_id,
            tags=sample_tags,
            page_id=provided_page_id,
        )

        # Check that one of the calls to execute is the INSERT with provided page_id
        insert_call_found = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            if "INSERT INTO pages" in call_args[0][0]:
                params = call_args[0][1]
                assert params[0] == provided_page_id  # page_id
                insert_call_found = True
                break

        assert insert_call_found, "INSERT query was not found in execute calls"

        # Check that the function returned the provided page_id
        assert page_id == provided_page_id


@pytest.mark.unit
@pytest.mark.async_test
async def test_store_page_error_handling(sample_url, sample_text, job_id, mock_duckdb_connection):
    """Test error handling when storing a page."""
    with patch("src.lib.database.duckdb.connect", return_value=mock_duckdb_connection):
        # Set up the mock to fail on execute
        # For store_page, first call is begin_transaction, second is insert
        mock_duckdb_connection.execute.side_effect = Exception("Database error")

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection
        db.transaction_active = True  # Set transaction active to true

        with pytest.raises(Exception, match="Database error"):
            await db.store_page(url=sample_url, text=sample_text, job_id=job_id)

        # Check that rollback was called
        mock_duckdb_connection.rollback.assert_called_once()


@pytest.mark.unit
def test_update_job_status_basic(job_id, mock_duckdb_connection):
    """Test updating a job status with basic options."""
    with (
        patch("src.lib.database.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.datetime.datetime") as mock_datetime,
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection

        # Call the method with just status
        db.update_job_status(job_id=job_id, status="running")

        # Check that the database operations were performed correctly
        update_call_found = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            query = call_args[0][0]
            if "UPDATE jobs SET" in query and "WHERE job_id = ?" in query:
                params = call_args[0][1]
                assert params[0] == "running"  # status
                assert params[1] == now  # updated_at
                assert params[-1] == job_id  # job_id (last parameter)
                update_call_found = True
                break

        assert update_call_found, "UPDATE query was not found in execute calls"
        mock_duckdb_connection.commit.assert_called_once()


@pytest.mark.unit
def test_update_job_status_with_all_options(job_id, mock_duckdb_connection):
    """Test updating a job status with all options."""
    with (
        patch("src.lib.database.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.datetime.datetime") as mock_datetime,
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection

        # Call the method with all options
        db.update_job_status(
            job_id=job_id,
            status="running",
            pages_discovered=10,
            pages_crawled=5,
            error_message="Some error occurred",
        )

        # Check that one of the calls to execute has the correct parameters
        update_call_found = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            query = call_args[0][0]
            if "UPDATE jobs SET" in query and "WHERE job_id = ?" in query:
                if len(call_args[0]) > 1:  # Make sure we have parameters
                    params = call_args[0][1]
                    status_param_found = "running" in params
                    pages_discovered_found = 10 in params
                    pages_crawled_found = 5 in params
                    error_message_found = "Some error occurred" in params
                    job_id_found = job_id in params

                    if (
                        status_param_found
                        and pages_discovered_found
                        and pages_crawled_found
                        and error_message_found
                        and job_id_found
                    ):
                        update_call_found = True
                        break

        assert update_call_found, "UPDATE query with correct parameters not found"
        mock_duckdb_connection.commit.assert_called_once()


@pytest.mark.unit
def test_update_job_status_checkpoint(job_id, mock_duckdb_connection):
    """Test that checkpoint is called for completed/failed status."""
    with patch("src.lib.database.duckdb.connect", return_value=mock_duckdb_connection):
        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor
        mock_duckdb_connection.execute.side_effect = None  # Remove any previous side effects

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection

        # Test with 'completed' status - checkpoint should be called
        db.update_job_status(job_id=job_id, status="completed")

        # Check that checkpoint was called
        checkpoint_called = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            if call_args[0][0] == "CHECKPOINT":
                checkpoint_called = True
                break

        assert checkpoint_called, "CHECKPOINT was not called for 'completed' status"

        # Reset and test with 'failed' status - should also call checkpoint
        mock_duckdb_connection.reset_mock()
        mock_duckdb_connection.execute.return_value = mock_cursor

        db.update_job_status(job_id=job_id, status="failed")

        # Check that checkpoint was called
        checkpoint_called = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            if call_args[0][0] == "CHECKPOINT":
                checkpoint_called = True
                break

        assert checkpoint_called, "CHECKPOINT was not called for 'failed' status"

        # Reset and test with other status - should not call checkpoint
        mock_duckdb_connection.reset_mock()
        mock_duckdb_connection.execute.return_value = mock_cursor

        db.update_job_status(job_id=job_id, status="running")

        # Check that checkpoint was not called
        checkpoint_called = False
        for call_args in mock_duckdb_connection.execute.call_args_list:
            if call_args[0][0] == "CHECKPOINT":
                checkpoint_called = True
                break

        assert not checkpoint_called, "CHECKPOINT was incorrectly called for 'running' status"


@pytest.mark.unit
def test_update_job_status_error_handling(job_id, mock_duckdb_connection):
    """Test error handling when updating a job status."""
    with patch("src.lib.database.duckdb.connect", return_value=mock_duckdb_connection):
        # Simulate a database error on execute
        mock_duckdb_connection.execute.side_effect = Exception("Database error")

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection
        db.transaction_active = True  # Set transaction active to true

        with pytest.raises(Exception, match="Database error"):
            db.update_job_status(job_id=job_id, status="running")

        # Check that rollback was called
        mock_duckdb_connection.rollback.assert_called_once()


@pytest.mark.unit
def test_serialize_tags():
    """Test the serialize_tags static method."""
    # Test with normal list
    tags = ["tag1", "tag2", "tag3"]
    serialized = Database.serialize_tags(tags)
    assert serialized == '["tag1", "tag2", "tag3"]'

    # Test with empty list
    empty_tags = []
    serialized = Database.serialize_tags(empty_tags)
    assert serialized == "[]"

    # Test with None
    none_tags = None
    serialized = Database.serialize_tags(none_tags)
    assert serialized == "[]"


@pytest.mark.unit
def test_deserialize_tags():
    """Test the deserialize_tags static method."""
    # Test with normal JSON array
    serialized = '["tag1", "tag2", "tag3"]'
    tags = Database.deserialize_tags(serialized)
    assert tags == ["tag1", "tag2", "tag3"]

    # Test with empty array
    empty_serialized = "[]"
    tags = Database.deserialize_tags(empty_serialized)
    assert tags == []

    # Test with None/empty string
    tags = Database.deserialize_tags(None)
    assert tags == []
    tags = Database.deserialize_tags("")
    assert tags == []

    # Test with invalid JSON
    invalid_json = "{not valid json"
    tags = Database.deserialize_tags(invalid_json)
    assert tags == []
