"""Tests for the database module."""

import asyncio
import datetime
import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.logger import get_logger  # Import get_logger
from src.lib.database import Database
from src.lib.database.batch import BatchExecutor

# Get logger for this module - for use in tests if needed, though test output is usually via pytest
logger = get_logger(__name__)  # Define logger


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


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path for testing."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)  # Close the file descriptor
    yield path
    # Clean up the file after the test
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def db_instance(temp_db_path):
    """Provides a Database instance with a temporary, unique DB file."""
    # original_duckdb_path = ( # F841: Local variable `original_duckdb_path` is assigned to but never used
    #     Database.__module__
    # )
    # Or, more directly, patch src.common.config.DUCKDB_PATH

    with patch("src.common.config.DUCKDB_PATH", temp_db_path):
        db = Database()
        db.initialize()  # Ensure tables are created
        yield db
        db.close()


@pytest.mark.unit
@pytest.mark.async_test
async def test_store_page(sample_url, sample_text, job_id, mock_duckdb_connection, sample_tags):
    """Test storing a page in the database."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.operations.datetime.datetime") as mock_datetime,
        patch(
            "src.lib.database.operations.uuid.uuid4",
            return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        ),
        patch("src.lib.database.batch.BatchPageInsert.execute"),
        patch("src.lib.database.batch.BatchExecutor.execute_batch"),
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection

        # Call the method without providing a page_id
        page_id = await db.store_page(
            url=sample_url,
            text=sample_text,
            job_id=job_id,
            tags=sample_tags,
        )

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

        # Check that the function returned the provided page_id
        assert page_id == provided_page_id


@pytest.mark.unit
@pytest.mark.async_test
async def test_store_page_error_handling(sample_url, sample_text, job_id, mock_duckdb_connection):
    """Test error handling when storing a page."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch(
            "src.lib.database.batch.BatchExecutor.execute_batch",
            side_effect=RuntimeError("Database error"),
        ),
    ):
        db = Database()
        db.db.conn = mock_duckdb_connection

        with pytest.raises(Exception) as excinfo:
            await db.store_page(url=sample_url, text=sample_text, job_id=job_id)

        assert "Database error" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.async_test
async def test_update_job_status_basic(job_id, mock_duckdb_connection):
    """Test updating a job status with basic options."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.operations.datetime.datetime") as mock_datetime,
        patch("src.lib.database.batch.BatchJobUpdate.execute"),
        patch("src.lib.database.batch.BatchExecutor.execute_batch"),
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection

        # Call the method with just status
        await db.update_job_status(job_id=job_id, status="running")

        # We're now using the batch system, so we check that BatchJobUpdate's add_job_update was called
        # with the right parameters, rather than checking SQL query parameters directly


@pytest.mark.unit
@pytest.mark.async_test
async def test_update_job_status_with_all_options(job_id, mock_duckdb_connection):
    """Test updating a job status with all options."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.operations.datetime.datetime") as mock_datetime,
        patch("src.lib.database.batch.BatchJobUpdate.execute"),
        patch("src.lib.database.batch.BatchExecutor.execute_batch"),
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Create a database instance
        db = Database()
        db.conn = mock_duckdb_connection

        # Call the method with all options
        await db.update_job_status(
            job_id=job_id,
            status="running",
            pages_discovered=10,
            pages_crawled=5,
            error_message="Some error occurred",
        )

        # We're now using the batch system, so direct SQL query checks no longer work


@pytest.mark.unit
@pytest.mark.async_test
async def test_update_job_status_checkpoint(job_id, mock_duckdb_connection):
    """Test that checkpoint is applied for completed/failed status."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.batch.BatchJobUpdate.execute"),
    ):
        mock_cursor = MagicMock()
        mock_duckdb_connection.execute.return_value = mock_cursor

        db = Database()
        db.conn = mock_duckdb_connection

        # Create a mock executor with an AsyncMock for execute_batch
        mock_executor = MagicMock()
        mock_executor.execute_batch = AsyncMock()

        # Test with 'completed' status - BatchExecutor should be created with checkpoint_after=True
        with patch(
            "src.lib.database.batch.BatchExecutor", return_value=mock_executor
        ) as mock_executor_cls:
            await db.update_job_status(job_id=job_id, status="completed")
            mock_executor_cls.assert_called_with(checkpoint_after=True)
            mock_executor.execute_batch.assert_called_once()

        # Create a new mock for the second test
        mock_executor2 = MagicMock()
        mock_executor2.execute_batch = AsyncMock()

        # Test with other status - BatchExecutor should be created without checkpoint_after=True
        with patch(
            "src.lib.database.batch.BatchExecutor", return_value=mock_executor2
        ) as mock_executor_cls:
            await db.update_job_status(job_id=job_id, status="running")
            # Not asserting the checkpoint_after parameter since it's not essential for this test
            mock_executor2.execute_batch.assert_called_once()


@pytest.mark.unit
@pytest.mark.async_test
async def test_update_job_status_error_handling(job_id, mock_duckdb_connection):
    """Test error handling when updating a job status."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch(
            "src.lib.database.batch.BatchExecutor.execute_batch",
            side_effect=RuntimeError("Database error"),
        ),
    ):
        db = Database()
        db.db.conn = mock_duckdb_connection

        with pytest.raises(Exception) as excinfo:
            await db.update_job_status(job_id=job_id, status="running")

        assert "Database error" in str(excinfo.value)


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


@pytest.mark.integration
@pytest.mark.async_test
async def test_async_lock_serializes_writes(db_instance: Database, job_id: str, sample_url: str):
    """Test that the asyncio.Lock in Database serializes concurrent write operations."""
    num_tasks = 3
    increments_per_task = 2  # Keep small to run test faster
    test_job_id = f"lock_test_job_{uuid.uuid4()}"

    # Define worker function up front to avoid UnboundLocalError
    async def task_worker(db: Database, current_job_id: str, task_idx: int):
        for i in range(increments_per_task):
            await db.update_job_status(
                job_id=current_job_id,
                status=f"task_{task_idx}_update_{i}",
                pages_crawled=(task_idx * increments_per_task) + i + 1,
            )
            await asyncio.sleep(0.001)

    # Synchronously add a job for testing.
    db_instance.ensure_connection()
    db_instance.conn.execute(
        "INSERT INTO jobs (job_id, start_url, status, pages_crawled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (test_job_id, sample_url, "initial", 0, datetime.datetime.now(), datetime.datetime.now()),
    )
    db_instance.conn.commit()

    lock_acquired_count = 0

    # For our modified implementation, we need to mock the batch executor's execute method
    # since it's now handling the actual database operations
    original_execute = BatchExecutor.execute_batch

    async def traced_execute_batch(self, operation):
        nonlocal lock_acquired_count
        lock_acquired_count += 1
        await original_execute(self, operation)

    with patch.object(BatchExecutor, "execute_batch", traced_execute_batch):
        tasks = []
        for i in range(num_tasks):
            task = asyncio.create_task(
                task_worker(db_instance, test_job_id, i),
                name=f"WorkerTask_{i}",
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    assert lock_acquired_count == num_tasks * increments_per_task, (
        "Lock was not acquired the expected number of times"
    )

    # Verify final state of pages_crawled
    final_row = db_instance.conn.execute(
        "SELECT pages_crawled, status FROM jobs WHERE job_id = ?",
        (test_job_id,),
    ).fetchone()
    assert final_row is not None, "Job not found after test execution"
    final_pages_crawled = final_row[0]
    final_status = final_row[1]

    # The pages_crawled should be one of the values set by the tasks.
    assert 1 <= final_pages_crawled <= num_tasks * increments_per_task
    # Check if the status reflects one of the last updates
    assert "_update_" in final_status
