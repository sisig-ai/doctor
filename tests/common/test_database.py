"""Tests for the database module."""

import asyncio
import datetime
import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.logger import get_logger  # Import get_logger
from src.lib.database import DatabaseOperations
from src.lib.database.utils import serialize_tags, deserialize_tags

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
        db = DatabaseOperations()
        db.db.initialize()  # Ensure tables are created by the connection manager
        yield db
        db.db.close()  # Close using the connection manager


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
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Create a database instance
        db = DatabaseOperations()
        db.db.conn = mock_duckdb_connection

        # Call the method without providing a page_id
        page_id = await db.store_page(
            url=sample_url,
            text=sample_text,
            job_id=job_id,
            tags=sample_tags,
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
    with patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection):
        # Mock execute for the sequence of calls:
        # 1. ensure_connection (from store_page) -> SELECT 1
        # 2. ensure_connection (from begin_transaction) -> SELECT 1
        # 3. begin_transaction -> BEGIN TRANSACTION
        # 4. store_page (INSERT) -> raises Exception
        mock_select_cursor1 = MagicMock()
        mock_select_cursor1.fetchone.return_value = (1,)
        mock_select_cursor2 = MagicMock()
        mock_select_cursor2.fetchone.return_value = (1,)
        mock_begin_tx_cursor = MagicMock()

        mock_duckdb_connection.execute.side_effect = [
            mock_select_cursor1,
            mock_select_cursor2,
            mock_begin_tx_cursor,
            Exception("Database error"),
        ]

        db = DatabaseOperations()
        # db.db.conn will be set by the first ensure_connection -> connect() call if not pre-set
        # For this test, we can pre-set it to ensure the mock is used as intended from the start
        db.db.conn = mock_duckdb_connection

        with pytest.raises(RuntimeError) as excinfo:  # Operations layer wraps with RuntimeError
            await db.store_page(url=sample_url, text=sample_text, job_id=job_id)

        assert "Unexpected error storing page" in str(excinfo.value)
        # Or, if duckdb.Error is expected to be re-raised directly:
        # assert "Database error" in str(excinfo.value)

        mock_duckdb_connection.rollback.assert_called_once()
        assert mock_duckdb_connection.execute.call_count == 4


@pytest.mark.unit
@pytest.mark.async_test
async def test_update_job_status_basic(job_id, mock_duckdb_connection):
    """Test updating a job status with basic options."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.operations.datetime.datetime") as mock_datetime,
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor

        # Create a database instance
        db = DatabaseOperations()
        db.db.conn = mock_duckdb_connection

        # Call the method with just status
        await db.update_job_status(job_id=job_id, status="running")

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
@pytest.mark.async_test
async def test_update_job_status_with_all_options(job_id, mock_duckdb_connection):
    """Test updating a job status with all options."""
    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch("src.lib.database.operations.datetime.datetime") as mock_datetime,
    ):
        # Set mock datetime
        now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now

        # Mock cursor with rowcount attribute
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_duckdb_connection.execute.return_value = mock_cursor

        # Create a database instance
        db = DatabaseOperations()
        db.db.conn = mock_duckdb_connection

        # Call the method with all options
        await db.update_job_status(
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
@pytest.mark.async_test
async def test_update_job_status_checkpoint(job_id, mock_duckdb_connection):
    """Test that checkpoint_async is effectively called for completed/failed status."""
    # update_job_status calls self.checkpoint_async(), which then calls
    # await asyncio.to_thread(self.conn.execute, "CHECKPOINT")
    # We will mock out checkpoint_async itself to see if it's called by update_job_status.

    with (
        patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection),
        patch.object(
            DatabaseOperations, "checkpoint_async", new_callable=AsyncMock
        ) as mock_checkpoint_async,
    ):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        # Ensure execute returns the mock_cursor for the main update, and then for the read-back verify
        # It might be called multiple times by update_job_status (update, then select to verify)
        mock_duckdb_connection.execute.return_value = mock_cursor

        db = DatabaseOperations()
        db.db.conn = mock_duckdb_connection  # Assign the mock connection

        # Test with 'completed' status
        await db.update_job_status(job_id=job_id, status="completed")
        mock_checkpoint_async.assert_called_once()

        # Reset and test with 'failed' status
        mock_checkpoint_async.reset_mock()
        await db.update_job_status(job_id=job_id, status="failed")
        mock_checkpoint_async.assert_called_once()

        # Reset and test with other status - should not call checkpoint_async
        mock_checkpoint_async.reset_mock()
        await db.update_job_status(job_id=job_id, status="running")
        mock_checkpoint_async.assert_not_called()


@pytest.mark.unit
@pytest.mark.async_test
async def test_update_job_status_error_handling(job_id, mock_duckdb_connection):
    """Test error handling when updating a job status."""
    with patch("src.lib.database.connection.duckdb.connect", return_value=mock_duckdb_connection):
        # Mock execute for the sequence of calls:
        # 1. ensure_connection (from update_job_status) -> SELECT 1
        # 2. ensure_connection (from begin_transaction) -> SELECT 1
        # 3. begin_transaction -> BEGIN TRANSACTION
        # 4. update_job_status (UPDATE) -> raises Exception
        mock_select_cursor1 = MagicMock()
        mock_select_cursor1.fetchone.return_value = (1,)
        mock_select_cursor2 = MagicMock()
        mock_select_cursor2.fetchone.return_value = (1,)
        mock_begin_tx_cursor = MagicMock()

        mock_duckdb_connection.execute.side_effect = [
            mock_select_cursor1,
            mock_select_cursor2,
            mock_begin_tx_cursor,
            Exception("Database error"),
        ]

        db = DatabaseOperations()
        db.db.conn = mock_duckdb_connection

        with pytest.raises(RuntimeError) as excinfo:  # Operations layer wraps with RuntimeError
            await db.update_job_status(job_id=job_id, status="running")

        assert "Unexpected error updating job" in str(excinfo.value)
        # Or, if duckdb.Error is expected to be re-raised directly:
        # assert "Database error" in str(excinfo.value)

        mock_duckdb_connection.rollback.assert_called_once()
        assert mock_duckdb_connection.execute.call_count == 4


@pytest.mark.unit
def test_serialize_tags():
    """Test the serialize_tags static method."""
    # Test with normal list
    tags = ["tag1", "tag2", "tag3"]
    serialized = serialize_tags(tags)
    assert serialized == '["tag1", "tag2", "tag3"]'

    # Test with empty list
    empty_tags = []
    serialized = serialize_tags(empty_tags)
    assert serialized == "[]"

    # Test with None
    none_tags = None
    serialized = serialize_tags(none_tags)
    assert serialized == "[]"


@pytest.mark.unit
def test_deserialize_tags():
    """Test the deserialize_tags static method."""
    # Test with normal JSON array
    serialized = '["tag1", "tag2", "tag3"]'
    tags = deserialize_tags(serialized)
    assert tags == ["tag1", "tag2", "tag3"]

    # Test with empty array
    empty_serialized = "[]"
    tags = deserialize_tags(empty_serialized)
    assert tags == []

    # Test with None/empty string
    tags = deserialize_tags(None)
    assert tags == []
    tags = deserialize_tags("")
    assert tags == []

    # Test with invalid JSON
    invalid_json = "{not valid json"
    tags = deserialize_tags(invalid_json)
    assert tags == []


@pytest.mark.integration
@pytest.mark.async_test
async def test_async_lock_serializes_writes(
    db_instance: DatabaseOperations, job_id: str, sample_url: str
):
    """Test that the asyncio.Lock in Database serializes concurrent write operations."""
    num_tasks = 3
    increments_per_task = 2  # Keep small to run test faster
    test_job_id = f"lock_test_job_{uuid.uuid4()}"

    # Synchronously add a job for testing.
    db_instance.db.ensure_connection()
    db_instance.db.conn.execute(
        "INSERT INTO jobs (job_id, start_url, status, pages_crawled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (test_job_id, sample_url, "initial", 0, datetime.datetime.now(), datetime.datetime.now()),
    )
    db_instance.db.conn.commit()

    active_critical_section_tasks = 0
    max_concurrent_critical_section_tasks = 0
    lock_acquired_count = 0

    original_acquire = db_instance._write_lock.acquire
    original_release = db_instance._write_lock.release

    async def traced_acquire(*args, **kwargs):
        nonlocal \
            active_critical_section_tasks, \
            max_concurrent_critical_section_tasks, \
            lock_acquired_count

        # About to call original acquire
        # print(f"Task {asyncio.current_task().get_name()} attempting to acquire lock")
        await original_acquire(*args, **kwargs)  # Actually acquire the lock
        # print(f"Task {asyncio.current_task().get_name()} acquired lock")

        lock_acquired_count += 1
        active_critical_section_tasks += 1
        max_concurrent_critical_section_tasks = max(
            max_concurrent_critical_section_tasks,
            active_critical_section_tasks,
        )

        # Simulate work done while holding the lock
        await asyncio.sleep(0.02)
        # No return needed from original_acquire usually (it's True or raises)

    def traced_release(*args, **kwargs):
        nonlocal active_critical_section_tasks
        # print(f"Task {asyncio.current_task().get_name()} releasing lock")
        active_critical_section_tasks -= 1
        original_release(*args, **kwargs)  # Actually release the lock

    async def task_worker(db: DatabaseOperations, current_job_id: str, task_idx: int):
        # print(f"Task Worker {task_idx} starting for job {current_job_id}")
        for i in range(increments_per_task):
            # Each task will try to update the same job record
            # The actual update logic (e.g. incrementing pages_crawled) needs to be robust (atomic)
            # or the test might pass due to DB atomicity rather than Python lock.
            # Here, we set pages_crawled to a unique value per call to see if all updates apply.
            # The primary goal is to test the lock serialization, not complex DB state.
            await db.update_job_status(
                job_id=current_job_id,
                status=f"task_{task_idx}_update_{i}",
                pages_crawled=(task_idx * increments_per_task) + i + 1,
            )
            await asyncio.sleep(
                0.001,
            )  # Brief yield to allow other tasks to run and contend for the lock
        # print(f"Task Worker {task_idx} finished")

    with (
        patch.object(db_instance._write_lock, "acquire", new=traced_acquire),
        patch.object(db_instance._write_lock, "release", new=traced_release),
    ):
        tasks = []
        for i in range(num_tasks):
            task = asyncio.create_task(
                task_worker(db_instance, test_job_id, i),
                name=f"WorkerTask_{i}",
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    # print(f"Max concurrent tasks in critical section: {max_concurrent_critical_section_tasks}")
    # print(f"Total lock acquisitions: {lock_acquired_count}")

    assert lock_acquired_count == num_tasks * increments_per_task, (
        "Lock was not acquired the expected number of times"
    )
    assert max_concurrent_critical_section_tasks == 1, (
        "Lock did not serialize access to critical section"
    )

    # Verify final state of pages_crawled - it should be the value from the last update by one of the tasks.
    # This part is tricky because the order of task completion isn't guaranteed.
    # However, since each task sets a specific pages_crawled value, we can check if it's one of the expected final values.
    # Expected final values range from 1 to num_tasks * increments_per_task.
    final_row = db_instance.db.conn.execute(
        "SELECT pages_crawled, status FROM jobs WHERE job_id = ?",
        (test_job_id,),
    ).fetchone()
    assert final_row is not None, "Job not found after test execution"
    final_pages_crawled = final_row[0]
    final_status = final_row[1]

    # print(f"Final pages_crawled: {final_pages_crawled}, status: {final_status}")

    # The pages_crawled should be one of the values set by the tasks.
    # The maximum value any task would set is num_tasks * increments_per_task.
    # The minimum is 1.
    assert 1 <= final_pages_crawled <= num_tasks * increments_per_task
    # Check if the status reflects one of the last updates
    assert "_update_" in final_status
