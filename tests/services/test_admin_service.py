"""Tests for the admin service."""

import pytest
from unittest.mock import patch, MagicMock
from rq import Queue

from src.web_service.services.admin_service import delete_docs


@pytest.fixture
def mock_queue():
    """Mock Redis queue."""
    mock = MagicMock(spec=Queue)
    mock.enqueue = MagicMock()
    return mock


@pytest.mark.unit
@pytest.mark.async_test
async def test_delete_docs_basic(mock_queue):
    """Test basic document deletion without filters."""
    # Set a fixed UUID for testing
    with patch("uuid.uuid4", return_value="delete-task-123"):
        # Call the function without filters
        task_id = await delete_docs(queue=mock_queue)

        # Verify queue.enqueue was called with the right parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0]
        assert call_args[0] == "src.crawl_worker.tasks.delete_docs"
        assert call_args[1] == "delete-task-123"  # task_id
        assert call_args[2] is None  # tags
        assert call_args[3] is None  # domain
        assert call_args[4] is None  # page_ids

        # Check that the function returned the expected task ID
        assert task_id == "delete-task-123"


@pytest.mark.unit
@pytest.mark.async_test
async def test_delete_docs_with_tags(mock_queue):
    """Test document deletion with tag filter."""
    # Set a fixed UUID for testing
    with patch("uuid.uuid4", return_value="delete-task-456"):
        # Call the function with tags filter
        task_id = await delete_docs(
            queue=mock_queue,
            tags=["test", "example"],
        )

        # Verify queue.enqueue was called with the right parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0]
        assert call_args[0] == "src.crawl_worker.tasks.delete_docs"
        assert call_args[1] == "delete-task-456"  # task_id
        assert call_args[2] == ["test", "example"]  # tags
        assert call_args[3] is None  # domain
        assert call_args[4] is None  # page_ids

        # Check that the function returned the expected task ID
        assert task_id == "delete-task-456"


@pytest.mark.unit
@pytest.mark.async_test
async def test_delete_docs_with_domain(mock_queue):
    """Test document deletion with domain filter."""
    # Set a fixed UUID for testing
    with patch("uuid.uuid4", return_value="delete-task-789"):
        # Call the function with domain filter
        task_id = await delete_docs(
            queue=mock_queue,
            domain="example.com",
        )

        # Verify queue.enqueue was called with the right parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0]
        assert call_args[0] == "src.crawl_worker.tasks.delete_docs"
        assert call_args[1] == "delete-task-789"  # task_id
        assert call_args[2] is None  # tags
        assert call_args[3] == "example.com"  # domain
        assert call_args[4] is None  # page_ids

        # Check that the function returned the expected task ID
        assert task_id == "delete-task-789"


@pytest.mark.unit
@pytest.mark.async_test
async def test_delete_docs_with_page_ids(mock_queue):
    """Test document deletion with specific page IDs."""
    # Set a fixed UUID for testing
    with patch("uuid.uuid4", return_value="delete-task-abc"):
        # Call the function with page_ids filter
        page_ids = ["page-1", "page-2", "page-3"]
        task_id = await delete_docs(
            queue=mock_queue,
            page_ids=page_ids,
        )

        # Verify queue.enqueue was called with the right parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0]
        assert call_args[0] == "src.crawl_worker.tasks.delete_docs"
        assert call_args[1] == "delete-task-abc"  # task_id
        assert call_args[2] is None  # tags
        assert call_args[3] is None  # domain
        assert call_args[4] == page_ids  # page_ids

        # Check that the function returned the expected task ID
        assert task_id == "delete-task-abc"


@pytest.mark.unit
@pytest.mark.async_test
async def test_delete_docs_with_all_filters(mock_queue):
    """Test document deletion with all filters."""
    # Set a fixed UUID for testing
    with patch("uuid.uuid4", return_value="delete-task-all"):
        # Call the function with all filters
        page_ids = ["page-1", "page-2"]
        task_id = await delete_docs(
            queue=mock_queue,
            tags=["test"],
            domain="example.com",
            page_ids=page_ids,
        )

        # Verify queue.enqueue was called with the right parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0]
        assert call_args[0] == "src.crawl_worker.tasks.delete_docs"
        assert call_args[1] == "delete-task-all"  # task_id
        assert call_args[2] == ["test"]  # tags
        assert call_args[3] == "example.com"  # domain
        assert call_args[4] == page_ids  # page_ids

        # Check that the function returned the expected task ID
        assert task_id == "delete-task-all"
