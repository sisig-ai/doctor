"""Tests for the processor module."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.common.processor import process_crawl_result, process_page_batch


@pytest.fixture
def mock_processor_dependencies():
    """Setup mock dependencies for processor tests."""
    extract_page_text_mock = MagicMock()
    extract_page_text_mock.return_value = "Extracted text content"

    # Mock for Database class store_page method
    store_page_mock = AsyncMock()
    store_page_mock.return_value = "test-page-123"

    # Mock for Database class update_job_status method
    update_job_status_mock = MagicMock()

    # Create a Database class mock that can be both used directly
    # and as a context manager
    database_mock = MagicMock(
        __enter__=lambda self: self,
        __exit__=lambda *args: None,
        store_page=store_page_mock,
        update_job_status=update_job_status_mock,
    )

    generate_embedding_mock = AsyncMock()
    generate_embedding_mock.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

    mocks = {
        "extract_page_text": extract_page_text_mock,
        "database": database_mock,
        "store_page": store_page_mock,
        "TextChunker": MagicMock(),
        "generate_embedding": generate_embedding_mock,
        "VectorIndexer": MagicMock(),
        "update_job_status": update_job_status_mock,
    }

    chunker_instance = MagicMock()
    chunker_instance.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
    mocks["TextChunker"].return_value = chunker_instance

    indexer_instance = MagicMock()
    indexer_instance.index_vector = AsyncMock(return_value="vector-id-123")
    mocks["VectorIndexer"].return_value = indexer_instance

    return mocks


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_crawl_result(
    sample_crawl_result, job_id, sample_tags, mock_processor_dependencies
):
    """Test processing a single crawl result."""
    with (
        patch(
            "src.common.processor.extract_page_text",
            mock_processor_dependencies["extract_page_text"],
        ),
        patch(
            "src.common.processor.Database", return_value=mock_processor_dependencies["database"]
        ),
        patch("src.common.processor.TextChunker", mock_processor_dependencies["TextChunker"]),
        patch(
            "src.common.processor.generate_embedding",
            mock_processor_dependencies["generate_embedding"],
        ),
        patch("src.common.processor.VectorIndexer", mock_processor_dependencies["VectorIndexer"]),
    ):
        page_id = await process_crawl_result(
            page_result=sample_crawl_result, job_id=job_id, tags=sample_tags
        )

        mock_processor_dependencies["extract_page_text"].assert_called_once_with(
            sample_crawl_result
        )
        mock_processor_dependencies["store_page"].assert_called_once_with(
            url=sample_crawl_result.url,
            text="Extracted text content",
            job_id=job_id,
            tags=sample_tags,
        )
        chunker_instance = mock_processor_dependencies["TextChunker"].return_value
        chunker_instance.split_text.assert_called_once_with("Extracted text content")
        assert mock_processor_dependencies["generate_embedding"].call_count == 3
        indexer_instance = mock_processor_dependencies["VectorIndexer"].return_value
        assert indexer_instance.index_vector.call_count == 3
        assert page_id == "test-page-123"


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_crawl_result_with_errors(
    sample_crawl_result, job_id, sample_tags, mock_processor_dependencies
):
    """Test processing a crawl result with errors during chunking/embedding."""
    with (
        patch(
            "src.common.processor.extract_page_text",
            mock_processor_dependencies["extract_page_text"],
        ),
        patch(
            "src.common.processor.Database", return_value=mock_processor_dependencies["database"]
        ),
        patch("src.common.processor.TextChunker", mock_processor_dependencies["TextChunker"]),
        patch("src.common.processor.generate_embedding", side_effect=Exception("Embedding error")),
        patch("src.common.processor.VectorIndexer", mock_processor_dependencies["VectorIndexer"]),
    ):
        page_id = await process_crawl_result(
            page_result=sample_crawl_result, job_id=job_id, tags=sample_tags
        )
        assert page_id == "test-page-123"
        mock_processor_dependencies["store_page"].assert_called_once()
        indexer_instance = mock_processor_dependencies["VectorIndexer"].return_value
        assert indexer_instance.index_vector.call_count == 0


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_page_batch(mock_processor_dependencies):
    """Test processing a batch of crawl results."""
    mock_results = [
        MagicMock(url="https://example.com/page1"),
        MagicMock(url="https://example.com/page2"),
        MagicMock(url="https://example.com/page3"),
    ]
    mock_process_result = AsyncMock(side_effect=["page-1", "page-2", "page-3"])

    with (
        patch("src.common.processor.process_crawl_result", mock_process_result),
        patch(
            "src.common.processor.Database", return_value=mock_processor_dependencies["database"]
        ),
    ):
        job_id = "test-job"
        tags = ["test", "batch"]
        page_ids = await process_page_batch(
            page_results=mock_results,
            job_id=job_id,
            tags=tags,
            batch_size=2,
        )
        assert mock_process_result.call_count == 3
        mock_process_result.assert_any_call(mock_results[0], job_id, tags)
        mock_process_result.assert_any_call(mock_results[1], job_id, tags)
        mock_process_result.assert_any_call(mock_results[2], job_id, tags)
        assert mock_processor_dependencies["update_job_status"].call_count == 3
        assert page_ids == ["page-1", "page-2", "page-3"]


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_page_batch_with_errors(mock_processor_dependencies):
    """Test processing a batch with errors on some pages."""
    mock_results = [
        MagicMock(url="https://example.com/page1"),
        MagicMock(url="https://example.com/page2"),
        MagicMock(url="https://example.com/page3"),
    ]
    mock_process_result = AsyncMock(
        side_effect=["page-1", Exception("Error processing page 2"), "page-3"]
    )

    with (
        patch("src.common.processor.process_crawl_result", mock_process_result),
        patch(
            "src.common.processor.Database", return_value=mock_processor_dependencies["database"]
        ),
    ):
        job_id = "test-job"
        tags = ["test", "batch"]
        page_ids = await process_page_batch(page_results=mock_results, job_id=job_id, tags=tags)
        assert mock_process_result.call_count == 3
        assert mock_processor_dependencies["update_job_status"].call_count == 2
        assert page_ids == ["page-1", "page-3"]


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_page_batch_empty():
    """Test processing an empty batch of pages."""
    page_ids = await process_page_batch(page_results=[], job_id="test-job", tags=["test"])
    assert page_ids == []
