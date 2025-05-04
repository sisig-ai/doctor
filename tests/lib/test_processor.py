"""Tests for the processor module."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.lib.processor import process_crawl_result, process_page_batch


@pytest.fixture
def mock_processor_dependencies():
    """Setup mock dependencies for processor tests."""
    # Create mock functions and return them in a dict
    # extract_page_text is not async, so use MagicMock instead of AsyncMock
    extract_page_text_mock = MagicMock()
    extract_page_text_mock.return_value = "Extracted text content"

    store_page_mock = AsyncMock()
    store_page_mock.return_value = "test-page-123"

    generate_embedding_mock = AsyncMock()
    generate_embedding_mock.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

    mocks = {
        "extract_page_text": extract_page_text_mock,
        "store_page": store_page_mock,
        "TextChunker": MagicMock(),
        "generate_embedding": generate_embedding_mock,
        "VectorIndexer": MagicMock(),
        "update_job_status": MagicMock(),
    }

    # Setup the chunker mock
    chunker_instance = MagicMock()
    chunker_instance.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
    mocks["TextChunker"].return_value = chunker_instance

    # Setup the indexer mock
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
            "src.lib.processor.extract_page_text", mock_processor_dependencies["extract_page_text"]
        ),
        patch("src.lib.processor.store_page", mock_processor_dependencies["store_page"]),
        patch("src.lib.processor.TextChunker", mock_processor_dependencies["TextChunker"]),
        patch(
            "src.lib.processor.generate_embedding",
            mock_processor_dependencies["generate_embedding"],
        ),
        patch("src.lib.processor.VectorIndexer", mock_processor_dependencies["VectorIndexer"]),
    ):
        # Call the function
        page_id = await process_crawl_result(
            page_result=sample_crawl_result, job_id=job_id, tags=sample_tags
        )

        # Check that extract_page_text was called with the crawl result
        mock_processor_dependencies["extract_page_text"].assert_called_once_with(
            sample_crawl_result
        )

        # Check that store_page was called with the extracted text
        mock_processor_dependencies["store_page"].assert_called_once_with(
            url=sample_crawl_result.url,
            text="Extracted text content",
            job_id=job_id,
            tags=sample_tags,
        )

        # Check that the chunker was called correctly
        chunker_instance = mock_processor_dependencies["TextChunker"].return_value
        chunker_instance.split_text.assert_called_once_with("Extracted text content")

        # Check that generate_embedding was called for each chunk
        assert mock_processor_dependencies["generate_embedding"].call_count == 3

        # Check that index_vector was called for each chunk
        indexer_instance = mock_processor_dependencies["VectorIndexer"].return_value
        assert indexer_instance.index_vector.call_count == 3

        # Check the function returned the expected page_id
        assert page_id == "test-page-123"


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_crawl_result_with_errors(
    sample_crawl_result, job_id, sample_tags, mock_processor_dependencies
):
    """Test processing a crawl result with errors during chunking/embedding."""
    with (
        patch(
            "src.lib.processor.extract_page_text", mock_processor_dependencies["extract_page_text"]
        ),
        patch("src.lib.processor.store_page", mock_processor_dependencies["store_page"]),
        patch("src.lib.processor.TextChunker", mock_processor_dependencies["TextChunker"]),
        patch("src.lib.processor.generate_embedding", side_effect=Exception("Embedding error")),
        patch("src.lib.processor.VectorIndexer", mock_processor_dependencies["VectorIndexer"]),
    ):
        # Call the function - it should still complete despite embedding errors
        page_id = await process_crawl_result(
            page_result=sample_crawl_result, job_id=job_id, tags=sample_tags
        )

        # Check that we still got a page_id back despite the errors
        assert page_id == "test-page-123"

        # The page should still have been stored
        mock_processor_dependencies["store_page"].assert_called_once()

        # But no vectors should have been indexed
        indexer_instance = mock_processor_dependencies["VectorIndexer"].return_value
        assert indexer_instance.index_vector.call_count == 0


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_page_batch(mock_processor_dependencies):
    """Test processing a batch of crawl results."""
    # Create mock crawl results
    mock_results = [
        MagicMock(url="https://example.com/page1"),
        MagicMock(url="https://example.com/page2"),
        MagicMock(url="https://example.com/page3"),
    ]

    # Mock the process_crawl_result function to return predictable page IDs
    mock_process_result = AsyncMock(side_effect=["page-1", "page-2", "page-3"])

    with (
        patch("src.lib.processor.process_crawl_result", mock_process_result),
        patch(
            "src.lib.processor.update_job_status", mock_processor_dependencies["update_job_status"]
        ),
    ):
        # Call the function
        job_id = "test-job"
        tags = ["test", "batch"]
        page_ids = await process_page_batch(
            page_results=mock_results,
            job_id=job_id,
            tags=tags,
            batch_size=2,  # Process in batches of 2
        )

        # Check that process_crawl_result was called for each page
        assert mock_process_result.call_count == 3
        mock_process_result.assert_any_call(mock_results[0], job_id, tags)
        mock_process_result.assert_any_call(mock_results[1], job_id, tags)
        mock_process_result.assert_any_call(mock_results[2], job_id, tags)

        # Check that update_job_status was called for each page
        assert mock_processor_dependencies["update_job_status"].call_count == 3

        # Check that the function returned the expected page IDs
        assert page_ids == ["page-1", "page-2", "page-3"]


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_page_batch_with_errors(mock_processor_dependencies):
    """Test processing a batch with errors on some pages."""
    # Create mock crawl results
    mock_results = [
        MagicMock(url="https://example.com/page1"),
        MagicMock(url="https://example.com/page2"),
        MagicMock(url="https://example.com/page3"),
    ]

    # Mock the process_crawl_result function to succeed for first page, fail for second, succeed for third
    mock_process_result = AsyncMock(
        side_effect=["page-1", Exception("Error processing page 2"), "page-3"]
    )

    with (
        patch("src.lib.processor.process_crawl_result", mock_process_result),
        patch(
            "src.lib.processor.update_job_status", mock_processor_dependencies["update_job_status"]
        ),
    ):
        # Call the function
        job_id = "test-job"
        tags = ["test", "batch"]
        page_ids = await process_page_batch(page_results=mock_results, job_id=job_id, tags=tags)

        # Check that process_crawl_result was called for each page
        assert mock_process_result.call_count == 3

        # Check that update_job_status was called only for successful pages
        assert mock_processor_dependencies["update_job_status"].call_count == 2

        # Check that the function returned only the successful page IDs
        assert page_ids == ["page-1", "page-3"]


@pytest.mark.unit
@pytest.mark.async_test
async def test_process_page_batch_empty():
    """Test processing an empty batch of pages."""
    page_ids = await process_page_batch(page_results=[], job_id="test-job", tags=["test"])

    # Should return an empty list
    assert page_ids == []
