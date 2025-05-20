"""Tests for the crawler module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.lib.crawler import crawl_url, extract_page_text


@pytest.mark.unit
@pytest.mark.async_test
async def test_crawl_url(sample_url):
    """Test crawling a URL."""
    # Create mock results
    mock_results = [
        MagicMock(url=sample_url),
        MagicMock(url=f"{sample_url}/page1"),
        MagicMock(url=f"{sample_url}/page2"),
    ]

    # Create a mock AsyncWebCrawler class
    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.arun = AsyncMock(return_value=mock_results)

    # Create a mock context manager
    mock_crawler_class = MagicMock()
    mock_crawler_class.return_value.__aenter__.return_value = mock_crawler_instance

    # Patch the AsyncWebCrawler
    with patch("src.lib.crawler.AsyncWebCrawler", mock_crawler_class):
        results = await crawl_url(sample_url, max_pages=3, max_depth=2)

        # Check that arun was called with the correct arguments
        mock_crawler_instance.arun.assert_called_once()
        args, kwargs = mock_crawler_instance.arun.call_args
        assert kwargs["url"] == sample_url
        assert "config" in kwargs

        # Check that we got the expected results
        assert results == mock_results
        assert len(results) == 3


@pytest.mark.unit
def test_extract_page_text_with_markdown(sample_crawl_result):
    """Test extracting text from a crawl result with markdown."""
    text = extract_page_text(sample_crawl_result)
    assert text == "# Example Page\n\nThis is some example content."


@pytest.mark.unit
def test_extract_page_text_with_extracted_content():
    """Test extracting text from a crawl result with extracted content."""
    # Create a mock result with only extracted content
    mock_markdown = MagicMock()
    mock_markdown.fit_markdown = "Example Page. This is some example content."

    mock_result = MagicMock(
        url="https://example.com",
        markdown=mock_markdown,
        _markdown=None,
        extracted_content="Fallback content that should not be used",
        html="<html>...</html>",
    )

    text = extract_page_text(mock_result)
    assert text == "Example Page. This is some example content."


@pytest.mark.unit
def test_extract_page_text_with_html_only():
    """Test extracting text from a crawl result with only HTML."""
    # Create a mock result with only HTML
    html_content = "<html><body><h1>Example</h1></body></html>"

    # Create mock with no markdown or extracted content, only HTML
    mock_result = MagicMock(
        url="https://example.com",
        markdown=None,
        _markdown=None,
        extracted_content=None,
        html=html_content,
    )

    text = extract_page_text(mock_result)
    assert text == html_content
