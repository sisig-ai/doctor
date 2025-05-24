"""Tests for the enhanced crawler with hierarchy tracking."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.lib.crawler_enhanced import CrawlResultWithHierarchy, crawl_url_with_hierarchy


class TestCrawlResultWithHierarchy:
    """Test the CrawlResultWithHierarchy class."""

    def test_init_with_parent_url(self) -> None:
        """Test initialization with parent URL in metadata.

        Args:
            None.

        Returns:
            None.
        """
        # Create mock base result
        base_result = Mock()
        base_result.url = "https://example.com/page1"
        base_result.html = "<html><title>Test Page</title></html>"
        base_result.metadata = {"parent_url": "https://example.com", "depth": 1}

        # Create enhanced result
        enhanced = CrawlResultWithHierarchy(base_result)

        assert enhanced.url == "https://example.com/page1"
        assert enhanced.parent_url == "https://example.com"
        assert enhanced.depth == 1
        assert enhanced.title == "Test Page"

    def test_init_without_metadata(self) -> None:
        """Test initialization without metadata.

        Args:
            None.

        Returns:
            None.
        """
        # Create mock base result without metadata
        base_result = Mock()
        base_result.url = "https://example.com"
        base_result.html = "<html><body>Content</body></html>"
        base_result.metadata = None

        # Mock hasattr to return False for metadata
        with patch("builtins.hasattr", side_effect=lambda obj, attr: attr != "metadata"):
            enhanced = CrawlResultWithHierarchy(base_result)

        assert enhanced.url == "https://example.com"
        assert enhanced.parent_url is None
        assert enhanced.depth == 0

    def test_extract_title_from_html(self) -> None:
        """Test extracting title from HTML.

        Args:
            None.

        Returns:
            None.
        """
        base_result = Mock()
        base_result.url = "https://example.com"
        base_result.html = "<html><title>My Test Page</title></html>"
        base_result.metadata = {}

        enhanced = CrawlResultWithHierarchy(base_result)
        assert enhanced.title == "My Test Page"

    def test_extract_title_from_markdown(self) -> None:
        """Test extracting title from markdown when no HTML title.

        Args:
            None.

        Returns:
            None.
        """
        base_result = Mock()
        base_result.url = "https://example.com"
        base_result.html = "<html><body>Content</body></html>"
        base_result.metadata = {}

        # Mock extract_page_text to return markdown
        with patch(
            "src.lib.crawler_enhanced.extract_page_text",
            return_value="# Main Heading\n\nSome content",
        ):
            enhanced = CrawlResultWithHierarchy(base_result)

        assert enhanced.title == "Main Heading"

    def test_extract_title_fallback_to_url(self) -> None:
        """Test title fallback to URL path when no title found.

        Args:
            None.

        Returns:
            None.
        """
        base_result = Mock()
        base_result.url = "https://example.com/docs/api"
        base_result.html = ""
        base_result.metadata = {}

        with patch("src.lib.crawler_enhanced.extract_page_text", return_value=""):
            enhanced = CrawlResultWithHierarchy(base_result)

        assert enhanced.title == "api"

    def test_extract_title_home_for_root(self) -> None:
        """Test title defaults to 'Home' for root URLs.

        Args:
            None.

        Returns:
            None.
        """
        base_result = Mock()
        base_result.url = "https://example.com/"
        base_result.html = ""
        base_result.metadata = {}

        with patch("src.lib.crawler_enhanced.extract_page_text", return_value=""):
            enhanced = CrawlResultWithHierarchy(base_result)

        assert enhanced.title == "Home"


@pytest.mark.asyncio
class TestCrawlUrlWithHierarchy:
    """Test the crawl_url_with_hierarchy function."""

    async def test_crawl_with_hierarchy(self) -> None:
        """Test crawling with hierarchy enhancement.

        Args:
            None.

        Returns:
            None.
        """
        # Mock base crawl results
        mock_results = [
            Mock(
                url="https://example.com",
                metadata={"parent_url": None, "depth": 0},
                html="<title>Home</title>",
            ),
            Mock(
                url="https://example.com/about",
                metadata={"parent_url": "https://example.com", "depth": 1},
                html="<title>About</title>",
            ),
            Mock(
                url="https://example.com/contact",
                metadata={"parent_url": "https://example.com", "depth": 1},
                html="<title>Contact</title>",
            ),
        ]

        with patch("src.lib.crawler_enhanced.base_crawl_url", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_results

            results = await crawl_url_with_hierarchy("https://example.com", max_pages=10)

        assert len(results) == 3
        assert all(isinstance(r, CrawlResultWithHierarchy) for r in results)

        # Check hierarchy information
        assert results[0].parent_url is None
        assert results[0].root_url == "https://example.com"
        assert results[0].depth == 0

        assert results[1].parent_url == "https://example.com"
        assert results[1].root_url == "https://example.com"
        assert results[1].depth == 1

    async def test_relative_path_calculation(self) -> None:
        """Test calculation of relative paths.

        Args:
            None.

        Returns:
            None.
        """
        # Mock results with parent-child relationship
        mock_results = [
            Mock(
                url="https://example.com",
                metadata={"parent_url": None, "depth": 0},
                html="<title>Home</title>",
            ),
            Mock(
                url="https://example.com/docs",
                metadata={"parent_url": "https://example.com", "depth": 1},
                html="<title>Documentation</title>",
            ),
            Mock(
                url="https://example.com/docs/api",
                metadata={"parent_url": "https://example.com/docs", "depth": 2},
                html="<title>API Reference</title>",
            ),
        ]

        with patch("src.lib.crawler_enhanced.base_crawl_url", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_results

            results = await crawl_url_with_hierarchy("https://example.com")

        # Check relative paths
        assert results[0].relative_path == "/"
        assert results[1].relative_path == "Documentation"
        assert results[2].relative_path == "Documentation/API Reference"

    async def test_empty_crawl_results(self) -> None:
        """Test handling of empty crawl results.

        Args:
            None.

        Returns:
            None.
        """
        with patch("src.lib.crawler_enhanced.base_crawl_url", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = []

            results = await crawl_url_with_hierarchy("https://example.com")

        assert results == []
