"""Enhanced web crawler with hierarchy tracking for the maps feature."""

from typing import Any, Optional
from urllib.parse import urlparse
import re

from src.common.logger import get_logger
from src.lib.crawler import crawl_url as base_crawl_url, extract_page_text

logger = get_logger(__name__)


class CrawlResultWithHierarchy:
    """Enhanced crawl result that includes hierarchy information."""

    def __init__(self, base_result: Any) -> None:
        """Initialize with a base crawl4ai result.

        Args:
            base_result: The original crawl4ai result object.

        Returns:
            None.
        """
        self.base_result = base_result
        self.url = base_result.url
        self.parent_url: Optional[str] = None
        self.root_url: Optional[str] = None
        self.depth: int = 0
        self.relative_path: str = ""
        self.title: Optional[str] = None

        # Extract hierarchy info from metadata if available
        if hasattr(base_result, "metadata"):
            self.parent_url = base_result.metadata.get("parent_url")
            self.depth = base_result.metadata.get("depth", 0)

        # Extract title from the page
        self._extract_title()

    def _extract_title(self) -> None:
        """Extract the page title from the HTML or markdown content.

        Args:
            None.

        Returns:
            None.
        """
        try:
            # Try to extract from HTML title tag first
            if hasattr(self.base_result, "html") and self.base_result.html:
                title_match = re.search(
                    r"<title[^>]*>(.*?)</title>", self.base_result.html, re.IGNORECASE | re.DOTALL
                )
                if title_match:
                    self.title = title_match.group(1).strip()
                    # Clean up common HTML entities
                    self.title = (
                        self.title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                    )
                    return

            # Fallback to first H1 in markdown
            text = extract_page_text(self.base_result)
            if text:
                # Look for first markdown heading
                h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if h1_match:
                    self.title = h1_match.group(1).strip()
                    return

                # Look for first line of non-empty text as last resort
                lines = text.strip().split("\n")
                for line in lines[:5]:  # Check first 5 lines
                    if line.strip() and len(line.strip()) > 10:
                        self.title = line.strip()[:100]  # Limit to 100 chars
                        break

        except Exception as e:
            logger.warning(f"Error extracting title from {self.url}: {e}")

        # Default to URL path if no title found
        if not self.title:
            path = urlparse(self.url).path
            self.title = path.strip("/").split("/")[-1] or "Home"


async def crawl_url_with_hierarchy(
    url: str,
    max_pages: int = 100,
    max_depth: int = 2,
    strip_urls: bool = True,
) -> list[CrawlResultWithHierarchy]:
    """Crawl a URL and return results with hierarchy information.

    Args:
        url: The URL to start crawling from.
        max_pages: Maximum number of pages to crawl.
        max_depth: Maximum depth for the BFS crawl.
        strip_urls: Whether to strip URLs from the returned markdown.

    Returns:
        List of crawl results with hierarchy information.
    """
    # Get base crawl results
    base_results = await base_crawl_url(url, max_pages, max_depth, strip_urls)

    # Enhance results with hierarchy information
    enhanced_results = []
    root_url = url  # The starting URL is the root

    # Build a map of URLs to results for quick lookup
    url_to_result = {}
    for result in base_results:
        enhanced = CrawlResultWithHierarchy(result)
        enhanced.root_url = root_url
        url_to_result[enhanced.url] = enhanced
        enhanced_results.append(enhanced)

    # Calculate relative paths
    for enhanced in enhanced_results:
        if enhanced.parent_url and enhanced.parent_url in url_to_result:
            parent = url_to_result[enhanced.parent_url]
            # Calculate relative path based on parent's path
            if parent.relative_path == "/":
                enhanced.relative_path = enhanced.title
            elif parent.relative_path:
                enhanced.relative_path = f"{parent.relative_path}/{enhanced.title}"
            else:
                enhanced.relative_path = enhanced.title
        elif enhanced.url == root_url:
            enhanced.relative_path = "/"
        else:
            # Fallback to URL-based path
            enhanced.relative_path = urlparse(enhanced.url).path

    logger.info(f"Enhanced {len(enhanced_results)} pages with hierarchy information")
    return enhanced_results
