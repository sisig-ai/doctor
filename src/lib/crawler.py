"""Web crawling functionality using crawl4ai."""

from typing import List, Any

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

from src.lib.logger import get_logger

# Configure logging
logger = get_logger(__name__)


async def crawl_url(url: str, max_pages: int = 100, max_depth: int = 2) -> List[Any]:
    """
    Crawl a URL and return the results.

    Args:
        url: The URL to start crawling from
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth for the BFS crawl

    Returns:
        List of crawled page results
    """
    logger.info(f"Starting crawl for URL: {url} with max_pages={max_pages}")

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            logger=get_logger("crawl4ai"),
            include_external=False,
        ),
        verbose=True,
    )

    # Initialize the crawler
    async with AsyncWebCrawler() as crawler:
        # Crawl the URL - returns a list of CrawlResult objects when deep crawling
        crawl_results = await crawler.arun(url=url, config=config)
        logger.info(f"Deep crawl discovered {len(crawl_results)} pages")
    return crawl_results


def extract_page_text(page_result: Any) -> str:
    """
    Extract the text content from a crawl4ai page result.

    Args:
        page_result: The crawl result for the page

    Returns:
        The extracted text content
    """
    # Use markdown if available, otherwise use extracted content or HTML
    if hasattr(page_result, "_markdown") and page_result._markdown:
        page_text = page_result._markdown.raw_markdown
        logger.debug(f"Using markdown text of length {len(page_text)}")
    elif page_result.extracted_content:
        page_text = page_result.extracted_content
        logger.debug(f"Using extracted content of length {len(page_text)}")
    else:
        page_text = page_result.html
        logger.debug(f"Using HTML content of length {len(page_text)}")

    return page_text
