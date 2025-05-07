"""Web crawling functionality using crawl4ai."""

from typing import List, Any

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from src.common.logger import get_logger

# Configure logging
logger = get_logger(__name__)


async def crawl_url(
    url: str, max_pages: int = 100, max_depth: int = 2, strip_urls: bool = True
) -> List[Any]:
    """
    Crawl a URL and return the results.

    Args:
        url: The URL to start crawling from
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth for the BFS crawl
        strip_urls: Whether to strip URLs from the returned markdown

    Returns:
        List of crawled page results
    """
    logger.info(f"Starting crawl for URL: {url} with max_pages={max_pages}")

    # Create content filter to remove navigation elements and other non-essential content
    content_filter = PruningContentFilter(threshold=0.6, threshold_type="fixed")

    # Configure markdown generator to ignore links and navigation elements
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=content_filter,
        options={
            "ignore_links": strip_urls,
            "body_width": 0,
            "ignore_images": True,
            "single_line_break": True,
        },
    )

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            logger=get_logger("crawl4ai"),
            include_external=False,
        ),
        markdown_generator=markdown_generator,
        excluded_tags=["nav", "footer", "aside", "header"],
        remove_overlay_elements=True,
        verbose=True,
    )

    # Initialize the crawler
    async with AsyncWebCrawler() as crawler:
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
    # Use filtered markdown if available, otherwise use raw markdown,
    # extracted content, or HTML as fallbacks
    if hasattr(page_result, "markdown") and page_result.markdown:
        if hasattr(page_result.markdown, "fit_markdown") and page_result.markdown.fit_markdown:
            page_text = page_result.markdown.fit_markdown
            logger.debug(f"Using fit markdown text of length {len(page_text)}")
        elif hasattr(page_result.markdown, "raw_markdown"):
            page_text = page_result.markdown.raw_markdown
            logger.debug(f"Using raw markdown text of length {len(page_text)}")
        else:
            # Handle string-like markdown (backward compatibility)
            page_text = str(page_result.markdown)
            logger.debug(f"Using string markdown text of length {len(page_text)}")
    elif hasattr(page_result, "_markdown") and page_result._markdown:
        if hasattr(page_result._markdown, "fit_markdown") and page_result._markdown.fit_markdown:
            page_text = page_result._markdown.fit_markdown
            logger.debug(f"Using fit markdown text from _markdown of length {len(page_text)}")
        else:
            page_text = page_result._markdown.raw_markdown
            logger.debug(f"Using raw markdown text from _markdown of length {len(page_text)}")
    elif page_result.extracted_content:
        page_text = page_result.extracted_content
        logger.debug(f"Using extracted content of length {len(page_text)}")
    else:
        page_text = page_result.html
        logger.debug(f"Using HTML content of length {len(page_text)}")

    return page_text
