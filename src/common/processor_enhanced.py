"""Enhanced page processing pipeline with hierarchy tracking."""

import asyncio
from itertools import islice
from urllib.parse import urlparse

from src.common.indexer import VectorIndexer
from src.common.logger import get_logger
from src.lib.chunker import TextChunker
from src.lib.crawler import extract_page_text
from src.lib.crawler_enhanced import CrawlResultWithHierarchy, crawl_url_with_hierarchy
from src.lib.database import DatabaseOperations
from src.lib.embedder import generate_embedding

# Configure logging
logger = get_logger(__name__)


async def process_crawl_result_with_hierarchy(
    page_result: CrawlResultWithHierarchy,
    job_id: str,
    tags: list[str] | None = None,
    max_concurrent_embeddings: int = 5,
    url_to_page_id: dict[str, str] | None = None,
) -> str:
    """Process a single crawled page result with hierarchy information.

    Args:
        page_result: The enhanced crawl result with hierarchy info.
        job_id: The ID of the crawl job.
        tags: Optional tags to associate with the page.
        max_concurrent_embeddings: Maximum number of concurrent embedding generations.
        url_to_page_id: Optional mapping of URLs to page IDs for parent lookup.

    Returns:
        The ID of the processed page.
    """
    if tags is None:
        tags = []
    if url_to_page_id is None:
        url_to_page_id = {}

    try:
        logger.info(f"Processing page with hierarchy: {page_result.url}")

        # Extract text from the crawl result
        page_text = extract_page_text(page_result.base_result)

        # Look up parent page ID if we have a parent URL
        parent_page_id = None
        if page_result.parent_url and page_result.parent_url in url_to_page_id:
            parent_page_id = url_to_page_id[page_result.parent_url]

        # Look up root page ID
        root_page_id = None
        if page_result.root_url and page_result.root_url in url_to_page_id:
            root_page_id = url_to_page_id[page_result.root_url]

        # Store the page in the database with hierarchy info
        db_ops = DatabaseOperations()
        page_id = await db_ops.store_page(
            url=page_result.url,
            text=page_text,
            job_id=job_id,
            tags=tags,
            parent_page_id=parent_page_id,
            root_page_id=root_page_id,
            depth=page_result.depth,
            path=page_result.relative_path,
            title=page_result.title,
        )

        # Store the mapping for future lookups
        url_to_page_id[page_result.url] = page_id

        # Initialize components
        chunker = TextChunker()
        indexer = VectorIndexer()

        # Split text into chunks
        chunks = chunker.split_text(page_text)
        logger.info(f"Split page into {len(chunks)} chunks")

        # Extract domain from URL
        domain = urlparse(page_result.url).netloc

        # Process chunks in parallel batches
        successful_chunks = 0

        async def process_chunk(chunk_text):
            try:
                # Generate embedding
                embedding = await generate_embedding(chunk_text)

                # Prepare payload
                payload = {
                    "text": chunk_text,
                    "page_id": page_id,
                    "url": page_result.url,
                    "domain": domain,
                    "tags": tags,
                    "job_id": job_id,
                }

                # Index the vector
                await indexer.index_vector(embedding, payload)
                return True
            except Exception as chunk_error:
                logger.error(f"Error processing chunk: {chunk_error!s}")
                return False

        # Process chunks in batches with limited concurrency
        i = 0
        while i < len(chunks):
            # Take up to max_concurrent_embeddings chunks
            batch_chunks = list(islice(chunks, i, i + max_concurrent_embeddings))
            i += max_concurrent_embeddings

            # Process this batch in parallel
            results = await asyncio.gather(
                *[process_chunk(chunk) for chunk in batch_chunks],
                return_exceptions=False,
            )

            successful_chunks += sum(1 for result in results if result)

        logger.info(
            f"Successfully indexed {successful_chunks}/{len(chunks)} chunks for page {page_id}",
        )

        return page_id

    except Exception as e:
        logger.error(f"Error processing page {page_result.url}: {e!s}")
        raise


async def process_crawl_with_hierarchy(
    url: str,
    job_id: str,
    tags: list[str] | None = None,
    max_pages: int = 100,
    max_depth: int = 2,
    strip_urls: bool = True,
) -> list[str]:
    """Crawl a URL and process all pages with hierarchy tracking.

    Args:
        url: The URL to start crawling from.
        job_id: The ID of the crawl job.
        tags: Optional tags to associate with the pages.
        max_pages: Maximum number of pages to crawl.
        max_depth: Maximum depth for the BFS crawl.
        strip_urls: Whether to strip URLs from the returned markdown.

    Returns:
        List of processed page IDs.
    """
    if tags is None:
        tags = []

    # Crawl with hierarchy tracking
    crawl_results = await crawl_url_with_hierarchy(url, max_pages, max_depth, strip_urls)

    # Sort results by depth to ensure parents are processed before children
    crawl_results.sort(key=lambda r: r.depth)

    # Map to track URL to page ID relationships
    url_to_page_id = {}
    processed_page_ids = []

    # Process pages in order
    for i, page_result in enumerate(crawl_results):
        try:
            page_id = await process_crawl_result_with_hierarchy(
                page_result, job_id, tags, url_to_page_id=url_to_page_id
            )
            processed_page_ids.append(page_id)

            # Update job progress
            db_ops_status = DatabaseOperations()
            await db_ops_status.update_job_status(
                job_id=job_id,
                status="running",
                pages_discovered=len(crawl_results),
                pages_crawled=len(processed_page_ids),
            )

        except Exception as page_error:
            logger.error(f"Error processing page {page_result.url}: {page_error!s}")
            continue

    return processed_page_ids
