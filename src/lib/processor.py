"""Page processing pipeline combining crawling, chunking, embedding, and indexing."""

import logging
from typing import List, Any
from urllib.parse import urlparse

from src.lib.crawler import extract_page_text
from src.lib.chunker import TextChunker
from src.lib.embedder import generate_embedding
from src.lib.indexer import VectorIndexer
from src.lib.database import store_page, update_job_status

# Configure logging
logger = logging.getLogger(__name__)


async def process_crawl_result(
    page_result: Any,
    job_id: str,
    tags: List[str] = None,
) -> str:
    """
    Process a single crawled page result through the entire pipeline.

    Args:
        page_result: The crawl result for the page
        job_id: The ID of the crawl job
        tags: Optional tags to associate with the page

    Returns:
        The ID of the processed page
    """
    if tags is None:
        tags = []

    try:
        logger.info(f"Processing page: {page_result.url}")

        # Extract text from the crawl result
        page_text = extract_page_text(page_result)

        # Store the page in the database
        page_id = await store_page(
            url=page_result.url,
            text=page_text,
            job_id=job_id,
            tags=tags,
        )

        # Initialize components
        chunker = TextChunker()
        indexer = VectorIndexer()

        # Split text into chunks
        chunks = chunker.split_text(page_text)
        logger.info(f"Split page into {len(chunks)} chunks")

        # Extract domain from URL
        domain = urlparse(page_result.url).netloc

        # Process each chunk
        successful_chunks = 0
        for chunk_text in chunks:
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
                successful_chunks += 1

            except Exception as chunk_error:
                logger.error(f"Error processing chunk: {str(chunk_error)}")
                continue

        logger.info(
            f"Successfully indexed {successful_chunks}/{len(chunks)} chunks for page {page_id}"
        )

        return page_id

    except Exception as e:
        logger.error(f"Error processing page {page_result.url}: {str(e)}")
        raise


async def process_page_batch(
    page_results: List[Any],
    job_id: str,
    tags: List[str] = None,
    batch_size: int = 10,
) -> List[str]:
    """
    Process a batch of crawled pages.

    Args:
        page_results: List of crawl results
        job_id: The ID of the crawl job
        tags: Optional tags to associate with the pages
        batch_size: Size of batches for processing

    Returns:
        List of processed page IDs
    """
    if tags is None:
        tags = []

    processed_page_ids = []

    # Process pages in smaller batches to avoid memory issues
    for i in range(0, len(page_results), batch_size):
        batch = page_results[i : i + batch_size]
        logger.info(f"Processing batch of {len(batch)} pages (batch {i // batch_size + 1})")

        for page_result in batch:
            try:
                page_id = await process_crawl_result(page_result, job_id, tags)
                processed_page_ids.append(page_id)

                # Update job progress
                update_job_status(
                    job_id=job_id,
                    status="running",
                    pages_discovered=len(page_results),
                    pages_crawled=len(processed_page_ids),
                )

            except Exception as page_error:
                logger.error(f"Error in batch processing for {page_result.url}: {str(page_error)}")
                continue

    return processed_page_ids
