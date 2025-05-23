"""Page processing pipeline combining crawling, chunking, embedding, and indexing."""

import asyncio
from itertools import islice
from typing import Any
from urllib.parse import urlparse

from src.common.indexer import VectorIndexer
from src.common.logger import get_logger
from src.lib.chunker import TextChunker
from src.lib.crawler import extract_page_text
from src.lib.database import DatabaseOperations
from src.lib.embedder import generate_embedding

# Configure logging
logger = get_logger(__name__)


async def process_crawl_result(
    page_result: Any,
    job_id: str,
    tags: list[str] | None = None,
    max_concurrent_embeddings: int = 5,
) -> str:
    """Process a single crawled page result through the entire pipeline.

    Args:
        page_result: The crawl result for the page
        job_id: The ID of the crawl job
        tags: Optional tags to associate with the page
        max_concurrent_embeddings: Maximum number of concurrent embedding generations

    Returns:
        The ID of the processed page

    """
    if tags is None:
        tags = []  # Initialize as empty list instead of None

    try:
        logger.info(f"Processing page: {page_result.url}")

        # Extract text from the crawl result
        page_text = extract_page_text(page_result)

        # Store the page in the database
        db_ops = DatabaseOperations()
        page_id = await db_ops.store_page(  # store_page now handles its own connection
            url=page_result.url,
            text=page_text,
            job_id=job_id,
            tags=tags,
        )
        # No explicit close needed as store_page uses context manager internally

        # Initialize components
        chunker = TextChunker()
        indexer = VectorIndexer()  # VectorIndexer will now create and manage its own connection.

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
    finally:
        # VectorIndexer now manages its own connection if instantiated without one.
        # Its __del__ method will attempt to close its connection if self._own_connection is True.
        # No explicit action needed here for the indexer's connection.
        pass


async def process_page_batch(
    page_results: list[Any],
    job_id: str,
    tags: list[str] | None = None,
    batch_size: int = 10,
) -> list[str]:
    """Process a batch of crawled pages.

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
                db_ops_status = DatabaseOperations()
                await (
                    db_ops_status.update_job_status(  # update_job_status handles its own connection
                        job_id=job_id,
                        status="running",
                        pages_discovered=len(page_results),
                        pages_crawled=len(processed_page_ids),
                    )
                )

            except Exception as page_error:
                logger.error(f"Error in batch processing for {page_result.url}: {page_error!s}")
                continue

    return processed_page_ids
