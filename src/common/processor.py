"""Page processing pipeline combining crawling, chunking, embedding, and indexing."""

import asyncio
from itertools import islice
from typing import Any
from urllib.parse import urlparse

from src.common.indexer import VectorIndexer
from src.common.logger import get_logger
from src.lib.chunker import TextChunker
from src.lib.crawler import extract_page_text
from src.lib.database import Database, batch_store_pages, BatchJobUpdate, BatchExecutor
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
        db = Database()
        try:
            page_id = await db.store_page(
                url=page_result.url,
                text=page_text,
                job_id=job_id,
                tags=tags,
            )
        finally:
            db.close()

        # Initialize components
        chunker = TextChunker()

        # Get a DuckDB connection for the vector indexer
        db = Database()
        duckdb_conn = db.connect()
        indexer = VectorIndexer(connection=duckdb_conn)

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
        # Close the DuckDB connection if it exists
        indexer_var = locals().get("indexer")
        if indexer_var and hasattr(indexer_var, "conn"):
            try:
                # The connection will be closed by the VectorIndexer's destructor
                # but we explicitly set _own_connection to False since we created it
                indexer_var._own_connection = True
            except Exception as close_error:
                logger.warning(f"Error marking connection for closure: {close_error}")


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
    total_pages = len(page_results)

    # Process pages in smaller batches to avoid memory issues
    for i in range(0, total_pages, batch_size):
        batch = page_results[i : i + batch_size]
        logger.info(f"Processing batch of {len(batch)} pages (batch {i // batch_size + 1})")

        # Extract text from all pages in this batch
        page_data = []
        for page_result in batch:
            try:
                page_text = extract_page_text(page_result)
                page_data.append((page_result.url, page_text, job_id, tags, None))
            except Exception as page_error:
                logger.error(f"Error extracting text from {page_result.url}: {page_error!s}")
                continue

        # Batch store all pages at once for better performance
        if page_data:
            try:
                batch_page_ids = await batch_store_pages(page_data, max_batch_size=batch_size)
                processed_page_ids.extend(batch_page_ids)

                # Update job progress using batch update
                job_update = BatchJobUpdate()
                job_update.add_job_update(
                    job_id=job_id,
                    status="running",
                    pages_discovered=total_pages,
                    pages_crawled=len(processed_page_ids),
                )

                # Execute the batch update
                executor = BatchExecutor(checkpoint_after=False)
                await executor.execute_batch(job_update)

                # Process the stored pages for embedding and indexing
                for idx, page_id in enumerate(batch_page_ids):
                    try:
                        # Initialize components
                        chunker = TextChunker()

                        # Get a DuckDB connection for the vector indexer
                        db = Database()
                        duckdb_conn = db.connect()
                        indexer = VectorIndexer(connection=duckdb_conn)

                        # Extract page data from our batch
                        page_url = page_data[idx][0]
                        page_text = page_data[idx][1]

                        # Split text into chunks
                        chunks = chunker.split_text(page_text)
                        logger.info(f"Split page into {len(chunks)} chunks")

                        # Extract domain from URL
                        domain = urlparse(page_url).netloc

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
                                    "url": page_url,
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
                        j = 0
                        while j < len(chunks):
                            # Take up to max_concurrent_embeddings chunks
                            batch_chunks = list(
                                islice(chunks, j, j + 5)
                            )  # Max 5 concurrent embeddings
                            j += 5

                            # Process this batch in parallel
                            results = await asyncio.gather(
                                *[process_chunk(chunk) for chunk in batch_chunks],
                                return_exceptions=False,
                            )

                            successful_chunks += sum(1 for result in results if result)

                        logger.info(
                            f"Successfully indexed {successful_chunks}/{len(chunks)} chunks for page {page_id}",
                        )
                    except Exception as e:
                        logger.error(f"Error processing embeddings for page {page_id}: {e!s}")
                    finally:
                        # Close the DuckDB connection
                        db.close()

            except Exception as batch_error:
                logger.error(f"Error in batch storage: {batch_error!s}")

    # Perform a final checkpoint after all batches are processed
    if processed_page_ids:
        checkpoint_executor = BatchExecutor(checkpoint_after=True)
        job_update = BatchJobUpdate()
        job_update.add_job_update(
            job_id=job_id,
            status="running",
            pages_discovered=total_pages,
            pages_crawled=len(processed_page_ids),
        )
        await checkpoint_executor.execute_batch(job_update)

    return processed_page_ids
