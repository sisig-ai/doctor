"""Page processing pipeline combining crawling, chunking, embedding, and indexing."""

from typing import List, Any
from urllib.parse import urlparse
import asyncio
from itertools import islice

from src.lib.crawler import extract_page_text
from src.lib.chunker import TextChunker
from src.lib.embedder import generate_embedding
from src.common.indexer import VectorIndexer
from src.common.logger import get_logger
from src.lib.database import Database

# Configure logging
logger = get_logger(__name__)


async def process_crawl_result(
    page_result: Any,
    job_id: str,
    tags: List[str] | None = None,
    max_concurrent_embeddings: int = 5,
) -> str:
    """
    Process a single crawled page result through the entire pipeline.

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
                logger.error(f"Error processing chunk: {str(chunk_error)}")
                return False

        # Process chunks in batches with limited concurrency
        i = 0
        while i < len(chunks):
            # Take up to max_concurrent_embeddings chunks
            batch_chunks = list(islice(chunks, i, i + max_concurrent_embeddings))
            i += max_concurrent_embeddings

            # Process this batch in parallel
            results = await asyncio.gather(
                *[process_chunk(chunk) for chunk in batch_chunks], return_exceptions=False
            )

            successful_chunks += sum(1 for result in results if result)

        logger.info(
            f"Successfully indexed {successful_chunks}/{len(chunks)} chunks for page {page_id}"
        )

        return page_id

    except Exception as e:
        logger.error(f"Error processing page {page_result.url}: {str(e)}")
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
    page_results: List[Any],
    job_id: str,
    tags: List[str] | None = None,
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
                with Database() as db:
                    db.update_job_status(
                        job_id=job_id,
                        status="running",
                        pages_discovered=len(page_results),
                        pages_crawled=len(processed_page_ids),
                    )

            except Exception as page_error:
                logger.error(f"Error in batch processing for {page_result.url}: {str(page_error)}")
                continue

    return processed_page_ids
