"""Task definitions for the Doctor Crawl Worker."""

import logging
import uuid
import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import duckdb
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
import litellm

from src.common.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.common.db_setup import (
    get_duckdb_connection,
    get_qdrant_client,
    QDRANT_COLLECTION_NAME,
    serialize_tags,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_job(
    url: str, tags: Optional[List[str]] = None, max_pages: int = 100, job_id: Optional[str] = None
) -> str:
    """
    Create a new crawl job in the database.

    Args:
        url: The URL to start crawling from
        tags: Optional tags to associate with the crawled pages
        max_pages: Maximum number of pages to crawl
        job_id: Optional pre-generated job ID

    Returns:
        The job ID
    """
    if tags is None:
        tags = []

    # Use provided job_id or generate a new one
    if job_id is None:
        job_id = str(uuid.uuid4())

    logger.info(f"Creating new job {job_id} for URL: {url}, max pages: {max_pages}")

    # Create job record in DuckDB
    conn = get_duckdb_connection()
    now = datetime.datetime.now()

    try:
        # Insert into DuckDB
        logger.info(f"Inserting job {job_id} into DuckDB")
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, start_url, status, pages_discovered, pages_crawled,
                max_pages, tags, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                str(url),
                "pending",
                0,
                0,
                max_pages,
                serialize_tags(tags),
                now,
                now,
            ),
        )
        # Ensure the changes are written to disk
        conn.commit()

        # Force a checkpoint to ensure changes are persisted
        conn.execute("CHECKPOINT")

        # Verify the job was created by reading it back
        job_record = conn.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not job_record:
            logger.error(f"Job {job_id} was not successfully created in the database!")
            raise Exception(f"Failed to create job {job_id} in the database")

        logger.info(f"Successfully created job {job_id} in DuckDB")

        # Enqueue the crawl task
        from rq import Queue
        import redis
        from src.common.config import REDIS_URI

        redis_conn = redis.from_url(REDIS_URI)
        default_queue = Queue("default", connection=redis_conn)

        # Clean up the job request from Redis if it exists
        job_request_key = f"job_request:{job_id}"
        if redis_conn.exists(job_request_key):
            logger.info(f"Cleaning up job request {job_id} from Redis")
            redis_conn.delete(job_request_key)

        # Enqueue the crawl task
        logger.info(f"Enqueueing crawl task for job {job_id}")
        default_queue.enqueue(
            perform_crawl,
            job_id,  # Positional argument
            url=url,
            tags=tags,
            max_pages=max_pages,
            job_timeout="1h",  # Set a reasonable timeout
        )

        logger.info(f"Enqueued crawl task for job {job_id}")

        return job_id
    except Exception as e:
        logger.exception(f"Error creating job for URL {url}: {str(e)}")
        raise
    finally:
        conn.close()


def perform_crawl(
    job_id: str, url: str, tags: Optional[List[str]] = None, max_pages: int = 100
) -> Dict[str, Any]:
    """
    Perform a crawl job.

    Args:
        job_id: The unique ID of the job
        url: The URL to start crawling from
        tags: Optional tags to associate with the crawled pages
        max_pages: Maximum number of pages to crawl

    Returns:
        Dict with job status information
    """
    if tags is None:
        tags = []

    logger.info(f"Starting crawl job {job_id} for URL: {url}, max pages: {max_pages}")

    # Update job status to running
    conn = get_duckdb_connection()
    now = datetime.datetime.now()

    # Update in DuckDB
    conn.execute(
        """
        UPDATE jobs
        SET status = 'running', updated_at = ?
        WHERE job_id = ?
        """,
        (now, job_id),
    )
    conn.commit()

    # Force a checkpoint to ensure changes are persisted
    conn.execute("CHECKPOINT")

    try:
        # Run the crawl
        result = asyncio.run(_crawl_and_process(job_id, url, tags, max_pages, conn))

        # Update job status to completed
        now = datetime.datetime.now()
        conn.execute(
            """
            UPDATE jobs
            SET status = 'completed', updated_at = ?
            WHERE job_id = ?
            """,
            (now, job_id),
        )
        conn.commit()

        # Force a checkpoint to ensure changes are persisted
        conn.execute("CHECKPOINT")

        logger.info(f"Completed crawl job {job_id}: {result}")
        return result

    except Exception as e:
        logger.exception(f"Error in crawl job {job_id}: {str(e)}")

        # Update job status to failed
        now = datetime.datetime.now()
        error_msg = str(e)

        # Update in DuckDB
        conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', error_message = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (error_msg, now, job_id),
        )
        conn.commit()

        # Force a checkpoint to ensure changes are persisted
        conn.execute("CHECKPOINT")

        return {"job_id": job_id, "status": "failed", "error": error_msg}
    finally:
        conn.close()


def _update_job_progress(
    conn: duckdb.DuckDBPyConnection, job_id: str, pages_discovered: int, pages_crawled: int
) -> None:
    """
    Update the job progress in the database.

    Args:
        conn: DuckDB connection
        job_id: The job ID
        pages_discovered: Number of pages discovered
        pages_crawled: Number of pages crawled
    """
    # Only update DuckDB
    now = datetime.datetime.now()
    conn.execute(
        """
        UPDATE jobs
        SET pages_discovered = ?, pages_crawled = ?, updated_at = ?
        WHERE job_id = ?
        """,
        (pages_discovered, pages_crawled, now, job_id),
    )
    conn.commit()

    # Periodically force a checkpoint to ensure changes are persisted
    # Don't do this every time to avoid performance impact
    if pages_crawled % 10 == 0 or pages_crawled == pages_discovered:
        try:
            conn.execute("CHECKPOINT")
            logger.info(f"Forced checkpoint after processing {pages_crawled} pages")
        except Exception as e:
            logger.warning(f"Failed to checkpoint: {str(e)}")


async def _crawl_and_process(
    job_id: str, url: str, tags: List[str], max_pages: int, conn: duckdb.DuckDBPyConnection
) -> Dict[str, Any]:
    """
    Crawl the URL and process the results.

    Args:
        job_id: The unique ID of the job
        url: The URL to start crawling from
        tags: Tags to associate with the crawled pages
        max_pages: Maximum number of pages to crawl
        conn: DuckDB connection

    Returns:
        Dict with job status information
    """
    # Initialize the crawler
    async with AsyncWebCrawler() as crawler:
        # Set up counters
        pages_discovered = 0
        pages_crawled = 0

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        # Initialize Qdrant client
        qdrant_client = get_qdrant_client()

        # Configure the crawl run for deep crawling
        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_pages=max_pages,
                max_depth=3,
                # include_external=False, # Optionally keep crawl within the same domain
            ),
            # Define other parameters if needed, e.g., scraping_strategy
            # verbose=True # Useful for more detailed crawl4ai logging
        )

        # Crawl the URL
        logger.info(f"Job {job_id}: Starting deep crawl for URL: {url} with max_pages={max_pages}")

        # arun now returns a list of CrawlResult objects when deep crawling
        crawl_results: List[Any] = await crawler.arun(url=url, config=config)

        pages_discovered = len(crawl_results)
        logger.info(f"Job {job_id}: Deep crawl discovered {pages_discovered} pages.")

        logger.error(f"DEBUGGING: Crawl complete, about to process {len(crawl_results)} results")

        # Process each crawled page result
        for i, page_result in enumerate(crawl_results):
            # Check max_pages again just in case crawl4ai returns more than requested
            if pages_crawled >= max_pages:
                logger.info(
                    f"Job {job_id}: Max pages limit ({max_pages}) reached during processing, stopping."
                )
                break

            logger.info(
                f"Job {job_id}: Processing page {i + 1}/{pages_discovered}: {page_result.url}"
            )
            try:
                await _process_page(page_result, text_splitter, qdrant_client, conn, job_id, tags)
                pages_crawled += 1
                # Update progress less frequently to avoid excessive DB writes
                if (
                    pages_crawled % 5 == 0
                    or pages_crawled == pages_discovered
                    or pages_discovered < 5
                ):
                    _update_job_progress(conn, job_id, pages_discovered, pages_crawled)
            except Exception as page_error:
                logger.error(f"Job {job_id}: Error processing page {page_result.url}: {page_error}")

        # Ensure final progress is updated
        _update_job_progress(conn, job_id, pages_discovered, pages_crawled)

        logger.info(
            f"Job {job_id}: Finished processing. Discovered: {pages_discovered}, Crawled & Processed: {pages_crawled}"
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "pages_discovered": pages_discovered,
            "pages_crawled": pages_crawled,
        }


async def _process_page(
    page_result, text_splitter, qdrant_client, conn, job_id, tags: List[str]
) -> None:
    """
    Process a single crawled page.

    Args:
        page_result: The crawl result for the page
        text_splitter: The text splitter to use
        qdrant_client: The Qdrant client
        conn: DuckDB connection
        job_id: The job ID
        tags: Tags to associate with the page
    """
    logger.error(f"DEBUGGING: Starting to process page {page_result.url}")
    # Generate a unique page ID
    page_id = str(uuid.uuid4())

    # Extract domain from URL
    domain = urlparse(page_result.url).netloc

    # Store page in DuckDB
    logger.info(f"Storing page {page_id} from {page_result.url}")

    # Use markdown if available, otherwise use extracted content or HTML
    if hasattr(page_result, "_markdown") and page_result._markdown:
        page_text = page_result._markdown.raw_markdown
        logger.error(f"DEBUGGING: Using markdown text of length {len(page_text)}")
    elif page_result.extracted_content:
        page_text = page_result.extracted_content
        logger.error(f"DEBUGGING: Using extracted content of length {len(page_text)}")
    else:
        page_text = page_result.html
        logger.error(f"DEBUGGING: Using HTML content of length {len(page_text)}")

    # Store page data
    logger.error("DEBUGGING: About to insert page into DuckDB")
    conn.execute(
        """
        INSERT INTO pages (id, url, domain, raw_text, crawl_date, tags)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            page_id,
            page_result.url,
            domain,
            page_text,
            datetime.datetime.now(),
            serialize_tags(tags),
        ),
    )
    try:
        # Database operations
        conn.commit()
        logger.error("DEBUGGING: Successfully committed page to DuckDB")
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise

    # Split text into chunks
    logger.error("DEBUGGING: Starting text splitting")
    chunks = text_splitter.split_text(page_text)
    logger.error(f"DEBUGGING: Split text into {len(chunks)} chunks")

    # Process each chunk
    chunk_count = 0
    success_count = 0
    for i, chunk_text in enumerate(chunks):
        # Skip empty chunks
        if not chunk_text.strip():
            logger.error(f"DEBUGGING: Skipping empty chunk #{i}")
            continue

        chunk_count += 1
        # Generate embedding
        try:
            logger.error(
                f"DEBUGGING: Generating embedding for chunk #{i} of length {len(chunk_text)}"
            )
            embedding_response = litellm.embedding(
                model=EMBEDDING_MODEL,
                input=[chunk_text],
                timeout=30,  # Add timeout in seconds
            )
            logger.error(f"DEBUGGING: Successfully received embedding response for chunk #{i}")
            embedding = embedding_response["data"][0]["embedding"]

            # Generate a unique point ID
            point_id = str(uuid.uuid4())

            # Prepare payload
            payload = {"text": chunk_text, "page_id": page_id, "domain": domain, "tags": tags}

            # Upsert the point into Qdrant
            logger.error(f"DEBUGGING: Upserting point into Qdrant collection for chunk #{i}")
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[{"id": point_id, "vector": embedding, "payload": payload}],
            )
            logger.error(f"DEBUGGING: Successfully stored point in Qdrant for chunk #{i}")
            success_count += 1

        except Exception as e:
            logger.error(f"Embedding error for chunk #{i}: {e}")
            continue

    logger.error(
        f"DEBUGGING: Completed processing page {page_result.url} - processed {chunk_count} chunks with {success_count} successful embeddings"
    )
