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
import redis

from src.common.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, REDIS_URI
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

    # Update in Redis
    _update_redis_job_status(job_id, "running", now)

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

        # Update in Redis
        _update_redis_job_status(job_id, "completed", now)

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

        # Update in Redis
        _update_redis_job_status(job_id, "failed", now, error_msg)

        return {"job_id": job_id, "status": "failed", "error": error_msg}
    finally:
        conn.close()


def _update_redis_job_status(
    job_id: str, status: str, updated_at: datetime.datetime, error_message: Optional[str] = None
) -> None:
    """
    Update job status in Redis.

    Args:
        job_id: The job ID
        status: Job status
        updated_at: Timestamp
        error_message: Optional error message
    """
    try:
        # Connect to Redis
        redis_conn = redis.from_url(REDIS_URI)

        # Get existing data if any
        job_key = f"job_status:{job_id}"

        # Use existing data or create new
        job_data = {"status": status, "updated_at": updated_at.isoformat()}

        if error_message:
            job_data["error_message"] = error_message

        # Store in Redis
        redis_conn.hset(job_key, mapping=job_data)
        # Set expiration to 24 hours
        redis_conn.expire(job_key, 86400)
    except Exception as e:
        # Log but continue - Redis storage is secondary
        logger.warning(f"Failed to update Redis with job status: {str(e)}")


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
                if pages_crawled % 5 == 0 or pages_crawled == pages_discovered:
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
    # Generate a unique page ID
    page_id = str(uuid.uuid4())

    # Extract domain from URL
    domain = urlparse(page_result.url).netloc

    # Store page in DuckDB
    logger.info(f"Storing page {page_id} from {page_result.url}")

    # Use markdown if available, otherwise use extracted content or HTML
    if hasattr(page_result, "_markdown") and page_result._markdown:
        page_text = page_result._markdown.raw_markdown
    elif page_result.extracted_content:
        page_text = page_result.extracted_content
    else:
        page_text = page_result.html

    # Store page data
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
    conn.commit()

    # Split text into chunks
    chunks = text_splitter.split_text(page_text)

    # Process each chunk
    for chunk_text in chunks:
        # Skip empty chunks
        if not chunk_text.strip():
            continue

        # Generate embedding
        try:
            embedding_response = litellm.embedding(model=EMBEDDING_MODEL, input=[chunk_text])
            embedding = embedding_response["data"][0]["embedding"]

            # Generate a unique point ID
            point_id = str(uuid.uuid4())

            # Prepare payload
            payload = {"text": chunk_text, "page_id": page_id, "domain": domain, "tags": tags}

            # Upsert the point into Qdrant
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[{"id": point_id, "vector": embedding, "payload": payload}],
            )

        except Exception as e:
            logger.error(f"Error generating embedding for chunk: {str(e)}")
            continue


def _update_job_progress(
    conn: duckdb.DuckDBPyConnection, job_id: str, pages_discovered: int, pages_crawled: int
) -> None:
    """
    Update the job progress in the database and Redis.

    Args:
        conn: DuckDB connection
        job_id: The job ID
        pages_discovered: Number of pages discovered
        pages_crawled: Number of pages crawled
    """
    # Update in DuckDB
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

    # Also update in Redis for real-time access
    try:
        # Get current job status from DuckDB
        status_result = conn.execute(
            "SELECT status, error_message FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

        if status_result:
            status, error_message = status_result

            # Connect to Redis
            redis_conn = redis.from_url(REDIS_URI)

            # Store job progress in Redis hash
            job_key = f"job_status:{job_id}"
            job_data = {
                "pages_discovered": str(pages_discovered),
                "pages_crawled": str(pages_crawled),
                "status": status,
                "updated_at": now.isoformat(),
            }

            if error_message:
                job_data["error_message"] = error_message

            # Store in Redis
            redis_conn.hset(job_key, mapping=job_data)
            # Set expiration to 24 hours
            redis_conn.expire(job_key, 86400)
    except Exception as e:
        # Log but continue - Redis storage is secondary to DuckDB
        logger.warning(f"Failed to update Redis with job progress: {str(e)}")
