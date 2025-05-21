"""Task definitions for the Doctor Crawl Worker."""

import asyncio
import datetime
from typing import Any

import redis
from rq import Queue

from src.common.config import REDIS_URI
from src.common.logger import get_logger
from src.common.processor import process_page_batch
from src.lib.crawler import crawl_url
from src.lib.database import Database, get_connection, BatchJobUpdate, BatchExecutor

# Get logger for this module
logger = get_logger(__name__)


async def create_job(
    url: str,
    job_id: str,
    tags: list[str] | None = None,
    max_pages: int = 100,
) -> str:
    """Create a new crawl job in the database.

    Args:
        url: The URL to start crawling from
        job_id: Pre-generated job ID
        tags: Optional tags to associate with the crawled pages
        max_pages: Maximum number of pages to crawl

    Returns:
        The job ID

    """
    if tags is None:
        tags = []

    logger.info(f"Creating new job {job_id} for URL: {url}, max pages: {max_pages}")
    now = datetime.datetime.now()

    try:
        # Get a connection from the pool (read-write)
        async with await get_connection(read_only=False) as conn_manager:
            # Insert into DuckDB
            logger.info(f"Inserting job {job_id} into DuckDB")
            conn = await conn_manager.async_ensure_connection()
            conn_manager.begin_transaction()

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
                    Database.serialize_tags(tags),
                    now,
                    now,
                ),
            )
            # Commit the transaction
            conn_manager.commit()

            # Ensure changes are persisted with a checkpoint
            conn.execute("CHECKPOINT")

            # Verify the job was created by reading it back
            job_record = conn.execute(
                "SELECT job_id FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not job_record:
                logger.error(f"Job {job_id} was not successfully created in the database!")
                raise Exception(f"Failed to create job {job_id} in the database")

        logger.info(f"Successfully created job {job_id} in DuckDB")

        # Enqueue the crawl task
        redis_conn = redis.from_url(REDIS_URI)
        queue = Queue("worker", connection=redis_conn)

        # Enqueue the crawl task
        logger.info(f"Enqueueing crawl task for job {job_id}")
        queue.enqueue(
            "src.crawl_worker.tasks.perform_crawl",
            job_id,
            url,
            tags,
            max_pages,
            job_timeout=600,
        )

        logger.info(f"Enqueued crawl task for job {job_id}")

        return job_id
    except Exception as e:
        logger.exception(f"Error creating job for URL {url}: {e!s}")
        raise


def perform_crawl(
    job_id: str,
    url: str,
    tags: list[str] | None = None,
    max_pages: int = 100,
) -> dict[str, Any]:
    """Perform a crawl job.

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

    # Update job status to running using batch operations
    batch_update = BatchJobUpdate()
    batch_update.add_job_update(job_id, "running")
    executor = BatchExecutor()

    # Use asyncio.run to execute the async batch operation in a sync context
    asyncio.run(executor.execute_batch(batch_update))

    try:
        # Run the crawl and processing pipeline
        result = asyncio.run(_execute_pipeline(job_id, url, tags, max_pages))

        # Update job status to completed
        batch_update = BatchJobUpdate()
        batch_update.add_job_update(job_id, "completed")
        executor = BatchExecutor(checkpoint_after=True)
        asyncio.run(executor.execute_batch(batch_update))

        logger.info(f"Completed crawl job {job_id}: {result}")
        return result

    except Exception as e:
        logger.exception(f"Error in crawl job {job_id}: {e!s}")

        # Update job status to failed
        batch_update = BatchJobUpdate()
        batch_update.add_job_update(job_id, "failed", error_message=str(e))
        executor = BatchExecutor(checkpoint_after=True)
        asyncio.run(executor.execute_batch(batch_update))

        return {"job_id": job_id, "status": "failed", "error": str(e)}


async def _execute_pipeline(
    job_id: str,
    url: str,
    tags: list[str],
    max_pages: int,
) -> dict[str, Any]:
    """Execute the crawl and processing pipeline.

    Args:
        job_id: The unique ID of the job
        url: The URL to start crawling from
        tags: Tags to associate with the crawled pages
        max_pages: Maximum number of pages to crawl

    Returns:
        Dict with job status information

    """
    logger.info(f"Job {job_id}: Starting pipeline for URL: {url} with max_pages={max_pages}")

    # Update job status using batch operations
    batch_update = BatchJobUpdate()
    batch_update.add_job_update(
        job_id=job_id,
        status="running",
        pages_discovered=0,
        pages_crawled=0,
    )
    executor = BatchExecutor()
    await executor.execute_batch(batch_update)

    # Step 1: Crawl the URL - this doesn't require database connections
    logger.info(f"Job {job_id}: Beginning crawling phase from URL: {url}")
    crawl_results = await crawl_url(url, max_pages=max_pages)
    pages_discovered = len(crawl_results)

    logger.info(f"Job {job_id}: Crawl discovered {pages_discovered} pages")

    # Update job progress after crawl phase using batch operations
    batch_update = BatchJobUpdate()
    batch_update.add_job_update(
        job_id=job_id,
        status="running",
        pages_discovered=pages_discovered,
        pages_crawled=0,
    )
    await executor.execute_batch(batch_update)

    # Step 2: Process the crawled pages
    if crawl_results:
        logger.info(f"Job {job_id}: Beginning processing phase for {pages_discovered} pages")

        # Process the pages using batch operations
        processed_page_ids = await process_page_batch(
            page_results=crawl_results,
            job_id=job_id,
            tags=tags,
        )

        pages_crawled = len(processed_page_ids)
        logger.info(
            f"Job {job_id}: Successfully processed {pages_crawled}/{pages_discovered} pages",
        )

        # Final update before completing
        batch_update = BatchJobUpdate()
        batch_update.add_job_update(
            job_id=job_id,
            status="running",
            pages_discovered=pages_discovered,
            pages_crawled=pages_crawled,
        )
        await executor.execute_batch(batch_update)
    else:
        logger.warning(f"Job {job_id}: No pages were discovered during crawl")
        pages_crawled = 0

        # Update status for empty crawl
        batch_update = BatchJobUpdate()
        batch_update.add_job_update(
            job_id=job_id,
            status="running",
            pages_discovered=0,
            pages_crawled=0,
            error_message="No pages were discovered during crawl",
        )
        await executor.execute_batch(batch_update)

    # Return final status
    logger.info(
        f"Job {job_id}: Pipeline completed with {pages_crawled}/{pages_discovered} pages processed",
    )
    return {
        "job_id": job_id,
        "status": "completed",
        "pages_discovered": pages_discovered,
        "pages_crawled": pages_crawled,
    }


def delete_docs(
    task_id: str,
    tags: list[str] | None = None,
    domain: str | None = None,
    page_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Delete documents from the database based on filters.

    Args:
        task_id: Unique ID for the delete task
        tags: Optional list of tags to filter by
        domain: Optional domain substring to filter by
        page_ids: Optional list of specific page IDs to delete

    Returns:
        Dictionary with deletion statistics

    """
    logger.info(
        f"Starting delete task {task_id} with filters: tags={tags}, domain={domain}, page_ids={page_ids}",
    )

    # Use asyncio.run to execute async operations in a sync context
    async def delete_async():
        # Build the SQL where clause and parameters outside connection scope
        # to minimize connection time
        conditions = []
        params = []

        if page_ids and len(page_ids) > 0:
            placeholders = ", ".join(["?" for _ in page_ids])
            conditions.append(f"id IN ({placeholders})")
            params.extend(page_ids)

        if domain:
            conditions.append("domain LIKE ?")
            params.append(f"%{domain}%")

        if tags and len(tags) > 0:
            tag_conditions = []
            for tag in tags:
                # Format the tag for JSON contains - note we need quotes around the value
                # First parameter is the haystack (tags), second is the needle (our tag)
                tag_conditions.append(f"json_valid(tags) AND json_contains(tags, '\"{tag}\"')")

            conditions.append(f"({' OR '.join(tag_conditions)})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Only acquire connection when ready to execute query
        async with await get_connection(read_only=False) as conn_manager:
            conn = await conn_manager.async_ensure_connection()
            conn_manager.begin_transaction()

            try:
                # Get the IDs of pages that will be deleted
                query = f"SELECT id FROM pages WHERE {where_clause}"
                logger.debug(f"Executing query to get page IDs: {query}")
                results = conn.execute(query, params).fetchall()
                page_ids_to_delete = [row[0] for row in results]

                # Then delete from SQL database
                delete_query = f"DELETE FROM pages WHERE {where_clause}"
                logger.debug(f"Executing delete query: {delete_query}")
                cursor = conn.execute(delete_query, params)
                deleted_pages = cursor.rowcount

                # Commit the changes
                conn_manager.commit()

                # Force a checkpoint for durability
                conn.execute("CHECKPOINT")

                logger.info(f"Deleted {deleted_pages} pages ")

                return {
                    "task_id": task_id,
                    "deleted_pages": deleted_pages,
                    "page_ids": page_ids_to_delete,
                }
            except Exception as e:
                conn_manager.rollback()
                logger.error(f"Error deleting documents: {e!s}")
                raise

    try:
        result = asyncio.run(delete_async())
        return result
    except Exception as e:
        logger.error(f"Error in delete_docs: {e!s}")
        raise
