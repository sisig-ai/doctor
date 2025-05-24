"""Task definitions for the Doctor Crawl Worker."""

import asyncio
import datetime
from typing import Any

import redis
from rq import Queue

from src.common.config import REDIS_URI
from src.common.logger import get_logger
from src.common.processor_enhanced import process_crawl_with_hierarchy
from src.lib.database import DatabaseOperations
from src.lib.database.schema import CREATE_FTS_INDEX_SQL
from src.lib.database.utils import serialize_tags

# Get logger for this module
logger = get_logger(__name__)


def create_job(
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

    # Create job record in DuckDB
    db_ops = DatabaseOperations()
    now = datetime.datetime.now()

    try:
        with db_ops.db as conn_manager:  # Use DuckDBConnectionManager as context manager
            actual_conn = conn_manager.conn
            if not actual_conn:
                raise RuntimeError("Failed to get DB connection for create_job")

            logger.info(f"Inserting job {job_id} into DuckDB")
            conn_manager.begin_transaction()
            actual_conn.execute(
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
            conn_manager.commit()

            # Force a checkpoint to ensure changes are persisted
            actual_conn.execute("CHECKPOINT")

            # Verify the job was created by reading it back
            job_record = actual_conn.execute(
                "SELECT job_id FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not job_record:
                logger.error(f"Job {job_id} was not successfully created in the database!")
                # Rollback if verification fails, though commit was done.
                # This state is problematic. Ideally, verification is part of the transaction.
                # For now, just log and raise.
                # conn_manager.rollback() # Not strictly correct as commit was done
                raise Exception(
                    f"Failed to create job {job_id} in the database after insert/commit"
                )

            logger.info(f"Successfully created job {job_id} in DuckDB")

        # Enqueue the crawl task (outside DB connection context)
        redis_conn = redis.from_url(REDIS_URI)
        queue = Queue("worker", connection=redis_conn)

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
    # No 'finally db.db.close()' needed, context manager handles it.


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

    # Update job status to running
    db_ops_running = DatabaseOperations()
    asyncio.run(db_ops_running.update_job_status(job_id, "running"))

    try:
        # Run the crawl and processing pipeline
        result = asyncio.run(_execute_pipeline(job_id, url, tags, max_pages))

        # Update job status to completed
        db_ops_completed = DatabaseOperations()
        asyncio.run(db_ops_completed.update_job_status(job_id, "completed"))

        logger.info(f"Completed crawl job {job_id}: {result}")
        return result

    except Exception as e:
        logger.exception(f"Error in crawl job {job_id}: {e!s}")

        # Update job status to failed
        db_ops_failed = DatabaseOperations()
        asyncio.run(db_ops_failed.update_job_status(job_id, "failed", error_message=str(e)))

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

    # Update job status to indicate crawling has started
    db_ops_crawl_start = DatabaseOperations()
    await db_ops_crawl_start.update_job_status(
        job_id=job_id,
        status="running",
        pages_discovered=0,
        pages_crawled=0,
    )

    # Use enhanced crawling with hierarchy tracking
    logger.info(
        f"Job {job_id}: Beginning enhanced crawling with hierarchy tracking from URL: {url}"
    )

    # Process everything with hierarchy tracking
    processed_page_ids = await process_crawl_with_hierarchy(
        url=url,
        job_id=job_id,
        tags=tags,
        max_pages=max_pages,
    )

    pages_crawled = len(processed_page_ids)
    pages_discovered = pages_crawled  # With the new approach, discovered == crawled

    logger.info(
        f"Job {job_id}: Successfully processed {pages_crawled} pages with hierarchy tracking",
    )

    if processed_page_ids:
        # Final update before completing
        db_ops_process_complete = DatabaseOperations()
        await db_ops_process_complete.update_job_status(
            job_id=job_id,
            status="running",
            pages_discovered=pages_discovered,
            pages_crawled=pages_crawled,
        )
    else:
        logger.warning(f"Job {job_id}: No pages were discovered during crawl")
        pages_crawled = 0

        # Update status for empty crawl
        db_ops_empty_crawl = DatabaseOperations()
        await db_ops_empty_crawl.update_job_status(
            job_id=job_id,
            status="running",
            pages_discovered=0,
            pages_crawled=0,
            error_message="No pages were discovered during crawl",
        )

    db_ops_fts = DatabaseOperations()
    with db_ops_fts.db as conn_manager:
        actual_conn = conn_manager.conn
        if not actual_conn:
            raise RuntimeError("Failed to get DB connection for create_job")

        actual_conn.execute(CREATE_FTS_INDEX_SQL)

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

    db_ops = DatabaseOperations()  # Instantiate, remove read_only
    deleted_pages = 0
    page_ids_to_delete = []

    try:
        with db_ops.db as conn_manager:  # Use DuckDBConnectionManager as context manager
            actual_conn = conn_manager.conn
            if not actual_conn:
                raise RuntimeError("Failed to get DB connection for delete_docs")

            conn_manager.begin_transaction()

            # Build the SQL where clause based on filters
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
                    tag_conditions.append(f"json_valid(tags) AND json_contains(tags, '\"{tag}\"')")
                if tag_conditions:  # Ensure there are conditions before joining
                    conditions.append(f"({' OR '.join(tag_conditions)})")

            where_clause = (
                " AND ".join(conditions) if conditions else "1=1"
            )  # Avoid "WHERE " if no conditions

            if not conditions:  # Safety: don't delete all if no filters provided
                logger.warning("Delete_docs called with no filters. Aborting delete operation.")
                conn_manager.rollback()  # Rollback the empty transaction
                return {
                    "task_id": task_id,
                    "deleted_pages": 0,
                    "page_ids": [],
                    "message": "No filters provided, no documents deleted.",
                }

            # Get the IDs of pages that will be deleted
            query = f"SELECT id FROM pages WHERE {where_clause}"
            logger.debug(f"Executing query to get page IDs: {query} with params: {params}")
            results = actual_conn.execute(query, params).fetchall()
            page_ids_to_delete = [row[0] for row in results]

            if not page_ids_to_delete:
                logger.info(f"No pages found matching delete criteria for task {task_id}.")
                conn_manager.commit()  # Commit (empty) transaction
                return {
                    "task_id": task_id,
                    "deleted_pages": 0,
                    "page_ids": [],
                }

            # Then delete from SQL database
            # Re-using where_clause and params is fine
            delete_query = f"DELETE FROM pages WHERE {where_clause}"
            logger.debug(f"Executing delete query: {delete_query} with params: {params}")
            cursor = actual_conn.execute(delete_query, params)
            deleted_pages = cursor.rowcount if cursor else 0

            conn_manager.commit()
            logger.info(f"Deleted {deleted_pages} pages for task {task_id}")

        return {
            "task_id": task_id,
            "deleted_pages": deleted_pages,
            "page_ids": page_ids_to_delete,
        }
    except Exception as e:
        logger.exception(f"Error deleting documents for task {task_id}: {e!s}")
        # Attempt to rollback if conn_manager was successfully entered and has an active transaction
        if "conn_manager" in locals() and conn_manager.transaction_active:
            try:
                logger.info("Attempting rollback due to error in delete_docs.")
                conn_manager.rollback()
            except Exception as rb_err:
                logger.error(f"Failed to rollback during error handling in delete_docs: {rb_err}")
        raise
    # No 'finally db.db.close()' needed, context manager handles it.
