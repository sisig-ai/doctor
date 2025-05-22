"""Job service for the web service."""

import uuid

import duckdb
from rq import Queue

from src.common.logger import get_logger
from src.common.models import (
    FetchUrlResponse,
    JobProgressResponse,
)

# Get logger for this module
logger = get_logger(__name__)


async def fetch_url(
    queue: Queue,
    url: str,
    tags: list[str] | None = None,
    max_pages: int = 100,
) -> FetchUrlResponse:
    """Initiate a fetch job to crawl a website.

    Args:
        queue: Redis queue for job processing
        url: The URL to start indexing from
        tags: Optional tags to assign this website
        max_pages: How many pages to index

    Returns:
        FetchUrlResponse: The job ID

    """
    # Generate a temporary job ID
    job_id = str(uuid.uuid4())

    # Enqueue the job creation task
    # The crawl worker will handle both creating the job record and enqueueing the crawl task
    queue.enqueue(
        "src.crawl_worker.tasks.create_job",
        url,
        job_id,
        tags=tags,
        max_pages=max_pages,
    )

    logger.info(f"Enqueued job creation for URL: {url}, job_id: {job_id}")

    return FetchUrlResponse(job_id=job_id)


async def get_job_progress(
    conn: duckdb.DuckDBPyConnection,
    job_id: str,
) -> JobProgressResponse | None:
    """Check the progress of a job.

    Args:
        conn: Connected DuckDB connection
        job_id: The job ID to check progress for

    Returns:
        JobProgressResponse: Job progress information or None if job not found

    """
    logger.info(f"Checking progress for job {job_id}")

    # First try exact job_id match
    logger.info(f"Looking for exact match for job ID: {job_id}")
    result = conn.execute(
        """
        SELECT job_id, start_url, status, pages_discovered, pages_crawled,
               max_pages, tags, created_at, updated_at, error_message
        FROM jobs
        WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()

    # If not found, try partial match (useful if client received truncated UUID)
    if not result and len(job_id) >= 8:
        logger.info(f"Exact match not found, trying partial match for job ID: {job_id}")
        result = conn.execute(
            """
            SELECT job_id, start_url, status, pages_discovered, pages_crawled,
                   max_pages, tags, created_at, updated_at, error_message
            FROM jobs
            WHERE job_id LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (f"{job_id}%",),
        ).fetchone()

    if not result:
        logger.warning(f"Job {job_id} not found")
        return None

    # --- Job found in database ---
    (
        job_id,
        url,
        status,
        pages_discovered,
        pages_crawled,
        max_pages,
        tags_json,
        created_at,
        updated_at,
        error_message,
    ) = result

    logger.info(
        f"Found job with ID: {job_id}, status: {status}, discovered: {pages_discovered}, crawled: {pages_crawled}, updated: {updated_at}",
    )

    # Determine if job is completed
    completed = status in ["completed", "failed"]

    # Calculate progress percentage
    progress_percent = 0
    if max_pages > 0 and pages_discovered > 0:
        progress_percent = min(100, int((pages_crawled / min(pages_discovered, max_pages)) * 100))

    logger.info(
        f"Job {job_id} progress: {pages_crawled}/{pages_discovered} pages, {progress_percent}% complete, status: {status}",
    )

    return JobProgressResponse(
        pages_crawled=pages_crawled,
        pages_total=pages_discovered,
        completed=completed,
        status=status,
        error_message=error_message,
        progress_percent=progress_percent,
        url=url,
        max_pages=max_pages,
        created_at=created_at,
        updated_at=updated_at,
    )


async def get_job_count() -> int:
    """Get the total number of jobs in the database.
    Used for debugging purposes.

    Returns:
        int: Total number of jobs

    """
    from src.lib.database import DatabaseOperations

    db = None
    try:
        db = DatabaseOperations(read_only=True)
        conn = db.db.ensure_connection()
        job_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        logger.info(f"Database contains {job_count} total jobs.")
        return job_count
    except Exception as count_error:
        logger.warning(f"Failed to count jobs in database: {count_error!s}")
        return -1
    finally:
        if db:
            db.db.close()
