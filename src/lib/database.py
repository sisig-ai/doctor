"""Database operations for storing crawled pages and job information."""

import uuid
import datetime
from typing import List, Optional
from urllib.parse import urlparse

from src.common.db_setup import get_duckdb_connection, serialize_tags
from src.common.logger import get_logger

# Configure logging
logger = get_logger(__name__)


async def store_page(
    url: str, text: str, job_id: str, tags: List[str] = None, page_id: Optional[str] = None
) -> str:
    """
    Store a crawled page in the database.

    Args:
        url: The URL of the page
        text: The extracted text content of the page
        job_id: The ID of the crawl job
        tags: Optional tags to associate with the page
        page_id: Optional ID for the page (generated if not provided)

    Returns:
        The ID of the stored page
    """
    if tags is None:
        tags = []

    # Generate a page ID if not provided
    if page_id is None:
        page_id = str(uuid.uuid4())

    # Extract domain from URL
    domain = urlparse(url).netloc

    logger.debug(f"Storing page {page_id} from {url} with {len(text)} characters")

    # Get DuckDB connection
    conn = get_duckdb_connection()
    transaction_active = False

    try:
        # Begin transaction explicitly
        conn.execute("BEGIN TRANSACTION")
        transaction_active = True

        # Store page data
        conn.execute(
            """
            INSERT INTO pages (id, url, domain, raw_text, crawl_date, tags, job_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                page_id,
                url,
                domain,
                text,
                datetime.datetime.now(),
                serialize_tags(tags),
                job_id,
            ),
        )

        # Commit the transaction
        conn.commit()
        transaction_active = False
        logger.debug(f"Successfully stored page {page_id} in database")

        return page_id

    except Exception as e:
        if transaction_active:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logger.warning(f"Rollback failed: {str(rollback_error)}")
        logger.error(f"Database error storing page: {str(e)}")
        raise

    finally:
        conn.close()


def update_job_status(
    job_id: str,
    status: str,
    pages_discovered: Optional[int] = None,
    pages_crawled: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Update the status of a crawl job.

    Args:
        job_id: The ID of the job
        status: The new status of the job
        pages_discovered: Optional number of pages discovered
        pages_crawled: Optional number of pages crawled
        error_message: Optional error message if the job failed
    """
    logger.info(
        f"Updating job {job_id} status to {status}, discovered={pages_discovered}, crawled={pages_crawled}"
    )

    # Get DuckDB connection
    conn = get_duckdb_connection()
    transaction_active = False
    update_successful = False

    try:
        # Begin transaction explicitly
        conn.execute("BEGIN TRANSACTION")
        transaction_active = True

        # Build the SQL query dynamically based on which fields are provided
        query_parts = ["UPDATE jobs SET status = ?, updated_at = ?"]
        params = [status, datetime.datetime.now()]

        if pages_discovered is not None:
            query_parts.append("pages_discovered = ?")
            params.append(pages_discovered)

        if pages_crawled is not None:
            query_parts.append("pages_crawled = ?")
            params.append(pages_crawled)

        if error_message is not None:
            query_parts.append("error_message = ?")
            params.append(error_message)

        query = f"{', '.join(query_parts)} WHERE job_id = ?"
        params.append(job_id)

        # Execute the query
        cursor = conn.execute(query, tuple(params))

        # Verify the update was applied by checking the row count
        rows_affected = cursor.rowcount
        if rows_affected == 0:
            logger.warning(
                f"Job status update for {job_id} didn't affect any rows. Job may not exist."
            )
        else:
            logger.debug(f"Job status update affected {rows_affected} rows")
            update_successful = True

        # Commit the transaction
        conn.commit()
        transaction_active = False

        # Periodically force a checkpoint to ensure changes are persisted
        # Only do this for important status transitions
        if status in ["completed", "failed"]:
            try:
                conn.execute("CHECKPOINT")
                logger.info(f"Forced checkpoint after updating job {job_id} to {status}")
            except Exception as e:
                logger.warning(f"Failed to checkpoint: {str(e)}")

        if update_successful:
            logger.info(f"Successfully updated job {job_id} status to {status}")

        # Verify the job status was updated correctly by reading it back
        if status in ["completed", "failed"]:
            try:
                job_data = conn.execute(
                    "SELECT status, pages_discovered, pages_crawled FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()

                if job_data:
                    current_status, current_discovered, current_crawled = job_data
                    logger.info(
                        f"Job {job_id} current state: status={current_status}, "
                        f"discovered={current_discovered}, crawled={current_crawled}"
                    )
                else:
                    logger.warning(
                        f"Could not verify job {job_id} status - job not found in database"
                    )
            except Exception as verify_error:
                logger.warning(f"Error verifying job status: {str(verify_error)}")

    except Exception as e:
        if transaction_active:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logger.warning(f"Rollback failed: {str(rollback_error)}")
        logger.error(f"Database error updating job status: {str(e)}")
        raise

    finally:
        conn.close()
