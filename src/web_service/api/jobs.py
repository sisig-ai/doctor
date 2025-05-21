"""Job API routes for the web service."""

import asyncio

import redis
from fastapi import APIRouter, Depends, HTTPException, Query
from rq import Queue

from src.common.config import REDIS_URI
from src.common.logger import get_logger
from src.common.models import (
    FetchUrlRequest,
    FetchUrlResponse,
    JobProgressResponse,
)
from src.lib.database import get_connection
from src.lib.database.sync import sync_write_to_read
from src.web_service.services.job_service import (
    fetch_url,
    get_job_count,
    get_job_progress,
)

# Get logger for this module
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["jobs"])


@router.post("/fetch_url", response_model=FetchUrlResponse, operation_id="fetch_url")
async def fetch_url_endpoint(
    request: FetchUrlRequest,
    queue: Queue = Depends(lambda: Queue("worker", connection=redis.from_url(REDIS_URI))),
):
    """Initiate a fetch job to crawl a website.

    Args:
        request: The fetch URL request

    Returns:
        The job ID

    """
    logger.info(f"API: Initiating fetch for URL: {request.url}")

    try:
        # Call the service function
        response = await fetch_url(
            queue=queue,
            url=request.url,
            tags=request.tags,
            max_pages=request.max_pages,
        )
        return response
    except Exception as e:
        logger.error(f"Error initiating fetch: {e!s}")
        raise HTTPException(status_code=500, detail=f"Error initiating fetch: {e!s}")


@router.get("/job_progress", response_model=JobProgressResponse, operation_id="job_progress")
async def job_progress_endpoint(
    job_id: str = Query(..., description="The job ID to check progress for"),
):
    """Check the progress of a job.

    Args:
        job_id: The job ID to check progress for

    Returns:
        Job progress information

    """
    logger.info(f"API: BEGIN job_progress for job {job_id}")

    # Force a manual sync from the write database to ensure we have the latest job status
    try:
        logger.info("Forcing manual sync from write database before checking job progress")
        sync_result = await sync_write_to_read()
        if sync_result:
            logger.info("Manual database sync completed successfully before job progress check")
        else:
            logger.warning(
                "Manual database sync failed before job progress check, proceeding with potentially stale data"
            )
    except Exception as e:
        logger.warning(
            f"Error during manual sync before job progress check, proceeding anyway: {e}"
        )

    # Use retries for handling potential transient database issues
    attempts = 0
    max_attempts = 3
    retry_delay = 0.1  # seconds

    logger.info(f"Starting check loop for job {job_id} (max_attempts={max_attempts})")

    while attempts < max_attempts:
        attempts += 1

        try:
            # Use proper connection pool with context manager
            async with await get_connection(read_only=True) as conn_manager:
                # Get connection using async method
                conn = await conn_manager.async_ensure_connection()
                logger.info(
                    f"Established fresh read-only connection to database (attempt {attempts})"
                )

                # Call the service function
                result = await get_job_progress(conn, job_id)

                if result is None:
                    # Job not found, try all attempts before returning 404
                    if attempts >= max_attempts:
                        # Log job count for debugging before raising 404
                        job_count = await get_job_count()
                        logger.info(
                            f"Database contains {job_count} total jobs during final 404 check."
                        )

                        logger.warning(f"Job {job_id} not found after {attempts} attempts")
                        raise HTTPException(
                            status_code=404,
                            detail=f"Job {job_id} not found",
                        )
                    logger.info(f"Job {job_id} not found on attempt {attempts}, will retry")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                # Job found successfully
                logger.info(f"API: END job_progress for job {job_id} (found)")
                return result

        except HTTPException:
            # Re-raise HTTP exceptions (like 404) directly
            raise
        except Exception as e:
            # Log errors during attempts but don't necessarily stop retrying unless it's the last attempt
            logger.error(
                f"Error checking job progress (attempt {attempts}) for job {job_id}: {e!s}",
            )
            if attempts < max_attempts:
                logger.info(f"Non-fatal error on attempt {attempts}, will retry after delay.")
                await asyncio.sleep(retry_delay)  # Wait before retry on error too
                retry_delay *= 2
                continue  # Continue to next attempt

            logger.error(f"Error on final attempt ({attempts}) for job {job_id}. Raising 500.")
            raise HTTPException(
                status_code=500,
                detail=f"Database error after retries: {e!s}",
            )

    # This code should be unreachable, but just in case
    logger.error(f"Unexpected exit from retry loop for job {job_id} without result or exception")
    raise HTTPException(
        status_code=500,
        detail="Unexpected internal error during job progress check",
    )
