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
from src.lib.database import DatabaseOperations
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

    # Create a fresh connection to get the latest data
    # This ensures we always get the most recent job state
    attempts = 0
    max_attempts = 3
    retry_delay = 0.1  # seconds

    logger.info(f"Starting check loop for job {job_id} (max_attempts={max_attempts})")
    while attempts < max_attempts:
        attempts += 1
        db_ops = DatabaseOperations()  # Instantiate inside the loop

        try:
            with db_ops.db as conn_manager:  # Use DuckDBConnectionManager as context manager
                actual_conn = conn_manager.conn
                if not actual_conn:
                    # This case should ideally be handled by DuckDBConnectionManager.connect() raising an error
                    logger.error(
                        f"Attempt {attempts}: Failed to obtain DB connection from manager for job {job_id}."
                    )
                    if attempts >= max_attempts:
                        raise HTTPException(
                            status_code=500, detail="Database connection error after retries."
                        )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue  # To next attempt

                logger.info(
                    f"Established fresh connection to database (attempt {attempts}) for job {job_id}"
                )

                # Call the service function
                result = await get_job_progress(actual_conn, job_id)

                if not result:
                    # Job not found on this attempt
                    logger.warning(
                        f"Job {job_id} not found (attempt {attempts}/{max_attempts}). Retrying in {retry_delay}s...",
                    )
                    # Connection is closed by conn_manager context exit
                    if attempts >= max_attempts:
                        logger.warning(f"Job {job_id} not found after {max_attempts} attempts.")
                        break  # Exit the while loop to raise 404
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue  # Go to next iteration of the while loop

                # Job found, return the result
                return result

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            logger.warning(
                f"HTTPException occurred during attempt {attempts} for job {job_id}, re-raising.",
            )
            raise
        except Exception as e:  # Includes duckdb.Error if connect fails in conn_manager
            logger.error(
                f"Error checking job progress (attempt {attempts}) for job {job_id}: {e!s}",
            )
            if attempts < max_attempts:
                logger.info(f"Non-fatal error on attempt {attempts}, will retry after delay.")
                # Connection (if opened by conn_manager) is closed on context exit
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue
            logger.error(f"Error on final attempt ({attempts}) for job {job_id}. Raising 500.")
            raise HTTPException(
                status_code=500,
                detail=f"Database error after retries: {e!s}",
            )
        # The 'finally' block for closing 'conn' is removed as conn_manager handles it.

    # If the loop finished without finding the job (i.e., break was hit after max attempts)
    # raise the 404 error.
    # Check job count for debugging before raising 404
    job_count = await get_job_count()
    logger.info(f"Database contains {job_count} total jobs during final 404 check.")

    logger.warning(f"Raising 404 for job {job_id} after all retries.")  # Add log before raising
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found after retries")
