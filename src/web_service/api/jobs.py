"""Job API routes for the web service."""

import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query
from rq import Queue

from src.common.models import (
    FetchUrlRequest,
    FetchUrlResponse,
    JobProgressResponse,
)
from src.lib.logger import get_logger
from src.common.db_setup import (
    get_duckdb_connection_with_retry,
)
from src.web_service.services.job_service import (
    fetch_url,
    get_job_progress,
    get_job_count,
)

# Get logger for this module
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["jobs"])


@router.post("/fetch_url", response_model=FetchUrlResponse, operation_id="fetch_url")
async def fetch_url_endpoint(
    request: FetchUrlRequest, queue: Queue = Depends(lambda: Queue("worker"))
):
    """
    Initiate a fetch job to crawl a website.

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
        logger.error(f"Error initiating fetch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initiating fetch: {str(e)}")


@router.get("/job_progress", response_model=JobProgressResponse, operation_id="job_progress")
async def job_progress_endpoint(
    job_id: str = Query(..., description="The job ID to check progress for"),
):
    """
    Check the progress of a job.

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
        conn = None

        try:
            # Get a fresh read-only connection each time
            conn = await get_duckdb_connection_with_retry()
            logger.info(f"Established fresh read-only connection to database (attempt {attempts})")

            # Call the service function
            result = await get_job_progress(conn, job_id)

            if not result:
                # Job not found on this attempt
                logger.warning(
                    f"Job {job_id} not found (attempt {attempts}/{max_attempts}). Retrying in {retry_delay}s..."
                )
                # Close connection before sleeping
                if conn:
                    conn.close()
                    conn = None  # Ensure we get a fresh one next iteration

                # If this was the last attempt, break the loop to raise 404 outside
                if attempts >= max_attempts:
                    logger.warning(f"Job {job_id} not found after {max_attempts} attempts.")
                    break  # Exit the while loop

                # Wait before the next attempt
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue  # Go to next iteration of the while loop

            # Job found, return the result
            return result

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            logger.warning(
                f"HTTPException occurred during attempt {attempts} for job {job_id}, re-raising."
            )
            raise
        except Exception as e:
            # Log errors during attempts but don't necessarily stop retrying unless it's the last attempt
            logger.error(
                f"Error checking job progress (attempt {attempts}) for job {job_id}: {str(e)}"
            )
            if attempts < max_attempts:
                logger.info(f"Non-fatal error on attempt {attempts}, will retry after delay.")
                # Clean up potentially broken connection before retrying
                if conn:
                    try:
                        conn.close()
                        conn = None
                    except Exception as close_err:
                        logger.warning(
                            f"Error closing connection during error handling: {close_err}"
                        )
                await asyncio.sleep(retry_delay)  # Wait before retry on error too
                retry_delay *= 2
                continue  # Continue to next attempt
            else:
                logger.error(f"Error on final attempt ({attempts}) for job {job_id}. Raising 500.")
                # Clean up connection if open
                if conn:
                    try:
                        conn.close()
                    except Exception as close_err:
                        logger.warning(
                            f"Error closing connection during final error handling: {close_err}"
                        )
                raise HTTPException(
                    status_code=500, detail=f"Database error after retries: {str(e)}"
                )

        finally:
            # Make sure to close the connection if it's still open *at the end of this attempt*
            if conn:
                conn.close()

    # If the loop finished without finding the job (i.e., break was hit after max attempts)
    # raise the 404 error.
    # Check job count for debugging before raising 404
    job_count = await get_job_count()
    logger.info(f"Database contains {job_count} total jobs during final 404 check.")

    logger.warning(f"Raising 404 for job {job_id} after all retries.")  # Add log before raising
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found after retries")
