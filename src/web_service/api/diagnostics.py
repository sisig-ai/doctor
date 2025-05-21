"""Diagnostic API routes for debugging purposes.
Important: Should be disabled in production.
"""

from fastapi import APIRouter, HTTPException, Query

from src.common.logger import get_logger
from src.lib.database import get_connection
from src.web_service.services.debug_bm25 import debug_bm25_search

# Get logger for this module
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["diagnostics"])


@router.get("/debug_bm25", operation_id="debug_bm25")
async def debug_bm25_endpoint(
    query: str = Query("test", description="The search query to test with"),
):
    """Diagnose BM25 search issues by executing tests and gathering debug information.
    This endpoint is for development and debugging purposes only.

    Args:
        query: Search query to test

    Returns:
        dict: Diagnostic information

    """
    logger.info(f"API: Running BM25 diagnostics with query: '{query}'")

    try:
        # Use connection pool with context manager
        async with await get_connection(read_only=True) as conn_manager:
            conn = await conn_manager.async_ensure_connection()

            # Call the diagnostic function
            results = await debug_bm25_search(conn, query)
            return results
    except Exception as e:
        logger.error(f"Error during BM25 diagnostics: {e!s}")
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {e!s}")
