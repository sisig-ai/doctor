"""Document API routes for the web service."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.common.models import (
    GetDocPageResponse,
    ListDocPagesResponse,
    SearchDocsResponse,
)
from src.lib.logger import get_logger
from src.common.db_setup import (
    get_duckdb_connection_with_retry,
    get_qdrant_client_with_retry,
)
from src.web_service.services.document_service import (
    search_docs,
    list_doc_pages,
    get_doc_page,
)

# Get logger for this module
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["documents"])


@router.get("/search_docs", response_model=SearchDocsResponse, operation_id="search_docs")
async def search_docs_endpoint(
    query: str = Query(..., description="The search string to query the database with"),
    tags: Optional[List[str]] = Query(None, description="Tags to limit the search with"),
    max_results: int = Query(10, description="Maximum number of results to return", ge=1, le=100),
):
    """
    Search for documents using semantic search.

    Args:
        query: The search query
        tags: Optional tags to filter by
        max_results: Maximum number of results to return

    Returns:
        Search results
    """
    logger.info(
        f"API: Searching docs with query: '{query}', tags: {tags}, max_results: {max_results}"
    )

    try:
        # Get a fresh connection to Qdrant for this search
        qdrant_client = await get_qdrant_client_with_retry()
        logger.debug("Connected to Qdrant for search")

        # Get a fresh DuckDB connection
        conn = await get_duckdb_connection_with_retry()
        try:
            # Call the service function
            response = await search_docs(qdrant_client, conn, query, tags, max_results)
            return response
        except Exception as db_error:
            logger.error(f"Database error during search: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        finally:
            # Close DuckDB connection
            conn.close()
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/list_doc_pages", response_model=ListDocPagesResponse, operation_id="list_doc_pages")
async def list_doc_pages_endpoint(
    page: int = Query(1, description="Page number", ge=1),
    tags: Optional[List[str]] = Query(None, description="Tags to filter by"),
):
    """
    List available indexed pages.

    Args:
        page: Page number (1-based)
        tags: Optional tags to filter by

    Returns:
        List of document pages
    """
    logger.info(f"API: Listing document pages (page={page}, tags={tags})")

    # Get a fresh connection for each request
    conn = await get_duckdb_connection_with_retry()

    try:
        # Call the service function
        response = await list_doc_pages(conn, page, tags)
        return response
    except Exception as e:
        logger.error(f"Error listing document pages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()


@router.get("/get_doc_page", response_model=GetDocPageResponse, operation_id="get_doc_page")
async def get_doc_page_endpoint(
    page_id: str = Query(..., description="The page ID to retrieve"),
    starting_line: int = Query(1, description="Line to view from", ge=1),
    ending_line: int = Query(100, description="Line to view up to", ge=1),
):
    """
    Get the full text of a document page.

    Args:
        page_id: The page ID
        starting_line: Line to view from (1-based)
        ending_line: Line to view up to (1-based)

    Returns:
        Document page text
    """
    logger.info(f"API: Retrieving document page {page_id} (lines {starting_line}-{ending_line})")

    # Get a fresh connection for each request
    conn = await get_duckdb_connection_with_retry()

    try:
        # Call the service function
        response = await get_doc_page(conn, page_id, starting_line, ending_line)
        if response is None:
            raise HTTPException(status_code=404, detail=f"Page {page_id} not found")
        return response
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error retrieving document page {page_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()
