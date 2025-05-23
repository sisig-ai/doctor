"""Document API routes for the web service."""

from fastapi import APIRouter, HTTPException, Query

from src.common.config import RETURN_FULL_DOCUMENT_TEXT
from src.common.logger import get_logger
from src.common.models import (
    GetDocPageResponse,
    ListDocPagesResponse,
    ListTagsResponse,
    SearchDocsResponse,
)
from src.lib.database import DatabaseOperations
from src.web_service.services.document_service import (
    get_doc_page,
    list_doc_pages,
    list_tags,
    search_docs,
)

# Get logger for this module
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["documents"])


@router.get("/search_docs", response_model=SearchDocsResponse, operation_id="search_docs")
async def search_docs_endpoint(
    query: str = Query(..., description="The search string to query the database with"),
    tags: list[str] | None = Query(None, description="Tags to limit the search with"),
    max_results: int = Query(10, description="Maximum number of results to return", ge=1, le=100),
    return_full_document_text: bool = Query(
        RETURN_FULL_DOCUMENT_TEXT,
        description="Whether to return the full document text instead of the matching chunks only",
    ),
):
    """Search for documents using semantic search. Use `get_doc_page` to get the full text of a document page.

    Args:
        query: The search query
        tags: Optional tags to filter by
        max_results: Maximum number of results to return

    Returns:
        Search results

    """
    logger.info(
        f"API: Searching docs with query: '{query}', tags: {tags}, max_results: {max_results}, return_full_document_text: {return_full_document_text}",
    )

    db_ops = DatabaseOperations()
    try:
        # Use DuckDBConnectionManager as a context manager
        with db_ops.db as conn_manager:
            actual_conn = conn_manager.conn
            if not actual_conn:
                logger.error("Failed to obtain database connection for search_docs.")
                raise HTTPException(status_code=500, detail="Database connection error.")
            # Call the service function
            response = await search_docs(
                actual_conn, query, tags, max_results, return_full_document_text
            )
            return response
    except HTTPException:  # Re-raise HTTP exceptions directly
        raise
    except (
        Exception
    ) as e:  # Catch other exceptions (like DB errors from service or connection issues)
        logger.error(f"Error searching documents: {e!s}")
        # Log the specific error, but return a generic server error to the client
        raise HTTPException(status_code=500, detail=f"Search error: {e!s}")
    # No finally block to close connection, context manager handles it.


@router.get("/list_doc_pages", response_model=ListDocPagesResponse, operation_id="list_doc_pages")
async def list_doc_pages_endpoint(
    page: int = Query(1, description="Page number", ge=1),
    tags: list[str] | None = Query(None, description="Tags to filter by"),
):
    """List all available indexed pages.

    Args:
        page: Page number (1-based)
        tags: Optional tags to filter by

    Returns:
        List of document pages

    """
    logger.info(f"API: Listing document pages (page={page}, tags={tags})")

    db_ops = DatabaseOperations()
    try:
        with db_ops.db as conn_manager:
            actual_conn = conn_manager.conn
            if not actual_conn:
                logger.error("Failed to obtain database connection for list_doc_pages.")
                raise HTTPException(status_code=500, detail="Database connection error.")
            # Call the service function
            response = await list_doc_pages(actual_conn, page, tags)
            return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing document pages: {e!s}")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}")
    # No finally block to close connection


@router.get("/get_doc_page", response_model=GetDocPageResponse, operation_id="get_doc_page")
async def get_doc_page_endpoint(
    page_id: str = Query(..., description="The page ID to retrieve"),
    starting_line: int = Query(1, description="Line to view from", ge=1),
    ending_line: int = Query(
        -1,
        description="Line to view up to. Set to -1 to view the entire page.",
    ),
):
    """Get the full text of a document page. Use `search_docs` or `list_doc_pages` to get the page IDs.

    Args:
        page_id: The page ID
        starting_line: Line to view from (1-based)
        ending_line: Line to view up to (1-based). Set to -1 to view the entire page.

    Returns:
        Document page text

    """
    logger.info(f"API: Retrieving document page {page_id} (lines {starting_line}-{ending_line})")

    db_ops = DatabaseOperations()
    try:
        with db_ops.db as conn_manager:
            actual_conn = conn_manager.conn
            if not actual_conn:
                logger.error("Failed to obtain database connection for get_doc_page.")
                raise HTTPException(status_code=500, detail="Database connection error.")
            # Call the service function
            response = await get_doc_page(actual_conn, page_id, starting_line, ending_line)
            if response is None:
                raise HTTPException(status_code=404, detail=f"Page {page_id} not found")
            return response
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error retrieving document page {page_id}: {e!s}")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}")
    # No finally block to close connection


@router.get("/list_tags", response_model=ListTagsResponse, operation_id="list_tags")
async def list_tags_endpoint(
    search_substring: str | None = Query(
        None,
        description="Optional substring to filter tags (case-insensitive fuzzy matching)",
    ),
):
    """List all unique tags available in the document database. Use `search_docs` or `list_doc_pages` to get the page IDs using the tags.

    Args:
        search_substring: Optional substring to filter tags using case-insensitive fuzzy matching

    Returns:
        List of unique tags

    """
    logger.info(
        f"API: Listing all unique document tags{' with filter: ' + search_substring if search_substring else ''}",
    )

    db_ops = DatabaseOperations()
    try:
        with db_ops.db as conn_manager:
            actual_conn = conn_manager.conn
            if not actual_conn:
                logger.error("Failed to obtain database connection for list_tags.")
                raise HTTPException(status_code=500, detail="Database connection error.")
            # Call the service function
            response = await list_tags(actual_conn, search_substring)
            return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing document tags: {e!s}")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}")
    # No finally block to close connection
