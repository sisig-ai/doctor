"""Document service for the web service."""

from typing import List, Optional
import duckdb
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.common.db_setup import (
    QDRANT_COLLECTION_NAME,
    deserialize_tags,
)
from src.common.models import (
    DocPageSummary,
    GetDocPageResponse,
    ListDocPagesResponse,
    SearchDocsResponse,
    SearchResult,
)
from src.lib.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


async def search_docs(
    qdrant_client: QdrantClient,
    conn: duckdb.DuckDBPyConnection,
    query: str,
    tags: Optional[List[str]] = None,
    max_results: int = 10,
) -> SearchDocsResponse:
    """
    Search for documents using semantic search.

    Args:
        qdrant_client: Connected Qdrant client
        conn: Connected DuckDB connection
        query: The search query
        tags: Optional tags to filter by
        max_results: Maximum number of results to return

    Returns:
        SearchDocsResponse: Search results
    """
    logger.info(f"Searching docs with query: '{query}', tags: {tags}, max_results: {max_results}")

    # Prepare filter based on tags
    filter_conditions = None
    if tags and len(tags) > 0:
        filter_conditions = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="tags",
                    match=qdrant_models.MatchAny(any=tags),
                )
            ]
        )
        logger.debug(f"Applied tag filters: {tags}")

    results = []

    try:
        # Try semantic search
        from src.lib.embedder import generate_embedding

        # Generate embedding for the query
        query_embedding = await generate_embedding(query)
        logger.debug("Generated embedding for search query")

        # Search Qdrant
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=max_results,
            query_filter=filter_conditions,
            with_payload=True,
        )
        logger.info(f"Found {len(search_results)} results from vector search")

        # Keep track of the highest scoring chunk per page
        page_best_results = {}

        # Process results and keep only the highest scoring chunk per page
        for result in search_results:
            chunk_id = result.id
            score = result.score
            page_id = result.payload.get("page_id")
            chunk_text = result.payload.get("text")
            result_tags = result.payload.get("tags", [])
            url = result.payload.get("url")

            if not page_id or not chunk_text:
                logger.warning(f"Result {chunk_id} missing page_id or text in payload")
                continue

            # Only keep this result if it's the first one for this page_id
            # or if it has a higher score than the current best for this page_id
            if page_id not in page_best_results or score > page_best_results[page_id]["score"]:
                page_best_results[page_id] = {
                    "chunk_text": chunk_text,
                    "tags": result_tags,
                    "score": score,
                    "url": url,
                }

        # Convert the dictionary of best results to a list of SearchResult objects
        for page_id, best_result in page_best_results.items():
            results.append(
                SearchResult(
                    chunk_text=best_result["chunk_text"],
                    page_id=page_id,
                    tags=best_result["tags"],
                    score=best_result["score"],
                    url=best_result["url"],
                )
            )

        # Sort results by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)

        # Limit results to max_results if needed
        if len(results) > max_results:
            results = results[:max_results]
    except Exception as vector_error:
        logger.warning(f"Vector search failed: {str(vector_error)}")

    logger.info(f"Returning {len(results)} search results")
    return SearchDocsResponse(results=results)


async def list_doc_pages(
    conn: duckdb.DuckDBPyConnection,
    page: int = 1,
    tags: Optional[List[str]] = None,
) -> ListDocPagesResponse:
    """
    List available indexed pages.

    Args:
        conn: Connected DuckDB connection
        page: Page number (1-based)
        tags: Optional tags to filter by

    Returns:
        ListDocPagesResponse: List of document pages
    """
    logger.info(f"Listing document pages (page={page}, tags={tags})")

    # Calculate offset
    offset = (page - 1) * 100

    # Prepare SQL query
    base_query = """
        SELECT id, url, domain, crawl_date, tags
        FROM pages
    """

    count_query = """
        SELECT COUNT(*)
        FROM pages
    """

    # Add tag filtering if provided
    where_clause = ""

    if tags and len(tags) > 0:
        # For each tag, check if it's in the JSON array
        tag_conditions = []
        for tag in tags:
            escaped_tag = tag.replace("'", "''")  # Escape single quotes for SQL
            tag_conditions.append(f"tags LIKE '%{escaped_tag}%'")

        where_clause = f"WHERE {' OR '.join(tag_conditions)}"
        logger.debug(f"Using tag filter: {where_clause}")

    # Execute count query
    count_sql = f"{count_query} {where_clause}"
    logger.debug(f"Executing count query: {count_sql}")
    total_count = conn.execute(count_sql).fetchone()[0]
    logger.info(f"Found {total_count} total document pages matching criteria")

    # Execute main query with pagination
    query = f"{base_query} {where_clause} ORDER BY crawl_date DESC LIMIT 100 OFFSET {offset}"
    logger.debug(f"Executing page query: {query}")
    results = conn.execute(query).fetchall()
    logger.info(f"Retrieved {len(results)} document pages for current page")

    # Format results
    doc_pages = []
    for row in results:
        id, url, domain, crawl_date, tags_json = row
        doc_pages.append(
            DocPageSummary(
                page_id=id,
                domain=domain,
                tags=deserialize_tags(tags_json),
                crawl_date=crawl_date,
                url=url,
            )
        )

    return ListDocPagesResponse(
        doc_pages=doc_pages, total_pages=total_count, current_page=page, pages_per_page=100
    )


async def get_doc_page(
    conn: duckdb.DuckDBPyConnection,
    page_id: str,
    starting_line: int = 1,
    ending_line: int = 100,
) -> GetDocPageResponse:
    """
    Get the full text of a document page.

    Args:
        conn: Connected DuckDB connection
        page_id: The page ID
        starting_line: Line to view from (1-based)
        ending_line: Line to view up to (1-based)

    Returns:
        GetDocPageResponse: Document page text
    """
    logger.info(f"Retrieving document page {page_id} (lines {starting_line}-{ending_line})")

    # Query for the page
    result = conn.execute(
        """
        SELECT raw_text
        FROM pages
        WHERE id = ?
        """,
        (page_id,),
    ).fetchone()

    if not result:
        logger.warning(f"Page {page_id} not found in database")
        return None  # Let the API layer handle the 404

    raw_text = result[0]
    logger.info(f"Found page {page_id} with {len(raw_text)} characters")

    # Split into lines
    lines = raw_text.splitlines()
    total_lines = len(lines)
    logger.debug(f"Page has {total_lines} total lines")

    # Validate line range
    if starting_line > total_lines:
        logger.warning(
            f"Starting line {starting_line} exceeds total lines {total_lines}, resetting to 1"
        )
        starting_line = 1

    if ending_line > total_lines:
        logger.debug(
            f"Ending line {ending_line} exceeds total lines {total_lines}, capping at {total_lines}"
        )
        ending_line = total_lines

    # Extract requested lines
    requested_lines = lines[starting_line - 1 : ending_line]
    logger.info(f"Returning {len(requested_lines)} lines from page {page_id}")

    return GetDocPageResponse(text="\n".join(requested_lines), total_lines=total_lines)
