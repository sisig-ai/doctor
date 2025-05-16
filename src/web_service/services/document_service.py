"""Document service for the web service."""

from typing import List, Optional
import duckdb

from src.lib.database import Database
from src.common.models import (
    DocPageSummary,
    GetDocPageResponse,
    ListDocPagesResponse,
    SearchDocsResponse,
    SearchResult,
    ListTagsResponse,
)
from src.common.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        int: Edit distance between the strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def is_fuzzy_match(s1: str, s2: str, threshold: float = 0.7) -> bool:
    """
    Determine if two strings are fuzzy matches based on Levenshtein distance.

    Args:
        s1: First string
        s2: Second string
        threshold: Similarity threshold (0.0 to 1.0)

    Returns:
        bool: True if the strings are similar enough
    """
    # Normalize strings
    s1_norm = s1.lower().replace(" ", "")
    s2_norm = s2.lower().replace(" ", "")

    # Check for substring match first (faster)
    if s1_norm in s2_norm or s2_norm in s1_norm:
        return True

    # If one string is empty, they can't be a fuzzy match
    if not s1_norm or not s2_norm:
        return False

    # For very different length strings, it's unlikely to be a match
    len_diff = abs(len(s1_norm) - len(s2_norm))
    max_len = max(len(s1_norm), len(s2_norm))

    # If length difference is too great, not a match
    if len_diff / max_len > (1 - threshold):
        return False

    # Calculate Levenshtein distance
    distance = levenshtein_distance(s1_norm, s2_norm)

    # Calculate similarity as percentage (1 - normalized_distance)
    similarity = 1 - (distance / max(len(s1_norm), len(s2_norm)))

    return similarity >= threshold


async def search_docs(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    tags: Optional[List[str]] = None,
    max_results: int = 10,
    return_full_document_text: bool = False,
) -> SearchDocsResponse:
    """
    Search for documents using semantic search.

    Args:
        conn: Connected DuckDB connection
        query: The search query
        tags: Optional tags to filter by
        max_results: Maximum number of results to return
        return_full_document_text: Whether to return the full document text instead of the matching chunks only
    Returns:
        SearchDocsResponse: Search results
    """
    logger.info(f"Searching docs with query: '{query}', tags: {tags}, max_results: {max_results}")

    # Prepare filter based on tags
    filter_conditions = None
    if tags and len(tags) > 0:
        # Create a filter compatible with VectorIndexer.search method
        filter_conditions = {"must": [{"key": "tags", "match": {"any": tags}}]}
        logger.debug(f"Applied tag filters: {tags}")

    results = []

    try:
        # Try semantic search
        from src.lib.embedder import generate_embedding
        from src.common.indexer import VectorIndexer

        # Generate embedding for the query
        query_embedding = await generate_embedding(query)
        logger.debug("Generated embedding for search query")

        # Initialize VectorIndexer with the existing connection
        indexer = VectorIndexer(connection=conn)

        # Search using the indexer
        search_results = await indexer.search(
            query_vector=query_embedding,
            limit=max_results,
            filter_payload=filter_conditions,
        )
        logger.info(f"Found {len(search_results)} results from vector search")

        # Keep track of the highest scoring chunk per page
        page_best_results = {}

        # Process results and keep only the highest scoring chunk per page
        for result in search_results:
            chunk_id = result["id"]
            score = result["score"]
            payload = result["payload"]
            page_id = payload.get("page_id")

            if return_full_document_text:
                doc = await get_doc_page(conn, page_id)
                chunk_text = doc.text
            else:
                chunk_text = payload.get("text")
            result_tags = payload.get("tags", [])
            url = payload.get("url")

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
                tags=Database.deserialize_tags(tags_json),
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
        ending_line: Line to view up to (1-based). Set to -1 to view the entire page.

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

    if (ending_line != -1 and ending_line > total_lines) or (ending_line == -1):
        logger.debug(
            f"Ending line {ending_line} exceeds total lines {total_lines} or is -1, capping at {total_lines}"
        )
        ending_line = total_lines

    # Extract requested lines
    requested_lines = lines[starting_line - 1 : ending_line]
    logger.info(f"Returning {len(requested_lines)} lines from page {page_id}")

    return GetDocPageResponse(text="\n".join(requested_lines), total_lines=total_lines)


async def list_tags(
    conn: duckdb.DuckDBPyConnection, search_substring: Optional[str] = None
) -> ListTagsResponse:
    """
    List all unique tags available in the document database.

    Args:
        conn: Connected DuckDB connection
        search_substring: Optional substring to filter tags using case-insensitive fuzzy matching

    Returns:
        ListTagsResponse: List of unique tags
    """
    logger.info(
        f"Retrieving all unique document tags{' with filter: ' + search_substring if search_substring else ''}"
    )

    # We need to extract unique tags from the JSON array in the 'tags' column
    # This is a bit complex in SQL since we need to unnest the JSON arrays
    try:
        # First, we get all the tags columns which are JSON arrays as strings
        result = conn.execute(
            """
            SELECT DISTINCT tags
            FROM pages
            WHERE tags IS NOT NULL AND tags != '[]'
            """
        ).fetchall()

        # Set to store unique tags
        unique_tags = set()

        # Process each JSON array and extract individual tags
        for row in result:
            tags_json = row[0]
            tags = Database.deserialize_tags(tags_json)
            for tag in tags:
                if tag:  # Ensure we don't add empty tags
                    unique_tags.add(tag)

        # Filter tags if search_substring is provided
        if search_substring:
            # Use fuzzy matching to filter tags
            filtered_tags = [tag for tag in unique_tags if is_fuzzy_match(search_substring, tag)]
            tags_list = sorted(filtered_tags)
            logger.info(
                f"Found {len(tags_list)} tags fuzzy-matching '{search_substring}' from total of {len(unique_tags)} tags"
            )
        else:
            # Convert to sorted list
            tags_list = sorted(list(unique_tags))
            logger.info(f"Found {len(tags_list)} unique tags")

        return ListTagsResponse(tags=tags_list)
    except Exception as e:
        logger.error(f"Error retrieving tags: {str(e)}")
        raise e
