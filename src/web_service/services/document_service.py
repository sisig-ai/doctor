"""Document service for the web service."""

import duckdb
import asyncio

from src.common.config import RETURN_FULL_DOCUMENT_TEXT
from src.common.logger import get_logger
from src.common.models import (
    DocPageSummary,
    GetDocPageResponse,
    ListDocPagesResponse,
    ListTagsResponse,
    SearchDocsResponse,
    SearchResult,
)
from src.lib.database.utils import deserialize_tags

# Get logger for this module
logger = get_logger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings.

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
    """Determine if two strings are fuzzy matches based on Levenshtein distance.

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


async def _perform_vector_search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    max_results: int,
    filter_conditions: dict | None,
    return_full_document_text: bool,
    hybrid_weight: float,
) -> dict:
    """
    Performs a vector search for documents.

    Args:
        conn: DuckDB connection object.
        query: The search query string.
        max_results: The maximum number of results to fetch initially (typically `max_results * 2`).
        filter_conditions: Payload filter conditions for the vector indexer.
        return_full_document_text: Boolean indicating if full document text should be returned.
        hybrid_weight: The weight to apply to the vector search scores.

    Returns:
        A dictionary of search results, keyed by page_id. Each value is a dictionary
        containing chunk_text, tags, score, url, and source ("vector").

    Raises:
        Exception: If the vector search process fails at any point.
    """
    from src.common.indexer import VectorIndexer
    from src.lib.embedder import generate_embedding

    vector_results_map = {}
    logger.debug("Starting vector search.")

    query_embedding = await generate_embedding(query, text_type="query")
    logger.debug("Generated embedding for search query.")
    indexer = VectorIndexer(connection=conn)
    vector_results_raw = await indexer.search(
        query_vector=query_embedding,
        limit=max_results,  # Already adjusted before calling
        filter_payload=filter_conditions,
    )
    logger.info(f"Found {len(vector_results_raw)} results from raw vector search.")

    for result in vector_results_raw:
        chunk_id = result["id"]
        score = result["score"] * hybrid_weight
        payload = result["payload"]
        page_id = payload.get("page_id")

        if not page_id:
            logger.warning(f"Vector result {chunk_id} missing page_id in payload.")
            continue

        if return_full_document_text:
            doc = await get_doc_page(conn, page_id)
            chunk_text = doc.text if doc else None
        else:
            chunk_text = payload.get("text")

        if not chunk_text:
            chunk_text = result.get("text_chunk")
            if not chunk_text:
                logger.warning(f"Vector result {chunk_id} missing text in payload and text_chunk.")
                continue

        result_tags = payload.get("tags", [])
        url = payload.get("url")

        # Store directly, combination logic will be handled after gather
        vector_results_map[page_id] = {
            "chunk_text": chunk_text,
            "tags": result_tags,
            "score": score,
            "url": url,
            "source": "vector",
        }
    logger.debug(f"Processed {len(vector_results_map)} results from vector search.")
    return vector_results_map


async def _perform_bm25_search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    max_results: int,
    tag_condition_sql: str,
    return_full_document_text: bool,
    hybrid_weight: float,
) -> dict:
    """
    Performs a BM25/FTS full-text search using DuckDB FTS extension.

    Args:
        conn: DuckDB connection object.
        query: The search query string.
        max_results: The maximum number of results to fetch initially (typically `max_results * 2`).
        tag_condition_sql: The SQL snippet for tag-based filtering.
        return_full_document_text: Boolean indicating if full document text should be returned.
        hybrid_weight: The weight for vector search (used to calculate BM25 weight as 1.0 - hybrid_weight).

    Returns:
        A dictionary of search results, keyed by page_id. Each value is a dictionary
        containing chunk_text, tags, score, url, and source ("bm25").

    Raises:
        Exception: If the BM25 search process fails at any point.
    """
    # This divisor is used to normalize raw BM25 scores. It's an empirical value
    # and may need tuning based on the typical range of BM25 scores observed
    # from DuckDB FTS for the specific dataset to bring them into a rough 0-1 scale
    # before weighting.
    BM25_SCORE_NORMALIZATION_DIVISOR = 10.0

    bm25_results_map = {}
    logger.debug("Starting BM25/FTS search.")
    escaped_query = query.replace("'", "''")

    # Always attempt FTS search using fts_main_pages.match_bm25
    bm25_sql = f"""
    SELECT
        p.id AS page_id,
        p.url,
        p.domain,
        p.tags,
        p.raw_text,
        fts_main_pages.match_bm25(p.id, '{escaped_query}') AS bm25_score
    FROM pages p
    WHERE bm25_score IS NOT NULL {tag_condition_sql}
    ORDER BY bm25_score DESC
    LIMIT {max_results}
    """  # limit is already max_results * 2 from caller
    bm25_results_raw = conn.execute(bm25_sql).fetchall()
    logger.info(f"FTS/BM25 search returned {len(bm25_results_raw)} raw results.")

    for row in bm25_results_raw:
        page_id, url, domain, tags_json, raw_text, bm25_score_raw = row
        # Apply weighting for BM25
        normalized_bm25_score = min(bm25_score_raw / BM25_SCORE_NORMALIZATION_DIVISOR, 1.0) * (
            1.0 - hybrid_weight
        )

        if return_full_document_text:
            chunk_text = raw_text
        else:
            start_pos = 0
            for term in query.lower().split():
                pos = raw_text.lower().find(term)
                if pos > 0:  # Found a term, set start_pos relative to it
                    start_pos = max(0, pos - 150)  # Context before term
                    break
            end_pos = min(start_pos + 500, len(raw_text))  # Snippet length
            chunk_text = (
                raw_text[start_pos:end_pos] + "..."
                if end_pos < len(raw_text)
                else raw_text[start_pos:end_pos]
            )

        result_tags = deserialize_tags(tags_json)
        bm25_results_map[page_id] = {
            "chunk_text": chunk_text,
            "tags": result_tags,
            "score": normalized_bm25_score,
            "url": url,
            "source": "bm25",
        }
    logger.debug(f"Processed {len(bm25_results_map)} results from BM25 search.")
    return bm25_results_map


async def search_docs(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    tags: list[str] | None = None,
    max_results: int = 10,
    return_full_document_text: bool = RETURN_FULL_DOCUMENT_TEXT,
    hybrid_weight: float = 0.7,  # Weight for vector search (1-hybrid_weight for BM25)
) -> SearchDocsResponse:
    """Search for documents using hybrid search (semantic + BM25), performed in parallel.

    Always attempts FTS/BM25 search using DuckDB's FTS extension, regardless of index visibility in system tables.
    Vector search is performed using a separate index.
    Results from both are combined and ranked.
    """
    logger.debug(
        f"Initiating parallel search with query: '{query}', tags: {tags}, max_results: {max_results}",
    )

    # Prepare filter based on tags
    filter_conditions = None
    tag_condition_sql = ""
    if tags and len(tags) > 0:
        filter_conditions = {"must": [{"key": "tags", "match": {"any": tags}}]}
        tag_conditions_list = []
        for tag_item in tags:
            escaped_tag = tag_item.replace("'", "''")
            tag_conditions_list.append(f"tags LIKE '%{escaped_tag}%'")
        if tag_conditions_list:
            tag_condition_sql = f"AND ({' OR '.join(tag_conditions_list)})"
        logger.debug(f"Applied tag filters: {tags}. SQL: {tag_condition_sql}")

    # Determine fetch limit for individual searches
    internal_fetch_limit = max_results * 2

    tasks = [
        _perform_vector_search(
            conn,
            query,
            internal_fetch_limit,
            filter_conditions,
            return_full_document_text,
            hybrid_weight,
        ),
        _perform_bm25_search(
            conn,
            query,
            internal_fetch_limit,
            tag_condition_sql,
            return_full_document_text,
            hybrid_weight,
        ),
    ]

    gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

    vector_search_results_map = {}
    bm25_search_results_map = {}
    vector_search_success = False
    bm25_search_success = False

    # Process vector search results
    if isinstance(gathered_results[0], Exception):
        logger.warning(f"Vector search task failed: {gathered_results[0]!s}")
    else:
        vector_search_results_map = gathered_results[0]
        vector_search_success = True
        logger.info(
            f"Vector search task completed, found {len(vector_search_results_map)} potential results."
        )

    # Process BM25 search results
    if isinstance(gathered_results[1], Exception):
        logger.warning(f"BM25 search task failed: {gathered_results[1]!s}")
    else:
        bm25_search_results_map = gathered_results[1]
        bm25_search_success = True
        logger.info(
            f"BM25 search task completed, found {len(bm25_search_results_map)} potential results."
        )

    # Combine results
    all_results = {}

    # Add vector results first
    for page_id, data in vector_search_results_map.items():
        all_results[page_id] = data

    # Add/merge BM25 results
    for page_id, bm25_data in bm25_search_results_map.items():
        if page_id in all_results:  # Hybrid result: found by both vector and BM25
            # Current entry in all_results is from vector. Its score is already weighted.
            # bm25_data['score'] is also already weighted.
            # We take the max of the two weighted scores for hybrid results.
            all_results[page_id]["score"] = max(all_results[page_id]["score"], bm25_data["score"])
            all_results[page_id]["source"] = "hybrid"
            # Keep chunk_text from vector search if present, as it might be from a specific chunk.
            # If vector search didn't provide chunk_text but BM25 did, this would need more complex merging.
            # For now, this matches original behavior of prioritizing vector's text if it exists.
        else:  # BM25 only result
            all_results[page_id] = bm25_data

    # Convert combined results to SearchResult objects
    final_results_list = []
    for page_id, result_data in all_results.items():
        final_results_list.append(
            SearchResult(
                chunk_text=result_data["chunk_text"],
                page_id=page_id,
                tags=result_data["tags"],
                score=result_data["score"],
                url=result_data["url"],
            ),
        )

    # Sort by score and limit
    final_results_list.sort(key=lambda x: x.score, reverse=True)
    if len(final_results_list) > max_results:
        final_results_list = final_results_list[:max_results]

    search_methods_used = []
    if vector_search_success:
        search_methods_used.append("vector")
    if bm25_search_success:
        search_methods_used.append("bm25")
    search_method_str = (
        " and ".join(search_methods_used) if search_methods_used else "no successful methods"
    )

    logger.info(
        f"Hybrid search processing completed. Used: {search_method_str}. Returning {len(final_results_list)} results.",
    )
    return SearchDocsResponse(results=final_results_list)


async def list_doc_pages(
    conn: duckdb.DuckDBPyConnection,
    page: int = 1,
    tags: list[str] | None = None,
) -> ListDocPagesResponse:
    """List available indexed pages.

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
            ),
        )

    return ListDocPagesResponse(
        doc_pages=doc_pages,
        total_pages=total_count,
        current_page=page,
        pages_per_page=100,
    )


async def get_doc_page(
    conn: duckdb.DuckDBPyConnection,
    page_id: str,
    starting_line: int = 1,
    ending_line: int = 100,
) -> GetDocPageResponse:
    """Get the full text of a document page.

    Args:
        conn: Connected DuckDB connection
        page_id: The page ID
        starting_line: Line to view from (1-based)
        ending_line: Line to view up to (1-based). Set to -1 to view the entire page.

    Returns:
        GetDocPageResponse: Document page text

    """
    logger.debug(f"Retrieving document page {page_id} (lines {starting_line}-{ending_line})")

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
    logger.debug(f"Found page {page_id} with {len(raw_text)} characters")

    # Split into lines
    lines = raw_text.splitlines()
    total_lines = len(lines)
    logger.debug(f"Page has {total_lines} total lines")

    # Validate line range
    if starting_line > total_lines:
        logger.warning(
            f"Starting line {starting_line} exceeds total lines {total_lines}, resetting to 1",
        )
        starting_line = 1

    if (ending_line != -1 and ending_line > total_lines) or (ending_line == -1):
        logger.debug(
            f"Ending line {ending_line} exceeds total lines {total_lines} or is -1, capping at {total_lines}",
        )
        ending_line = total_lines

    # Extract requested lines
    requested_lines = lines[starting_line - 1 : ending_line]
    logger.debug(f"Returning {len(requested_lines)} lines from page {page_id}")

    return GetDocPageResponse(text="\n".join(requested_lines), total_lines=total_lines)


# Function moved to src/web_service/services/debug_bm25.py


async def list_tags(
    conn: duckdb.DuckDBPyConnection,
    search_substring: str | None = None,
) -> ListTagsResponse:
    """List all unique tags available in the document database.

    Args:
        conn: Connected DuckDB connection
        search_substring: Optional substring to filter tags using case-insensitive fuzzy matching

    Returns:
        ListTagsResponse: List of unique tags

    """
    logger.info(
        f"Retrieving all unique document tags{' with filter: ' + search_substring if search_substring else ''}",
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
            """,
        ).fetchall()

        # Set to store unique tags
        unique_tags = set()

        # Process each JSON array and extract individual tags
        for row in result:
            tags_json = row[0]
            tags = deserialize_tags(tags_json)
            for tag in tags:
                if tag:  # Ensure we don't add empty tags
                    unique_tags.add(tag)

        # Filter tags if search_substring is provided
        if search_substring:
            # Use fuzzy matching to filter tags
            filtered_tags = [tag for tag in unique_tags if is_fuzzy_match(search_substring, tag)]
            tags_list = sorted(filtered_tags)
            logger.info(
                f"Found {len(tags_list)} tags fuzzy-matching '{search_substring}' from total of {len(unique_tags)} tags",
            )
        else:
            # Convert to sorted list
            tags_list = sorted(list(unique_tags))
            logger.info(f"Found {len(tags_list)} unique tags")

        return ListTagsResponse(tags=tags_list)
    except Exception as e:
        logger.error(f"Error retrieving tags: {e!s}")
        raise e
