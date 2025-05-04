"""Main module for the Doctor Web Service."""

from contextlib import asynccontextmanager
import datetime
import json
import logging
import uuid
from typing import List, Optional

from mcp.server.session import ServerSession
import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi_mcp import FastApiMCP
from rq import Queue

from src.common.config import (
    REDIS_URI,
    WEB_SERVICE_HOST,
    WEB_SERVICE_PORT,
)
from src.common.db_setup import (
    QDRANT_COLLECTION_NAME,
    deserialize_tags,
    get_qdrant_client,
    get_read_only_connection,
    init_databases,
    ensure_qdrant_collection,
)
from src.common.models import (
    DocPageSummary,
    FetchUrlRequest,
    FetchUrlResponse,
    GetDocPageResponse,
    JobProgressResponse,
    ListDocPagesResponse,
    SearchDocsResponse,
    SearchResult,
)
from qdrant_client.http import models as qdrant_models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Temporary monkeypatch which avoids crashing when a POST message is received
# before a connection has been initialized, e.g: after a deployment.
# pylint: disable-next=protected-access
# https://github.com/modelcontextprotocol/python-sdk/issues/423
old__received_request = ServerSession._received_request


async def _received_request(self, *args, **kwargs):
    try:
        return await old__received_request(self, *args, **kwargs)
    except RuntimeError:
        pass


ServerSession._received_request = _received_request


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize databases in read-only mode for the web service
    init_databases(read_only=True)
    yield


app = FastAPI(
    title="Doctor API",
    description="API for the Doctor web crawling and indexing system",
    version="0.1.0",
)

redis_conn = redis.from_url(REDIS_URI)

default_queue = Queue("default", connection=redis_conn)
high_queue = Queue("high", connection=redis_conn)
low_queue = Queue("low", connection=redis_conn)


@app.post("/fetch_url", response_model=FetchUrlResponse, operation_id="fetch_url")
async def fetch_url(request: FetchUrlRequest):
    """
    Initiate a fetch job to crawl a website.

    Args:
        request: The fetch URL request

    Returns:
        The job ID
    """
    # Generate a temporary job ID
    job_id = str(uuid.uuid4())

    # Store the request details in Redis for the worker to pick up
    redis_conn = redis.from_url(REDIS_URI)
    job_request_key = f"job_request:{job_id}"

    # Store job request details
    job_request_data = {
        "url": str(request.url),
        "max_pages": str(request.max_pages),
        "tags": json.dumps(request.tags if request.tags else []),
        "status": "requested",
        "created_at": datetime.datetime.now().isoformat(),
    }

    # Store in Redis
    redis_conn.hset(job_request_key, mapping=job_request_data)
    # Set expiration to 24 hours
    redis_conn.expire(job_request_key, 86400)

    # Enqueue the job creation task
    # The crawl worker will handle both creating the job record and enqueueing the crawl task
    high_queue.enqueue(
        "src.crawl_worker.tasks.create_job",
        url=str(request.url),
        tags=request.tags,
        max_pages=request.max_pages,
        job_id=job_id,  # Pass the pre-generated job_id
        job_timeout="5m",  # Set a reasonable timeout for job creation
    )

    logger.info(f"Enqueued job creation for URL: {request.url}, job_id: {job_id}")

    return FetchUrlResponse(job_id=job_id)


@app.get("/search_docs", response_model=SearchDocsResponse, operation_id="search_docs")
async def search_docs(
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
    logger.info(f"Searching docs with query: '{query}', tags: {tags}, max_results: {max_results}")

    # Get a fresh connection to Qdrant for this search
    try:
        qdrant_client = get_qdrant_client()
        logger.debug("Connected to Qdrant for search")

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

        # Get a fresh DuckDB connection for verification and for getting the document text
        conn = get_read_only_connection()
        try:
            # First verify the Qdrant collection exists
            ensure_qdrant_collection(qdrant_client)

            # Get embeddings for the query
            from src.common.embeddings import get_embedding

            results = []

            try:
                # Try semantic search first
                query_embedding = get_embedding(query)
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

                # Format results
                for result in search_results:
                    chunk_id = result.id
                    score = result.score
                    page_id = result.payload.get("page_id")

                    if not page_id:
                        logger.warning(f"Result {chunk_id} missing page_id in payload")
                        continue

                    # Get chunk text from DuckDB
                    try:
                        chunk_query_result = conn.execute(
                            """
                            SELECT raw_text, tags
                            FROM pages
                            WHERE id = ?
                            """,
                            (page_id,),
                        ).fetchone()

                        if not chunk_query_result:
                            logger.warning(f"Page {page_id} not found in database")
                            continue

                        raw_text, tags_json = chunk_query_result
                        result_tags = deserialize_tags(tags_json)

                        # Summarize text if it's too long for display
                        chunk_text = raw_text
                        if len(chunk_text) > 1000:
                            chunk_text = chunk_text[:997] + "..."

                        logger.debug(f"Retrieved text for page {page_id} ({len(raw_text)} chars)")

                        results.append(
                            SearchResult(
                                chunk_text=chunk_text,
                                page_id=page_id,
                                tags=result_tags,
                                score=score,
                            )
                        )
                    except Exception as text_error:
                        logger.error(f"Error retrieving text for page {page_id}: {str(text_error)}")
                        # Continue with other results
            except Exception as vector_error:
                logger.warning(f"Vector search failed: {str(vector_error)}")

            # If no results from vector search, try text search fallback
            if not results:
                logger.info("No results from vector search, trying text search fallback")

                # Prepare tags filter for SQL
                tags_filter = ""
                if tags and len(tags) > 0:
                    tag_conditions = []
                    for tag in tags:
                        escaped_tag = tag.replace("'", "''")
                        tag_conditions.append(f"tags LIKE '%{escaped_tag}%'")
                    tags_filter = f"AND ({' OR '.join(tag_conditions)})"

                # Prepare search terms for SQL
                search_terms = query.split()
                if search_terms:
                    # Build a query that searches for each word in the raw_text
                    search_conditions = []
                    for term in search_terms:
                        if len(term) >= 3:  # Only use terms with at least 3 chars
                            escaped_term = term.replace("'", "''")
                            search_conditions.append(f"raw_text ILIKE '%{escaped_term}%'")

                    if search_conditions:
                        search_filter = f"({' OR '.join(search_conditions)})"

                        # Execute the text search
                        text_search_query = f"""
                        SELECT id, url, raw_text, tags
                        FROM pages
                        WHERE {search_filter} {tags_filter}
                        ORDER BY
                            CASE
                                WHEN url ILIKE '%{query.replace("'", "''")}%' THEN 1
                                ELSE 2
                            END,
                            LENGTH(raw_text) DESC
                        LIMIT {max_results}
                        """

                        logger.debug(f"Executing text search query: {text_search_query}")
                        text_search_results = conn.execute(text_search_query).fetchall()
                        logger.info(f"Found {len(text_search_results)} results from text search")

                        # Process text search results
                        for row in text_search_results:
                            page_id, url, raw_text, tags_json = row
                            result_tags = deserialize_tags(tags_json)

                            # Calculate a relevance score based on term frequency
                            term_count = 0
                            for term in search_terms:
                                if len(term) >= 3:
                                    term_count += raw_text.lower().count(term.lower())

                            # Normalize score to be between 0 and 1
                            text_score = min(0.8, term_count / 100) if term_count > 0 else 0.5

                            # Summarize text
                            chunk_text = raw_text
                            if len(chunk_text) > 1000:
                                chunk_text = chunk_text[:997] + "..."

                            # Try to find the context around the first search term
                            highlight_context = None
                            for term in search_terms:
                                if len(term) >= 3:
                                    term_pos = raw_text.lower().find(term.lower())
                                    if term_pos >= 0:
                                        start_pos = max(0, term_pos - 100)
                                        end_pos = min(len(raw_text), term_pos + 100)
                                        highlight_context = f"...{raw_text[start_pos:end_pos]}..."
                                        break

                            # Use highlight context if found
                            if highlight_context:
                                chunk_text = highlight_context

                            results.append(
                                SearchResult(
                                    chunk_text=chunk_text,
                                    page_id=page_id,
                                    tags=result_tags,
                                    score=text_score,
                                )
                            )

            logger.info(f"Returning {len(results)} search results")
            return SearchDocsResponse(results=results)
        except Exception as db_error:
            logger.error(f"Database error during search: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        finally:
            # Close DuckDB connection
            conn.close()
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/job_progress", response_model=JobProgressResponse, operation_id="job_progress")
async def job_progress(job_id: str = Query(..., description="The job ID to check progress for")):
    """
    Get the progress of a crawl job.

    Args:
        job_id: The job ID

    Returns:
        Job progress information
    """
    logger.info(f"Checking progress for job {job_id}")

    # Create a fresh connection to get the latest data
    # This ensures we always get the most recent job state
    conn = get_read_only_connection()

    try:
        # First try exact job_id match
        logger.info(f"Looking for exact match for job ID: {job_id}")
        result = conn.execute(
            """
            SELECT job_id, start_url, status, pages_discovered, pages_crawled,
                   max_pages, tags, created_at, updated_at, error_message
            FROM jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

        # If not found, try partial match (useful if client received truncated UUID)
        if not result and len(job_id) >= 8:
            logger.info(f"Exact match not found, trying partial match for job ID: {job_id}")
            result = conn.execute(
                """
                SELECT job_id, start_url, status, pages_discovered, pages_crawled,
                       max_pages, tags, created_at, updated_at, error_message
                FROM jobs
                WHERE job_id LIKE ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (f"{job_id}%",),
            ).fetchone()

        if not result:
            # Log the failure to find the job
            logger.warning(f"Job {job_id} not found in database (tried exact and partial match)")
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        (
            job_id,
            url,
            status,
            pages_discovered,
            pages_crawled,
            max_pages,
            tags_json,
            created_at,
            updated_at,
            error_message,
        ) = result

        logger.info(
            f"Found job with ID: {job_id}, status: {status}, discovered: {pages_discovered}, crawled: {pages_crawled}"
        )

        # Determine if job is completed
        completed = status in ["completed", "failed"]

        # Calculate progress percentage
        progress_percent = 0
        if max_pages > 0 and pages_discovered > 0:
            progress_percent = min(
                100, int((pages_crawled / min(pages_discovered, max_pages)) * 100)
            )

        logger.info(
            f"Job {job_id} progress: {pages_crawled}/{pages_discovered} pages, {progress_percent}% complete, status: {status}"
        )

        return JobProgressResponse(
            pages_crawled=pages_crawled,
            pages_total=pages_discovered,
            completed=completed,
            status=status,
            error_message=error_message,
            progress_percent=progress_percent,
            url=url,
            max_pages=max_pages,
            created_at=created_at,
            updated_at=updated_at,
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error checking job progress for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        # Make sure to close the connection to free memory
        conn.close()


@app.get("/list_doc_pages", response_model=ListDocPagesResponse, operation_id="list_doc_pages")
async def list_doc_pages(
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

    # Get a fresh connection for each request
    conn = get_read_only_connection()

    try:
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
    except Exception as e:
        logger.error(f"Error listing document pages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()


@app.get("/get_doc_page", response_model=GetDocPageResponse, operation_id="get_doc_page")
async def get_doc_page(
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
    logger.info(f"Retrieving document page {page_id} (lines {starting_line}-{ending_line})")

    # Get a fresh connection for each request
    conn = get_read_only_connection()

    try:
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
            raise HTTPException(status_code=404, detail=f"Page {page_id} not found")

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
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error retrieving document page {page_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()


mcp = FastApiMCP(
    app,
    name="Doctor",
    description="API for the Doctor web crawling and indexing system",
    describe_all_responses=True,
    describe_full_response_schema=True,
    exclude_operations=["fetch_url"],
)

mcp.mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_SERVICE_HOST, port=WEB_SERVICE_PORT)
