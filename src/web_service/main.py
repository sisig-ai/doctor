"""Main module for the Doctor Web Service."""

from contextlib import asynccontextmanager
import datetime
import logging
import uuid
from typing import List, Optional

from mcp.server.session import ServerSession
import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi_mcp import FastApiMCP
from rq import Queue

from src.common.config import (
    EMBEDDING_MODEL,
    REDIS_URI,
    WEB_SERVICE_HOST,
    WEB_SERVICE_PORT,
)
from src.common.db_setup import (
    QDRANT_COLLECTION_NAME,
    deserialize_tags,
    get_duckdb_connection,
    get_qdrant_client,
    get_read_only_connection,
    init_databases,
    serialize_tags,
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
from src.crawl_worker.tasks import perform_crawl

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
    init_databases()
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
    # Generate a job ID
    job_id = str(uuid.uuid4())

    # Create job record in DuckDB
    conn = get_duckdb_connection()
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, start_url, status, pages_discovered, pages_crawled, 
            max_pages, tags, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            str(request.url),
            "pending",
            0,
            0,
            request.max_pages,
            serialize_tags(request.tags),
            datetime.datetime.now(),
            datetime.datetime.now(),
        ),
    )
    conn.commit()
    conn.close()

    # Enqueue the crawl task
    default_queue.enqueue(
        perform_crawl,
        job_id,  # Positional argument
        url=str(request.url),
        tags=request.tags,
        max_pages=request.max_pages,
        job_timeout="1h",  # Set a reasonable timeout
    )

    logger.info(f"Enqueued fetch job {job_id} for URL: {request.url}")

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
    import litellm

    # Generate query embedding
    try:
        embedding_response = litellm.embedding(model=EMBEDDING_MODEL, input=[query])
        query_vector = embedding_response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

    # Prepare filter if tags are provided
    filter_query = None
    if tags and len(tags) > 0:
        filter_query = {"must": [{"key": "tags", "match": {"any": tags}}]}

    # Search Qdrant - this doesn't use DuckDB so no changes needed
    try:
        qdrant_client = get_qdrant_client()
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=max_results,
            query_filter=filter_query,
            with_payload=True,
            with_vectors=False,
        )

        # Format results
        results = []
        for hit in search_results:
            results.append(
                SearchResult(
                    chunk_text=hit.payload.get("text", ""),
                    page_id=hit.payload.get("page_id", ""),
                    tags=hit.payload.get("tags", []),
                    score=hit.score,
                )
            )

        return SearchDocsResponse(results=results)
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


@app.get("/job_progress", response_model=JobProgressResponse, operation_id="job_progress")
async def job_progress(job_id: str = Query(..., description="The job ID to check progress for")):
    """
    Get the progress of a crawl job.

    Args:
        job_id: The job ID

    Returns:
        Job progress information
    """
    # First try to get the status from Redis for real-time updates
    try:
        redis_conn = redis.from_url(REDIS_URI)
        job_key = f"job_status:{job_id}"

        # Check if the job exists in Redis
        if redis_conn.exists(job_key):
            # Get all fields from the hash
            job_data = redis_conn.hgetall(job_key)

            # Convert byte keys/values to strings if needed
            if isinstance(next(iter(job_data.keys()), b""), bytes):
                job_data = {k.decode("utf-8"): v.decode("utf-8") for k, v in job_data.items()}

            # Extract values with proper types
            status = job_data.get("status", "unknown")
            pages_discovered = int(job_data.get("pages_discovered", 0))
            pages_crawled = int(job_data.get("pages_crawled", 0))
            error_message = job_data.get("error_message")

            # Determine if job is completed
            completed = status in ["completed", "failed"]

            logger.info(
                f"Retrieved job {job_id} progress from Redis: {pages_crawled}/{pages_discovered} pages"
            )

            return JobProgressResponse(
                pages_crawled=pages_crawled,
                pages_total=pages_discovered,
                completed=completed,
                status=status,
                error_message=error_message,
            )

    except Exception as e:
        logger.warning(f"Could not retrieve job progress from Redis: {str(e)}")
        # Continue to the DuckDB fallback

    # Fallback to DuckDB if Redis failed or doesn't have the data
    logger.info(f"Falling back to DuckDB for job {job_id} progress")

    # Query DuckDB for job status using the read-only connection
    conn = get_read_only_connection()

    try:
        # The query remains the same as we've created views with the same names
        result = conn.execute(
            """
            SELECT status, pages_discovered, pages_crawled, error_message
            FROM jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        status, pages_discovered, pages_crawled, error_message = result

        # Determine if job is completed
        completed = status in ["completed", "failed"]

        return JobProgressResponse(
            pages_crawled=pages_crawled,
            pages_total=pages_discovered,
            completed=completed,
            status=status,
            error_message=error_message,
        )
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
            tag_conditions.append(f"tags LIKE '%{tag}%'")

        where_clause = f"WHERE {' OR '.join(tag_conditions)}"

    # Execute count query
    conn = get_read_only_connection()

    try:
        total_count = conn.execute(f"{count_query} {where_clause}").fetchone()[0]

        # Execute main query
        query = f"{base_query} {where_clause} ORDER BY crawl_date DESC LIMIT 100 OFFSET {offset}"
        results = conn.execute(query).fetchall()

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
    # Query DuckDB for page text
    conn = get_read_only_connection()

    try:
        result = conn.execute(
            """
            SELECT raw_text
            FROM pages
            WHERE id = ?
            """,
            (page_id,),
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"Page {page_id} not found")

        raw_text = result[0]

        # Split into lines
        lines = raw_text.splitlines()
        total_lines = len(lines)

        # Validate line range
        if starting_line > total_lines:
            starting_line = 1

        if ending_line > total_lines:
            ending_line = total_lines

        # Extract requested lines
        requested_lines = lines[starting_line - 1 : ending_line]

        return GetDocPageResponse(text="\n".join(requested_lines), total_lines=total_lines)
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
