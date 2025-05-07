"""Main module for the Doctor Web Service."""

from contextlib import asynccontextmanager
import uuid
from typing import List, Optional
import asyncio

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
    DeleteDocsRequest,
    FetchUrlRequest,
    FetchUrlResponse,
    GetDocPageResponse,
    JobProgressResponse,
    ListDocPagesResponse,
    SearchDocsResponse,
    SearchResult,
)
from src.lib.logger import get_logger
from qdrant_client.http import models as qdrant_models

# Get logger for this module
logger = get_logger(__name__)

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

queue = Queue("worker", connection=redis_conn)


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

    # Enqueue the job creation task
    # The crawl worker will handle both creating the job record and enqueueing the crawl task
    queue.enqueue(
        "src.crawl_worker.tasks.create_job",
        request.url,
        job_id,
        tags=request.tags,
        max_pages=request.max_pages,
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
    Search for documents using semantic search. Use the `get_doc_page` endpoint to get the full text of a document page.

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
            from src.lib.embedder import generate_embedding

            results = []

            try:
                # Try semantic search first
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
                    if (
                        page_id not in page_best_results
                        or score > page_best_results[page_id]["score"]
                    ):
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
    logger.info(f"BEGIN job_progress for job {job_id}")

    # Create a fresh connection to get the latest data
    # This ensures we always get the most recent job state
    attempts = 0
    max_attempts = 3
    retry_delay = 0.1  # seconds
    result = None  # Initialize result to None outside the loop

    logger.info(f"Starting check loop for job {job_id} (max_attempts={max_attempts})")
    while attempts < max_attempts:
        attempts += 1
        conn = None

        try:
            # Get a fresh read-only connection each time
            conn = get_read_only_connection()
            logger.info(f"Established fresh read-only connection to database (attempt {attempts})")

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

            # --- Job found in database ---
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
                f"Found job with ID: {job_id}, status: {status}, discovered: {pages_discovered}, crawled: {pages_crawled}, updated: {updated_at}"
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
    # We check 'result' one last time for clarity, though the loop logic ensures it's None here.
    if not result:
        # Check job count for debugging before raising 404
        conn_check = None
        try:
            conn_check = get_read_only_connection()
            job_count = conn_check.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            logger.info(f"Database contains {job_count} total jobs during final 404 check.")
        except Exception as count_error:
            logger.warning(f"Failed to count jobs in database during 404 check: {str(count_error)}")
        finally:
            if conn_check:
                conn_check.close()
        logger.warning(f"Raising 404 for job {job_id} after all retries.")  # Add log before raising
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found after retries")


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


@app.post("/delete_docs", status_code=204, operation_id="delete_docs")
async def delete_docs(request: DeleteDocsRequest):
    """
    Delete documents from the database based on filters.

    Args:
        request: The delete request with optional filters
            tags: Optional list of tags to filter by
            domain: Optional domain substring to filter by
            page_ids: Optional list of specific page IDs to delete

    Returns:
        204 No Content response
    """
    logger.info(
        f"Enqueueing delete task with filters: tags={request.tags}, domain={request.domain}, page_ids={request.page_ids}"
    )

    # Generate a task ID for tracking logs
    task_id = str(uuid.uuid4())

    # Enqueue the delete task
    queue.enqueue(
        "src.crawl_worker.tasks.delete_docs",
        task_id,
        request.tags,
        request.domain,
        request.page_ids,
    )

    logger.info(f"Enqueued delete task with ID: {task_id}")

    # Return 204 No Content
    return None


mcp = FastApiMCP(
    app,
    name="Doctor",
    description="API for the Doctor web crawling and indexing system",
    describe_all_responses=True,
    describe_full_response_schema=True,
    exclude_operations=["fetch_url", "job_progress", "delete_docs"],
)

mcp.mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_SERVICE_HOST, port=WEB_SERVICE_PORT)
