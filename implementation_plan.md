# Implementation Plan: Migrating Vector Storage from Qdrant to DuckDB

## Introduction

This document provides a detailed implementation plan for migrating the vector storage system from Qdrant to DuckDB. The goal is to consolidate the technology stack, simplify data management, and leverage DuckDB's in-process nature along with its `vss` extension for vector operations. This migration will replace the existing Qdrant integration used for storing and searching 1536-dimension text embeddings.

## Prerequisites

Before beginning the implementation, the following dependencies must be added to the project:

1. Update `pyproject.toml` to include:
   ```toml
   [tool.poetry.dependencies]
   duckdb-vss = "^0.1.0"  # Version may need to be adjusted based on availability
   ```

2. Install the required packages:
   ```bash
   uv pip install duckdb-vss
   ```

## File-by-File Changes

### 1. [`src/common/config.py`](src/common/config.py)

**Changes needed:**
- Keep the existing DuckDB configurations (already present)
- Add a new configuration variable for the vector embeddings table
- Remove Qdrant-specific configurations after the migration is complete

```python
# Current configuration:
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = "doctor_chunks"
VECTOR_SIZE = 1536  # OpenAI ada-002 embedding size

# DuckDB settings
DATA_DIR = os.getenv("DATA_DIR", "data")
DUCKDB_PATH = os.path.join(DATA_DIR, "doctor.duckdb")

# Add this new configuration:
DUCKDB_EMBEDDINGS_TABLE = "document_embeddings"
### 2. [`src/common/db_setup.py`](src/common/db_setup.py)

**Changes needed:**
- Add the document_embeddings table creation SQL
- Implement function to ensure VSS extension is loaded
- Implement function to ensure the HNSW index is created
- Update database initialization to include the embeddings table and index

```python
# Add this new SQL definition:
CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_embeddings (
    id BIGSERIAL PRIMARY KEY,
    embedding FLOAT4[1536] NOT NULL,
    text_chunk VARCHAR,
    page_id VARCHAR,
    url VARCHAR,
    domain VARCHAR,
    tags VARCHAR[],
    job_id VARCHAR
);
"""

# Add a new function to ensure VSS extension is loaded and embeddings table exists
def ensure_duckdb_vss_extension(conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Ensures the DuckDB VSS extension is loaded and embeddings table exists.

    Args:
        conn: An optional DuckDB connection. If None, a new connection is created.
    """
    close_conn = False
    if conn is None:
        try:
            conn = get_duckdb_connection()
            close_conn = True
        except Exception:
            logger.error("Cannot ensure DuckDB VSS extension without a valid connection.")
            return

    try:
        # Load VSS extension
        conn.execute("LOAD vss;")
        logger.info("DuckDB VSS extension loaded")

        # Create embeddings table
        conn.execute(CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL)
        logger.info("DuckDB document_embeddings table created/verified")

        # Create HNSW index if it doesn't exist
        # Note: DuckDB will ignore this if the index already exists (no IF NOT EXISTS support for indexes)
        try:
            conn.execute("""
            CREATE INDEX hnsw_index_on_embeddings
            ON document_embeddings
            USING HNSW (embedding)
            WITH (metric = 'cosine');
            """)
            logger.info("Created HNSW index on document_embeddings.embedding")
        except Exception as e:
            # Check if error is related to index already existing
            if "already exists" in str(e):
                logger.info("HNSW index on document_embeddings.embedding already exists")
            else:
                raise
    except Exception as e:
        logger.error(f"Failed to set up DuckDB VSS and embeddings table: {e}")
    finally:
        if close_conn and conn:
            conn.close()
```

**Update the `init_databases` function:**

```python
def init_databases(read_only: bool = False) -> None:
    """Initializes all databases (DuckDB and Qdrant), creating them if they don't exist.

    Args:
        read_only: If True, initializes DuckDB in read-only mode (tables are not created).
    """
    logger.info("Initializing databases")

    try:
        # Ensure database directory exists
        os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)

        # Only set up database tables if not in read-only mode
        if not read_only:
            # Ensure the DuckDB tables exist with the correct schema
            conn = get_duckdb_connection()
            ensure_duckdb_tables(conn)

            # Ensure VSS extension is loaded and embeddings table exists
            ensure_duckdb_vss_extension(conn)

            conn.close()

        # Eventually this will be removed:
        # Ensure the Qdrant collection exists
        ensure_qdrant_collection()

        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
### 3. [`src/common/indexer.py`](src/common/indexer.py)

The `VectorIndexer` class needs to be refactored to use DuckDB instead of Qdrant:

```python
"""Vector storage functionality using DuckDB."""

import logging
import uuid
from typing import List, Dict, Any, Optional, cast

import duckdb
from src.common.db_setup import get_duckdb_connection
from src.common.config import DUCKDB_EMBEDDINGS_TABLE, VECTOR_SIZE

# Configure logging
logger = logging.getLogger(__name__)


class VectorIndexer:
    """Class for indexing vectors and payloads in DuckDB."""

    def __init__(self, table_name: str = None, connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize the vector indexer.

        Args:
            table_name: Name of the DuckDB table to use (defaults to config value)
            connection: Optional DuckDB connection to use (creates a new one if not provided)
        """
        self._own_connection = connection is None
        self.conn = connection if connection is not None else get_duckdb_connection()

        # Ensure VSS extension is loaded
        self.conn.execute("LOAD vss;")

        self.table_name = table_name or DUCKDB_EMBEDDINGS_TABLE
        logger.debug(f"Initialized VectorIndexer with table={self.table_name}")

    def __del__(self):
        """Clean up resources on destruction."""
        if hasattr(self, '_own_connection') and self._own_connection and hasattr(self, 'conn'):
            try:
                self.conn.close()
                logger.debug("Closed DuckDB connection in VectorIndexer destructor")
            except Exception:
                pass

    async def index_vector(
        self, vector: List[float], payload: Dict[str, Any], point_id: Optional[str] = None
    ) -> str:
        """
        Index a single vector with its payload in DuckDB.

        Args:
            vector: The embedding vector
            payload: Additional data to store with the vector
            point_id: Optional ID for the point (generated if not provided)

        Returns:
            The ID of the indexed point
        """
        if not vector:
            logger.error("Cannot index empty vector")
            raise ValueError("Vector cannot be empty")

        if len(vector) != VECTOR_SIZE:
            logger.error(f"Vector dimension mismatch: expected {VECTOR_SIZE}, got {len(vector)}")
            raise ValueError(f"Vector dimension must be {VECTOR_SIZE}")

        # Generate a point ID if not provided
        if point_id is None:
            point_id = str(uuid.uuid4())

        logger.debug(f"Indexing vector of dimension {len(vector)} with point_id={point_id}")

        try:
            # Extract payload fields
            text_chunk = payload.get("text", "")
            page_id = payload.get("page_id", "")
            url = payload.get("url", "")
            domain = payload.get("domain", "")
            tags = payload.get("tags", [])
            job_id = payload.get("job_id", "")

            # Insert the vector and payload into the table
            self.conn.execute(
                f"""
                INSERT INTO {self.table_name}
                (id, embedding, text_chunk, page_id, url, domain, tags, job_id)
                VALUES (?, ?::FLOAT4[{VECTOR_SIZE}], ?, ?, ?, ?, ?::VARCHAR[], ?)
                """,
                (point_id, vector, text_chunk, page_id, url, domain, tags, job_id)
            )

            logger.debug(f"Successfully indexed point {point_id}")
            return point_id

        except Exception as e:
            logger.error(f"Error indexing vector: {str(e)}")
            raise

    async def index_batch(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        point_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Index a batch of vectors with their payloads in DuckDB.

        Args:
            vectors: List of embedding vectors
            payloads: List of payloads corresponding to the vectors
            point_ids: Optional list of IDs for the points (generated if not provided)

        Returns:
            List of IDs of the indexed points
        """
        if not vectors or not payloads:
            logger.error("Cannot index empty batch")
            raise ValueError("Vectors and payloads cannot be empty")

        if len(vectors) != len(payloads):
            logger.error("Length mismatch between vectors and payloads")
            raise ValueError("Number of vectors must match number of payloads")

        # Generate point IDs if not provided
        if point_ids is None:
            point_ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(point_ids) != len(vectors):
            logger.error("Length mismatch between vectors and point_ids")
            raise ValueError("Number of point_ids must match number of vectors")

        logger.debug(f"Indexing batch of {len(vectors)} vectors")

        try:
            # Prepare batch data for insertion
            batch_data = []
            for i, (point_id, vector, payload) in enumerate(zip(point_ids, vectors, payloads)):
                if len(vector) != VECTOR_SIZE:
                    logger.error(f"Vector {i} dimension mismatch: expected {VECTOR_SIZE}, got {len(vector)}")
                    raise ValueError(f"Vector dimension must be {VECTOR_SIZE}")

                text_chunk = payload.get("text", "")
                page_id = payload.get("page_id", "")
                url = payload.get("url", "")
                domain = payload.get("domain", "")
                tags = payload.get("tags", [])
                job_id = payload.get("job_id", "")

                batch_data.append((point_id, vector, text_chunk, page_id, url, domain, tags, job_id))

            # Use executemany for efficient batch insertion
            self.conn.executemany(
                f"""
                INSERT INTO {self.table_name}
                (id, embedding, text_chunk, page_id, url, domain, tags, job_id)
                VALUES (?, ?::FLOAT4[{VECTOR_SIZE}], ?, ?, ?, ?, ?::VARCHAR[], ?)
                """,
                batch_data
            )

            logger.debug(f"Successfully indexed batch of {len(batch_data)} points")
            return point_ids

        except Exception as e:
            logger.error(f"Error indexing vector batch: {str(e)}")
            raise

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in DuckDB.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            filter_payload: Optional filter to apply to the search (tags filter supported)

        Returns:
            List of search results, each containing the point ID, score, and payload
        """
        logger.debug(f"Searching for similar vectors with limit={limit}")

        try:
            # Build the SQL query with optional tag filtering
            query = f"""
            SELECT
                id,
                text_chunk,
                page_id,
                url,
                domain,
                tags,
                job_id,
                array_cosine_distance(embedding, ?::FLOAT4[{VECTOR_SIZE}]) AS cosine_distance
            FROM {self.table_name}
            """

            params = [query_vector]

            # Add tag filtering if present in filter_payload
            if filter_payload and 'must' in filter_payload:
                for condition in filter_payload['must']:
                    if condition.get('key') == 'tags' and condition.get('match', {}).get('any'):
                        tag_list = condition['match']['any']
                        if tag_list:
                            query += " WHERE array_has_any(tags, ?::VARCHAR[])"
                            params.append(tag_list)
                            logger.debug(f"Added tag filter: {tag_list}")
                            break

            # Add ordering and limit
            query += " ORDER BY cosine_distance ASC LIMIT ?"
            params.append(limit)

            # Execute the query
            result = self.conn.execute(query, params).fetchall()
            logger.debug(f"Search returned {len(result)} results")

            # Format results to match the expected structure
            search_results = []
            for row in result:
                id, text_chunk, page_id, url, domain, tags, job_id, distance = row

                # Convert cosine distance to similarity score (1 - distance)
                # This matches Qdrant's convention where higher scores are better
                similarity_score = 1.0 - float(distance)

                search_results.append({
                    "id": id,
                    "score": similarity_score,
                    "payload": {
                        "text": text_chunk,
                        "page_id": page_id,
                        "url": url,
                        "domain": domain,
                        "tags": tags,
                        "job_id": job_id
                    }
                })

            return search_results

        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise
```
```
### 4. [`src/web_service/services/document_service.py`](src/web_service/services/document_service.py)

**Changes needed:**
- Refactor the `search_docs` function to use the new DuckDB-based VectorIndexer

```python
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

    results = []

    try:
        # Try semantic search
        from src.lib.embedder import generate_embedding
        from src.common.indexer import VectorIndexer

        # Generate embedding for the query
        query_embedding = await generate_embedding(query)
        logger.debug("Generated embedding for search query")

        # Prepare filter based on tags
        filter_conditions = None
        if tags and len(tags) > 0:
            # Create a filter compatible with VectorIndexer.search method
            filter_conditions = {
                'must': [
                    {
                        'key': 'tags',
                        'match': {'any': tags}
                    }
                ]
            }
            logger.debug(f"Applied tag filters: {tags}")

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
```

### 5. [`src/web_service/api/documents.py`](src/web_service/api/documents.py)

**Changes needed:**
- Update `search_docs_endpoint` to remove Qdrant client usage

```python
@router.get("/search_docs", response_model=SearchDocsResponse, operation_id="search_docs")
async def search_docs_endpoint(
    query: str = Query(..., description="The search string to query the database with"),
    tags: Optional[List[str]] = Query(None, description="Tags to limit the search with"),
    max_results: int = Query(10, description="Maximum number of results to return", ge=1, le=100),
    return_full_document_text: bool = Query(
        False,
        description="Whether to return the full document text instead of the matching chunks only",
    ),
):
    """
    Search for documents using semantic search. Use `get_doc_page` to get the full text of a document page.

    Args:
        query: The search query
        tags: Optional tags to filter by
        max_results: Maximum number of results to return

    Returns:
        Search results
    """
    logger.info(
        f"API: Searching docs with query: '{query}', tags: {tags}, max_results: {max_results}, return_full_document_text: {return_full_document_text}"
    )

    try:
        # Get a fresh DuckDB connection
        conn = await get_duckdb_connection_with_retry()
        try:
            # Call the service function
            response = await search_docs(conn, query, tags, max_results, return_full_document_text)
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
## Data Migration Script

Create a new file `scripts/migrate_qdrant_to_duckdb.py` to handle the one-time data migration:

```python
#!/usr/bin/env python
"""
Migration script to transfer vector data from Qdrant to DuckDB.
"""

import asyncio
import argparse
from typing import List, Dict, Any
import time

from qdrant_client import QdrantClient
import duckdb

from src.common.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    DUCKDB_PATH,
    DUCKDB_EMBEDDINGS_TABLE,
    VECTOR_SIZE,
)
from src.common.db_setup import (
    get_qdrant_client,
    get_duckdb_connection,
    ensure_duckdb_vss_extension,
)
from src.common.logger import get_logger

# Configure logging
logger = get_logger(__name__)


def get_points_count_from_qdrant(client: QdrantClient, collection_name: str) -> int:
    """Get the total number of points in a Qdrant collection.

    Args:
        client: Qdrant client
        collection_name: Collection name

    Returns:
        int: Total number of points
    """
    collection_info = client.get_collection(collection_name=collection_name)
    return collection_info.points_count


def get_all_points_from_qdrant(
    client: QdrantClient, collection_name: str, batch_size: int = 100
) -> List[Dict[str, Any]]:
    """Get all points from a Qdrant collection in batches.

    Args:
        client: Qdrant client
        collection_name: Collection name
        batch_size: Number of points to retrieve per batch

    Returns:
        List of points, each containing id, vector, and payload
    """
    offset = 0
    all_points = []

    while True:
        # Scroll gives us a batch of points from the collection
        points = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )[0]

        if not points:
            break

        all_points.extend(points)
        offset += len(points)
        logger.info(f"Retrieved {offset} points from Qdrant")

    return all_points


def insert_points_into_duckdb(
    conn: duckdb.DuckDBPyConnection, table_name: str, points: List[Dict[str, Any]], batch_size: int = 100
) -> None:
    """Insert points into DuckDB embeddings table.

    Args:
        conn: DuckDB connection
        table_name: Table name
        points: List of points from Qdrant
        batch_size: Number of points to insert per batch
    """
    # Ensure VSS extension is loaded
    conn.execute("LOAD vss;")

    total_points = len(points)
    num_batches = (total_points + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_points)
        batch = points[start_idx:end_idx]

        # Prepare batch data
        batch_data = []
        for point in batch:
            point_id = point.id
            vector = point.vector
            payload = point.payload

            # Extract payload fields
            text_chunk = payload.get("text", "")
            page_id = payload.get("page_id", "")
            url = payload.get("url", "")
            domain = payload.get("domain", "")
            tags = payload.get("tags", [])
            job_id = payload.get("job_id", "")

            batch_data.append((point_id, vector, text_chunk, page_id, url, domain, tags, job_id))

        # Insert batch
        conn.executemany(
            f"""
            INSERT INTO {table_name}
            (id, embedding, text_chunk, page_id, url, domain, tags, job_id)
            VALUES (?, ?::FLOAT4[{VECTOR_SIZE}], ?, ?, ?, ?, ?::VARCHAR[], ?)
            """,
            batch_data
        )

        logger.info(f"Inserted batch {batch_idx + 1}/{num_batches} ({len(batch_data)} points) into DuckDB")


async def run_migration(batch_size: int = 100) -> None:
    """Run the migration from Qdrant to DuckDB.

    Args:
        batch_size: Number of points to process per batch
    """
    start_time = time.time()
    logger.info("Starting migration from Qdrant to DuckDB")

    # Connect to Qdrant
    qdrant_client = get_qdrant_client()
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    # Check if collection exists
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if QDRANT_COLLECTION_NAME not in collection_names:
        logger.error(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found")
        return

    # Get total points count
    total_points = get_points_count_from_qdrant(qdrant_client, QDRANT_COLLECTION_NAME)
    logger.info(f"Found {total_points} points in Qdrant collection '{QDRANT_COLLECTION_NAME}'")

    if total_points == 0:
        logger.warning("No points to migrate")
        return

    # Connect to DuckDB
    duckdb_conn = get_duckdb_connection()
    logger.info(f"Connected to DuckDB at {DUCKDB_PATH}")

    # Ensure VSS extension is loaded and embeddings table exists
    ensure_duckdb_vss_extension(duckdb_conn)

    try:
        # Get all points from Qdrant
        logger.info(f"Retrieving points from Qdrant in batches of {batch_size}")
        points = get_all_points_from_qdrant(qdrant_client, QDRANT_COLLECTION_NAME, batch_size)

        # Insert points into DuckDB
        logger.info(f"Inserting {len(points)} points into DuckDB")
        insert_points_into_duckdb(duckdb_conn, DUCKDB_EMBEDDINGS_TABLE, points, batch_size)

        # Verify migration
        count_result = duckdb_conn.execute(f"SELECT COUNT(*) FROM {DUCKDB_EMBEDDINGS_TABLE}").fetchone()
        duckdb_count = count_result[0] if count_result else 0

        logger.info(f"Migration complete: {duckdb_count}/{total_points} points migrated")

        if duckdb_count != total_points:
            logger.warning(f"Count mismatch: Qdrant has {total_points} points, DuckDB has {duckdb_count} points")

        elapsed_time = time.time() - start_time
        logger.info(f"Migration completed in {elapsed_time:.2f} seconds")

    finally:
        duckdb_conn.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Migrate vector data from Qdrant to DuckDB")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of points to process per batch",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_migration(batch_size=args.batch_size))
```

## Testing Considerations

### Unit Tests

Update tests in `tests/common/test_indexer.py` to test the new DuckDB-based `VectorIndexer`:

```python
async def test_vectorindexer_duckdb():
    """Test the DuckDB implementation of VectorIndexer."""
    # Create in-memory DuckDB connection for testing
    conn = duckdb.connect(":memory:")
    conn.execute("LOAD vss;")
    conn.execute("""
    CREATE TABLE document_embeddings (
        id BIGSERIAL PRIMARY KEY,
        embedding FLOAT4[1536] NOT NULL,
        text_chunk VARCHAR,
        page_id VARCHAR,
        url VARCHAR,
        domain VARCHAR,
        tags VARCHAR[],
        job_id VARCHAR
    );
    """)

    # Create VectorIndexer with the test connection
    indexer = VectorIndexer(connection=conn)

    # Test vector with random values
    test_vector = [0.1] * 1536
    test_payload = {
        "text": "Test chunk",
        "page_id": "test-page-id",
        "url": "https://example.com",
        "domain": "example.com",
        "tags": ["test", "example"],
        "job_id": "test-job",
    }

    # Test index_vector
    point_id = await indexer.index_vector(test_vector, test_payload)
    assert point_id is not None

    # Test search
    results = await indexer.search(test_vector, limit=1)
    assert len(results) == 1
    assert results[0]["id"] == point_id
    assert results[0]["payload"]["text"] == "Test chunk"

    # Clean up
    conn.close()
```

### Integration Tests

Update tests in `tests/services/test_document_service.py` to validate the search functionality with DuckDB instead of Qdrant:

```python
async def test_search_docs_with_duckdb():
    """Test search_docs function with DuckDB backend."""
    # Create in-memory DuckDB database for testing
    conn = duckdb.connect(":memory:")
    conn.execute("LOAD vss;")
    conn.execute("""
    CREATE TABLE document_embeddings (
        id BIGSERIAL PRIMARY KEY,
        embedding FLOAT4[1536] NOT NULL,
        text_chunk VARCHAR,
        page_id VARCHAR,
        url VARCHAR,
        domain VARCHAR,
        tags VARCHAR[],
        job_id VARCHAR
    );
    """)

    # Add test data
    from src.common.indexer import VectorIndexer
    indexer = VectorIndexer(connection=conn)

    test_vector = [0.1] * 1536
    test_payload = {
        "text": "Test document about artificial intelligence",
        "page_id": "page-1",
        "url": "https://example.com/ai",
        "domain": "example.com",
        "tags": ["ai", "test"],
        "job_id": "test-job",
    }

    # Index test data
    await indexer.index_vector(test_vector, test_payload)

    # Mock embedding generation to return a similar vector
    with patch('src.lib.embedder.generate_embedding', return_value=[0.11] * 1536):
        # Search with a query that should match
        result = await search_docs(conn, "artificial intelligence", tags=["ai"], max_results=5)

        # Validate results
        assert len(result.results) == 1
        assert "artificial intelligence" in result.results[0].chunk_text
        assert result.results[0].page_id == "page-1"
        assert "ai" in result.results[0].tags

    # Clean up
    conn.close()
```

## Deployment Considerations

### 1. Environment Setup

- Ensure the DuckDB `vss` extension is available in the deployment environment:
  - Update the Dockerfile to install the necessary dependencies:
    ```dockerfile
    # Add to both Dockerfiles where needed
    RUN pip install duckdb-vss
    ```

### 2. Database File Persistence

- Ensure the DuckDB database file is stored in a persistent volume in production environments:
  ```yaml
  # docker-compose.yml addition
  volumes:
    - ./data:/app/data  # Mount data directory to persist DuckDB files
  ```

### 3. Migration Process

- The migration should be executed as a one-time operation during a maintenance window:
  1. Stop all services that access Qdrant or the data being migrated
  2. Run the migration script
  3. Verify the migration was successful
  4. Start services with the updated DuckDB configuration

### 4. Rollback Plan

- Maintain the Qdrant service during initial deployment to allow for rollback if needed
- After successful validation in production, Qdrant can be decommissioned

## Conclusion

This implementation plan provides a comprehensive approach to migrate from Qdrant to DuckDB for vector storage. By following these steps, the system will benefit from a simplified technology stack, consolidated data environment, and reduced operational overhead while maintaining the existing functionality.

The migration preserves the current API contract, ensuring backward compatibility for client applications. Once completed, the system will use DuckDB's VSS extension for efficient vector similarity search operations, with all vector data stored alongside other application data in a single DuckDB database.
```
