"""Vector storage functionality using DuckDB."""

import logging
import uuid
from typing import Any

import duckdb

from src.common.config import DUCKDB_EMBEDDINGS_TABLE, VECTOR_SIZE
from src.lib.database import DatabaseOperations

# Configure logging
logger = logging.getLogger(__name__)


class VectorIndexer:
    """Class for indexing vectors and payloads in DuckDB."""

    def __init__(
        self,
        table_name: str = None,
        connection: duckdb.DuckDBPyConnection | None = None,
    ):
        """Initialize the vector indexer.

        Args:
            table_name: Name of the DuckDB table to use (defaults to config value)
            connection: Optional DuckDB connection to use (creates a new one if not provided)

        """
        self._own_connection = connection is None
        if connection is not None:
            self.conn = connection
        else:
            db = DatabaseOperations()
            self.conn = db.db.ensure_connection()

        # Ensure VSS extension is installed and loaded
        try:
            self.conn.execute("INSTALL vss;")
            logger.debug("VSS extension installed")
        except Exception as e:
            # If extension is already installed, this will fail, but that's okay
            logger.debug(f"VSS extension installation attempt: {e!s}")

        self.conn.execute("LOAD vss;")
        logger.debug("VSS extension loaded")

        self.table_name = table_name or DUCKDB_EMBEDDINGS_TABLE
        logger.debug(f"Initialized VectorIndexer with table={self.table_name}")

    def __del__(self):
        """Clean up resources on destruction."""
        if hasattr(self, "_own_connection") and self._own_connection and hasattr(self, "conn"):
            try:
                self.conn.close()
                logger.debug("Closed DuckDB connection in VectorIndexer destructor")
            except Exception:
                pass

    async def index_vector(
        self,
        vector: list[float],
        payload: dict[str, Any],
        point_id: str | None = None,
    ) -> str:
        """Index a single vector with its payload in DuckDB.

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
            # First check if the table exists
            try:
                table_exists = self.conn.execute(
                    f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{self.table_name}'",
                ).fetchone()[0]

                if not table_exists:
                    logger.error(
                        f"Table '{self.table_name}' does not exist. Ensure database setup is complete.",
                    )
                    raise Exception(
                        f"Table '{self.table_name}' does not exist. Run database initialization first.",
                    )
            except Exception as check_err:
                logger.error(f"Error checking if table exists: {check_err!s}")
                raise

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
                VALUES (?::VARCHAR, ?::FLOAT4[{VECTOR_SIZE}], ?, ?, ?, ?, ?::VARCHAR[], ?)
                """,
                (point_id, vector, text_chunk, page_id, url, domain, tags, job_id),
            )

            logger.debug(f"Successfully indexed point {point_id}")
            return point_id

        except Exception as e:
            logger.error(f"Error indexing vector: {e!s}")
            raise

    async def index_batch(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        point_ids: list[str] | None = None,
    ) -> list[str]:
        """Index a batch of vectors with their payloads in DuckDB.

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
            for i, (point_id, vector, payload) in enumerate(
                zip(point_ids, vectors, payloads, strict=False)
            ):
                if len(vector) != VECTOR_SIZE:
                    logger.error(
                        f"Vector {i} dimension mismatch: expected {VECTOR_SIZE}, got {len(vector)}",
                    )
                    raise ValueError(f"Vector dimension must be {VECTOR_SIZE}")

                text_chunk = payload.get("text", "")
                page_id = payload.get("page_id", "")
                url = payload.get("url", "")
                domain = payload.get("domain", "")
                tags = payload.get("tags", [])
                job_id = payload.get("job_id", "")

                batch_data.append(
                    (point_id, vector, text_chunk, page_id, url, domain, tags, job_id),
                )

            # Use executemany for efficient batch insertion
            self.conn.executemany(
                f"""
                INSERT INTO {self.table_name}
                (id, embedding, text_chunk, page_id, url, domain, tags, job_id)
                VALUES (?::VARCHAR, ?::FLOAT4[{VECTOR_SIZE}], ?, ?, ?, ?, ?::VARCHAR[], ?)
                """,
                batch_data,
            )

            logger.debug(f"Successfully indexed batch of {len(batch_data)} points")
            return point_ids

        except Exception as e:
            logger.error(f"Error indexing vector batch: {e!s}")
            raise

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in DuckDB.

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
                array_cosine_distance(embedding, ?::FLOAT[{VECTOR_SIZE}]) AS cosine_distance
            FROM {self.table_name}
            """

            params = [query_vector]

            # Add tag filtering if present in filter_payload
            if filter_payload and "must" in filter_payload:
                for condition in filter_payload["must"]:
                    if condition.get("key") == "tags" and condition.get("match", {}).get("any"):
                        tag_list = condition["match"]["any"]
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

                search_results.append(
                    {
                        "id": id,
                        "score": similarity_score,
                        "payload": {
                            "text": text_chunk,
                            "page_id": page_id,
                            "url": url,
                            "domain": domain,
                            "tags": tags,
                            "job_id": job_id,
                        },
                    },
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching vectors: {e!s}")
            raise
