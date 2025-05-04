"""Vector storage functionality using Qdrant."""

import logging
import uuid
from typing import List, Dict, Any, Optional

from src.common.db_setup import get_qdrant_client, QDRANT_COLLECTION_NAME

# Configure logging
logger = logging.getLogger(__name__)


class VectorIndexer:
    """Class for indexing vectors and payloads in Qdrant."""

    def __init__(self, collection_name: str = None):
        """
        Initialize the vector indexer.

        Args:
            collection_name: Name of the Qdrant collection to use (defaults to config value)
        """
        self.client = get_qdrant_client()
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        logger.debug(f"Initialized VectorIndexer with collection={self.collection_name}")

    async def index_vector(
        self, vector: List[float], payload: Dict[str, Any], point_id: Optional[str] = None
    ) -> str:
        """
        Index a single vector with its payload in Qdrant.

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

        # Generate a point ID if not provided
        if point_id is None:
            point_id = str(uuid.uuid4())

        logger.debug(f"Indexing vector of dimension {len(vector)} with point_id={point_id}")

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[{"id": point_id, "vector": vector, "payload": payload}],
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
        Index a batch of vectors with their payloads in Qdrant.

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
            points = [
                {"id": point_id, "vector": vector, "payload": payload}
                for point_id, vector, payload in zip(point_ids, vectors, payloads)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.debug(f"Successfully indexed batch of {len(points)} points")
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
        Search for similar vectors in Qdrant.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            filter_payload: Optional filter to apply to the search

        Returns:
            List of search results, each containing the point ID, score, and payload
        """
        logger.debug(f"Searching for similar vectors with limit={limit}")

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_payload,
            )

            logger.debug(f"Search returned {len(search_result)} results")
            return search_result

        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise
