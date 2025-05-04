"""Text embedding functionality using LiteLLM."""

import logging
from typing import List

import litellm
from src.common.config import EMBEDDING_MODEL

# Configure logging
logger = logging.getLogger(__name__)


async def generate_embedding(text: str, model: str = None, timeout: int = 30) -> List[float]:
    """
    Generate an embedding for a text chunk.

    Args:
        text: The text to embed
        model: The embedding model to use (defaults to config value)
        timeout: Timeout in seconds for the embedding API call

    Returns:
        The generated embedding as a list of floats
    """
    if not text.strip():
        logger.warning("Received empty text for embedding, cannot proceed")
        raise ValueError("Cannot generate embedding for empty text")

    model_name = model or EMBEDDING_MODEL
    logger.debug(f"Generating embedding for text of length {len(text)} using model {model_name}")

    try:
        embedding_response = await litellm.aembedding(
            model=model_name,
            input=[text],
            timeout=timeout,
        )

        # Extract the embedding vector from the response
        embedding = embedding_response["data"][0]["embedding"]
        logger.debug(f"Successfully generated embedding of dimension {len(embedding)}")

        return embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


class BatchEmbedder:
    """Class for batch processing text embeddings."""

    def __init__(self, model: str = None, batch_size: int = 10, timeout: int = 30):
        """
        Initialize the batch embedder.

        Args:
            model: The embedding model to use (defaults to config value)
            batch_size: Size of batches for embedding generation
            timeout: Timeout in seconds for embedding API calls
        """
        self.model = model or EMBEDDING_MODEL
        self.batch_size = batch_size
        self.timeout = timeout

        logger.debug(
            f"Initialized BatchEmbedder with model={self.model}, batch_size={self.batch_size}"
        )

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Received empty batch for embedding, returning empty list")
            return []

        # Filter out empty texts
        filtered_texts = [text for text in texts if text.strip()]
        if len(filtered_texts) < len(texts):
            logger.debug(
                f"Filtered out {len(texts) - len(filtered_texts)} empty texts before embedding"
            )

        if not filtered_texts:
            logger.warning("All texts in batch were empty, returning empty list")
            return []

        logger.debug(f"Generating embeddings for {len(filtered_texts)} texts")

        try:
            embedding_response = await litellm.aembedding(
                model=self.model,
                input=filtered_texts,
                timeout=self.timeout,
            )

            # Extract the embedding vectors from the response
            embeddings = [item["embedding"] for item in embedding_response["data"]]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
