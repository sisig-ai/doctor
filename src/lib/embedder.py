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
