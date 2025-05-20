"""Text embedding functionality using LiteLLM."""

from typing import Literal

import litellm

from src.common.config import DOC_EMBEDDING_MODEL, QUERY_EMBEDDING_MODEL
from src.common.logger import get_logger

# Configure logging
logger = get_logger(__name__)


async def generate_embedding(
    text: str,
    model: str = None,
    timeout: int = 30,
    text_type: Literal["doc", "query"] = "doc",
) -> list[float]:
    """Generate an embedding for a text chunk.

    Args:
        text: The text to embed
        model: The embedding model to use (defaults to config value)
        timeout: Timeout in seconds for the embedding API call
        text_type: The type of text to embed (defaults to "doc")

    Returns:
        The generated embedding as a list of floats

    """
    if not text.strip():
        logger.warning("Received empty text for embedding, cannot proceed")
        raise ValueError("Cannot generate embedding for empty text")

    model_name = model or (DOC_EMBEDDING_MODEL if text_type == "doc" else QUERY_EMBEDDING_MODEL)
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
        logger.error(f"Error generating embedding: {e!s}")
        raise
