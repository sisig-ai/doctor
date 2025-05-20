"""Tests for the embedder module."""

from unittest.mock import AsyncMock, patch

import pytest

from src.lib.embedder import generate_embedding


@pytest.fixture
def mock_embedding_response():
    """Mock response from litellm.aembedding."""
    return {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]}


@pytest.mark.unit
@pytest.mark.async_test
async def test_generate_embedding(sample_text, mock_embedding_response):
    """Test generating an embedding for a text chunk."""
    with patch("src.lib.embedder.litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
        mock_aembedding.return_value = mock_embedding_response

        # Test with default model
        with patch("src.lib.embedder.DOC_EMBEDDING_MODEL", "text-embedding-3-small"):
            embedding = await generate_embedding(sample_text)
            assert embedding == mock_embedding_response["data"][0]["embedding"]
            mock_aembedding.assert_called_once_with(
                model="text-embedding-3-small",
                input=[sample_text],
                timeout=30,
            )

        # Test with custom model and timeout
        mock_aembedding.reset_mock()
        embedding = await generate_embedding(sample_text, model="custom-model", timeout=60)

        # Check that aembedding was called with the correct arguments
        mock_aembedding.assert_called_once_with(
            model="custom-model",
            input=[sample_text],
            timeout=60,
        )


@pytest.mark.unit
@pytest.mark.async_test
async def test_generate_embedding_with_empty_text():
    """Test generating an embedding with empty text."""
    with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
        await generate_embedding("")

    with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
        await generate_embedding("   ")


@pytest.mark.unit
@pytest.mark.async_test
async def test_generate_embedding_error_handling():
    """Test error handling when generating an embedding."""
    with patch("src.lib.embedder.litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
        # Simulate an API error
        mock_aembedding.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            await generate_embedding("Some text")
