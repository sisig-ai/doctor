"""Tests for the embedder module."""

import pytest
from unittest.mock import patch, AsyncMock

from src.lib.embedder import generate_embedding, BatchEmbedder


@pytest.fixture
def mock_embedding_response():
    """Mock response from litellm.aembedding."""
    return {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]}


@pytest.fixture
def mock_batch_embedding_response():
    """Mock response from litellm.aembedding for batch processing."""
    return {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
            {"embedding": [0.5, 0.4, 0.3, 0.2, 0.1]},
            {"embedding": [0.3, 0.3, 0.3, 0.3, 0.3]},
        ]
    }


@pytest.mark.unit
@pytest.mark.async_test
async def test_generate_embedding(sample_text, mock_embedding_response):
    """Test generating an embedding for a text chunk."""
    with patch("src.lib.embedder.litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
        mock_aembedding.return_value = mock_embedding_response

        # Test with default model
        with patch("src.lib.embedder.EMBEDDING_MODEL", "text-embedding-3-small"):
            embedding = await generate_embedding(sample_text)

            # Check that aembedding was called with the correct arguments
            mock_aembedding.assert_called_once_with(
                model="text-embedding-3-small",
                input=[sample_text],
                timeout=30,
            )

            # Check that we got the expected embedding
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

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


@pytest.mark.unit
def test_batch_embedder_initialization():
    """Test BatchEmbedder initialization."""
    # Test with default values
    with patch("src.lib.embedder.EMBEDDING_MODEL", "text-embedding-3-small"):
        embedder = BatchEmbedder()
        assert embedder.model == "text-embedding-3-small"
        assert embedder.batch_size == 10
        assert embedder.timeout == 30

    # Test with custom values
    embedder = BatchEmbedder(model="custom-model", batch_size=20, timeout=60)
    assert embedder.model == "custom-model"
    assert embedder.batch_size == 20
    assert embedder.timeout == 60


@pytest.mark.unit
@pytest.mark.async_test
async def test_batch_embedder_embed_batch(mock_batch_embedding_response):
    """Test generating embeddings for a batch of texts."""
    with patch("src.lib.embedder.litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
        mock_aembedding.return_value = mock_batch_embedding_response

        embedder = BatchEmbedder(model="test-model")
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await embedder.embed_batch(texts)

        # Check that aembedding was called with the correct arguments
        mock_aembedding.assert_called_once_with(
            model="test-model",
            input=texts,
            timeout=30,
        )

        # Check that we got the expected embeddings
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embeddings[1] == [0.5, 0.4, 0.3, 0.2, 0.1]
        assert embeddings[2] == [0.3, 0.3, 0.3, 0.3, 0.3]


@pytest.mark.unit
@pytest.mark.async_test
async def test_batch_embedder_empty_inputs():
    """Test batch processing with empty inputs."""
    embedder = BatchEmbedder()

    # Test with empty list
    embeddings = await embedder.embed_batch([])
    assert embeddings == []

    # Test with list of empty strings
    embeddings = await embedder.embed_batch(["", "   ", "\n"])
    assert embeddings == []


@pytest.mark.unit
@pytest.mark.async_test
async def test_batch_embedder_filtering_empty_texts(mock_batch_embedding_response):
    """Test filtering empty texts before batch processing."""
    with patch("src.lib.embedder.litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
        mock_aembedding.return_value = mock_batch_embedding_response

        embedder = BatchEmbedder()
        texts = ["Text 1", "", "Text 2", "   ", "Text 3"]

        # Only the non-empty texts should be processed
        expected_filtered_texts = ["Text 1", "Text 2", "Text 3"]

        embeddings = await embedder.embed_batch(texts)

        # Check that aembedding was called with the filtered texts
        mock_aembedding.assert_called_once_with(
            model=embedder.model,
            input=expected_filtered_texts,
            timeout=30,
        )

        # Check that we got the expected embeddings
        assert len(embeddings) == 3
