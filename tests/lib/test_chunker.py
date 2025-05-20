"""Tests for the chunker module."""

from unittest.mock import MagicMock, patch

import pytest

from src.lib.chunker import TextChunker


@pytest.fixture
def mock_text_splitter():
    """Mock for the RecursiveCharacterTextSplitter."""
    mock = MagicMock()
    mock.split_text.return_value = [
        "This is chunk 1.",
        "This is chunk 2.",
        "This is chunk 3.",
        "   ",  # Empty chunk (with whitespace) that should be filtered
    ]
    return mock


@pytest.mark.unit
def test_text_chunker_initialization():
    """Test TextChunker initialization with default values."""
    with (
        patch("src.lib.chunker.CHUNK_SIZE", 1000),
        patch("src.lib.chunker.CHUNK_OVERLAP", 100),
        patch("src.lib.chunker.RecursiveCharacterTextSplitter") as mock_splitter_class,
    ):
        chunker = TextChunker()

        # Check that the chunker was initialized with the correct values
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

        # Check that the text splitter was initialized with the correct values
        mock_splitter_class.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )


@pytest.mark.unit
def test_text_chunker_initialization_with_custom_values():
    """Test TextChunker initialization with custom values."""
    with patch("src.lib.chunker.RecursiveCharacterTextSplitter") as mock_splitter_class:
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        # Check that the chunker was initialized with the correct values
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

        # Check that the text splitter was initialized with the correct values
        mock_splitter_class.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )


@pytest.mark.unit
def test_split_text(sample_text, mock_text_splitter):
    """Test splitting text into chunks."""
    with patch("src.lib.chunker.RecursiveCharacterTextSplitter", return_value=mock_text_splitter):
        chunker = TextChunker()
        chunks = chunker.split_text(sample_text)

        # Check that the text splitter was called with the correct input
        mock_text_splitter.split_text.assert_called_once_with(sample_text)

        # Check that we got the expected chunks (empty chunks filtered out)
        assert chunks == ["This is chunk 1.", "This is chunk 2.", "This is chunk 3."]
        assert len(chunks) == 3


@pytest.mark.unit
def test_split_text_empty_input():
    """Test splitting empty text."""
    chunker = TextChunker()

    # Test with empty string
    chunks = chunker.split_text("")
    assert chunks == []

    # Test with whitespace only
    chunks = chunker.split_text("   \n   ")
    assert chunks == []


@pytest.mark.unit
def test_split_text_all_empty_chunks():
    """Test when all chunks are empty after filtering."""
    with patch("src.lib.chunker.RecursiveCharacterTextSplitter") as mock_splitter_class:
        # Mock splitter to return only empty chunks
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["   ", "\n", "  \t  "]
        mock_splitter_class.return_value = mock_splitter

        chunker = TextChunker()
        chunks = chunker.split_text("Some text")

        # Check that we got an empty list
        assert chunks == []
