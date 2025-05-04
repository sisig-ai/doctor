"""Text chunking functionality using LangChain."""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.common.config import CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logger = logging.getLogger(__name__)


class TextChunker:
    """Class for splitting text into semantic chunks."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Size of each text chunk (defaults to config value)
            chunk_overlap: Overlap between chunks (defaults to config value)
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        logger.debug(f"Initialized TextChunker with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Received empty text for chunking, returning empty list")
            return []
            
        chunks = self.text_splitter.split_text(text)
        logger.debug(f"Split text of length {len(text)} into {len(chunks)} chunks")
        
        # Filter out empty chunks
        non_empty_chunks = [chunk for chunk in chunks if chunk.strip()]
        if len(non_empty_chunks) < len(chunks):
            logger.debug(f"Filtered out {len(chunks) - len(non_empty_chunks)} empty chunks")
            
        return non_empty_chunks 