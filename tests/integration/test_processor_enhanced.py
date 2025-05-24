"""Integration tests for the enhanced processor with hierarchy tracking."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.common.processor_enhanced import (
    process_crawl_result_with_hierarchy,
    process_crawl_with_hierarchy,
)
from src.lib.crawler_enhanced import CrawlResultWithHierarchy


@pytest.mark.asyncio
@pytest.mark.integration
class TestProcessorEnhancedIntegration:
    """Integration tests for enhanced processor."""

    async def test_process_crawl_result_with_hierarchy(self) -> None:
        """Test processing a single crawl result with hierarchy.

        Args:
            None.

        Returns:
            None.
        """
        # Create a mock enhanced crawl result
        base_result = Mock()
        base_result.url = "https://example.com/docs"
        base_result.html = "<html><title>Documentation</title></html>"

        enhanced_result = CrawlResultWithHierarchy(base_result)
        enhanced_result.parent_url = "https://example.com"
        enhanced_result.root_url = "https://example.com"
        enhanced_result.depth = 1
        enhanced_result.relative_path = "/docs"
        enhanced_result.title = "Documentation"

        # Mock dependencies
        with patch("src.common.processor_enhanced.extract_page_text") as mock_extract:
            mock_extract.return_value = "# Documentation\n\nThis is the documentation."

            with patch("src.common.processor_enhanced.DatabaseOperations") as MockDB:
                mock_db = MockDB.return_value
                mock_db.store_page = AsyncMock(return_value="page-123")

                with patch("src.common.processor_enhanced.TextChunker") as MockChunker:
                    mock_chunker = MockChunker.return_value
                    mock_chunker.split_text.return_value = ["Chunk 1 content", "Chunk 2 content"]

                    with patch("src.common.processor_enhanced.generate_embedding") as mock_embed:
                        mock_embed.return_value = [0.1, 0.2, 0.3]

                        with patch("src.common.processor_enhanced.VectorIndexer") as MockIndexer:
                            mock_indexer = MockIndexer.return_value
                            mock_indexer.index_vector = AsyncMock()

                            # Process with hierarchy tracking
                            url_to_page_id = {"https://example.com": "root-123"}
                            page_id = await process_crawl_result_with_hierarchy(
                                enhanced_result,
                                job_id="test-job",
                                tags=["test"],
                                url_to_page_id=url_to_page_id,
                            )

        # Verify results
        assert page_id == "page-123"
        assert url_to_page_id[enhanced_result.url] == "page-123"

        # Verify store_page was called with hierarchy info
        mock_db.store_page.assert_called_once_with(
            url="https://example.com/docs",
            text="# Documentation\n\nThis is the documentation.",
            job_id="test-job",
            tags=["test"],
            parent_page_id="root-123",  # Parent ID was looked up
            root_page_id="root-123",
            depth=1,
            path="/docs",
            title="Documentation",
        )

        # Verify chunks were indexed
        assert mock_indexer.index_vector.call_count == 2

    async def test_process_crawl_with_hierarchy_full_flow(self) -> None:
        """Test the full crawl and process flow with hierarchy.

        Args:
            None.

        Returns:
            None.
        """
        # Mock crawl results with hierarchy
        mock_crawl_results = []
        for i, (url, parent_url, depth) in enumerate(
            [
                ("https://example.com", None, 0),
                ("https://example.com/about", "https://example.com", 1),
                ("https://example.com/docs", "https://example.com", 1),
                ("https://example.com/docs/api", "https://example.com/docs", 2),
            ]
        ):
            base = Mock()
            base.url = url
            base.html = f"<title>Page {i}</title>"

            enhanced = CrawlResultWithHierarchy(base)
            enhanced.url = url
            enhanced.parent_url = parent_url
            enhanced.root_url = "https://example.com"
            enhanced.depth = depth
            enhanced.title = f"Page {i}"
            enhanced.relative_path = url.replace("https://example.com", "") or "/"

            mock_crawl_results.append(enhanced)

        # Mock the enhanced crawler
        with patch("src.common.processor_enhanced.crawl_url_with_hierarchy") as mock_crawl:
            mock_crawl.return_value = mock_crawl_results

            # Mock other dependencies
            with patch("src.common.processor_enhanced.extract_page_text") as mock_extract:
                mock_extract.return_value = "Page content"

                with patch("src.common.processor_enhanced.DatabaseOperations") as MockDB:
                    mock_db = MockDB.return_value
                    # Return unique IDs for each page
                    mock_db.store_page = AsyncMock(
                        side_effect=["page-0", "page-1", "page-2", "page-3"]
                    )
                    mock_db.update_job_status = AsyncMock()

                    with patch("src.common.processor_enhanced.TextChunker") as MockChunker:
                        mock_chunker = MockChunker.return_value
                        mock_chunker.split_text.return_value = ["chunk"]

                        with patch(
                            "src.common.processor_enhanced.generate_embedding"
                        ) as mock_embed:
                            mock_embed.return_value = [0.1, 0.2]

                            with patch(
                                "src.common.processor_enhanced.VectorIndexer"
                            ) as MockIndexer:
                                mock_indexer = MockIndexer.return_value
                                mock_indexer.index_vector = AsyncMock()

                                # Process everything
                                page_ids = await process_crawl_with_hierarchy(
                                    url="https://example.com",
                                    job_id="test-job",
                                    tags=["test"],
                                    max_pages=10,
                                )

        # Verify results
        assert len(page_ids) == 4
        assert page_ids == ["page-0", "page-1", "page-2", "page-3"]

        # Verify pages were stored in order (parent before child)
        assert mock_db.store_page.call_count == 4

        # Check that parent IDs were properly resolved
        # First page (root) has no parent
        first_call = mock_db.store_page.call_args_list[0]
        assert first_call.kwargs["parent_page_id"] is None

        # Second page's parent should be the root's ID
        second_call = mock_db.store_page.call_args_list[1]
        assert second_call.kwargs["parent_page_id"] == "page-0"

        # Fourth page's parent should be the third page's ID
        fourth_call = mock_db.store_page.call_args_list[3]
        assert fourth_call.kwargs["parent_page_id"] == "page-2"

    async def test_process_crawl_with_hierarchy_error_handling(self) -> None:
        """Test error handling in hierarchy processing.

        Args:
            None.

        Returns:
            None.
        """
        # Create a result that will fail
        base_result = Mock()
        base_result.url = "https://example.com/bad"

        enhanced_result = CrawlResultWithHierarchy(base_result)
        enhanced_result.url = "https://example.com/bad"
        enhanced_result.parent_url = None
        enhanced_result.depth = 0
        enhanced_result.title = "Bad Page"

        with patch("src.common.processor_enhanced.crawl_url_with_hierarchy") as mock_crawl:
            mock_crawl.return_value = [enhanced_result]

            with patch("src.common.processor_enhanced.extract_page_text") as mock_extract:
                # Make extraction fail
                mock_extract.side_effect = Exception("Extraction failed")

                with patch("src.common.processor_enhanced.DatabaseOperations") as MockDB:
                    mock_db = MockDB.return_value
                    mock_db.update_job_status = AsyncMock()

                    # Process should continue despite error
                    page_ids = await process_crawl_with_hierarchy(
                        url="https://example.com", job_id="test-job"
                    )

                    # Should return empty list due to error
                    assert page_ids == []

                    # Job status won't be updated if all pages fail
                    mock_db.update_job_status.assert_not_called()
