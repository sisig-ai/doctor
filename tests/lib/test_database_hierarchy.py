"""Tests for database operations with hierarchy support."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import datetime

from src.lib.database.operations import DatabaseOperations


@pytest.mark.asyncio
class TestDatabaseHierarchyOperations:
    """Test database operations for hierarchy features."""

    async def test_store_page_with_hierarchy(self) -> None:
        """Test storing a page with hierarchy information.

        Args:
            None.

        Returns:
            None.
        """
        db_ops = DatabaseOperations()

        # Mock the database connection and operations
        mock_conn = Mock()
        mock_conn_manager = Mock()
        mock_conn_manager.conn = mock_conn
        mock_conn_manager.begin_transaction = Mock()
        mock_conn_manager.commit = Mock()

        with patch.object(db_ops, "db") as mock_db:
            mock_db.__enter__.return_value = mock_conn_manager
            mock_db.__exit__.return_value = None

            page_id = await db_ops.store_page(
                url="https://example.com/docs",
                text="Documentation content",
                job_id="test-job",
                tags=["docs"],
                parent_page_id="parent-123",
                root_page_id="root-456",
                depth=1,
                path="/docs",
                title="Documentation",
            )

        # Verify the page was stored with hierarchy info
        assert page_id is not None
        mock_conn.execute.assert_called_once()

        # Check the parameters passed to execute
        call_args = mock_conn.execute.call_args[0]
        params = call_args[1]

        # Verify hierarchy parameters are included
        assert params[7] == "parent-123"  # parent_page_id
        assert params[8] == "root-456"  # root_page_id
        assert params[9] == 1  # depth
        assert params[10] == "/docs"  # path
        assert params[11] == "Documentation"  # title

    async def test_store_root_page_sets_root_id(self) -> None:
        """Test that storing a root page sets root_page_id to self.

        Args:
            None.

        Returns:
            None.
        """
        db_ops = DatabaseOperations()

        mock_conn = Mock()
        mock_conn_manager = Mock()
        mock_conn_manager.conn = mock_conn
        mock_conn_manager.begin_transaction = Mock()
        mock_conn_manager.commit = Mock()

        with patch.object(db_ops, "db") as mock_db:
            mock_db.__enter__.return_value = mock_conn_manager
            mock_db.__exit__.return_value = None

            await db_ops.store_page(
                url="https://example.com",
                text="Home content",
                job_id="test-job",
                parent_page_id=None,  # No parent - this is root
                root_page_id=None,
                title="Home",
            )

        # Check that root_page_id was set to the page's own ID
        call_args = mock_conn.execute.call_args[0]
        params = call_args[1]

        assert params[0] == params[8]  # page_id == root_page_id

    async def test_get_root_pages(self) -> None:
        """Test getting all root pages.

        Args:
            None.

        Returns:
            None.
        """
        db_ops = DatabaseOperations()

        # Mock database results
        mock_rows = [
            (
                "id1",
                "https://example1.com",
                "example1.com",
                "text1",
                datetime.datetime.now(),
                "[]",
                "job1",
                None,
                "id1",
                0,
                "/",
                "Site 1",
            ),
            (
                "id2",
                "https://example2.com",
                "example2.com",
                "text2",
                datetime.datetime.now(),
                "[]",
                "job2",
                None,
                "id2",
                0,
                "/",
                "Site 2",
            ),
        ]

        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_result.description = [
            ("id",),
            ("url",),
            ("domain",),
            ("raw_text",),
            ("crawl_date",),
            ("tags",),
            ("job_id",),
            ("parent_page_id",),
            ("root_page_id",),
            ("depth",),
            ("path",),
            ("title",),
        ]

        mock_conn = Mock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_conn_manager = Mock()
        mock_conn_manager.conn = mock_conn

        with patch.object(db_ops, "db") as mock_db:
            mock_db.__enter__.return_value = mock_conn_manager
            mock_db.__exit__.return_value = None

            root_pages = await db_ops.get_root_pages()

        assert len(root_pages) == 2
        assert root_pages[0]["title"] == "Site 1"
        assert root_pages[1]["title"] == "Site 2"
        assert all(page["parent_page_id"] is None for page in root_pages)

    async def test_get_page_hierarchy(self) -> None:
        """Test getting a complete page hierarchy.

        Args:
            None.

        Returns:
            None.
        """
        db_ops = DatabaseOperations()
        root_id = "root-123"

        # Mock pages in hierarchy
        mock_rows = [
            (
                "root-123",
                "https://example.com",
                "example.com",
                "root",
                datetime.datetime.now(),
                "[]",
                "job1",
                None,
                "root-123",
                0,
                "/",
                "Home",
            ),
            (
                "page-1",
                "https://example.com/about",
                "example.com",
                "about",
                datetime.datetime.now(),
                "[]",
                "job1",
                "root-123",
                "root-123",
                1,
                "/about",
                "About",
            ),
            (
                "page-2",
                "https://example.com/docs",
                "example.com",
                "docs",
                datetime.datetime.now(),
                "[]",
                "job1",
                "root-123",
                "root-123",
                1,
                "/docs",
                "Docs",
            ),
        ]

        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_result.description = [
            ("id",),
            ("url",),
            ("domain",),
            ("raw_text",),
            ("crawl_date",),
            ("tags",),
            ("job_id",),
            ("parent_page_id",),
            ("root_page_id",),
            ("depth",),
            ("path",),
            ("title",),
        ]

        mock_conn = Mock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_conn_manager = Mock()
        mock_conn_manager.conn = mock_conn

        with patch.object(db_ops, "db") as mock_db:
            mock_db.__enter__.return_value = mock_conn_manager
            mock_db.__exit__.return_value = None

            hierarchy = await db_ops.get_page_hierarchy(root_id)

        assert len(hierarchy) == 3
        assert hierarchy[0]["depth"] == 0
        assert hierarchy[1]["depth"] == 1
        assert hierarchy[2]["depth"] == 1

    async def test_get_child_pages(self) -> None:
        """Test getting child pages of a parent.

        Args:
            None.

        Returns:
            None.
        """
        db_ops = DatabaseOperations()
        parent_id = "parent-123"

        mock_rows = [
            (
                "child-1",
                "https://example.com/docs/api",
                "example.com",
                "api",
                datetime.datetime.now(),
                "[]",
                "job1",
                parent_id,
                "root-123",
                2,
                "/docs/api",
                "API",
            ),
            (
                "child-2",
                "https://example.com/docs/guide",
                "example.com",
                "guide",
                datetime.datetime.now(),
                "[]",
                "job1",
                parent_id,
                "root-123",
                2,
                "/docs/guide",
                "Guide",
            ),
        ]

        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_result.description = [
            ("id",),
            ("url",),
            ("domain",),
            ("raw_text",),
            ("crawl_date",),
            ("tags",),
            ("job_id",),
            ("parent_page_id",),
            ("root_page_id",),
            ("depth",),
            ("path",),
            ("title",),
        ]

        mock_conn = Mock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_conn_manager = Mock()
        mock_conn_manager.conn = mock_conn

        with patch.object(db_ops, "db") as mock_db:
            mock_db.__enter__.return_value = mock_conn_manager
            mock_db.__exit__.return_value = None

            children = await db_ops.get_child_pages(parent_id)

        assert len(children) == 2
        assert all(child["parent_page_id"] == parent_id for child in children)

    async def test_get_sibling_pages(self) -> None:
        """Test getting sibling pages.

        Args:
            None.

        Returns:
            None.
        """
        db_ops = DatabaseOperations()
        page_id = "page-1"
        parent_id = "parent-123"

        # First query returns the parent_page_id
        mock_parent_result = Mock()
        mock_parent_result.fetchone.return_value = (parent_id,)

        # Second query returns siblings
        mock_siblings_rows = [
            (
                "page-2",
                "https://example.com/page2",
                "example.com",
                "page2",
                datetime.datetime.now(),
                "[]",
                "job1",
                parent_id,
                "root-123",
                1,
                "/page2",
                "Page 2",
            ),
            (
                "page-3",
                "https://example.com/page3",
                "example.com",
                "page3",
                datetime.datetime.now(),
                "[]",
                "job1",
                parent_id,
                "root-123",
                1,
                "/page3",
                "Page 3",
            ),
        ]

        mock_siblings_result = Mock()
        mock_siblings_result.fetchall.return_value = mock_siblings_rows
        mock_siblings_result.description = [
            ("id",),
            ("url",),
            ("domain",),
            ("raw_text",),
            ("crawl_date",),
            ("tags",),
            ("job_id",),
            ("parent_page_id",),
            ("root_page_id",),
            ("depth",),
            ("path",),
            ("title",),
        ]

        mock_conn = Mock()
        mock_conn.execute = AsyncMock(side_effect=[mock_parent_result, mock_siblings_result])

        mock_conn_manager = Mock()
        mock_conn_manager.conn = mock_conn

        with patch.object(db_ops, "db") as mock_db:
            mock_db.__enter__.return_value = mock_conn_manager
            mock_db.__exit__.return_value = None

            siblings = await db_ops.get_sibling_pages(page_id)

        assert len(siblings) == 2
        assert page_id not in [s["id"] for s in siblings]

    async def test_get_sibling_pages_no_parent(self) -> None:
        """Test getting siblings when page has no parent.

        Args:
            None.

        Returns:
            None.
        """
        db_ops = DatabaseOperations()

        # Mock result with no parent
        mock_result = Mock()
        mock_result.fetchone.return_value = (None,)

        mock_conn = Mock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_conn_manager = Mock()
        mock_conn_manager.conn = mock_conn

        with patch.object(db_ops, "db") as mock_db:
            mock_db.__enter__.return_value = mock_conn_manager
            mock_db.__exit__.return_value = None

            siblings = await db_ops.get_sibling_pages("orphan-page")

        assert siblings == []
