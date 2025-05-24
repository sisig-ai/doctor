"""Tests for the map service."""

import pytest
from unittest.mock import AsyncMock, patch
import datetime

from src.web_service.services.map_service import MapService


@pytest.mark.asyncio
class TestMapService:
    """Test the MapService class."""

    async def test_get_all_sites(self) -> None:
        """Test getting all sites.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        # Mock the database response - sites from different domains
        mock_sites = [
            {
                "id": "site1",
                "url": "https://example1.com",
                "title": "Site 1",
                "domain": "example1.com",
            },
            {
                "id": "site2",
                "url": "https://example2.com",
                "title": "Site 2",
                "domain": "example2.com",
            },
        ]

        with patch.object(service.db_ops, "get_root_pages", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_sites

            sites = await service.get_all_sites()

        # Since they're from different domains, they should not be grouped
        assert len(sites) == 2
        assert sites[0]["id"] == "site1"
        assert sites[1]["id"] == "site2"
        mock_get.assert_called_once()

    async def test_build_page_tree(self) -> None:
        """Test building a page tree structure.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()
        root_id = "root-123"

        # Mock hierarchy data
        mock_pages = [
            {
                "id": "root-123",
                "parent_page_id": None,
                "title": "Home",
                "url": "https://example.com",
            },
            {
                "id": "page-1",
                "parent_page_id": "root-123",
                "title": "About",
                "url": "https://example.com/about",
            },
            {
                "id": "page-2",
                "parent_page_id": "root-123",
                "title": "Docs",
                "url": "https://example.com/docs",
            },
            {
                "id": "page-3",
                "parent_page_id": "page-2",
                "title": "API",
                "url": "https://example.com/docs/api",
            },
        ]

        with patch.object(service.db_ops, "get_page_hierarchy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_pages

            tree = await service.build_page_tree(root_id)

        # Verify tree structure
        assert tree["id"] == "root-123"
        assert tree["title"] == "Home"
        assert len(tree["children"]) == 2

        # Check first level children
        about_page = next(c for c in tree["children"] if c["id"] == "page-1")
        docs_page = next(c for c in tree["children"] if c["id"] == "page-2")

        assert about_page["title"] == "About"
        assert len(about_page["children"]) == 0

        assert docs_page["title"] == "Docs"
        assert len(docs_page["children"]) == 1
        assert docs_page["children"][0]["title"] == "API"

    async def test_get_navigation_context(self) -> None:
        """Test getting navigation context for a page.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()
        page_id = "page-123"

        # Mock current page
        mock_page = {
            "id": page_id,
            "parent_page_id": "parent-123",
            "root_page_id": "root-123",
            "title": "Current Page",
        }

        # Mock related pages
        mock_parent = {"id": "parent-123", "title": "Parent Page"}
        mock_siblings = [
            {"id": "sibling-1", "title": "Sibling 1"},
            {"id": "sibling-2", "title": "Sibling 2"},
        ]
        mock_children = [
            {"id": "child-1", "title": "Child 1"},
        ]
        mock_root = {"id": "root-123", "title": "Home"}

        with patch.object(
            service.db_ops, "get_page_by_id", new_callable=AsyncMock
        ) as mock_get_page:
            mock_get_page.side_effect = [mock_page, mock_parent, mock_root]

            with patch.object(
                service.db_ops, "get_sibling_pages", new_callable=AsyncMock
            ) as mock_siblings_fn:
                mock_siblings_fn.return_value = mock_siblings

                with patch.object(
                    service.db_ops, "get_child_pages", new_callable=AsyncMock
                ) as mock_children_fn:
                    mock_children_fn.return_value = mock_children

                    context = await service.get_navigation_context(page_id)

        assert context["current_page"] == mock_page
        assert context["parent"] == mock_parent
        assert context["siblings"] == mock_siblings
        assert context["children"] == mock_children
        assert context["root"] == mock_root

    def test_render_page_html(self) -> None:
        """Test rendering a page as HTML.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        page = {
            "id": "page-123",
            "title": "Test Page",
            "raw_text": "# Test Page\n\nThis is **markdown** content.",
        }

        navigation = {
            "current_page": page,
            "parent": {"id": "parent-123", "title": "Parent"},
            "siblings": [
                {"id": "sib-1", "title": "Sibling 1"},
            ],
            "children": [
                {"id": "child-1", "title": "Child 1"},
            ],
            "root": {"id": "root-123", "title": "Home"},
        }

        html = service.render_page_html(page, navigation)

        # Check that HTML contains expected elements
        assert "<title>Test Page</title>" in html
        assert "<h1>Test Page</h1>" in html
        assert "<strong>markdown</strong>" in html  # Markdown was rendered
        assert 'href="/map/page/parent-123"' in html  # Parent link
        assert 'href="/map/page/sib-1"' in html  # Sibling link
        assert 'href="/map/page/child-1"' in html  # Child link
        assert 'href="/map/site/root-123"' in html  # Site map link

    def test_format_site_list(self) -> None:
        """Test formatting a list of sites.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        sites = [
            {
                "id": "site1",
                "url": "https://example1.com",
                "title": "Example Site 1",
                "crawl_date": datetime.datetime(2024, 1, 1, 12, 0, 0),
            },
            {
                "id": "site2",
                "url": "https://example2.com",
                "title": "Example Site 2",
                "crawl_date": datetime.datetime(2024, 1, 2, 12, 0, 0),
            },
        ]

        html = service.format_site_list(sites)

        # Check HTML content
        assert "Site Map - All Sites" in html
        assert "Example Site 1" in html
        assert "Example Site 2" in html
        assert 'href="/map/site/site1"' in html
        assert 'href="/map/site/site2"' in html
        assert "https://example1.com" in html
        assert "https://example2.com" in html

    def test_format_site_list_empty(self) -> None:
        """Test formatting empty site list.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        html = service.format_site_list([])

        assert "No Sites Found" in html
        assert "No crawled sites are available" in html

    def test_format_site_tree(self) -> None:
        """Test formatting a site tree.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        tree = {
            "id": "root-123",
            "title": "My Site",
            "children": [
                {"id": "page-1", "title": "About", "children": []},
                {
                    "id": "page-2",
                    "title": "Docs",
                    "children": [{"id": "page-3", "title": "API", "children": []}],
                },
            ],
        }

        html = service.format_site_tree(tree)

        # Check HTML structure
        assert "My Site - Site Map" in html
        assert 'href="/map/page/root-123"' in html
        assert 'href="/map/page/page-1"' in html
        assert 'href="/map/page/page-2"' in html
        assert 'href="/map/page/page-3"' in html
        assert "<details>" in html  # Collapsible sections
        assert "Back to all sites" in html

    def test_build_tree_html_leaf_node(self) -> None:
        """Test building HTML for a leaf node.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        node = {"id": "leaf-123", "title": "Leaf Page", "children": []}

        html = service._build_tree_html(node, is_root=False)

        assert '<a href="/map/page/leaf-123">Leaf Page</a>' in html
        assert '<span class="tree-line">' in html
        assert '<span class="tree-node">' in html
        assert "<details>" not in html  # No collapsible section for leaf

    def test_build_tree_html_with_children(self) -> None:
        """Test building HTML for a node with children.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        node = {
            "id": "parent-123",
            "title": "Parent Page",
            "children": [
                {"id": "child-1", "title": "Child 1", "children": []},
                {"id": "child-2", "title": "Child 2", "children": []},
            ],
        }

        html = service._build_tree_html(node, is_root=False)

        assert "<details>" in html
        assert "<summary>" in html
        assert "Parent Page" in html
        assert "Child 1" in html
        assert "Child 2" in html
