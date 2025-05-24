"""Tests for the map service legacy page handling."""

import pytest
from unittest.mock import AsyncMock, patch
import datetime

from src.web_service.services.map_service import MapService


@pytest.mark.asyncio
class TestMapServiceLegacy:
    """Test the MapService legacy page handling."""

    async def test_get_all_sites_with_legacy(self) -> None:
        """Test getting all sites including legacy domain groups.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        # Mock root pages (pages with hierarchy)
        mock_root_pages = [
            {
                "id": "site1",
                "url": "https://docs.example.com",
                "title": "Documentation Site",
                "domain": "docs.example.com",
                "crawl_date": datetime.datetime(2024, 1, 2),
            }
        ]

        # Mock legacy pages
        mock_legacy_pages = [
            {
                "id": "page1",
                "url": "https://blog.example.com/post1",
                "domain": "blog.example.com",
                "title": "Blog Post 1",
                "crawl_date": datetime.datetime(2024, 1, 1),
                "root_page_id": None,
                "parent_page_id": None,
            },
            {
                "id": "page2",
                "url": "https://blog.example.com/post2",
                "domain": "blog.example.com",
                "title": "Blog Post 2",
                "crawl_date": datetime.datetime(2024, 1, 3),
                "root_page_id": None,
                "parent_page_id": None,
            },
            {
                "id": "page3",
                "url": "https://shop.example.com/product1",
                "domain": "shop.example.com",
                "title": "Product 1",
                "crawl_date": datetime.datetime(2024, 1, 1),
                "root_page_id": None,
                "parent_page_id": None,
            },
        ]

        with patch.object(
            service.db_ops, "get_root_pages", new_callable=AsyncMock
        ) as mock_get_root:
            mock_get_root.return_value = mock_root_pages

            with patch.object(
                service.db_ops, "get_legacy_pages", new_callable=AsyncMock
            ) as mock_get_legacy:
                mock_get_legacy.return_value = mock_legacy_pages

                sites = await service.get_all_sites()

        # Should have 1 root page + 2 domain groups
        assert len(sites) == 3

        # Check that we have the regular site
        regular_sites = [s for s in sites if not s.get("is_synthetic")]
        assert len(regular_sites) == 1
        assert regular_sites[0]["title"] == "Documentation Site"

        # Check that we have domain groups
        domain_groups = [s for s in sites if s.get("is_synthetic")]
        assert len(domain_groups) == 2

        # Check blog domain group
        blog_group = next(s for s in domain_groups if "blog.example.com" in s["title"])
        assert blog_group["id"] == "legacy-domain-blog.example.com"
        assert blog_group["page_count"] == 2
        assert blog_group["is_synthetic"] is True

        # Check shop domain group
        shop_group = next(s for s in domain_groups if "shop.example.com" in s["title"])
        assert shop_group["id"] == "legacy-domain-shop.example.com"
        assert shop_group["page_count"] == 1

    async def test_build_domain_tree(self) -> None:
        """Test building a tree for a domain group.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()
        domain = "blog.example.com"

        # Mock pages for the domain
        mock_domain_pages = [
            {
                "id": "page1",
                "url": "https://blog.example.com/post1",
                "title": "Post 1",
                "root_page_id": None,
            },
            {
                "id": "page2",
                "url": "https://blog.example.com/post2",
                "title": "Post 2",
                "root_page_id": None,
            },
            {
                "id": "page3",
                "url": "https://blog.example.com/post3",
                "title": "Post 3",
                "root_page_id": "page3",  # This is a proper root, not legacy
                "parent_page_id": None,
                "depth": 0,
            },
        ]

        with patch.object(
            service.db_ops, "get_pages_by_domain", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_domain_pages

            tree = await service.build_page_tree(f"legacy-domain-{domain}")

        # Check the synthetic root
        assert tree["id"] == f"legacy-domain-{domain}"
        assert tree["is_synthetic"] is True
        assert tree["title"] == f"{domain} (3 Pages)"

        # Should only include legacy pages (not page3)
        assert len(tree["children"]) == 2
        assert all(child["id"] in ["page1", "page2"] for child in tree["children"])

        # Children should be sorted by URL
        assert tree["children"][0]["url"] < tree["children"][1]["url"]

    def test_format_site_list_with_legacy(self) -> None:
        """Test formatting site list with legacy domain groups.

        Args:
            None.

        Returns:
            None.
        """
        service = MapService()

        sites = [
            {
                "id": "site1",
                "url": "https://docs.example.com",
                "title": "Documentation",
                "crawl_date": datetime.datetime(2024, 1, 1),
                "is_synthetic": False,
            },
            {
                "id": "legacy-domain-blog.example.com",
                "url": "https://blog.example.com",
                "title": "blog.example.com (Legacy Pages)",
                "crawl_date": datetime.datetime(2024, 1, 1),
                "is_synthetic": True,
                "page_count": 5,
            },
        ]

        html = service.format_site_list(sites)

        # Check regular site formatting
        assert "Documentation" in html
        assert "Crawled: 2024-01-01" in html

        # Check legacy domain group formatting
        assert "blog.example.com (Legacy Pages)" in html
        assert "Domain Group • 5 pages" in html
        assert "First crawled: 2024-01-01" in html
