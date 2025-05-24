"""Tests for the map API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.web_service.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app.

    Args:
        None.

    Returns:
        TestClient: Test client instance.
    """
    return TestClient(app)


class TestMapAPI:
    """Test the map API endpoints."""

    def test_get_site_index(self, client: TestClient) -> None:
        """Test the /map endpoint for site index.

        Args:
            client: The test client.

        Returns:
            None.
        """
        # Mock the map service
        with patch("src.web_service.api.map.MapService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_all_sites = AsyncMock(
                return_value=[
                    {"id": "site1", "title": "Site 1", "url": "https://example1.com"},
                    {"id": "site2", "title": "Site 2", "url": "https://example2.com"},
                ]
            )
            mock_service.format_site_list.return_value = "<html>Site List</html>"

            response = client.get("/map")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert response.text == "<html>Site List</html>"

    def test_get_site_index_error(self, client: TestClient) -> None:
        """Test /map endpoint error handling.

        Args:
            client: The test client.

        Returns:
            None.
        """
        with patch("src.web_service.api.map.MapService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_all_sites = AsyncMock(side_effect=Exception("Database error"))

            response = client.get("/map")

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_get_site_tree(self, client: TestClient) -> None:
        """Test the /map/site/{root_page_id} endpoint.

        Args:
            client: The test client.

        Returns:
            None.
        """
        root_id = "root-123"

        with patch("src.web_service.api.map.MapService") as MockService:
            mock_service = MockService.return_value
            mock_service.build_page_tree = AsyncMock(
                return_value={"id": root_id, "title": "Test Site", "children": []}
            )
            mock_service.format_site_tree.return_value = "<html>Site Tree</html>"

            response = client.get(f"/map/site/{root_id}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert response.text == "<html>Site Tree</html>"

    def test_view_page(self, client: TestClient) -> None:
        """Test the /map/page/{page_id} endpoint.

        Args:
            client: The test client.

        Returns:
            None.
        """
        page_id = "page-123"

        with patch("src.web_service.api.map.MapService") as MockService:
            mock_service = MockService.return_value
            mock_service.db_ops.get_page_by_id = AsyncMock(
                return_value={
                    "id": page_id,
                    "title": "Test Page",
                    "raw_text": "# Test Content",
                }
            )
            mock_service.get_navigation_context = AsyncMock(
                return_value={
                    "current_page": {"id": page_id, "title": "Test Page"},
                    "parent": None,
                    "siblings": [],
                    "children": [],
                    "root": None,
                }
            )
            mock_service.render_page_html.return_value = "<html>Rendered Page</html>"

            response = client.get(f"/map/page/{page_id}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert response.text == "<html>Rendered Page</html>"

    def test_view_page_not_found(self, client: TestClient) -> None:
        """Test viewing a non-existent page.

        Args:
            client: The test client.

        Returns:
            None.
        """
        page_id = "nonexistent"

        with patch("src.web_service.api.map.MapService") as MockService:
            mock_service = MockService.return_value
            mock_service.db_ops.get_page_by_id = AsyncMock(return_value=None)

            response = client.get(f"/map/page/{page_id}")

        assert response.status_code == 404
        assert response.json()["detail"] == "Page not found"

    def test_get_page_raw(self, client: TestClient) -> None:
        """Test the /map/page/{page_id}/raw endpoint.

        Args:
            client: The test client.

        Returns:
            None.
        """
        page_id = "page-123"
        raw_content = "# Test Page\n\nThis is markdown content."

        with patch("src.web_service.api.map.MapService") as MockService:
            mock_service = MockService.return_value
            mock_service.db_ops.get_page_by_id = AsyncMock(
                return_value={
                    "id": page_id,
                    "title": "Test Page",
                    "raw_text": raw_content,
                }
            )

            response = client.get(f"/map/page/{page_id}/raw")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/markdown"
        assert response.text == raw_content
        assert 'filename="Test Page.md"' in response.headers["content-disposition"]

    def test_get_page_raw_not_found(self, client: TestClient) -> None:
        """Test getting raw content for non-existent page.

        Args:
            client: The test client.

        Returns:
            None.
        """
        page_id = "nonexistent"

        with patch("src.web_service.api.map.MapService") as MockService:
            mock_service = MockService.return_value
            mock_service.db_ops.get_page_by_id = AsyncMock(return_value=None)

            response = client.get(f"/map/page/{page_id}/raw")

        assert response.status_code == 404
        assert response.json()["detail"] == "Page not found"
