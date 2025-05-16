"""Tests for the document API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import duckdb

from src.web_service.main import app
from src.common.models import (
    SearchDocsResponse,
    SearchResult,
    ListDocPagesResponse,
    DocPageSummary,
    GetDocPageResponse,
    ListTagsResponse,
)


@pytest.fixture
def test_client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_duckdb_connection():
    """Mock DuckDB connection."""
    return MagicMock(spec=duckdb.DuckDBPyConnection)


@pytest.mark.unit
@pytest.mark.api
def test_search_docs_endpoint(test_client, mock_duckdb_connection):
    """Test the search_docs endpoint."""
    # Create mock search results that match the expected schema
    mock_results = SearchDocsResponse(
        results=[
            SearchResult(
                chunk_text="This is a document about artificial intelligence",
                page_id="page1",
                tags=["ai", "tech"],
                score=0.95,
                url="https://example.com/ai",
            ),
            SearchResult(
                chunk_text="This is a document about machine learning",
                page_id="page2",
                tags=["ml", "tech"],
                score=0.85,
                url="https://example.com/ml",
            ),
        ]
    )

    # Mock the search_docs service function and DuckDB connection
    with (
        patch("src.web_service.api.documents.search_docs", return_value=mock_results),
        patch(
            "src.web_service.api.documents.Database.connect_with_retry",
            return_value=mock_duckdb_connection,
        ),
    ):
        # Call the API endpoint
        response = test_client.get(
            "/search_docs",
            params={"query": "artificial intelligence", "max_results": 5},
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert (
            data["results"][0]["chunk_text"] == "This is a document about artificial intelligence"
        )
        assert data["results"][0]["page_id"] == "page1"
        assert data["results"][0]["score"] == 0.95
        assert "ai" in data["results"][0]["tags"]


@pytest.mark.unit
@pytest.mark.api
def test_search_docs_endpoint_error(test_client, mock_duckdb_connection):
    """Test error handling in the search_docs endpoint."""
    # Mock an exception in the search_docs service function
    with (
        patch("src.web_service.api.documents.search_docs", side_effect=Exception("Database error")),
        patch(
            "src.web_service.api.documents.Database.connect_with_retry",
            return_value=mock_duckdb_connection,
        ),
    ):
        # Call the API endpoint
        response = test_client.get(
            "/search_docs",
            params={"query": "artificial intelligence", "max_results": 5},
        )

        # Should return 500 for server errors
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI adds error details


@pytest.mark.unit
@pytest.mark.api
def test_list_doc_pages_endpoint(test_client, mock_duckdb_connection):
    """Test the list_doc_pages endpoint."""
    # Create mock list_doc_pages response
    mock_response = ListDocPagesResponse(
        doc_pages=[
            DocPageSummary(
                page_id="page1",
                domain="example.com",
                tags=["tag1", "tag2"],
                crawl_date="2023-01-01T00:00:00",
                url="https://example.com/page1",
            ),
            DocPageSummary(
                page_id="page2",
                domain="example.com",
                tags=["tag2", "tag3"],
                crawl_date="2023-01-02T00:00:00",
                url="https://example.com/page2",
            ),
        ],
        total_pages=1,
        current_page=1,
        pages_per_page=100,
    )

    # Mock the list_doc_pages service function and DuckDB connection
    with (
        patch("src.web_service.api.documents.list_doc_pages", return_value=mock_response),
        patch(
            "src.web_service.api.documents.Database.connect_with_retry",
            return_value=mock_duckdb_connection,
        ),
    ):
        # Call the API endpoint
        response = test_client.get("/list_doc_pages", params={"page": 1})

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "doc_pages" in data
        assert len(data["doc_pages"]) == 2
        assert data["doc_pages"][0]["page_id"] == "page1"
        assert data["doc_pages"][0]["url"] == "https://example.com/page1"
        assert "tag1" in data["doc_pages"][0]["tags"]


@pytest.mark.unit
@pytest.mark.api
def test_get_doc_page_endpoint(test_client, mock_duckdb_connection):
    """Test the get_doc_page endpoint."""
    # Create mock get_doc_page response
    mock_response = GetDocPageResponse(
        text="This is the document content.\nIt has multiple lines.\nLine 3 is here.",
        total_lines=3,
    )

    # Mock the get_doc_page service function and DuckDB connection
    with (
        patch("src.web_service.api.documents.get_doc_page", return_value=mock_response),
        patch(
            "src.web_service.api.documents.Database.connect_with_retry",
            return_value=mock_duckdb_connection,
        ),
    ):
        # Call the API endpoint
        response = test_client.get(
            "/get_doc_page",
            params={"page_id": "page1", "starting_line": 1, "ending_line": 2},
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "This is the document content." in data["text"]
        assert data["total_lines"] == 3


@pytest.mark.unit
@pytest.mark.api
def test_get_doc_page_endpoint_not_found(test_client, mock_duckdb_connection):
    """Test the get_doc_page endpoint when page is not found."""
    # Mock the get_doc_page service function to return None (page not found)
    with (
        patch("src.web_service.api.documents.get_doc_page", return_value=None),
        patch(
            "src.web_service.api.documents.Database.connect_with_retry",
            return_value=mock_duckdb_connection,
        ),
    ):
        # Call the API endpoint
        response = test_client.get(
            "/get_doc_page",
            params={"page_id": "nonexistent", "starting_line": 1, "ending_line": 2},
        )

        # Check response
        assert response.status_code == 404


@pytest.mark.unit
@pytest.mark.api
def test_list_tags_endpoint(test_client, mock_duckdb_connection):
    """Test the list_tags endpoint."""
    # Create mock list_tags response
    mock_response = ListTagsResponse(tags=["tag1", "tag2", "tag3"])

    # Mock the list_tags service function and DuckDB connection
    with (
        patch("src.web_service.api.documents.list_tags", return_value=mock_response),
        patch(
            "src.web_service.api.documents.Database.connect_with_retry",
            return_value=mock_duckdb_connection,
        ),
    ):
        # Call the API endpoint
        response = test_client.get("/list_tags")

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "tags" in data
        assert len(data["tags"]) == 3
        assert "tag1" in data["tags"]
        assert "tag2" in data["tags"]
        assert "tag3" in data["tags"]


@pytest.mark.unit
@pytest.mark.api
def test_document_api_integration(test_client):
    """Test the integration of all document API endpoints.

    This test uses patching at a high level to simulate a complete API workflow.
    We don't test actual functionality here, just that the endpoints are connected correctly.
    """
    # Create mock data for all endpoints
    search_results = SearchDocsResponse(
        results=[
            SearchResult(
                chunk_text="Sample text",
                page_id="page1",
                tags=["tag1", "tag2"],
                score=0.95,
                url="https://example.com/page1",
            )
        ]
    )

    doc_pages = ListDocPagesResponse(
        doc_pages=[
            DocPageSummary(
                page_id="page1",
                domain="example.com",
                tags=["tag1", "tag2"],
                crawl_date="2023-01-01T00:00:00",
                url="https://example.com/page1",
            )
        ],
        total_pages=1,
        current_page=1,
        pages_per_page=10,
    )

    doc_page = GetDocPageResponse(
        text="Page content",
        total_lines=1,
    )

    tags = ListTagsResponse(tags=["tag1", "tag2"])

    # Patch all service functions to return mock data
    with (
        patch("src.web_service.api.documents.search_docs", return_value=search_results),
        patch("src.web_service.api.documents.list_doc_pages", return_value=doc_pages),
        patch("src.web_service.api.documents.get_doc_page", return_value=doc_page),
        patch("src.web_service.api.documents.list_tags", return_value=tags),
    ):
        # Test search_docs endpoint
        response = test_client.get("/search_docs", params={"query": "test"})
        assert response.status_code == 200

        # Test list_doc_pages endpoint
        response = test_client.get("/list_doc_pages")
        assert response.status_code == 200

        # Test get_doc_page endpoint
        response = test_client.get("/get_doc_page", params={"page_id": "page1"})
        assert response.status_code == 200

        # Test list_tags endpoint
        response = test_client.get("/list_tags")
        assert response.status_code == 200
