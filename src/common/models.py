"""Pydantic models for the Doctor project."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class FetchUrlRequest(BaseModel):
    """Request model for the /fetch_url endpoint."""

    url: HttpUrl = Field(..., description="The URL to start indexing from")
    tags: Optional[List[str]] = Field(default=None, description="Tags to assign this website")
    max_pages: int = Field(default=100, description="How many pages to index", ge=1, le=1000)


class FetchUrlResponse(BaseModel):
    """Response model for the /fetch_url endpoint."""

    job_id: str = Field(..., description="The job ID of the index job")


class SearchDocsRequest(BaseModel):
    """Request model for the /search_docs endpoint."""

    query: str = Field(..., description="The search string to query the database with")
    tags: Optional[List[str]] = Field(default=None, description="Tags to limit the search with")
    max_results: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )


class SearchResult(BaseModel):
    """A single search result."""

    chunk_text: str = Field(..., description="The text of the chunk")
    page_id: str = Field(..., description="Reference to the original page")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the chunk")
    score: float = Field(..., description="Similarity score")


class SearchDocsResponse(BaseModel):
    """Response model for the /search_docs endpoint."""

    results: List[SearchResult] = Field(default_factory=list, description="Search results")


class JobProgressRequest(BaseModel):
    """Request model for the /job_progress endpoint."""

    job_id: str = Field(..., description="The job ID to check progress for")


class JobProgressResponse(BaseModel):
    """Response model for the /job_progress endpoint."""

    pages_crawled: int = Field(..., description="Number of pages crawled so far")
    pages_total: int = Field(..., description="Total number of pages discovered")
    completed: bool = Field(..., description="Whether the job is completed")
    status: str = Field(..., description="Current job status")
    error_message: Optional[str] = Field(default=None, description="Error message if job failed")
    progress_percent: Optional[int] = Field(
        default=None, description="Percentage of crawl completed"
    )
    url: Optional[str] = Field(default=None, description="URL being crawled")
    max_pages: Optional[int] = Field(default=None, description="Maximum pages to crawl")
    created_at: Optional[datetime] = Field(default=None, description="When the job was created")
    updated_at: Optional[datetime] = Field(
        default=None, description="When the job was last updated"
    )


class ListDocPagesRequest(BaseModel):
    """Request model for the /list_doc_pages endpoint."""

    page: int = Field(default=1, description="Page number", ge=1)
    tags: Optional[List[str]] = Field(default=None, description="Tags to filter by")


class DocPageSummary(BaseModel):
    """Summary information about a document page."""

    page_id: str = Field(..., description="Unique page ID")
    domain: str = Field(..., description="Domain of the page")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the page")
    crawl_date: datetime = Field(..., description="When the page was crawled")
    url: str = Field(..., description="URL of the page")


class ListDocPagesResponse(BaseModel):
    """Response model for the /list_doc_pages endpoint."""

    doc_pages: List[DocPageSummary] = Field(
        default_factory=list, description="List of document pages"
    )
    total_pages: int = Field(..., description="Total number of pages matching the query")
    current_page: int = Field(..., description="Current page number")
    pages_per_page: int = Field(default=100, description="Number of items per page")


class GetDocPageRequest(BaseModel):
    """Request model for the /get_doc_page endpoint."""

    page_id: str = Field(..., description="The page ID to retrieve")
    starting_line: int = Field(default=1, description="Line to view from", ge=1)
    ending_line: int = Field(default=100, description="Line to view up to", ge=1)


class GetDocPageResponse(BaseModel):
    """Response model for the /get_doc_page endpoint."""

    text: str = Field(..., description="The document page text")
    total_lines: int = Field(..., description="Total number of lines in the document")


class Job(BaseModel):
    """Internal model for a crawl job."""

    job_id: str
    start_url: str
    status: str = "pending"  # pending, running, completed, failed
    pages_discovered: int = 0
    pages_crawled: int = 0
    max_pages: int
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None


class Page(BaseModel):
    """Internal model for a crawled page."""

    id: str
    url: str
    domain: str
    raw_text: str
    crawl_date: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """Internal model for a text chunk with its embedding."""

    id: str
    text: str
    page_id: str
    domain: str
    tags: List[str] = Field(default_factory=list)
    embedding: List[float]
