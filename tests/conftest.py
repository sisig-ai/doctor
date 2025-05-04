"""Common fixtures for tests."""

import pytest
import asyncio


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_url():
    """Sample URL for testing."""
    return "https://example.com"


@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam auctor,
    nisl nec ultricies lacinia, nisl nisl aliquet nisl, nec ultricies nisl
    nisl nec ultricies lacinia, nisl nisl aliquet nisl, nec ultricies nisl
    nisl nec ultricies lacinia, nisl nisl aliquet nisl, nec ultricies nisl.

    Pellentesque habitant morbi tristique senectus et netus et malesuada
    fames ac turpis egestas. Sed euismod, nisl nec ultricies lacinia, nisl
    nisl aliquet nisl, nec ultricies nisl nisl nec ultricies lacinia.
    """


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    import random

    random.seed(42)  # For reproducibility
    return [random.random() for _ in range(384)]


@pytest.fixture
def sample_crawl_result():
    """Sample crawl result for testing."""

    class MockCrawlResult:
        def __init__(self, url, markdown=None, extracted_content=None, html=None):
            self.url = url
            self._markdown = markdown
            self.extracted_content = extracted_content
            self.html = html

            # Create a mock _markdown attribute if provided
            if markdown:

                class MockMarkdown:
                    def __init__(self, raw_markdown):
                        self.raw_markdown = raw_markdown

                self._markdown = MockMarkdown(markdown)

    return MockCrawlResult(
        url="https://example.com",
        markdown="# Example Page\n\nThis is some example content.",
        extracted_content="Example Page. This is some example content.",
        html="<html><head><title>Example</title></head><body><h1>Example Page</h1><p>This is some example content.</p></body></html>",
    )


@pytest.fixture
def job_id():
    """Sample job ID for testing."""
    return "test-job-123"


@pytest.fixture
def page_id():
    """Sample page ID for testing."""
    return "test-page-456"


@pytest.fixture
def sample_tags():
    """Sample tags for testing."""
    return ["test", "example", "documentation"]
