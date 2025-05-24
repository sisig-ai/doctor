<div align="center">
  <picture>
    <img alt="Doctor Logo" src="doctor.png" height="30%" width="30%">
  </picture>
<br>

<h2>🩺 Doctor</h2>

[![Python Version](https://img.shields.io/badge/python-%3E=3.12-3776ab?style=flat&labelColor=333333&logo=python&logoColor=white)](https://github.com/sisig-ai/doctor)
[![License](https://img.shields.io/badge/license-MIT-00acc1?style=flat&labelColor=333333&logo=open-source-initiative&logoColor=white)](LICENSE.md)
[![Python Tests](https://github.com/sisig-ai/doctor/actions/workflows/pytest.yml/badge.svg)](https://github.com/sisig-ai/doctor/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/sisig-ai/doctor/branch/main/graph/badge.svg)](https://codecov.io/gh/sisig-ai/doctor)

A tool for discovering, crawl, and indexing web sites to be exposed as an MCP server for LLM agents for better and more up-to-date reasoning and code generation.

</div>

---

### 🔍 Overview

Doctor provides a complete stack for:
- Crawling web pages using crawl4ai with hierarchy tracking
- Chunking text with LangChain
- Creating embeddings with OpenAI via litellm
- Storing data in DuckDB with vector search support
- Exposing search functionality via a FastAPI web service
- Making these capabilities available to LLMs through an MCP server
- Navigating crawled sites with hierarchical site maps

---

### 🏗️ Core Infrastructure

#### 🗄️ DuckDB
- Database for storing document data and embeddings with vector search capabilities
- Managed by unified Database class

#### 📨 Redis
- Message broker for asynchronous task processing

#### 🕸️ Crawl Worker
- Processes crawl jobs
- Chunks text
- Creates embeddings

#### 🌐 Web Server
- FastAPI service exposing endpoints
- Fetching, searching, and viewing data
- Exposing the MCP server

---

### 💻 Setup

#### ⚙️ Prerequisites
- Docker and Docker Compose
- Python 3.10+
- uv (Python package manager)
- OpenAI API key

#### 📦 Installation
1. Clone this repository
2. Set up environment variables:
   ```
   export OPENAI_API_KEY=your-openai-key
   ```
3. Run the stack:
   ```
   docker compose up
   ```

---

### 👁 Usage
1. Go to http://localhost:9111/docs to see the OpenAPI docs
2. Look for the `/fetch_url` endpoint and start a crawl job by providing a URL
3. Use `/job_progress` to see the current job status
4. Configure your editor to use `http://localhost:9111/mcp` as an MCP server

---

### ☁️ Web API

#### Core Endpoints
- `POST /fetch_url`: Start crawling a URL
- `GET /search_docs`: Search indexed documents
- `GET /job_progress`: Check crawl job progress
- `GET /list_doc_pages`: List indexed pages
- `GET /get_doc_page`: Get full text of a page

#### Site Map Feature
The Maps feature provides a hierarchical view of crawled websites, making it easy to navigate and explore the structure of indexed sites.

**Endpoints:**
- `GET /map`: View an index of all crawled sites
- `GET /map/site/{root_page_id}`: View the hierarchical tree structure of a specific site
- `GET /map/page/{page_id}`: View a specific page with navigation (parent, siblings, children)
- `GET /map/page/{page_id}/raw`: Get the raw markdown content of a page

**Features:**
- **Hierarchical Navigation**: Pages maintain parent-child relationships, allowing you to navigate through the site structure
- **Automatic Title Extraction**: Page titles are extracted from HTML or markdown content
- **Breadcrumb Navigation**: Easy navigation with breadcrumbs showing the path from root to current page
- **Sibling Navigation**: Quick access to pages at the same level in the hierarchy
- **No JavaScript Required**: All navigation works with pure HTML and CSS for maximum compatibility

**Usage Example:**
1. Crawl a website using the `/fetch_url` endpoint
2. Visit `/map` to see all crawled sites
3. Click on a site to view its hierarchical structure
4. Navigate through pages using the provided links

---

### 🔧 MCP Integration
Ensure that your Docker Compose stack is up, and then add to your Cursor or VSCode MCP Servers configuration:

```json
"doctor": {
    "type": "sse",
    "url": "http://localhost:9111/mcp"
}
```

---

### 🧪 Testing

#### Running Tests
To run all tests:
```bash
# Run all tests with coverage report
pytest
```

To run specific test categories:
```bash
# Run only unit tests
pytest -m unit

# Run only async tests
pytest -m async_test

# Run tests for a specific component
pytest tests/lib/test_crawler.py
```

#### Test Coverage
The project is configured to generate coverage reports automatically:
```bash
# Run tests with detailed coverage report
pytest --cov=src --cov-report=term-missing
```

#### Test Structure
- `tests/conftest.py`: Common fixtures for all tests
- `tests/lib/`: Tests for library components
  - `test_crawler.py`: Tests for the crawler module
  - `test_crawler_enhanced.py`: Tests for enhanced crawler with hierarchy tracking
  - `test_chunker.py`: Tests for the chunker module
  - `test_embedder.py`: Tests for the embedder module
  - `test_database.py`: Tests for the unified Database class
  - `test_database_hierarchy.py`: Tests for database hierarchy operations
- `tests/common/`: Tests for common modules
- `tests/services/`: Tests for service layer
  - `test_map_service.py`: Tests for the map service
- `tests/api/`: Tests for API endpoints
  - `test_map_api.py`: Tests for map API endpoints
- `tests/integration/`: Integration tests
  - `test_processor_enhanced.py`: Tests for enhanced processor with hierarchy

---

### 🐞 Code Quality

#### Pre-commit Hooks
The project is configured with pre-commit hooks that run automatically before each commit:
- `ruff check --fix`: Lints code and automatically fixes issues
- `ruff format`: Formats code according to project style
- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Large file checks

#### Setup Pre-commit
To set up pre-commit hooks:
```bash
# Install pre-commit
uv pip install pre-commit

# Install the git hooks
pre-commit install
```

#### Running Pre-commit Manually
You can run the pre-commit hooks manually on all files:
```bash
# Run all pre-commit hooks
pre-commit run --all-files
```

Or on staged files only:
```bash
# Run on staged files
pre-commit run
```

---

### ⚖️ License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
