# Doctor

Doctor is a system that lets LLM agents discover, crawl, and index web sites for better and more up-to-date reasoning and code generation.

## Overview

Doctor provides a complete stack for:
- Crawling web pages using crawl4ai
- Chunking text with LangChain
- Creating embeddings with OpenAI via litellm
- Storing data in DuckDB and Qdrant
- Exposing search functionality via a FastAPI web service
- Making these capabilities available to LLMs through an MCP server

## Components

- **Qdrant Server**: Vector database for storing and searching embeddings
- **Redis**: Message broker for asynchronous task processing
- **Crawl Worker**: Processes crawl jobs, chunks text, creates embeddings
- **Web Server**: FastAPI service exposing endpoints for fetching, searching, and viewing data, and exposing the MCP server.

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- uv (Python package manager)
- OpenAI API key

### Installation

1. Clone this repository
2. Set up environment variables:
   ```
   export OPENAI_API_KEY=your-openai-key
   ```
3. Run the stack:
   ```
   docker compose up
   ```

## Usage

1. Go to http://localhost:9111/docs to see the OpenAPI docs.
2. Look for the `/fetch_url` endpoint and start a crawl job by providing a URL.
3. Use `/job_progress` to see the current job status.
4. Configure your editor to use `http://localhost:9111/mcp` as an MCP server

### Web API

- `POST /fetch_url`: Start crawling a URL
- `GET /search_docs`: Search indexed documents
- `GET /job_progress`: Check crawl job progress
- `GET /list_doc_pages`: List indexed pages
- `GET /get_doc_page`: Get full text of a page

### MCP Integration

Ensure that your Docker Compose stack is up, and then add to your Cursor or VSCode MCP Servers configuration:

```json
"doctor": {
    "type": "sse",
    "url": "http://localhost:9111/mcp" 
}
```