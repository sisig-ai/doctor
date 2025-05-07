"""Main module for the Doctor Web Service."""

from contextlib import asynccontextmanager

from typing import AsyncIterator
import redis
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from src.common.config import (
    REDIS_URI,
    WEB_SERVICE_HOST,
    WEB_SERVICE_PORT,
)
from src.common.db_setup import (
    init_databases,
)
from src.lib.logger import get_logger
from src.web_service.api import api_router
from mcp.server.session import ServerSession

# Get logger for this module
logger = get_logger(__name__)

# Temporary monkeypatch which avoids crashing when a POST message is received
# before a connection has been initialized, e.g: after a deployment.
# pylint: disable-next=protected-access
# https://github.com/modelcontextprotocol/python-sdk/issues/423
old__received_request = ServerSession._received_request


async def _received_request(self, *args, **kwargs):
    """Monkeypatch to handle RuntimeError during request reception."""
    try:
        return await old__received_request(self, *args, **kwargs)
    except RuntimeError:
        pass


ServerSession._received_request = _received_request


@asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for the FastAPI application.

    Handles startup and shutdown events.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Indicates the application is ready.
    """
    # Initialize databases in read-only mode for the web service
    init_databases(read_only=True)
    logger.info("Database initialization complete")
    yield
    logger.info("Shutting down application")


def create_application() -> FastAPI:
    """Creates and configures the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="Doctor API",
        description="API for the Doctor web crawling and indexing system",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include the API router
    app.include_router(api_router)

    # Set up MCP
    mcp = FastApiMCP(
        app,
        name="Doctor",
        description="API for the Doctor web crawling and indexing system",
        describe_all_responses=True,
        describe_full_response_schema=True,
        exclude_operations=["fetch_url", "job_progress", "delete_docs"],
    )

    mcp.mount()

    return app


# Create the application
app = create_application()

# Create Redis connection
redis_conn = redis.from_url(REDIS_URI)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_SERVICE_HOST, port=WEB_SERVICE_PORT)
