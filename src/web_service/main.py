"""Main module for the Doctor Web Service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import pathlib

import redis
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from src.common.config import (
    DUCKDB_PATH,
    DUCKDB_READ_PATH,
    REDIS_URI,
    WEB_SERVICE_HOST,
    WEB_SERVICE_PORT,
    check_config,
)
from src.common.logger import get_logger
from src.lib.database import get_connection_pool
from src.web_service.api import api_router

# Get logger for this module
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for the FastAPI application.

    Handles startup and shutdown events. Ensures configuration is valid
    and checks if the database exists, but does not create it.

    Args:
        app: The FastAPI application.

    Yields:
        None: This is an asynccontextmanager.
    """
    logger.info("Starting web service")

    # Check configuration
    if not check_config():
        logger.error("Invalid configuration. Check logs for details.")
        return

    # Check if read database exists
    read_db_path = pathlib.Path(DUCKDB_READ_PATH)
    original_db_path = pathlib.Path(DUCKDB_PATH)

    # If read DB doesn't exist but original does, copy it
    if not read_db_path.exists() and original_db_path.exists():
        logger.info("Copying original database to read database")
        import shutil

        try:
            # Make sure parent directory exists
            read_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the database
            shutil.copy2(original_db_path, read_db_path)
            logger.info("Copied original database to read database")

            # Also copy any associated WAL files if they exist
            original_wal_path = original_db_path.with_suffix(".wal")
            read_wal_path = read_db_path.with_suffix(".wal")
            if original_wal_path.exists():
                logger.info("Copying original WAL file to read database")
                shutil.copy2(original_wal_path, read_wal_path)

        except Exception as e:
            logger.error(f"Failed to copy original database: {e}")
            # Will try to use original DB as fallback

    # Get actual DB path for logging
    actual_db_path = (
        read_db_path
        if read_db_path.exists()
        else (original_db_path if original_db_path.exists() else None)
    )

    if actual_db_path and actual_db_path.exists():
        logger.info(f"Database file exists at {actual_db_path}")

        # Initialize a connection pool for the application to use
        # We create read-only connections so they don't conflict with the Crawl Worker
        conn_pool = await get_connection_pool()
    else:
        logger.warning("Database file not found. Web service may not function correctly.")

        # Still create a connection pool, it will fail appropriately when used
        conn_pool = await get_connection_pool()

    # Connect to Redis for job management
    try:
        redis_client = redis.from_url(REDIS_URI)
        info = redis_client.info()
        logger.info(f"Connected to Redis version {info['redis_version']}")
    except Exception as e:
        logger.error(f"Redis connection error: {e}")

    yield

    # Shutdown: close the connection pool
    logger.info("Shutting down web service")
    await conn_pool.stop()


app = FastAPI(
    title="Doctor API",
    description="API for the Doctor documentation manager",
    lifespan=lifespan,
)

# Enable MCP
app_mcp = FastApiMCP(app)

# Include the API router
app.include_router(api_router)


def main() -> None:
    """Run the web service with Uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.web_service.main:app",
        host=WEB_SERVICE_HOST,
        port=WEB_SERVICE_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
