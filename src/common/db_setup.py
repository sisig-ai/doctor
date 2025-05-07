"""Database setup and connection utilities for the Doctor project."""

import os
from typing import Optional, List
import json

import duckdb
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from .config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    VECTOR_SIZE,
    DUCKDB_PATH,
    DATA_DIR,
)
from src.lib.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Define table creation schemas
CREATE_JOBS_TABLE_SQL = """
CREATE OR REPLACE TABLE jobs (
    job_id VARCHAR PRIMARY KEY,
    start_url VARCHAR,
    status VARCHAR,
    pages_discovered INTEGER DEFAULT 0,
    pages_crawled INTEGER DEFAULT 0,
    max_pages INTEGER,
    tags VARCHAR, -- JSON string array
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    error_message VARCHAR
)
"""

CREATE_PAGES_TABLE_SQL = """
CREATE OR REPLACE TABLE pages (
    id VARCHAR PRIMARY KEY,
    url VARCHAR,
    domain VARCHAR,
    raw_text TEXT,
    crawl_date TIMESTAMP,
    tags VARCHAR,  -- JSON string array
    job_id VARCHAR  -- Reference to the job that crawled this page
)
"""


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client using configuration settings."""
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_qdrant_collection(client: Optional[QdrantClient] = None) -> None:
    """Ensure the Qdrant collection exists, creating it if necessary."""
    if client is None:
        client = get_qdrant_client()

    # Check if collection exists
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
    except Exception as e:
        logger.error(f"Could not connect to Qdrant to check collections: {e}")
        # Optionally, re-raise or handle connection failure
        return

    if QDRANT_COLLECTION_NAME not in collection_names:
        logger.info(f"Creating Qdrant collection: {QDRANT_COLLECTION_NAME}")
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection '{QDRANT_COLLECTION_NAME}': {e}")
    else:
        logger.info(f"Qdrant collection already exists: {QDRANT_COLLECTION_NAME}")


def get_duckdb_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Get a DuckDB connection, creating the database file if necessary.

    Args:
        read_only: Whether to open the database in read-only mode.
                  Setting to True allows multiple concurrent readers.

    Returns:
        DuckDB connection
    """
    # Ensure data directory exists (using DATA_DIR from config)
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info(f"Connecting to DuckDB at {DUCKDB_PATH} (read_only={read_only})")
    try:
        return duckdb.connect(DUCKDB_PATH, read_only=read_only)
    except Exception as e:
        logger.error(f"Failed to connect to DuckDB at {DUCKDB_PATH}: {e}")
        raise  # Re-raise the exception as connection failure is critical


def get_read_only_connection() -> duckdb.DuckDBPyConnection:
    """
    Get a read-only DuckDB connection that directly accesses the database file.

    This function requires the database file to exist and will raise an error
    if it cannot be found or opened in read-only mode.

    Returns:
        Read-only DuckDB connection to the database

    Raises:
        FileNotFoundError: If the database file does not exist.
        duckdb.IOException: If there's an issue opening the file (permissions, etc.).
        Exception: For other potential connection errors.
    """
    # Check if the database file exists first
    if not os.path.exists(DUCKDB_PATH):
        logger.error(
            f"Database file {DUCKDB_PATH} does not exist. Cannot create read-only connection."
        )
        raise FileNotFoundError(f"Required database file not found: {DUCKDB_PATH}")

    # Log file stats to help debug
    file_size = os.path.getsize(DUCKDB_PATH)
    file_mtime = os.path.getmtime(DUCKDB_PATH)
    logger.info(
        f"Attempting to open read-only database file: path={DUCKDB_PATH}, size={file_size} bytes, last_modified={file_mtime}"
    )

    try:
        # Open a direct connection with read_only=True
        # Use access_mode='READ_ONLY' for stricter filesystem-level read-only access
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        logger.info(f"Successfully opened DuckDB database {DUCKDB_PATH} in read-only mode")

        # Optional: Test the connection
        try:
            test_query = conn.execute("SELECT 1").fetchone()
            if not (test_query and test_query[0] == 1):
                logger.warning("Read-only connection test returned unexpected result.")
        except Exception as e:
            logger.warning(f"Read-only connection test failed: {e}. Proceeding anyway.")

        return conn

    except duckdb.IOException as e:
        logger.error(f"Failed to open DuckDB file {DUCKDB_PATH} in read-only mode: {e}")
        raise  # Re-raise the specific IO error
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while creating read-only connection to {DUCKDB_PATH}: {e}"
        )
        raise  # Re-raise any other connection errors


def ensure_duckdb_tables(conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Ensure the DuckDB tables exist, creating them if necessary."""
    close_conn = False
    if conn is None:
        try:
            conn = get_duckdb_connection()
            close_conn = True
        except Exception:
            logger.error("Cannot ensure DuckDB tables without a valid connection.")
            return

    try:
        # Create pages table
        conn.execute(CREATE_PAGES_TABLE_SQL)

        # Create jobs table
        conn.execute(CREATE_JOBS_TABLE_SQL)

        logger.info("DuckDB tables created/verified")
    except Exception as e:
        logger.error(f"Failed to create/verify DuckDB tables: {e}")
    finally:
        if close_conn and conn:
            conn.close()


def serialize_tags(tags: Optional[List[str]]) -> str:
    """Serialize tags list to JSON string for storage in DuckDB."""
    if tags is None:
        return json.dumps([])
    return json.dumps(tags)


def deserialize_tags(tags_json: str) -> List[str]:
    """Deserialize tags JSON string from DuckDB to a list."""
    if not tags_json:
        return []
    try:
        return json.loads(tags_json)
    except json.JSONDecodeError:
        logger.warning(f"Could not decode tags JSON: {tags_json}")
        return []  # Return empty list on decode error


def init_databases(read_only: bool = False) -> None:
    """Initialize all databases, creating them if they don't exist."""
    logger.info("Initializing databases")

    try:
        # Ensure database directory exists
        os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)

        # Only set up database tables if not in read-only mode
        if not read_only:
            # Ensure the DuckDB tables exist with the correct schema
            conn = get_duckdb_connection()
            ensure_duckdb_tables(conn)
            conn.close()

        # Ensure the Qdrant collection exists
        ensure_qdrant_collection()

        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    # When run directly, initialize all databases
    init_databases()
