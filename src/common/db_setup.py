"""Database setup and connection utilities for the Doctor project."""

import os
import asyncio
from typing import Optional, List
import json

import duckdb

from .config import (
    DUCKDB_PATH,
    DATA_DIR,
)
from src.common.logger import get_logger

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


# Define document embeddings table creation schema
CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_embeddings (
    id BIGSERIAL PRIMARY KEY,
    embedding FLOAT4[1536] NOT NULL,
    text_chunk VARCHAR,
    page_id VARCHAR,
    url VARCHAR,
    domain VARCHAR,
    tags VARCHAR[],
    job_id VARCHAR
);
"""


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
    """Ensures the DuckDB tables exist, creating them if necessary.

    Args:
        conn: An optional DuckDB connection. If None, a new connection is created.
    """
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
    """Serializes a list of tags to a JSON string for storage in DuckDB.

    Args:
        tags: A list of tags (strings) or None.

    Returns:
        str: A JSON string representation of the tags list.
    """
    if tags is None:
        return json.dumps([])
    return json.dumps(tags)


def deserialize_tags(tags_json: str) -> List[str]:
    """Deserializes a tags JSON string from DuckDB to a list of strings.

    Args:
        tags_json: The JSON string containing the tags.

    Returns:
        List[str]: A list of tags (strings). Returns an empty list if the input is empty or invalid JSON.
    """
    if not tags_json:
        return []
    try:
        return json.loads(tags_json)
    except json.JSONDecodeError:
        logger.warning(f"Could not decode tags JSON: {tags_json}")
        return []  # Return empty list on decode error


def ensure_duckdb_vss_extension(conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Ensures the DuckDB VSS extension is loaded and embeddings table exists.

    Args:
        conn: An optional DuckDB connection. If None, a new connection is created.
    """
    close_conn = False
    if conn is None:
        try:
            conn = get_duckdb_connection()
            close_conn = True
        except Exception:
            logger.error("Cannot ensure DuckDB VSS extension without a valid connection.")
            return

    try:
        # Install and load VSS extension
        try:
            conn.execute("INSTALL vss;")
            logger.info("DuckDB VSS extension installed")
        except Exception as e:
            # If extension is already installed, this will fail, but that's okay
            logger.debug(f"VSS extension installation attempt: {str(e)}")

        conn.execute("LOAD vss;")
        logger.info("DuckDB VSS extension loaded")

        # Create embeddings table
        conn.execute(CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL)
        logger.info("DuckDB document_embeddings table created/verified")

        # Create HNSW index if it doesn't exist
        # Note: DuckDB will ignore this if the index already exists (no IF NOT EXISTS support for indexes)
        try:
            conn.execute("""
            CREATE INDEX hnsw_index_on_embeddings
            ON document_embeddings
            USING HNSW (embedding)
            WITH (metric = 'cosine');
            """)
            logger.info("Created HNSW index on document_embeddings.embedding")
        except Exception as e:
            # Check if error is related to index already existing
            if "already exists" in str(e):
                logger.info("HNSW index on document_embeddings.embedding already exists")
            else:
                raise
    except Exception as e:
        logger.error(f"Failed to set up DuckDB VSS and embeddings table: {e}")
    finally:
        if close_conn and conn:
            conn.close()


def init_databases(read_only: bool = False) -> None:
    """Initializes all databases (DuckDB), creating them if they don't exist.

    Args:
        read_only: If True, initializes DuckDB in read-only mode (tables are not created).
    """
    logger.info("Initializing databases")

    try:
        # Ensure database directory exists
        os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)

        # Only set up database tables if not in read-only mode
        if not read_only:
            # Ensure the DuckDB tables exist with the correct schema
            conn = get_duckdb_connection()
            ensure_duckdb_tables(conn)

            # Ensure VSS extension is loaded and embeddings table exists
            ensure_duckdb_vss_extension(conn)

            conn.close()

        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def get_duckdb_connection_with_retry(
    max_attempts: int = 3, retry_delay: float = 0.1
) -> duckdb.DuckDBPyConnection:
    """
    Get a DuckDB connection with retry logic.

    Args:
        max_attempts: Maximum number of connection attempts
        retry_delay: Initial delay between retries (doubles after each attempt)

    Returns:
        duckdb.DuckDBPyConnection: Connected DuckDB connection

    Raises:
        Exception: If connection fails after all retries
    """
    attempts = 0
    last_error = None
    conn = None

    while attempts < max_attempts:
        attempts += 1
        try:
            # Get a fresh read-only connection each time
            conn = get_read_only_connection()
            # Verify the connection works with a simple query
            conn.execute("SELECT 1").fetchone()
            return conn
        except Exception as e:
            last_error = e
            logger.warning(f"DuckDB connection attempt {attempts}/{max_attempts} failed: {str(e)}")
            # Clean up potentially broken connection
            if conn:
                try:
                    conn.close()
                except Exception as close_err:
                    logger.warning(f"Error closing connection during error handling: {close_err}")

            if attempts < max_attempts:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    # If we get here, all attempts failed
    logger.error(f"Failed to connect to DuckDB after {max_attempts} attempts")
    raise last_error or Exception("Failed to connect to DuckDB")


if __name__ == "__main__":
    # When run directly, initialize all databases
    init_databases()
