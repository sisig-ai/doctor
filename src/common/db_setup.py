"""Database setup and connection utilities for the Doctor project."""

import os
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    This approach ensures we always get the most up-to-date data by directly
    connecting to the database file with read_only=True, which allows concurrent
    access even while the file is being written to by other processes.

    Returns:
        Read-only DuckDB connection to the database
    """
    try:
        # Check if the database file exists first
        if not os.path.exists(DUCKDB_PATH):
            logger.warning(
                f"Database file {DUCKDB_PATH} does not exist yet, returning empty in-memory database"
            )
            conn = duckdb.connect(":memory:")
            # Create empty tables with the right schema
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id VARCHAR PRIMARY KEY,
                start_url VARCHAR,
                status VARCHAR,
                pages_discovered INTEGER DEFAULT 0,
                pages_crawled INTEGER DEFAULT 0,
                max_pages INTEGER,
                tags VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                error_message VARCHAR
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id VARCHAR PRIMARY KEY,
                url VARCHAR,
                domain VARCHAR,
                raw_text TEXT,
                crawl_date TIMESTAMP,
                tags VARCHAR
            )
            """)
            return conn

        # Log file stats to help debug
        file_size = os.path.getsize(DUCKDB_PATH)
        file_mtime = os.path.getmtime(DUCKDB_PATH)
        logger.info(f"Opening database file: size={file_size} bytes, last_modified={file_mtime}")

        # Open a direct connection with read_only=True
        logger.info(f"Connecting to DuckDB file {DUCKDB_PATH} in read-only mode")

        # Use access_mode='READ_ONLY' to ensure filesystem-level read-only access
        # This is safer than the Python API read_only parameter which just sets a flag
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)

        # Test the connection to verify it's usable
        try:
            test_query = conn.execute("SELECT 1").fetchone()
            if test_query and test_query[0] == 1:
                logger.info("Successfully opened DuckDB database in read-only mode")
            else:
                logger.warning("Connection test returned unexpected result")
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            # Connection exists but query failed - we'll keep the connection anyway
            # as future queries might succeed if the DB becomes available

        return conn

    except Exception as e:
        logger.error(f"Failed to create read-only connection: {str(e)}")
        # Fallback to memory-only database with correct schema but no data
        try:
            conn = duckdb.connect(":memory:")
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id VARCHAR PRIMARY KEY,
                start_url VARCHAR,
                status VARCHAR,
                pages_discovered INTEGER DEFAULT 0,
                pages_crawled INTEGER DEFAULT 0,
                max_pages INTEGER,
                tags VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                error_message VARCHAR
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id VARCHAR PRIMARY KEY,
                url VARCHAR,
                domain VARCHAR,
                raw_text TEXT,
                crawl_date TIMESTAMP,
                tags VARCHAR
            )
            """)
            return conn
        except Exception as fallback_e:
            logger.error(f"Failed to create fallback in-memory database: {str(fallback_e)}")
            raise


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
        conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id VARCHAR PRIMARY KEY,
            url VARCHAR,
            domain VARCHAR,
            raw_text TEXT,
            crawl_date TIMESTAMP,
            tags VARCHAR  -- JSON string array
        )
        """)

        # Create jobs table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id VARCHAR PRIMARY KEY,
            start_url VARCHAR,
            status VARCHAR,
            pages_discovered INTEGER DEFAULT 0,
            pages_crawled INTEGER DEFAULT 0,
            max_pages INTEGER,
            tags VARCHAR,  -- JSON string array
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            error_message VARCHAR
        )
        """)

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
    """
    Initialize all databases.

    Args:
        read_only: Whether to initialize databases in read-only mode.
                  Web service should use read_only=True.
                  Crawl worker should use read_only=False (default).
    """
    # Initialize Qdrant
    try:
        qdrant_client = get_qdrant_client()
        ensure_qdrant_collection(qdrant_client)
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")

    # Initialize DuckDB
    try:
        if not read_only:
            # For crawl worker, ensure tables exist
            ensure_duckdb_tables()
        else:
            # For web service, just log that we're using read-only mode
            logger.info("Initializing databases in read-only mode")
    except Exception as e:
        logger.error(f"Failed to initialize DuckDB: {e}")

    logger.info(f"Database initialization process completed (read_only={read_only}).")


if __name__ == "__main__":
    # When run directly, initialize all databases
    init_databases()
