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
    Get a read-only DuckDB connection by making a temporary copy of the database.

    This approach avoids lock conflicts entirely by creating a point-in-time
    copy of the database specifically for reading. The copy is made in memory
    so no additional disk space is used.

    Returns:
        Read-only DuckDB connection to a copy of the database
    """

    try:
        # Check if the database file exists first
        if not os.path.exists(DUCKDB_PATH):
            logger.warning(
                f"Database file {DUCKDB_PATH} does not exist yet, returning empty in-memory database"
            )
            return duckdb.connect(":memory:")

        # Create an in-memory database
        logger.info("Creating in-memory database for read-only operations")
        conn = duckdb.connect(":memory:")

        # Import the entire database from the file
        logger.info(f"Importing data from {DUCKDB_PATH} to in-memory database")

        # Import schema and data from the main database
        try:
            conn.execute(f"IMPORT DATABASE '{DUCKDB_PATH}'")
            logger.info("Successfully imported database to in-memory copy")
            return conn
        except Exception as e:
            # If IMPORT DATABASE fails, try a less efficient but more reliable approach
            logger.warning(f"IMPORT DATABASE failed: {str(e)}. Trying alternative approach...")

            # Close the failed connection and start fresh
            conn.close()

            # Create a new in-memory connection
            conn = duckdb.connect(":memory:")

            # Create the tables manually
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

            # Try to read data with a separate connection and copy it
            try:
                logger.info("Attempting to directly read from database file...")
                orig_conn = duckdb.connect(database=DUCKDB_PATH, read_only=True)

                # Copy jobs data
                jobs_data = orig_conn.execute("SELECT * FROM jobs").fetchall()
                if jobs_data:
                    logger.info(f"Copying {len(jobs_data)} job records to in-memory database")
                    for job in jobs_data:
                        conn.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", job)

                # Copy pages data in batches to avoid memory issues with large datasets
                batch_size = 100
                offset = 0
                while True:
                    pages_data = orig_conn.execute(
                        f"SELECT * FROM pages LIMIT {batch_size} OFFSET {offset}"
                    ).fetchall()

                    if not pages_data:
                        break

                    logger.info(
                        f"Copying {len(pages_data)} page records to in-memory database (offset {offset})"
                    )
                    for page in pages_data:
                        conn.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?)", page)

                    offset += batch_size
                    if len(pages_data) < batch_size:
                        break

                # Close the original connection
                orig_conn.close()
                logger.info("Successfully copied database to in-memory database")

            except Exception as inner_e:
                logger.warning(f"Could not copy data from main database: {str(inner_e)}")
                # Return the empty database anyway, which will have the correct schema

            return conn

    except Exception as e:
        logger.error(f"Failed to create read-only in-memory copy: {str(e)}")
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


def init_databases() -> None:
    """Initialize all databases."""
    # Initialize Qdrant
    try:
        qdrant_client = get_qdrant_client()
        ensure_qdrant_collection(qdrant_client)
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")

    # Initialize DuckDB
    try:
        # ensure_duckdb_tables handles getting and closing the connection if needed
        ensure_duckdb_tables()
    except Exception as e:
        logger.error(f"Failed to initialize DuckDB: {e}")

    logger.info("Database initialization process completed.")


if __name__ == "__main__":
    # When run directly, initialize all databases
    init_databases()
