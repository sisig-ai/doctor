"""Database management for the Doctor project with a unified Database class."""

import os
import asyncio
import uuid
import datetime
from typing import Optional, List
import json
from urllib.parse import urlparse

import duckdb

from src.common.config import (
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
CREATE OR REPLACE TABLE document_embeddings (
    id VARCHAR PRIMARY KEY,
    embedding FLOAT4[1536] NOT NULL,
    text_chunk VARCHAR,
    page_id VARCHAR,
    url VARCHAR,
    domain VARCHAR,
    tags VARCHAR[],
    job_id VARCHAR
);
"""


class Database:
    """Unified database management for Doctor project.

    This class handles all database operations including connection management,
    table creation, and core data operations for pages and jobs.
    """

    def __init__(self, read_only: bool = False):
        """Initialize the database manager.

        Args:
            read_only: Whether to open in read-only mode.
                      Setting to True allows multiple concurrent readers.
        """
        self.read_only = read_only
        self.conn = None
        self.transaction_active = False

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish a database connection.

        Creates a new connection to the DuckDB database using the configured path.

        Returns:
            duckdb.DuckDBPyConnection: Active DuckDB connection object

        Raises:
            FileNotFoundError: If in read-only mode and the database file doesn't exist
            Exception: On connection failure
        """
        # Ensure data directory exists (using DATA_DIR from config)
        os.makedirs(DATA_DIR, exist_ok=True)

        logger.info(f"Connecting to DuckDB at {DUCKDB_PATH} (read_only={self.read_only})")

        if self.read_only and not os.path.exists(DUCKDB_PATH):
            logger.error(
                f"Database file {DUCKDB_PATH} does not exist. Cannot create read-only connection."
            )
            raise FileNotFoundError(f"Required database file not found: {DUCKDB_PATH}")

        try:
            self.conn = duckdb.connect(DUCKDB_PATH, read_only=self.read_only)
            # Test connection
            self.conn.execute("SELECT 1").fetchone()
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB at {DUCKDB_PATH}: {e}")
            self.conn = None
            raise

    def close(self) -> None:
        """Close the database connection if open.

        Attempts to roll back any active transactions before closing.
        Suppresses but logs any errors that occur during closing.
        """
        if self.conn:
            try:
                # If a transaction is active, try to roll it back
                if self.transaction_active:
                    try:
                        self.conn.rollback()
                        self.transaction_active = False
                    except Exception as e:
                        logger.warning(f"Error during rollback: {e}")

                self.conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self.conn = None

    def ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Ensure a valid database connection exists.

        Checks if current connection is valid and creates a new one if needed.

        Returns:
            duckdb.DuckDBPyConnection: Active DuckDB connection object
        """
        if self.conn is None:
            return self.connect()
        return self.conn

    def begin_transaction(self) -> None:
        """Begin a database transaction.

        Starts a new transaction if one is not already active.
        Ensures a connection exists before attempting to start the transaction.
        """
        if self.transaction_active:
            logger.warning("Transaction already active, not beginning a new one")
            return

        self.ensure_connection()
        self.conn.execute("BEGIN TRANSACTION")
        self.transaction_active = True

    def commit(self) -> None:
        """Commit the current transaction."""
        if not self.transaction_active:
            logger.warning("No active transaction to commit")
            return

        if self.conn:
            self.conn.commit()
            self.transaction_active = False

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if not self.transaction_active:
            logger.warning("No active transaction to rollback")
            return

        if self.conn:
            self.conn.rollback()
            self.transaction_active = False

    def checkpoint(self) -> None:
        """Force a database checkpoint to ensure changes are persisted."""
        try:
            self.ensure_connection()
            self.conn.execute("CHECKPOINT")
            logger.info("Forced database checkpoint")
        except Exception as e:
            logger.warning(f"Failed to checkpoint: {str(e)}")

    def ensure_tables(self) -> None:
        """Ensure all required database tables exist."""
        if self.read_only:
            logger.warning("Cannot ensure tables in read-only mode")
            return

        self.ensure_connection()

        try:
            # Create pages table
            self.conn.execute(CREATE_PAGES_TABLE_SQL)

            # Create jobs table
            self.conn.execute(CREATE_JOBS_TABLE_SQL)

            logger.info("DuckDB tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create/verify DuckDB tables: {e}")
            raise

    def ensure_vss_extension(self) -> None:
        """Ensure the VSS extension is loaded and embeddings table exists."""
        if self.read_only:
            logger.warning("Cannot ensure VSS extension in read-only mode")
            return

        self.ensure_connection()

        try:
            # Install and load VSS extension
            try:
                self.conn.execute("INSTALL vss;")
                logger.info("DuckDB VSS extension installed")
            except Exception:
                logger.info("VSS extension already installed")

            # Load the VSS extension
            self.conn.execute("LOAD vss;")
            logger.info("DuckDB VSS extension loaded")

            # Enable experimental persistence for HNSW indexes
            try:
                self.conn.execute("SET hnsw_enable_experimental_persistence = true;")
                logger.info("Enabled experimental persistence for HNSW indexes")
            except Exception as e:
                logger.error(f"Failed to enable HNSW persistence: {str(e)}")
                logger.warning("HNSW indexes may not work in persistent mode")

            # Verify the VSS extension is loaded
            try:
                self.conn.execute("""
                    WITH test_data AS (SELECT [0.1, 0.2, 0.3]::FLOAT[] AS v)
                    SELECT * FROM test_data LIMIT 1;
                """).fetchone()
                logger.info("VSS extension functionality verified successfully")
            except Exception as e:
                logger.error(f"VSS extension verification failed: {str(e)}")
                logger.warning("Will attempt to continue with table creation anyway")

            # Ensure the document_embeddings table exists
            try:
                # Check if the table exists first
                table_exists = self.conn.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name = 'document_embeddings'"
                ).fetchone()[0]

                if table_exists:
                    logger.info("document_embeddings table already exists")
                else:
                    # Create embeddings table
                    self.conn.execute(CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL)
                    logger.info("DuckDB document_embeddings table created successfully")

                    # Verify the table was created
                    verify_table = self.conn.execute(
                        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'document_embeddings'"
                    ).fetchone()[0]

                    if verify_table != 1:
                        raise Exception("Failed to create document_embeddings table")
            except Exception as table_err:
                logger.error(f"Error creating document_embeddings table: {str(table_err)}")
                raise

            # Try to create HNSW index, but don't fail if it can't be created
            try:
                # First check for existing index to avoid error
                index_exists = False
                try:
                    index_exists = (
                        self.conn.execute("""
                        SELECT count(*)
                        FROM duckdb_indexes()
                        WHERE index_name = 'hnsw_index_on_embeddings'
                    """).fetchone()[0]
                        > 0
                    )
                except Exception:
                    # If duckdb_indexes() function isn't available, just try creating the index
                    pass

                if index_exists:
                    logger.info("HNSW index on document_embeddings.embedding already exists")
                else:
                    self.conn.execute("""
                    CREATE INDEX hnsw_index_on_embeddings
                    ON document_embeddings
                    USING HNSW (embedding)
                    WITH (metric = 'cosine');
                    """)
                    logger.info("Created HNSW index on document_embeddings.embedding")
            except Exception as e:
                # Log but continue if index creation fails - we can still use the table
                logger.warning(f"Failed to create HNSW index, but continuing without it: {str(e)}")
                logger.info("Vector searches will work but may be slower without the HNSW index")
        except Exception as e:
            logger.error(f"Failed to set up DuckDB VSS and embeddings table: {e}")
            raise

    def initialize(self) -> None:
        """Initialize the database, creating all required tables and extensions."""
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)

            # Only set up database tables if not in read-only mode
            if not self.read_only:
                self.ensure_connection()
                self.ensure_tables()
                self.ensure_vss_extension()

            logger.info("Database initialization complete")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    @staticmethod
    def serialize_tags(tags: Optional[List[str]]) -> str:
        """Serialize a list of tags to a JSON string.

        Args:
            tags: A list of tags (strings) or None.

        Returns:
            JSON string representation of the tags list
        """
        if tags is None:
            return json.dumps([])
        return json.dumps(tags)

    @staticmethod
    def deserialize_tags(tags_json: str) -> List[str]:
        """Deserialize a tags JSON string to a list of strings.

        Args:
            tags_json: The JSON string containing the tags.

        Returns:
            A list of tags (strings). Returns an empty list if input is empty or invalid.
        """
        if not tags_json:
            return []
        try:
            return json.loads(tags_json)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode tags JSON: {tags_json}")
            return []  # Return empty list on decode error

    async def connect_with_retry(
        self, max_attempts: int = 3, retry_delay: float = 0.1
    ) -> duckdb.DuckDBPyConnection:
        """Get a DuckDB connection with retry logic.

        Args:
            max_attempts: Maximum number of connection attempts
            retry_delay: Initial delay between retries (doubles after each attempt)

        Returns:
            Connected DuckDB connection

        Raises:
            Exception: If connection fails after all retries
        """
        attempts = 0
        last_error = None
        self.conn = None

        while attempts < max_attempts:
            attempts += 1
            try:
                # Try to connect to the database
                if self.read_only:
                    # This is a read-only mode, so expect the file to exist
                    if not os.path.exists(DUCKDB_PATH):
                        raise FileNotFoundError(f"Required database file not found: {DUCKDB_PATH}")

                self.conn = duckdb.connect(DUCKDB_PATH, read_only=self.read_only)
                # Verify the connection works with a simple query
                self.conn.execute("SELECT 1").fetchone()
                return self.conn
            except Exception as e:
                last_error = e
                logger.warning(
                    f"DuckDB connection attempt {attempts}/{max_attempts} failed: {str(e)}"
                )
                # Clean up potentially broken connection
                self.close()

                if attempts < max_attempts:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        # If we get here, all attempts failed
        logger.error(f"Failed to connect to DuckDB after {max_attempts} attempts")
        raise last_error or Exception("Failed to connect to DuckDB")

    async def store_page(
        self,
        url: str,
        text: str,
        job_id: str,
        tags: Optional[List[str]] = None,
        page_id: Optional[str] = None,
    ) -> str:
        """Store a crawled page in the database.

        Adds a crawled page's content and metadata to the pages table.

        Args:
            url (str): The URL of the page
            text (str): The extracted text content of the page
            job_id (str): The ID of the crawl job
            tags (Optional[List[str]]): Tags to associate with the page. Defaults to None.
            page_id (Optional[str]): ID for the page. Defaults to None (UUID generated).

        Returns:
            str: The ID of the stored page

        Raises:
            Exception: If database errors occur during storage
        """
        if tags is None:
            tags = []

        # Generate a page ID if not provided
        if page_id is None:
            page_id = str(uuid.uuid4())

        # Extract domain from URL
        domain = urlparse(url).netloc

        logger.debug(f"Storing page {page_id} from {url} with {len(text)} characters")

        # Get DuckDB connection
        self.ensure_connection()

        try:
            # Begin transaction explicitly
            self.begin_transaction()

            # Store page data
            self.conn.execute(
                """
                INSERT INTO pages (id, url, domain, raw_text, crawl_date, tags, job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    page_id,
                    url,
                    domain,
                    text,
                    datetime.datetime.now(),
                    self.serialize_tags(tags),
                    job_id,
                ),
            )

            # Commit the transaction
            self.commit()
            logger.debug(f"Successfully stored page {page_id} in database")

            return page_id

        except Exception as e:
            self.rollback()
            logger.error(f"Database error storing page: {str(e)}")
            raise

    def update_job_status(
        self,
        job_id: str,
        status: str,
        pages_discovered: Optional[int] = None,
        pages_crawled: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the status of a crawl job.

        Updates job metadata in the database with new status information.
        For completed/failed jobs, a checkpoint is forced to ensure persistence.

        Args:
            job_id (str): The ID of the job to update
            status (str): The new status of the job (e.g., "running", "completed", "failed")
            pages_discovered (Optional[int]): Number of pages discovered. Defaults to None.
            pages_crawled (Optional[int]): Number of pages crawled. Defaults to None.
            error_message (Optional[str]): Error message if the job failed. Defaults to None.

        Raises:
            Exception: If database errors occur during update
        """
        logger.info(
            f"Updating job {job_id} status to {status}, discovered={pages_discovered}, crawled={pages_crawled}"
        )

        # Get DuckDB connection
        self.ensure_connection()
        update_successful = False

        try:
            # Begin transaction explicitly
            self.begin_transaction()

            # Build the SQL query dynamically based on which fields are provided
            query_parts = ["UPDATE jobs SET status = ?, updated_at = ?"]
            params = [status, datetime.datetime.now()]

            if pages_discovered is not None:
                query_parts.append("pages_discovered = ?")
                params.append(pages_discovered)

            if pages_crawled is not None:
                query_parts.append("pages_crawled = ?")
                params.append(pages_crawled)

            if error_message is not None:
                query_parts.append("error_message = ?")
                params.append(error_message)

            query = f"{', '.join(query_parts)} WHERE job_id = ?"
            params.append(job_id)

            # Execute the query
            cursor = self.conn.execute(query, tuple(params))

            # Verify the update was applied by checking the row count
            rows_affected = cursor.rowcount
            if rows_affected == 0:
                logger.warning(
                    f"Job status update for {job_id} didn't affect any rows. Job may not exist."
                )
            else:
                logger.debug(f"Job status update affected {rows_affected} rows")
                update_successful = True

            # Commit the transaction
            self.commit()

            # Periodically force a checkpoint to ensure changes are persisted
            # Only do this for important status transitions
            if status in ["completed", "failed"]:
                self.checkpoint()

            if update_successful:
                logger.info(f"Successfully updated job {job_id} status to {status}")

            # Verify the job status was updated correctly by reading it back
            if status in ["completed", "failed"]:
                try:
                    job_data = self.conn.execute(
                        "SELECT status, pages_discovered, pages_crawled FROM jobs WHERE job_id = ?",
                        (job_id,),
                    ).fetchone()

                    if job_data:
                        current_status, current_discovered, current_crawled = job_data
                        logger.info(
                            f"Job {job_id} current state: status={current_status}, "
                            f"discovered={current_discovered}, crawled={current_crawled}"
                        )
                    else:
                        logger.warning(
                            f"Could not verify job {job_id} status - job not found in database"
                        )
                except Exception as verify_error:
                    logger.warning(f"Error verifying job status: {str(verify_error)}")

        except Exception as e:
            self.rollback()
            logger.error(f"Database error updating job status: {str(e)}")
            raise

    def __enter__(self) -> "Database":
        """Context manager entry point.

        Ensures a database connection is established when entering the context.

        Returns:
            Database: Self reference for use in with statements.
        """
        self.ensure_connection()
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]
    ) -> None:
        """Context manager exit point.

        Closes the database connection when exiting the context.
        Will roll back any active transaction if an exception occurred.

        Args:
            exc_type (Optional[type]): Exception type if an exception was raised
            exc_val (Optional[Exception]): Exception value if an exception was raised
            exc_tb (Optional[object]): Exception traceback if an exception was raised
        """
        if exc_type and self.transaction_active:
            self.rollback()
        self.close()


if __name__ == "__main__":
    # When run directly, initialize all databases
    db = Database()
    db.initialize()
    db.close()
