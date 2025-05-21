"""Manages DuckDB database connections, setup, and low-level operations.

This module provides the DuckDBConnectionManager class, responsible for establishing
connections to a DuckDB database, loading necessary extensions (FTS, VSS),
managing database schema (tables, indexes, triggers), and handling transactions.
It is the foundational layer for database interactions in the project.
"""

import asyncio
import pathlib
import time
from types import TracebackType

import duckdb

from src.common.config import DATA_DIR, DUCKDB_PATH, DUCKDB_READ_PATH, DUCKDB_WRITE_PATH
from src.common.logger import get_logger

logger = get_logger(__name__)


def connect_with_retry(
    read_only: bool = False, max_retries: int = -1, retry_delay: float = 1.0
) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB with retry logic for lock conflicts.

    This function will retry the connection if it encounters a lock conflict,
    which happens when different processes try to access the database
    simultaneously with different access modes.

    Note:
        This is a blocking operation. For async code, use async_connect_with_retry instead.

    Args:
        read_only: Whether to open the connection in read-only mode. Defaults to False.
        max_retries: Maximum number of retries, or -1 for infinite retries. Defaults to -1.
        retry_delay: Delay between retries in seconds. Defaults to 1.0.

    Returns:
        An active DuckDB connection.

    Raises:
        Exception: If the connection fails after max_retries attempts.
    """
    # Select the appropriate database file based on read_only flag
    if read_only:
        db_path = pathlib.Path(DUCKDB_READ_PATH)
        # For backwards compatibility, if read file doesn't exist, try the original path
        if not db_path.exists() and pathlib.Path(DUCKDB_PATH).exists():
            db_path = pathlib.Path(DUCKDB_PATH)
    else:
        db_path = pathlib.Path(DUCKDB_WRITE_PATH)
        # For backwards compatibility, if write file doesn't exist, use the original path
        if not db_path.exists() and pathlib.Path(DUCKDB_PATH).exists():
            db_path = pathlib.Path(DUCKDB_PATH)

    # If database doesn't exist and in read-only mode, fail immediately
    if not db_path.exists() and read_only:
        raise FileNotFoundError(
            f"Database file {db_path} does not exist and read-only mode requested"
        )

    # Make sure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    retries = 0
    last_error = None

    # -1 means retry forever
    while max_retries == -1 or retries < max_retries:
        try:
            # Attempt to connect
            conn = duckdb.connect(str(db_path), read_only=read_only)
            # Test the connection
            conn.execute("SELECT 1").fetchone()
            logger.info(f"Successfully connected to DuckDB at {db_path} (read_only={read_only})")
            return conn
        except duckdb.IOException as e:
            # This is likely a lock conflict
            if "Conflicting lock is held in PID" in str(e):
                retries += 1
                last_error = e
                logger.warning(
                    f"Connection attempt {retries} failed due to lock conflict, "
                    f"retrying in {retry_delay} seconds: {e}"
                )
                time.sleep(retry_delay)
            else:
                # Some other I/O error, don't retry
                logger.error(f"Failed to connect to DuckDB due to I/O error: {e}")
                raise
        except Exception as e:
            # Other exceptions we don't retry
            logger.error(f"Failed to connect to DuckDB due to unexpected error: {e}")
            raise

    # If we get here, we've exceeded max_retries
    if last_error:
        logger.error(f"Failed to connect after {retries} attempts")
        raise last_error
    else:
        raise RuntimeError("Failed to connect to DuckDB, but no error was recorded")


async def async_connect_with_retry(
    read_only: bool = False, max_retries: int = -1, retry_delay: float = 1.0
) -> duckdb.DuckDBPyConnection:
    """Asynchronously connect to DuckDB with retry logic for lock conflicts.

    This function will retry the connection if it encounters a lock conflict,
    which happens when different processes try to access the database
    simultaneously with different access modes. Unlike the synchronous version,
    this does not block the event loop during retries.

    Args:
        read_only: Whether to open the connection in read-only mode. Defaults to False.
        max_retries: Maximum number of retries, or -1 for infinite retries. Defaults to -1.
        retry_delay: Delay between retries in seconds. Defaults to 1.0.

    Returns:
        An active DuckDB connection.

    Raises:
        Exception: If the connection fails after max_retries attempts.
    """
    # Select the appropriate database file based on read_only flag
    if read_only:
        db_path = pathlib.Path(DUCKDB_READ_PATH)
        # For backwards compatibility, if read file doesn't exist, try the original path
        if not db_path.exists() and pathlib.Path(DUCKDB_PATH).exists():
            db_path = pathlib.Path(DUCKDB_PATH)
    else:
        db_path = pathlib.Path(DUCKDB_WRITE_PATH)
        # For backwards compatibility, if write file doesn't exist, use the original path
        if not db_path.exists() and pathlib.Path(DUCKDB_PATH).exists():
            db_path = pathlib.Path(DUCKDB_PATH)

    # If database doesn't exist and in read-only mode, fail immediately
    if not db_path.exists() and read_only:
        raise FileNotFoundError(
            f"Database file {db_path} does not exist and read-only mode requested"
        )

    # Make sure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    retries = 0
    last_error = None

    # -1 means retry forever
    while max_retries == -1 or retries < max_retries:
        try:
            # Attempt to connect - DuckDB itself is not async, so we don't need to await this
            conn = duckdb.connect(str(db_path), read_only=read_only)
            # Test the connection
            conn.execute("SELECT 1").fetchone()
            logger.info(f"Successfully connected to DuckDB at {db_path} (read_only={read_only})")
            return conn
        except duckdb.IOException as e:
            # This is likely a lock conflict
            if "Conflicting lock is held in PID" in str(e):
                retries += 1
                last_error = e
                logger.warning(
                    f"Connection attempt {retries} failed due to lock conflict, "
                    f"retrying in {retry_delay} seconds: {e}"
                )
                # Use async sleep instead of blocking sleep
                await asyncio.sleep(retry_delay)
            else:
                # Some other I/O error, don't retry
                logger.error(f"Failed to connect to DuckDB due to I/O error: {e}")
                raise
        except Exception as e:
            # Other exceptions we don't retry
            logger.error(f"Failed to connect to DuckDB due to unexpected error: {e}")
            raise

    # If we get here, we've exceeded max_retries
    if last_error:
        logger.error(f"Failed to connect after {retries} attempts")
        raise last_error
    else:
        raise RuntimeError("Failed to connect to DuckDB, but no error was recorded")


class DuckDBConnectionManager:
    """Manages DuckDB connections, extensions, transactions, and schema.

    Handles the low-level details of interacting with a DuckDB database instance,
    including setting up necessary tables and extensions like FTS and VSS.
    """

    def __init__(self, *, read_only: bool = False) -> None:
        """Initialize the DuckDBConnectionManager.

        Args:
            read_only: If True, the database connection will be opened in
                read-only mode. Defaults to False.

        """
        self.read_only: bool = read_only
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._transaction_active: bool = False
        # These will be set by the connection pool
        self._lock_manager = None
        self._is_read_only = read_only

    def _load_single_extension(
        self,
        conn: duckdb.DuckDBPyConnection,
        ext_name: str,
    ) -> None:
        """Install (if necessary) and load a single DuckDB extension.

        Args:
            conn: The active DuckDB connection.
            ext_name: The name of the extension to load (e.g., 'fts', 'vss').

        """
        try:
            conn.execute(f"INSTALL {ext_name};")
            logger.info(
                f"{ext_name.upper()} extension installation attempted/verified.",
            )
        except duckdb.Error as e_install_db:  # More specific
            logger.info(
                f"{ext_name.upper()} extension already installed or DB error during "
                f"install, proceeding to load: {e_install_db}",
            )
        except Exception as e_install_generic:  # pragma: no cover # noqa: BLE001
            logger.warning(
                f"Unexpected error during {ext_name.upper()} install: {e_install_generic}, "
                "proceeding to load.",
            )
        try:
            conn.execute(f"LOAD {ext_name};")
            logger.info(f"{ext_name.upper()} extension loaded for current connection.")
        except duckdb.Error as e_load_db:  # More specific
            logger.warning(
                f"Could not load {ext_name.upper()} extension (DB error): {e_load_db}. "
                "Functionality requiring this extension may fail or be slow.",
            )
        except Exception as e_load_generic:  # pragma: no cover # noqa: BLE001
            logger.warning(
                f"Unexpected error loading {ext_name.upper()} extension: {e_load_generic}. "
                "Functionality may be impacted.",
            )

    def _load_extensions(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Install (if necessary) and loads required DuckDB extensions (FTS, VSS)."""
        self._load_single_extension(conn, "fts")
        self._load_single_extension(conn, "vss")

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish and return a DuckDB connection.

        Ensures the data directory exists and loads necessary extensions (FTS, VSS)
        upon successful connection.

        Returns:
            duckdb.DuckDBPyConnection: An active DuckDB connection object.

        Raises:
            FileNotFoundError: If in read-only mode and the database file doesn't exist.
            IOError: For OS-level I/O errors during directory creation.
            duckdb.Error: For DuckDB-specific connection failures.

        """
        data_dir_path = pathlib.Path(DATA_DIR)
        db_file_path = pathlib.Path(DUCKDB_PATH)
        try:
            data_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e_mkdir:  # pragma: no cover
            msg = f"Failed to create data directory {DATA_DIR}: {e_mkdir}"
            logger.exception(msg)
            raise OSError(msg) from e_mkdir

        logger.info(
            f"Connecting to DuckDB at {db_file_path} (read_only={self.read_only})",
        )
        if self.read_only and not db_file_path.exists():
            msg = (
                f"Database file {db_file_path} does not exist. Cannot create read-only connection."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            # Use connect_with_retry for better handling of lock conflicts
            self.conn = connect_with_retry(read_only=self.read_only)
            self._load_extensions(self.conn)
        except duckdb.Error:
            logger.exception(f"Failed to connect to DuckDB at {db_file_path}")
            self.conn = None
            raise
        except Exception as e_other:  # pragma: no cover
            logger.exception(
                f"An unexpected error occurred connecting to DuckDB at {db_file_path}: {e_other}",
            )
            self.conn = None
            raise
        else:
            return self.conn

    async def async_connect(self) -> duckdb.DuckDBPyConnection:
        """Asynchronously establish and return a DuckDB connection.

        Ensures the data directory exists and loads necessary extensions (FTS, VSS)
        upon successful connection.

        Returns:
            duckdb.DuckDBPyConnection: An active DuckDB connection object.

        Raises:
            FileNotFoundError: If in read-only mode and the database file doesn't exist.
            IOError: For OS-level I/O errors during directory creation.
            duckdb.Error: For DuckDB-specific connection failures.

        """
        data_dir_path = pathlib.Path(DATA_DIR)
        db_file_path = pathlib.Path(DUCKDB_PATH)
        try:
            data_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e_mkdir:  # pragma: no cover
            msg = f"Failed to create data directory {DATA_DIR}: {e_mkdir}"
            logger.exception(msg)
            raise OSError(msg) from e_mkdir

        logger.info(
            f"Connecting asynchronously to DuckDB at {db_file_path} (read_only={self.read_only})",
        )
        if self.read_only and not db_file_path.exists():
            msg = (
                f"Database file {db_file_path} does not exist. Cannot create read-only connection."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            # Use async_connect_with_retry for better handling of lock conflicts
            self.conn = await async_connect_with_retry(read_only=self.read_only)
            self._load_extensions(self.conn)
        except duckdb.Error:
            logger.exception(f"Failed to connect to DuckDB at {db_file_path}")
            self.conn = None
            raise
        except Exception as e_other:  # pragma: no cover
            logger.exception(
                f"An unexpected error occurred connecting to DuckDB at {db_file_path}: {e_other}",
            )
            self.conn = None
            raise
        else:
            return self.conn

    async def async_ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Asynchronously ensure an active database connection exists, creating one if necessary.

        Returns:
            duckdb.DuckDBPyConnection: An active DuckDB connection object.

        """
        if self.conn is not None:
            try:
                self.conn.execute("SELECT 1").fetchone()
            except (duckdb.Error, AttributeError) as e:
                logger.warning(
                    f"Existing connection is not usable ({e}), attempting to reconnect.",
                )
                self.conn = None
            else:
                return self.conn  # Connection is alive and usable

        return await self.async_connect()

    def ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Ensure an active database connection exists, creating one if necessary.

        Returns:
            duckdb.DuckDBPyConnection: An active DuckDB connection object.

        """
        if self.conn is not None:
            try:
                self.conn.execute("SELECT 1").fetchone()
            except (duckdb.Error, AttributeError) as e:
                logger.warning(
                    f"Existing connection is not usable ({e}), attempting to reconnect.",
                )
                self.conn = None
            else:
                return self.conn  # Connection is alive and usable

        return self.connect()

    def close(self) -> None:
        """Close the database connection.

        This also rolls back any active transaction before closing.

        Note:
            After closing, the connection cannot be used anymore.
            To get a new connection, use connect() again.
        """
        if self.conn is None:
            return  # Already closed

        try:
            # Roll back any active transaction before closing
            if self._transaction_active:
                try:
                    self.rollback()
                except Exception as e:
                    logger.warning(f"Error rolling back transaction during close: {e}")

            # Close the connection - this releases any locks
            self.conn.close()
            logger.info("DuckDB connection closed.")
        except Exception as e:
            logger.error(f"Error while closing database connection: {e}")
        finally:
            # Always set conn to None to prevent reuse of closed connection
            self.conn = None
            self._transaction_active = False

    def begin_transaction(self) -> None:
        """Begin a database transaction if one is not already active."""
        if self._transaction_active:
            logger.warning("Transaction already active, not beginning a new one.")
            return
        conn = self.ensure_connection()
        conn.execute("BEGIN TRANSACTION")
        self._transaction_active = True
        logger.debug("Began new database transaction.")

    def commit(self) -> None:
        """Commit the current active transaction."""
        if not self._transaction_active:
            logger.warning("No active transaction to commit.")
            return
        if self.conn:
            self.conn.commit()
            self._transaction_active = False
            logger.debug("Database transaction committed.")
        else:  # pragma: no cover
            logger.warning("Commit called but no active or open connection.")
            self._transaction_active = False

    def rollback(self) -> None:
        """Roll back the current active transaction."""
        if not self._transaction_active:
            logger.warning("No active transaction to rollback.")
            return
        if self.conn:
            self.conn.rollback()
            self._transaction_active = False
            logger.debug("Database transaction rolled back.")
        else:  # pragma: no cover
            logger.warning("Rollback called but no active or open connection.")
            self._transaction_active = False

    def _create_fts_objects(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create FTS related table, function and triggers."""
        logger.info("Creating FTS table 'fts_main_pages'...")
        conn.execute("CREATE TABLE fts_main_pages(id VARCHAR, raw_text TEXT);")

        logger.info("Populating FTS table from existing 'pages' data...")
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                "INSERT INTO fts_main_pages (id, raw_text) SELECT id, raw_text FROM pages;",
            )
            conn.execute("COMMIT")
            logger.info("FTS table population committed successfully.")
        except duckdb.Error as tx_err:  # pragma: no cover
            conn.execute("ROLLBACK")
            logger.warning(f"Failed to populate FTS table: {tx_err}")

        fts_count_result = conn.execute(
            "SELECT COUNT(*) FROM fts_main_pages",
        ).fetchone()
        fts_count = fts_count_result[0] if fts_count_result else 0
        logger.info(
            f"FTS table 'fts_main_pages' contains {fts_count} records.",
        )

        logger.info("Creating FTS match_bm25 function...")
        conn.execute(
            """
        CREATE OR REPLACE FUNCTION fts_main_pages.match_bm25(
            doc_id VARCHAR, query VARCHAR
        )
        RETURNS DOUBLE AS
        $$
            SELECT fts_match_bm25(raw_text, query, 1.2, 0.75)
            FROM fts_main_pages
            WHERE id = doc_id
        $$;
        """,
        )
        logger.info("FTS match_bm25 function created/verified.")

        logger.info("Ensuring FTS sync triggers for 'pages' table...")
        trigger_exists_result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='fts_sync_pages'",
        ).fetchone()
        trigger_exists = trigger_exists_result is not None

        if trigger_exists:
            logger.info(
                "FTS insert sync trigger 'fts_sync_pages' already exists.",
            )
        else:
            conn.execute(
                """
            CREATE TRIGGER fts_sync_pages AFTER INSERT ON pages
            BEGIN
                INSERT INTO fts_main_pages (id, raw_text)
                VALUES (NEW.id, NEW.raw_text);
            END;
            """,
            )
            logger.info("FTS insert sync trigger 'fts_sync_pages' created.")

        conn.execute(
            """
        CREATE TRIGGER IF NOT EXISTS fts_delete_sync AFTER DELETE ON pages
        BEGIN
            DELETE FROM fts_main_pages WHERE id = OLD.id;
        END;
        """,
        )
        logger.info(
            "FTS delete sync trigger 'fts_delete_sync' created/verified.",
        )

    def ensure_tables(self) -> None:
        """Ensure all required base tables (jobs, pages) and FTS setup exist.

        Creates the 'jobs' and 'pages' tables.
        Sets up the 'fts_main_pages' table for Full-Text Search, including
        a function for BM25 matching and triggers for synchronization.
        No-op if in read-only mode.
        """
        if self.read_only:
            logger.warning("Cannot ensure tables in read-only mode.")
            return

        conn = self.ensure_connection()
        from .schema import CREATE_JOBS_TABLE_SQL, CREATE_PAGES_TABLE_SQL

        try:
            logger.info("Ensuring base tables (pages, jobs) exist...")
            conn.execute(CREATE_PAGES_TABLE_SQL)
            conn.execute(CREATE_JOBS_TABLE_SQL)
            logger.info("'pages' and 'jobs' tables created/verified.")

            logger.info("Ensuring FTS setup for 'pages' table...")
            try:
                table_exists_result = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = 'fts_main_pages'",
                ).fetchone()
                table_exists = table_exists_result[0] > 0 if table_exists_result else False

                if not table_exists:
                    self._create_fts_objects(conn)
                else:
                    logger.info(
                        "FTS table 'fts_main_pages' already exists, ensuring triggers.",
                    )
                    # Still ensure triggers exist even if table exists
                    trigger_exists_result = conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='trigger' AND name='fts_sync_pages'",
                    ).fetchone()
                    if not (trigger_exists_result is not None):
                        conn.execute(
                            """
                        CREATE TRIGGER fts_sync_pages AFTER INSERT ON pages
                        BEGIN
                            INSERT INTO fts_main_pages (id, raw_text)
                            VALUES (NEW.id, NEW.raw_text);
                        END;
                        """,
                        )
                        logger.info(
                            "FTS insert sync trigger 'fts_sync_pages' created.",
                        )
                    conn.execute(
                        """
                    CREATE TRIGGER IF NOT EXISTS fts_delete_sync AFTER DELETE ON pages
                    BEGIN
                        DELETE FROM fts_main_pages WHERE id = OLD.id;
                    END;
                    """,
                    )
                    logger.info(
                        "FTS delete sync trigger 'fts_delete_sync' created/verified.",
                    )

            except Exception as fts_err:  # pragma: no cover # noqa: BLE001
                logger.warning(
                    f"Failed to create/verify FTS setup: {fts_err}. BM25 search may be impacted.",
                )
        except Exception:  # pragma: no cover
            logger.exception("Failed to create/verify base DuckDB tables")
            raise

    def _create_hnsw_index(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create HNSW index on document_embeddings.embedding."""
        try:
            index_exists = False
            try:
                index_exists_result = conn.execute(
                    "SELECT count(*) FROM duckdb_indexes() "
                    "WHERE index_name = 'hnsw_index_on_embeddings'",
                ).fetchone()
                index_exists = index_exists_result[0] > 0 if index_exists_result else False
            except duckdb.Error:  # More specific
                logger.info(
                    "'duckdb_indexes()' function not available or errored, "
                    "will attempt to create HNSW index regardless.",
                )

            if index_exists:
                logger.info(
                    "HNSW index 'hnsw_index_on_embeddings' already exists.",
                )
            else:
                conn.execute(
                    """
                CREATE INDEX hnsw_index_on_embeddings
                ON document_embeddings
                USING HNSW (embedding)
                WITH (metric = 'cosine');
                """,
                )
                logger.info("HNSW index 'hnsw_index_on_embeddings' created.")
        except Exception as e_hnsw:  # pragma: no cover # noqa: BLE001
            logger.warning(
                f"Failed to create HNSW index: {e_hnsw}. "
                "Vector searches will work but may be slower.",
            )

    def ensure_vss_extension(self) -> None:
        """Ensure VSS extension is functional and document_embeddings table exists with HNSW index.

        Enables experimental HNSW persistence.
        Creates the 'document_embeddings' table if it doesn't exist.
        Creates an HNSW index on the 'embedding' column.
        No-op if in read-only mode.
        """
        if self.read_only:
            logger.warning("Cannot ensure VSS extension in read-only mode.")
            return

        conn = self.ensure_connection()
        from .schema import CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL

        try:
            logger.info("Configuring and verifying VSS extension...")
            try:
                conn.execute("SET hnsw_enable_experimental_persistence = true;")
                logger.info("Enabled experimental persistence for HNSW indexes.")
            except Exception:  # pragma: no cover
                logger.exception(
                    "Failed to enable HNSW persistence. "  # Removed {e_persist}
                    "HNSW indexes may not persist.",
                )

            try:
                conn.execute(
                    "SELECT array_to_string([0.1, 0.2]::FLOAT[], ', ');",
                ).fetchone()
                conn.execute(
                    "SELECT list_cosine_similarity([0.1,0.2],[0.2,0.3]);",
                ).fetchone()
                logger.info(
                    "VSS extension functionality verified "
                    "(array_to_string, list_cosine_similarity).",
                )
            except Exception:  # pragma: no cover
                logger.exception(
                    "VSS extension verification failed. "  # Removed {e_vss_verify}
                    "Vector search may fail.",
                )

            logger.info("Ensuring 'document_embeddings' table exists...")
            try:
                table_exists_result = conn.execute(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_name = 'document_embeddings'",
                ).fetchone()
                table_exists = table_exists_result[0] > 0 if table_exists_result else False

                if table_exists:
                    logger.info("'document_embeddings' table already exists.")
                else:
                    conn.execute(CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL)
                    logger.info(
                        "'document_embeddings' table created successfully.",
                    )
                    verify_table_result = conn.execute(
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_name = 'document_embeddings'",
                    ).fetchone()
                    verify_table = verify_table_result[0] > 0 if verify_table_result else False
                    if not verify_table:  # pragma: no cover
                        msg = (
                            "Failed to create 'document_embeddings' table after explicit creation."
                        )
                        raise RuntimeError(msg)  # noqa: TRY301 (Clearer here)
            except Exception:  # pragma: no cover
                logger.exception(
                    "Error creating/verifying 'document_embeddings' table",
                )
                raise

            self._create_hnsw_index(conn)

        except Exception:  # pragma: no cover
            logger.exception("Failed to set up DuckDB VSS and embeddings table")
            raise

    def initialize(self) -> None:
        """Initialize the database: creates directory, tables, and extensions.

        This is the main setup method to ensure the database is ready for use.
        No-op if in read-only mode, except for directory creation.
        """
        logger.info(
            f"Initializing DuckDBConnectionManager (read_only={self.read_only})...",
        )
        db_dir = pathlib.Path(DUCKDB_PATH).parent
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e_mkdir:  # pragma: no cover
            msg = f"Failed to create database directory {db_dir}: {e_mkdir}"
            logger.exception(msg)
            raise OSError(msg) from e_mkdir

        if not self.read_only:
            self.ensure_connection()
            self.ensure_tables()
            self.ensure_vss_extension()
        logger.info("DuckDBConnectionManager initialization complete.")

    async def async_initialize(self) -> None:
        """Asynchronously initialize the database: creates directory, tables, and extensions.

        This is the main async setup method to ensure the database is ready for use.
        No-op if in read-only mode, except for directory creation.
        """
        logger.info(
            f"Asynchronously initializing DuckDBConnectionManager (read_only={self.read_only})...",
        )
        db_dir = pathlib.Path(DUCKDB_PATH).parent
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e_mkdir:  # pragma: no cover
            msg = f"Failed to create database directory {db_dir}: {e_mkdir}"
            logger.exception(msg)
            raise OSError(msg) from e_mkdir

        if not self.read_only:
            await self.async_ensure_connection()
            self.ensure_tables()  # These methods are not async but they're fast
            self.ensure_vss_extension()
        logger.info("DuckDBConnectionManager async initialization complete.")

    def __enter__(self) -> "DuckDBConnectionManager":
        """Enter the runtime context related to this object. Ensures connection."""
        self.ensure_connection()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context related to this object. Closes connection.

        Rolls back active transaction if an exception occurred within the context.
        """
        if exc_type and self._transaction_active:
            exception_name = exc_type.__name__ if exc_type else "UnknownException"
            logger.warning(
                f"Exception '{exception_name}' occurred in context, "
                "rolling back active transaction.",
            )
            self.rollback()
        self.close()

    async def __aenter__(self) -> "DuckDBConnectionManager":
        """Enter the async runtime context related to this object. Ensures connection asynchronously."""
        await self.async_ensure_connection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async runtime context related to this object. Closes connection.

        Rolls back active transaction if an exception occurred within the context.
        """
        if exc_type and self._transaction_active:
            exception_name = exc_type.__name__ if exc_type else "UnknownException"
            logger.warning(
                f"Exception '{exception_name}' occurred in async context, "
                "rolling back active transaction.",
            )
            self.rollback()
        self.close()

    @property
    def transaction_active(self) -> bool:
        """Whether there's currently an active transaction.

        Returns:
            bool: True if a transaction is active, False otherwise.
        """
        return self._transaction_active

    def checkpoint_database(self) -> bool:
        """Perform a checkpoint operation on the database.

        This forces writing all changes from the WAL to the main database file
        and truncates the WAL file. This can be used to ensure database consistency
        before copying the database file.

        Returns:
            bool: True if checkpoint was successful, False otherwise.
        """
        if self.read_only:
            logger.warning("Cannot checkpoint a read-only database.")
            return False

        try:
            conn = self.ensure_connection()
            logger.info("Performing database checkpoint...")

            # DuckDB uses force_checkpoint instead of wal_checkpoint
            conn.execute("PRAGMA force_checkpoint")
            logger.info("Checkpoint completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during database checkpoint: {e}")
            return False
