"""Manages DuckDB database connections, setup, and low-level operations.

This module provides the DuckDBConnectionManager class, responsible for establishing
connections to a DuckDB database, loading necessary extensions (FTS, VSS),
managing database schema (tables, indexes, triggers), and handling transactions.
It is the foundational layer for database interactions in the project.
"""

import pathlib
import time  # Added for retry sleep
from types import TracebackType

import duckdb

from src.common.config import DATA_DIR, DUCKDB_PATH
from src.common.logger import get_logger
from .schema import (
    BEGIN_TRANSACTION_SQL,
    CHECK_EXTENSION_LOADED_SQL,
    CHECK_HNSW_INDEX_SQL,
    CHECK_TABLE_EXISTS_SQL,
    CHECKPOINT_SQL,
    CREATE_HNSW_INDEX_SQL,
    INSTALL_EXTENSION_SQL,
    LOAD_EXTENSION_SQL,
    SET_HNSW_PERSISTENCE_SQL,
    TEST_CONNECTION_SQL,
    VSS_ARRAY_TO_STRING_TEST_SQL,
    VSS_COSINE_SIMILARITY_TEST_SQL,
)

logger = get_logger(__name__)


class DuckDBConnectionManager:
    """Manages DuckDB connections, extensions, transactions, and schema.

    Handles the low-level details of interacting with a DuckDB database instance,
    including setting up necessary tables and extensions like FTS and VSS.
    """

    def __init__(self) -> None:
        """Initialize the DuckDBConnectionManager.

        Args:
            None.
        Returns:
            None.
        """
        self.conn: duckdb.DuckDBPyConnection | None = None
        self.transaction_active: bool = False

    def _load_extension(self, conn: duckdb.DuckDBPyConnection, ext_name: str) -> None:
        """Install and load a DuckDB extension if not already loaded.

        Args:
            conn: The active DuckDB connection.
            ext_name: The name of the extension to load (e.g., 'fts', 'vss').

        Returns:
            None.
        """
        try:
            result = conn.execute(CHECK_EXTENSION_LOADED_SQL.format(ext_name)).fetchone()
            if result:
                logger.debug(f"{ext_name.upper()} extension already loaded")
                return
        except Exception:
            pass

        try:
            conn.execute(INSTALL_EXTENSION_SQL.format(ext_name))
            conn.execute(LOAD_EXTENSION_SQL.format(ext_name))
            logger.debug(f"{ext_name.upper()} extension loaded")
        except duckdb.Error as e:
            logger.warning(f"Failed to load {ext_name.upper()} extension: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error loading {ext_name.upper()}: {e}")

    def _load_extensions(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Load required DuckDB extensions (FTS, VSS).

        Args:
            conn: The active DuckDB connection.

        Returns:
            None.
        """
        for ext in ["fts", "vss"]:
            self._load_extension(conn, ext)

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish and return a DuckDB connection.

        Args:
            None.

        Returns:
            duckdb.DuckDBPyConnection: An active DuckDB connection object.

        Raises:
            duckdb.Error: For DuckDB-specific connection failures after retries.
            Exception: For other unexpected errors during connection.
        """
        pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        db_path = pathlib.Path(DUCKDB_PATH)

        logger.debug(f"Connecting to DuckDB at {db_path}")

        for attempt in range(3):
            try:
                self.conn = duckdb.connect(str(db_path), read_only=False)
                self.conn.execute(TEST_CONNECTION_SQL).fetchone()
                self._load_extensions(self.conn)
                logger.debug(f"Connected to DuckDB at {db_path}")
                return self.conn
            except duckdb.Error as e:
                logger.warning(f"Connection attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(1)
                else:
                    self.conn = None
                    raise
            except Exception as e:
                logger.exception(f"Unexpected error connecting to DuckDB: {e}")
                self.conn = None
                raise

        raise duckdb.Error("Failed to connect after retries")

    def close(self) -> None:
        """Close the database connection if it is open.

        Args:
            None.
        Returns:
            None.
        """
        if self.conn:
            try:
                if self.transaction_active:
                    try:
                        self.conn.rollback()
                        logger.debug("Active transaction rolled back during close.")
                    except duckdb.Error as e_rollback:
                        logger.warning(
                            f"Error during DuckDB rollback on close: {e_rollback}",
                        )
                    except Exception as e_generic_rollback:  # pragma: no cover # noqa: BLE001
                        logger.warning(
                            f"Generic error during rollback on close: {e_generic_rollback}",
                        )
                    finally:
                        self.transaction_active = False

                # Force a checkpoint to ensure data is persisted before closing
                try:
                    self.conn.execute(CHECKPOINT_SQL)
                    logger.debug("Database checkpoint executed during close.")
                except Exception as e_checkpoint:
                    logger.warning(f"Error during checkpoint on close: {e_checkpoint}")

                self.conn.close()
                logger.debug("DuckDB connection closed.")
            except duckdb.Error as e_db_close:
                logger.warning(f"DuckDB error closing connection: {e_db_close}")
            except Exception as e_generic_close:  # pragma: no cover # noqa: BLE001
                logger.warning(
                    f"Generic error closing connection: {e_generic_close}",
                )
            finally:
                self.conn = None

    def ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Ensure an active database connection exists.

        Args:
            None.

        Returns:
            duckdb.DuckDBPyConnection: An active DuckDB connection object.
        """
        if self.conn:
            try:
                self.conn.execute(TEST_CONNECTION_SQL).fetchone()
                return self.conn
            except (duckdb.Error, AttributeError) as e:
                logger.warning(f"Connection unusable ({e}), reconnecting")
                self.conn = None

        return self.connect()

    def begin_transaction(self) -> None:
        """Begin a database transaction if one is not already active.

        Args:
            None.
        Returns:
            None.
        """
        if self.transaction_active:
            logger.warning("Transaction already active, not beginning a new one.")
            return
        conn = self.ensure_connection()
        conn.execute(BEGIN_TRANSACTION_SQL)
        self.transaction_active = True
        logger.debug("Began new database transaction.")

    def commit(self) -> None:
        """Commit the current active transaction.

        Args:
            None.
        Returns:
            None.
        """
        if not self.transaction_active:
            logger.warning("No active transaction to commit.")
            return
        if self.conn:
            self.conn.commit()
            self.transaction_active = False
            logger.debug("Database transaction committed.")
        else:  # pragma: no cover
            logger.warning("Commit called but no active or open connection.")
            self.transaction_active = False

    def rollback(self) -> None:
        """Roll back the current active transaction.

        Args:
            None.
        Returns:
            None.
        """
        if not self.transaction_active:
            logger.warning("No active transaction to rollback.")
            return
        if self.conn:
            self.conn.rollback()
            self.transaction_active = False
            logger.debug("Database transaction rolled back.")
        else:  # pragma: no cover
            logger.warning("Rollback called but no active or open connection.")
            self.transaction_active = False

    def ensure_tables(self) -> None:
        """Ensure all required base tables (jobs, pages) and FTS setup exist.

        Args:
            None.

        Returns:
            None.

        Args:
            None.
        Returns:
            None.
        """
        conn = self.ensure_connection()
        from .schema import CREATE_JOBS_TABLE_SQL, CREATE_PAGES_TABLE_SQL

        try:
            logger.debug("Ensuring base tables (pages, jobs) exist...")
            conn.execute(CREATE_PAGES_TABLE_SQL)
            conn.execute(CREATE_JOBS_TABLE_SQL)
            logger.debug("'pages' and 'jobs' tables created/verified.")
        except Exception:
            logger.exception("Failed to create/verify base DuckDB tables")
            raise

    def _create_hnsw_index(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create HNSW index on document_embeddings.embedding if it doesn't exist.

        Args:
            conn: The active DuckDB connection.

        Returns:
            None.
        """
        try:
            result = conn.execute(CHECK_HNSW_INDEX_SQL).fetchone()
            if result and result[0] > 0:
                logger.debug("HNSW index already exists")
                return
        except duckdb.Error:
            logger.debug("Cannot check existing indexes, attempting creation")

        try:
            conn.execute(CREATE_HNSW_INDEX_SQL)
            logger.info("Created HNSW index")
        except Exception as e:
            logger.warning(f"Failed to create HNSW index: {e}")

    def ensure_vss_extension(self) -> None:
        """Ensure VSS extension is functional and document_embeddings table exists with HNSW index.

        Args:
            None.

        Returns:
            None.

        Args:
            None.
        Returns:
            None.
        """
        conn = self.ensure_connection()
        from .schema import CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL

        try:
            logger.debug("Configuring and verifying VSS extension...")
            # Check if the VSS extension is already loaded
            self._load_extension(conn, "vss")

            try:
                conn.execute(SET_HNSW_PERSISTENCE_SQL)
                logger.debug("Enabled experimental persistence for HNSW indexes.")
            except Exception:  # pragma: no cover
                logger.exception(
                    "Failed to enable HNSW persistence. "  # Removed {e_persist}
                    "HNSW indexes may not persist.",
                )

            try:
                conn.execute(VSS_ARRAY_TO_STRING_TEST_SQL).fetchone()
                conn.execute(VSS_COSINE_SIMILARITY_TEST_SQL).fetchone()
                logger.debug(
                    "VSS extension functionality verified "
                    "(array_to_string, list_cosine_similarity).",
                )
            except Exception:  # pragma: no cover
                logger.exception(
                    "VSS extension verification failed. "  # Removed {e_vss_verify}
                    "Vector search may fail.",
                )

            logger.debug("Ensuring 'document_embeddings' table exists...")
            try:
                table_exists_result = conn.execute(
                    CHECK_TABLE_EXISTS_SQL.format("document_embeddings")
                ).fetchone()
                table_exists = table_exists_result[0] > 0 if table_exists_result else False

                if table_exists:
                    logger.debug("'document_embeddings' table already exists.")
                else:
                    conn.execute(CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL)
                    logger.debug(
                        "'document_embeddings' table created successfully.",
                    )
                    verify_table_result = conn.execute(
                        CHECK_TABLE_EXISTS_SQL.format("document_embeddings")
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

        Args:
            None.

        Returns:
            None.

        Args:
            None.
        Returns:
            None.
        """
        logger.debug("Initializing DuckDBConnectionManager...")
        # The directory creation is handled by ensure_connection -> connect
        conn = self.ensure_connection()

        self._load_extensions(conn)

        # Always ensure tables and VSS extension are set up
        self.ensure_tables()
        self.ensure_vss_extension()

        logger.debug("DuckDBConnectionManager initialization complete.")

    def __enter__(self) -> "DuckDBConnectionManager":
        """Enter the runtime context related to this object. Ensures connection.

        Args:
            None.

        Returns:
            DuckDBConnectionManager: The context manager instance.

        Args:
            None.
        Returns:
            DuckDBConnectionManager: The context manager instance.
        """
        self.ensure_connection()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context related to this object. Closes connection.

        Args:
            exc_type: Exception type if raised in context, else None.
            _exc_val: Exception value if raised in context, else None.
            _exc_tb: Traceback if exception raised, else None.

        Returns:
            None.
        """
        if exc_type and self.transaction_active:
            exception_name = exc_type.__name__ if exc_type else "UnknownException"
            logger.warning(
                f"Exception '{exception_name}' occurred in context, "
                "rolling back active transaction.",
            )
            self.rollback()
        self.close()
