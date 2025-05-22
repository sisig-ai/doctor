"""High-level database operations for the Doctor project.

This module provides the `DatabaseOperations` class, which serves as the primary
interface for interacting with the database. It abstracts away the lower-level
connection and schema management details, offering methods for common tasks like
storing crawled pages, updating job statuses, and managing checkpoints.

It relies on `DuckDBConnectionManager` for actual database communication and includes
several methods and properties for backward compatibility with older versions of the
original `Database` class, primarily to support existing tests.
"""

import asyncio
import datetime
import uuid
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import duckdb  # For type hinting duckdb.DuckDBPyConnection

from src.common.logger import get_logger

from .connection import DuckDBConnectionManager
from .schema import CHECKPOINT_SQL, INSERT_PAGE_SQL, UPDATE_JOB_STATUS_BASE_SQL

logger = get_logger(__name__)


class DatabaseOperations:
    """Provides a high-level API for database operations.

    This class encapsulates common database interactions such as storing pages,
    updating job statuses, and managing database connections and schemas through
    an underlying DuckDBConnectionManager.

    It also includes several methods and properties for backward compatibility
    with older versions of the Database class, primarily for testing purposes.
    These compatibility layers may be refactored or removed in future versions.
    """

    def __init__(self, *, read_only: bool = False) -> None:
        """Initialize the DatabaseOperations instance.

        Args:
            read_only: If True, the underlying database connection will be
                opened in read-only mode. Defaults to False.

        """
        self.db: DuckDBConnectionManager = DuckDBConnectionManager(read_only=read_only)
        self._write_lock: asyncio.Lock = asyncio.Lock()
        self.db.initialize()  # Ensure tables and extensions are set up

    def __enter__(self) -> "DatabaseOperations":
        """Enter the context manager, ensuring a database connection.

        Returns:
            Self for use in a with statement.
        """
        self.db.ensure_connection()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager, closing the database connection.

        Args:
            exc_type: The exception type if an exception was raised in the with block, else None.
            exc_val: The exception value if an exception was raised, else None.
            exc_tb: The exception traceback if an exception was raised, else None.
        """
        self.db.close()

    async def store_page(
        self,
        url: str,
        text: str,
        job_id: str,
        tags: list[str] | None = None,
        page_id: str | None = None,
    ) -> str:
        """Store a crawled page in the database.

        Args:
            url: The URL of the page.
            text: The extracted text content of the page.
            job_id: The ID of the crawl job this page belongs to.
            tags: Optional list of tags to associate with the page.
            page_id: Optional ID for the page. If None, a UUID will be generated.

        Returns:
            The ID of the stored page (either provided or generated).

        Raises:
            IOError: If an I/O error occurs during database operations.
            duckdb.Error: For DuckDB specific errors.
            RuntimeError: For other unexpected errors during the storage process.

        """
        if tags is None:
            tags = []
        current_page_id: str = page_id if page_id is not None else str(uuid.uuid4())
        domain: str = urlparse(url).netloc

        logger.debug(
            f"Storing page {current_page_id} from {url} with {len(text)} characters.",
        )

        async with self._write_lock:
            conn = self.db.ensure_connection()
            try:
                await asyncio.to_thread(self.db.begin_transaction)
                from .utils import serialize_tags

                await asyncio.to_thread(
                    conn.execute,
                    INSERT_PAGE_SQL,
                    (
                        current_page_id,
                        url,
                        domain,
                        text,
                        datetime.datetime.now(datetime.UTC),
                        serialize_tags(tags),
                        job_id,
                    ),
                )
                await asyncio.to_thread(self.db.commit)
            except duckdb.Error:
                logger.exception(
                    f"DuckDB error storing page {current_page_id} for URL {url}",
                )
                await asyncio.to_thread(self.db.rollback)
                if conn and hasattr(conn, "rollback") and self.db.transaction_active:
                    try:  # pragma: no cover
                        logger.debug(
                            "Attempting direct rollback on connection object due to DB error.",
                        )
                        conn.rollback()
                    except Exception as direct_rollback_err:  # pragma: no cover
                        logger.warning(
                            f"Direct rollback on connection failed: {direct_rollback_err}",
                        )
                raise
            except Exception as e_generic:  # pragma: no cover
                msg = f"Unexpected error storing page {current_page_id} for URL {url}"
                logger.exception(msg)
                await asyncio.to_thread(self.db.rollback)
                raise RuntimeError(msg) from e_generic  # Use RuntimeError for generic
            else:
                logger.debug(
                    f"Successfully stored page {current_page_id} in database.",
                )
                return current_page_id

    def _build_update_job_query(
        self,
        job_id: str,
        status: str,
        pages_discovered: int | None,
        pages_crawled: int | None,
        error_message: str | None,
    ) -> tuple[str, list[Any]]:
        """Build dynamic SQL query and parameters for updating a job."""
        query_parts = [UPDATE_JOB_STATUS_BASE_SQL]
        params: list[Any] = [
            status,
            datetime.datetime.now(datetime.UTC),
        ]
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
        return query, params

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        pages_discovered: int | None = None,
        pages_crawled: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status and other metadata of a crawl job in the database.

        Args:
            job_id: The ID of the job to update.
            status: The new status of the job (e.g., "running", "completed", "failed").
            pages_discovered: Optional number of pages discovered.
            pages_crawled: Optional number of pages crawled.
            error_message: Optional error message if the job failed.

        Raises:
            IOError: If an I/O error occurs during database operations.
            duckdb.Error: For DuckDB specific errors.
            RuntimeError: For other unexpected errors during the update.

        """
        logger.info(
            f"Updating job {job_id}: status='{status}', discovered={pages_discovered}, "
            f"crawled={pages_crawled}, error='{error_message is not None}'",
        )
        update_successful = False
        async with self._write_lock:
            conn = self.db.ensure_connection()
            try:
                await asyncio.to_thread(self.db.begin_transaction)

                query, params_tuple = self._build_update_job_query(
                    job_id,
                    status,
                    pages_discovered,
                    pages_crawled,
                    error_message,
                )

                cursor = await asyncio.to_thread(conn.execute, query, params_tuple)
                rows_affected = cursor.rowcount if cursor else 0

                if rows_affected == 0:
                    logger.warning(
                        f"Job status update for {job_id} affected 0 rows. Job may not exist.",
                    )
                else:
                    logger.debug(
                        f"Job status update for {job_id} affected {rows_affected} rows.",
                    )
                    update_successful = True
                await asyncio.to_thread(self.db.commit)
            except duckdb.Error:
                logger.exception(
                    f"DuckDB error updating job {job_id} to status {status}",
                )
                await asyncio.to_thread(self.db.rollback)
                if conn and hasattr(conn, "rollback") and self.db.transaction_active:
                    try:  # pragma: no cover
                        logger.debug(
                            "Attempting direct rollback on connection object due to DB error.",
                        )
                        conn.rollback()
                    except Exception as direct_rollback_err:  # pragma: no cover
                        logger.warning(
                            f"Direct rollback on connection failed: {direct_rollback_err}",
                        )
                raise
            except Exception as e_generic:  # pragma: no cover
                msg = f"Unexpected error updating job {job_id} to status {status}"
                logger.exception(msg)
                await asyncio.to_thread(self.db.rollback)
                raise RuntimeError(msg) from e_generic  # Use RuntimeError
            else:
                if update_successful and status in ["completed", "failed"]:
                    await self.checkpoint_async()
                    logger.info(
                        f"Job {job_id} successfully updated to {status} and checkpointed.",
                    )
                elif update_successful:
                    logger.info(f"Job {job_id} successfully updated to {status}.")

    async def checkpoint_async(self) -> None:
        """Force a database checkpoint to ensure changes are persisted.

        This operation is typically called after critical updates, such as when
        a job status changes to "completed" or "failed".
        """
        logger.info("Attempting to force database checkpoint...")
        async with self._write_lock:
            conn = self.db.ensure_connection()
            try:
                await asyncio.to_thread(conn.execute, CHECKPOINT_SQL)
                logger.info("Database checkpoint successful.")
            except duckdb.Error:  # pragma: no cover
                logger.exception("Failed to force database checkpoint due to DB error")
            except Exception:  # pragma: no cover
                logger.exception("Unexpected error during database checkpoint")
