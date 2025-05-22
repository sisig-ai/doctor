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
from typing import Any
from urllib.parse import urlparse

import duckdb  # For type hinting duckdb.DuckDBPyConnection

from src.common.logger import get_logger

from .connection import DuckDBConnectionManager
from .schema import (
    CHECKPOINT_SQL,
    INSERT_PAGE_SQL,
    UPDATE_JOB_STATUS_BASE_SQL,
)

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

    def __init__(self) -> None:
        """Initialize the DatabaseOperations instance."""
        self.db: DuckDBConnectionManager = DuckDBConnectionManager()
        self._write_lock: asyncio.Lock = asyncio.Lock()
        # Initialize the database (ensures tables/extensions exist)
        # This will open and close a connection once.
        with self.db as conn_manager:
            conn_manager.initialize()

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
            with self.db as conn_manager:  # DuckDBConnectionManager is now a context manager
                actual_conn = conn_manager.conn  # Get the actual DuckDBPyConnection
                if not actual_conn:  # Should not happen if __enter__ works
                    raise RuntimeError("Failed to obtain database connection from manager.")
                try:
                    await asyncio.to_thread(conn_manager.begin_transaction)
                    from .utils import serialize_tags

                    await asyncio.to_thread(
                        actual_conn.execute,
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
                    await asyncio.to_thread(conn_manager.commit)
                except duckdb.Error:
                    logger.exception(
                        f"DuckDB error storing page {current_page_id} for URL {url}",
                    )
                    await asyncio.to_thread(conn_manager.rollback)
                    # The conn_manager handles its own connection's state during rollback
                    raise
                except Exception as e_generic:  # pragma: no cover
                    msg = f"Unexpected error storing page {current_page_id} for URL {url}"
                    logger.exception(msg)
                    await asyncio.to_thread(conn_manager.rollback)
                    raise RuntimeError(msg) from e_generic
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
            with self.db as conn_manager:  # DuckDBConnectionManager is now a context manager
                actual_conn = conn_manager.conn
                if not actual_conn:
                    raise RuntimeError("Failed to obtain database connection from manager.")
                try:
                    await asyncio.to_thread(conn_manager.begin_transaction)

                    query, params_tuple = self._build_update_job_query(
                        job_id,
                        status,
                        pages_discovered,
                        pages_crawled,
                        error_message,
                    )

                    cursor = await asyncio.to_thread(actual_conn.execute, query, params_tuple)
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
                    await asyncio.to_thread(conn_manager.commit)
                except duckdb.Error:
                    logger.exception(
                        f"DuckDB error updating job {job_id} to status {status}",
                    )
                    await asyncio.to_thread(conn_manager.rollback)
                    raise
                except Exception as e_generic:  # pragma: no cover
                    msg = f"Unexpected error updating job {job_id} to status {status}"
                    logger.exception(msg)
                    await asyncio.to_thread(conn_manager.rollback)
                    raise RuntimeError(msg) from e_generic
                else:
                    if update_successful and status in ["completed", "failed"]:
                        # Call checkpoint_async which will handle its own connection context
                        await self._checkpoint_internal_async()
                        logger.info(
                            f"Job {job_id} successfully updated to {status} and checkpointed.",
                        )
                    elif update_successful:
                        logger.info(f"Job {job_id} successfully updated to {status}.")

    async def _checkpoint_internal_async(self) -> None:
        """Internal method to force a database checkpoint. Assumes lock may be held."""
        logger.debug("Attempting to force database checkpoint (internal)...")
        # This method is called from within an existing write_lock context if called
        # after a successful job update. If called directly, it needs its own context.
        # For simplicity, we'll assume the lock is managed by the caller or not strictly needed
        # if this were to be public, but since it's an internal helper for now:
        with self.db as conn_manager:
            actual_conn = conn_manager.conn
            if not actual_conn:
                raise RuntimeError("Failed to obtain database connection for checkpoint.")
            try:
                await asyncio.to_thread(actual_conn.execute, CHECKPOINT_SQL)
                logger.info("Database checkpoint successful (internal).")
            except duckdb.Error as e:  # pragma: no cover
                logger.exception(f"Failed to force database checkpoint due to DB error: {e}")
                raise
            except Exception as e:  # pragma: no cover
                logger.exception(f"Unexpected error during database checkpoint: {e}")
                raise

    async def checkpoint_async(self) -> None:
        """Force a database checkpoint to ensure changes are persisted.

        This operation is typically called after critical updates, such as when
        a job status changes to "completed" or "failed".
        This is a public method and will acquire the write lock.
        """
        logger.info("Attempting to force database checkpoint (public)...")
        async with self._write_lock:
            await self._checkpoint_internal_async()
