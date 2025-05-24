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
        """Initialize the DatabaseOperations instance.

        Args:
            None.
        Returns:
            None.
        """
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
        parent_page_id: str | None = None,
        root_page_id: str | None = None,
        depth: int = 0,
        path: str | None = None,
        title: str | None = None,
    ) -> str:
        """Store a crawled page in the database.

        Args:
            url: The URL of the page.
            text: The extracted text content of the page.
            job_id: The ID of the crawl job this page belongs to.
            tags: Optional list of tags to associate with the page.
            page_id: Optional ID for the page. If None, a UUID will be generated.
            parent_page_id: Optional ID of the parent page in the hierarchy.
            root_page_id: Optional ID of the root page of the site.
            depth: Distance from the root page (default 0).
            path: Relative path from the root page.
            title: Extracted page title.

        Returns:
            str: The ID of the stored page (either provided or generated).

        Raises:
            duckdb.Error: For DuckDB specific errors.
            RuntimeError: For other unexpected errors during the storage process.
        """
        page_id = page_id or str(uuid.uuid4())
        domain = urlparse(url).netloc
        tags = tags or []

        # If this is a root page (no parent), set root_page_id to self
        if parent_page_id is None and root_page_id is None:
            root_page_id = page_id

        logger.debug(f"Storing page {page_id} from {url}")

        async with self._write_lock:
            with self.db as conn_manager:
                conn = conn_manager.conn
                if not conn:
                    raise RuntimeError("Failed to obtain database connection")

                try:
                    await asyncio.to_thread(conn_manager.begin_transaction)
                    from .utils import serialize_tags

                    await asyncio.to_thread(
                        conn.execute,
                        INSERT_PAGE_SQL,
                        (
                            page_id,
                            url,
                            domain,
                            text,
                            datetime.datetime.now(datetime.UTC),
                            serialize_tags(tags),
                            job_id,
                            parent_page_id,
                            root_page_id,
                            depth,
                            path,
                            title,
                        ),
                    )
                    await asyncio.to_thread(conn_manager.commit)
                    logger.debug(f"Stored page {page_id}")
                    return page_id
                except duckdb.Error:
                    logger.exception(f"DuckDB error storing page {page_id}")
                    await asyncio.to_thread(conn_manager.rollback)
                    raise
                except Exception as e:
                    logger.exception(f"Error storing page {page_id}: {e}")
                    await asyncio.to_thread(conn_manager.rollback)
                    raise RuntimeError(f"Failed to store page {page_id}") from e

    def _build_update_job_query(
        self,
        job_id: str,
        status: str,
        pages_discovered: int | None,
        pages_crawled: int | None,
        error_message: str | None,
    ) -> tuple[str, list[Any]]:
        """Build dynamic SQL query and parameters for updating a job.

        Args:
            job_id: The ID of the job to update.
            status: The new status of the job.
            pages_discovered: Optional number of pages discovered.
            pages_crawled: Optional number of pages crawled.
            error_message: Optional error message if the job failed.

        Returns:
            tuple[str, list[Any]]: The SQL query and parameters for updating the job.
        """
        query_parts = [UPDATE_JOB_STATUS_BASE_SQL]
        params = [status, datetime.datetime.now(datetime.UTC)]

        for field, value in [
            ("pages_discovered", pages_discovered),
            ("pages_crawled", pages_crawled),
            ("error_message", error_message),
        ]:
            if value is not None:
                query_parts.append(f"{field} = ?")
                params.append(value)

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
        """Update the status and other metadata of a crawl job.

        Args:
            job_id: The ID of the job to update.
            status: The new status of the job (e.g., "running", "completed", "failed").
            pages_discovered: Optional number of pages discovered.
            pages_crawled: Optional number of pages crawled.
            error_message: Optional error message if the job failed.

        Returns:
            None.

        Raises:
            duckdb.Error: For DuckDB specific errors.
            RuntimeError: For other unexpected errors during the update.
        """
        logger.info(f"Updating job {job_id} to status '{status}'")

        async with self._write_lock:
            with self.db as conn_manager:
                conn = conn_manager.conn
                if not conn:
                    raise RuntimeError("Failed to obtain database connection")

                try:
                    await asyncio.to_thread(conn_manager.begin_transaction)

                    query, params = self._build_update_job_query(
                        job_id, status, pages_discovered, pages_crawled, error_message
                    )

                    cursor = await asyncio.to_thread(conn.execute, query, params)
                    rows_affected = cursor.rowcount if cursor else 0

                    if rows_affected == 0:
                        logger.warning(f"Job {job_id} not found for update")
                    else:
                        logger.debug(f"Updated job {job_id}, {rows_affected} rows affected")

                    await asyncio.to_thread(conn_manager.commit)

                    if rows_affected > 0 and status in ["completed", "failed"]:
                        await self._checkpoint_internal_async()
                        logger.info(f"Job {job_id} updated to {status} and checkpointed")

                except duckdb.Error:
                    logger.exception(f"DuckDB error updating job {job_id}")
                    await asyncio.to_thread(conn_manager.rollback)
                    raise
                except Exception as e:
                    logger.exception(f"Error updating job {job_id}: {e}")
                    await asyncio.to_thread(conn_manager.rollback)
                    raise RuntimeError(f"Failed to update job {job_id}") from e

    async def _checkpoint_internal_async(self) -> None:
        """Internal method to force a database checkpoint.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If the database connection cannot be obtained.
            Exception: For unexpected errors during the checkpoint.
        """
        logger.debug("Executing database checkpoint")
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection for checkpoint")
            try:
                await asyncio.to_thread(conn.execute, CHECKPOINT_SQL)
                logger.debug("Database checkpoint successful")
            except Exception as e:
                logger.exception(f"Database checkpoint failed: {e}")
                raise

    async def checkpoint_async(self) -> None:
        """Force a database checkpoint to ensure changes are persisted.

        Args:
            None.

        Returns:
            None.
        """
        logger.info("Forcing database checkpoint")
        async with self._write_lock:
            await self._checkpoint_internal_async()

    async def get_page_by_url(self, url: str) -> dict[str, Any] | None:
        """Get a page by its URL.

        Args:
            url: The URL of the page to retrieve.

        Returns:
            dict[str, Any] | None: The page data if found, None otherwise.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                result = await asyncio.to_thread(
                    conn.execute, "SELECT * FROM pages WHERE url = ?", [url]
                )
                row = result.fetchone()
                if row:
                    columns = [desc[0] for desc in result.description]
                    return dict(zip(columns, row))
                return None
            except Exception as e:
                logger.error(f"Error getting page by URL {url}: {e}")
                raise

    async def get_root_pages(self) -> list[dict[str, Any]]:
        """Get all root pages (pages with no parent) that have proper hierarchy tracking.

        This excludes legacy pages that don't have hierarchy information.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: List of root page records with hierarchy.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                result = await asyncio.to_thread(
                    conn.execute,
                    """SELECT * FROM pages
                       WHERE parent_page_id IS NULL
                       AND root_page_id IS NOT NULL
                       AND root_page_id = id
                       AND (depth IS NOT NULL AND depth >= 0)
                       ORDER BY crawl_date DESC""",
                )
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error(f"Error getting root pages: {e}")
                raise

    async def get_page_hierarchy(self, root_page_id: str) -> list[dict[str, Any]]:
        """Get all pages in a hierarchy starting from a root page.

        Args:
            root_page_id: The ID of the root page.

        Returns:
            list[dict[str, Any]]: List of all pages in the hierarchy.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                result = await asyncio.to_thread(
                    conn.execute,
                    """SELECT * FROM pages
                       WHERE root_page_id = ?
                       ORDER BY depth, path""",
                    [root_page_id],
                )
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error(f"Error getting page hierarchy for {root_page_id}: {e}")
                raise

    async def get_child_pages(self, parent_page_id: str) -> list[dict[str, Any]]:
        """Get direct child pages of a parent page.

        Args:
            parent_page_id: The ID of the parent page.

        Returns:
            list[dict[str, Any]]: List of child page records.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                result = await asyncio.to_thread(
                    conn.execute,
                    """SELECT * FROM pages
                       WHERE parent_page_id = ?
                       ORDER BY title""",
                    [parent_page_id],
                )
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error(f"Error getting child pages for {parent_page_id}: {e}")
                raise

    async def get_sibling_pages(self, page_id: str) -> list[dict[str, Any]]:
        """Get sibling pages (pages with the same parent).

        Args:
            page_id: The ID of the page whose siblings to find.

        Returns:
            list[dict[str, Any]]: List of sibling page records.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                # First get the parent_page_id of the given page
                result = await asyncio.to_thread(
                    conn.execute, "SELECT parent_page_id FROM pages WHERE id = ?", [page_id]
                )
                row = result.fetchone()
                if not row or row[0] is None:
                    return []  # No parent means no siblings

                parent_page_id = row[0]

                # Get all pages with the same parent
                result = await asyncio.to_thread(
                    conn.execute,
                    """SELECT * FROM pages
                       WHERE parent_page_id = ? AND id != ?
                       ORDER BY title""",
                    [parent_page_id, page_id],
                )
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error(f"Error getting sibling pages for {page_id}: {e}")
                raise

    async def get_page_by_id(self, page_id: str) -> dict[str, Any] | None:
        """Get a page by its ID.

        Args:
            page_id: The ID of the page to retrieve.

        Returns:
            dict[str, Any] | None: The page data if found, None otherwise.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                result = await asyncio.to_thread(
                    conn.execute, "SELECT * FROM pages WHERE id = ?", [page_id]
                )
                row = result.fetchone()
                if row:
                    columns = [desc[0] for desc in result.description]
                    return dict(zip(columns, row))
                return None
            except Exception as e:
                logger.error(f"Error getting page by ID {page_id}: {e}")
                raise

    async def get_legacy_pages(self) -> list[dict[str, Any]]:
        """Get all legacy pages (pages without proper hierarchy information).

        A legacy page is one that doesn't have complete hierarchy tracking -
        specifically pages that have no root_page_id or NULL depth, indicating
        they were crawled before hierarchy tracking was implemented.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: List of legacy page records.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                result = await asyncio.to_thread(
                    conn.execute,
                    """SELECT * FROM pages
                       WHERE root_page_id IS NULL
                          OR depth IS NULL
                       ORDER BY domain, url""",
                )
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error(f"Error getting legacy pages: {e}")
                raise

    async def get_pages_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Get all pages for a specific domain.

        Args:
            domain: The domain to filter by.

        Returns:
            list[dict[str, Any]]: List of page records for the domain.
        """
        with self.db as conn_manager:
            conn = conn_manager.conn
            if not conn:
                raise RuntimeError("Failed to obtain database connection")

            try:
                result = await asyncio.to_thread(
                    conn.execute,
                    """SELECT * FROM pages
                       WHERE domain = ?
                       ORDER BY url""",
                    [domain],
                )
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error(f"Error getting pages for domain {domain}: {e}")
                raise
