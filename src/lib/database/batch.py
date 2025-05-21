"""Batch operation utilities for database operations.

This module provides batch operation capabilities for the database, allowing
for efficient bulk inserts, updates, and other operations. It reduces transaction
overhead and improves performance for large operations.
"""

import datetime
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from src.common.logger import get_logger
from .connection import DuckDBConnectionManager
from .connection_pool import get_connection
from .utils import serialize_tags

logger = get_logger(__name__)


class BatchOperation:
    """Base class for batch database operations.

    This abstract class defines the interface for batch operations.
    Subclasses must implement the execute method.
    """

    async def execute(self, conn_manager: DuckDBConnectionManager) -> None:
        """Execute the batch operation.

        Args:
            conn_manager: The connection manager to use for the operation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute()")


class BatchPageInsert(BatchOperation):
    """Batch operation for inserting multiple pages at once."""

    def __init__(self) -> None:
        """Initialize an empty batch of pages to insert."""
        self.pages: List[Dict[str, Any]] = []
        self.job_ids: Set[str] = set()

    def add_page(
        self,
        url: str,
        text: str,
        job_id: str,
        tags: Optional[List[str]] = None,
        page_id: Optional[str] = None,
    ) -> str:
        """Add a page to the batch.

        Args:
            url: The URL of the page.
            text: The extracted text content of the page.
            job_id: The ID of the crawl job this page belongs to.
            tags: Optional list of tags to associate with the page.
            page_id: Optional ID for the page. If None, a UUID will be generated.

        Returns:
            The ID of the page (either provided or generated).
        """
        current_page_id = page_id if page_id is not None else str(uuid.uuid4())
        domain = urlparse(url).netloc

        self.pages.append(
            {
                "id": current_page_id,
                "url": url,
                "domain": domain,
                "raw_text": text,
                "crawl_date": datetime.datetime.now(datetime.UTC),
                "tags": serialize_tags(tags),
                "job_id": job_id,
            }
        )
        self.job_ids.add(job_id)

        return current_page_id

    async def execute(self, conn_manager: DuckDBConnectionManager) -> None:
        """Execute the batch page insert.

        Args:
            conn_manager: The connection manager to use for the operation.

        Raises:
            RuntimeError: If the batch insert fails.
        """
        if not self.pages:
            logger.debug("Skipping batch page insert - no pages to insert")
            return

        logger.info(f"Executing batch insert of {len(self.pages)} pages")

        try:
            conn = await conn_manager.async_ensure_connection()
            conn_manager.begin_transaction()

            # Instead of using zip function, insert rows one by one
            # This is less efficient but more compatible with different DuckDB versions
            for page in self.pages:
                conn.execute(
                    """
                    INSERT INTO pages (id, url, domain, raw_text, crawl_date, tags, job_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        page["id"],
                        page["url"],
                        page["domain"],
                        page["raw_text"],
                        page["crawl_date"],
                        page["tags"],
                        page["job_id"],
                    ],
                )

            conn_manager.commit()
            logger.info(f"Successfully inserted {len(self.pages)} pages in batch")
        except Exception as e:
            logger.exception(f"Error during batch page insert: {e}")
            conn_manager.rollback()
            raise RuntimeError(f"Batch page insert failed: {e}") from e


class BatchJobUpdate(BatchOperation):
    """Batch operation for updating job status information."""

    def __init__(self) -> None:
        """Initialize an empty batch of job updates."""
        self.updates: Dict[str, Dict[str, Any]] = {}

    def add_job_update(
        self,
        job_id: str,
        status: Optional[str] = None,
        pages_discovered: Optional[int] = None,
        pages_crawled: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Add a job update to the batch.

        Args:
            job_id: The ID of the job to update.
            status: Optional new status of the job.
            pages_discovered: Optional number of pages discovered.
            pages_crawled: Optional number of pages crawled.
            error_message: Optional error message if the job failed.
        """
        # Create the job entry if it doesn't exist
        if job_id not in self.updates:
            self.updates[job_id] = {"updated_at": datetime.datetime.now(datetime.UTC)}

        # Add only the values that are specified
        if status is not None:
            self.updates[job_id]["status"] = status
        if pages_discovered is not None:
            self.updates[job_id]["pages_discovered"] = pages_discovered
        if pages_crawled is not None:
            self.updates[job_id]["pages_crawled"] = pages_crawled
        if error_message is not None:
            self.updates[job_id]["error_message"] = error_message

    async def execute(self, conn_manager: DuckDBConnectionManager) -> None:
        """Execute the batch job updates.

        Args:
            conn_manager: The connection manager to use for the operation.

        Raises:
            RuntimeError: If the batch update fails.
        """
        if not self.updates:
            logger.debug("Skipping batch job update - no updates to perform")
            return

        logger.info(f"Executing batch update of {len(self.updates)} jobs")

        try:
            conn = await conn_manager.async_ensure_connection()
            conn_manager.begin_transaction()

            for job_id, updates in self.updates.items():
                # Construct dynamic SQL
                set_clauses = []
                params = []

                for key, value in updates.items():
                    set_clauses.append(f"{key} = ?")
                    params.append(value)

                sql = f"UPDATE jobs SET {', '.join(set_clauses)} WHERE job_id = ?"
                params.append(job_id)

                # Execute the update
                conn.execute(sql, params)

            conn_manager.commit()
            logger.info(f"Successfully updated {len(self.updates)} jobs in batch")
        except Exception as e:
            logger.exception(f"Error during batch job update: {e}")
            conn_manager.rollback()
            raise RuntimeError(f"Batch job update failed: {e}") from e


class BatchExecutor:
    """Executor for batch database operations.

    Provides methods to execute batches of operations efficiently,
    with proper transaction handling and checkpoint management.
    """

    def __init__(self, max_batch_size: int = 100, checkpoint_after: bool = True) -> None:
        """Initialize the batch executor.

        Args:
            max_batch_size: Maximum number of items to process in a single batch.
            checkpoint_after: Whether to force a checkpoint after executing batches.
        """
        self.max_batch_size = max_batch_size
        self.checkpoint_after = checkpoint_after

    async def execute_batch(self, operation: BatchOperation) -> None:
        """Execute a single batch operation.

        Args:
            operation: The batch operation to execute.

        Raises:
            RuntimeError: If the batch operation fails.
        """
        # Only acquire connection when the operation is ready to be executed
        async with await get_connection(read_only=False) as conn_manager:
            await operation.execute(conn_manager)

            if self.checkpoint_after:
                logger.info("Forcing checkpoint after batch operation")
                conn = await conn_manager.async_ensure_connection()
                conn.execute("PRAGMA force_checkpoint")

    async def execute_batches(self, operations: List[BatchOperation]) -> None:
        """Execute multiple batch operations.

        Each operation is executed in its own transaction.

        Args:
            operations: List of batch operations to execute.

        Raises:
            RuntimeError: If any batch operation fails.
        """
        for operation in operations:
            await self.execute_batch(operation)


# Convenience functions for common batch operations


async def batch_store_pages(
    pages: List[Tuple[str, str, str, Optional[List[str]], Optional[str]]],
    max_batch_size: int = 100,
) -> List[str]:
    """Store multiple pages in batches.

    Args:
        pages: List of tuples (url, text, job_id, tags, page_id)
        max_batch_size: Maximum number of pages per batch.

    Returns:
        List of page IDs (either provided or generated).

    Raises:
        RuntimeError: If the batch operation fails.
    """
    page_ids = []
    current_batch = BatchPageInsert()
    batches = []

    for url, text, job_id, tags, page_id in pages:
        if len(current_batch.pages) >= max_batch_size:
            batches.append(current_batch)
            current_batch = BatchPageInsert()

        page_id = current_batch.add_page(url, text, job_id, tags, page_id)
        page_ids.append(page_id)

    if current_batch.pages:
        batches.append(current_batch)

    executor = BatchExecutor(max_batch_size)
    await executor.execute_batches(batches)

    return page_ids
