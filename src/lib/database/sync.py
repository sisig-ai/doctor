"""Database synchronization utilities.

This module provides utilities for synchronizing the write database
to the read database, ensuring read operations have access to the
latest data without lock conflicts.
"""

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from src.common.config import (
    DUCKDB_PATH,
    DUCKDB_READ_PATH,
    DUCKDB_WRITE_PATH,
    DB_SYNC_INTERVAL_SEC,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


async def sync_write_to_read() -> bool:
    """Synchronize the write database to the read database.

    Copies the write database file to the read database file,
    ensuring it's consistent and retrying if there are lock issues.

    Returns:
        bool: True if sync was successful, False otherwise.
    """
    write_path = Path(DUCKDB_WRITE_PATH)
    read_path = Path(DUCKDB_READ_PATH)

    # If write DB doesn't exist but original does, sync from original
    if not write_path.exists() and Path(DUCKDB_PATH).exists():
        logger.info(f"Write database not found at {write_path}, using original at {DUCKDB_PATH}")
        write_path = Path(DUCKDB_PATH)

    if not write_path.exists():
        logger.warning(
            f"Write database at {write_path} does not exist, cannot sync to read database"
        )
        return False

    # Create temp file first, then move it to reduce chance of corruption
    temp_path = read_path.with_suffix(".tmp")

    try:
        # First prepare for sync by forcing a checkpoint on the write DB
        # This ensures the WAL is flushed to the main database file
        from src.lib.database import get_connection

        try:
            async with await get_connection(read_only=False) as conn_manager:
                conn = await conn_manager.async_ensure_connection()
                # Try to force a checkpoint first to consolidate WAL into main DB
                try:
                    logger.info("Forcing checkpoint on write database before sync")
                    conn.execute("PRAGMA force_checkpoint")
                    logger.info("Checkpoint completed successfully")
                except Exception as e:
                    logger.warning(f"Checkpoint failed, will copy as-is: {e}")
        except Exception as e:
            logger.warning(f"Could not get connection to force checkpoint: {e}")

        # Copy write DB to temp file
        logger.info(f"Copying write database from {write_path} to temp file {temp_path}")
        shutil.copy2(write_path, temp_path)

        # Also copy any associated WAL files if they exist
        wal_path = write_path.with_suffix(".wal")
        temp_wal_path = temp_path.with_suffix(".wal")
        if wal_path.exists():
            logger.info(f"Copying WAL file from {wal_path} to {temp_wal_path}")
            shutil.copy2(wal_path, temp_wal_path)
        else:
            logger.info(f"No WAL file found at {wal_path}, continuing without it")

        # Move temp file to read DB
        logger.info(f"Moving temp file to read database at {read_path}")
        if read_path.exists():
            read_path.unlink()
        os.rename(temp_path, read_path)

        # Move WAL file if it exists
        if temp_wal_path.exists():
            read_wal_path = read_path.with_suffix(".wal")
            if read_wal_path.exists():
                read_wal_path.unlink()
            os.rename(temp_wal_path, read_wal_path)

        logger.info(
            f"Database synchronization completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return True
    except Exception as e:
        logger.error(f"Error during database synchronization: {e}")

        # Clean up temp files
        for temp_file in [temp_path, temp_path.with_suffix(".wal")]:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as cleanup_err:
                    logger.error(f"Failed to clean up temp file {temp_file}: {cleanup_err}")
        return False


class DatabaseSyncService:
    """Service for periodically synchronizing the write database to the read database."""

    def __init__(self, sync_interval: int = DB_SYNC_INTERVAL_SEC) -> None:
        """Initialize the database sync service.

        Args:
            sync_interval: Seconds between sync operations. Defaults to DB_SYNC_INTERVAL_SEC.
        """
        self.sync_interval = sync_interval
        self._sync_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        # Track last successful sync time
        self._last_successful_sync = 0

    async def start(self) -> None:
        """Start the database sync service."""
        if self._sync_task is not None:
            logger.warning("Database sync service is already running")
            return

        logger.info(f"Starting database sync service with interval {self.sync_interval}s")
        self._stop_event.clear()
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop the database sync service."""
        if self._sync_task is None:
            logger.warning("Database sync service is not running")
            return

        logger.info("Stopping database sync service")
        self._stop_event.set()
        await self._sync_task
        self._sync_task = None
        logger.info("Database sync service stopped")

    async def _sync_loop(self) -> None:
        """Background task that periodically syncs the databases."""
        try:
            # Initial sync on start
            success = await sync_write_to_read()
            if success:
                self._last_successful_sync = time.time()
                logger.info("Initial database sync completed successfully")

            while not self._stop_event.is_set():
                # Sleep until next sync
                try:
                    await asyncio.wait_for(self._stop_event.wait(), self.sync_interval)
                    if self._stop_event.is_set():
                        break
                except asyncio.TimeoutError:
                    # Timeout means it's time to sync
                    pass

                # Perform sync
                logger.info(
                    f"Running scheduled database sync at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                success = await sync_write_to_read()

                if success:
                    self._last_successful_sync = time.time()
                    seconds_since_start = int(time.time() - self._last_successful_sync)
                    logger.info(
                        f"Sync completed successfully. Service running for {seconds_since_start}s"
                    )
                else:
                    # If sync fails, log how long it's been since last success
                    if self._last_successful_sync > 0:
                        seconds_since_sync = int(time.time() - self._last_successful_sync)
                        logger.warning(
                            f"Sync failed. Last successful sync was {seconds_since_sync}s ago"
                        )

        except asyncio.CancelledError:
            logger.info("Database sync task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in database sync loop: {e}")
            # Try to restart the sync loop after a delay
            if not self._stop_event.is_set():
                logger.info("Attempting to restart sync loop after error")
                await asyncio.sleep(5)  # Wait a bit before restarting
                asyncio.create_task(self._sync_loop())  # Create new task


# Global singleton sync service
_sync_service: Optional[DatabaseSyncService] = None


async def start_sync_service() -> None:
    """Start the global database sync service."""
    global _sync_service

    if _sync_service is None:
        _sync_service = DatabaseSyncService()

    await _sync_service.start()


async def stop_sync_service() -> None:
    """Stop the global database sync service."""
    global _sync_service

    if _sync_service is not None:
        await _sync_service.stop()
