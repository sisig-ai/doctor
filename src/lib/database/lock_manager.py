"""File-based lock manager for DuckDB access coordination.

This module provides a file-based locking mechanism to coordinate database access
between the web service (read-only) and crawler (read-write) processes.

It uses a simple file lock approach to prevent the concurrency conflicts that
DuckDB experiences when different processes try to access the database in different modes.
"""

import asyncio
import os
import fcntl
import pathlib
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Optional

from src.common.config import DATA_DIR
from src.common.logger import get_logger

logger = get_logger(__name__)

# Lock file paths
WRITE_LOCK_FILE = os.path.join(DATA_DIR, "duckdb_write.lock")
READ_LOCKS_DIR = os.path.join(DATA_DIR, "read_locks")


def ensure_lock_directories() -> None:
    """Ensure the lock directories exist.

    This creates the lock files directory if it doesn't exist.
    """
    # Ensure the data directory exists
    pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Ensure the read locks directory exists
    pathlib.Path(READ_LOCKS_DIR).mkdir(parents=True, exist_ok=True)


@contextmanager
def file_lock(lock_file: str, mode: str = "w") -> None:
    """A context manager for file-based locking.

    This uses fcntl to acquire an exclusive lock on a file, ensuring
    no other process can acquire the same lock.

    Args:
        lock_file: Path to the lock file
        mode: File open mode ('r' for read, 'w' for write)
    """
    ensure_lock_directories()

    lock_fd = None
    try:
        # Create lock file if it doesn't exist
        if not os.path.exists(lock_file):
            open(lock_file, "w").close()

        # Open the lock file
        lock_fd = open(lock_file, mode)

        # Try to acquire the lock (non-blocking)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Yield control back to the caller
        yield
    except IOError:
        # Lock couldn't be acquired
        if lock_fd:
            lock_fd.close()
        raise
    finally:
        # Release the lock
        if lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


class DatabaseLockManager:
    """Manages database access locks between processes.

    This class provides methods to acquire and release read and write locks
    for the database, ensuring that we never have concurrency conflicts.
    """

    def __init__(self) -> None:
        """Initialize the database lock manager."""
        self.process_id = os.getpid()
        self.read_lock_file = os.path.join(READ_LOCKS_DIR, f"read_{self.process_id}.lock")
        self.read_lock_fd: Optional[int] = None
        self.write_lock_fd: Optional[int] = None

        # Ensure the lock directories exist
        ensure_lock_directories()

    @contextmanager
    def acquire_read_lock(self, wait_timeout: float = 5.0) -> None:
        """Acquire a read lock on the database.

        This will wait for any write lock to be released before acquiring
        the read lock. Multiple processes can hold read locks simultaneously.

        Args:
            wait_timeout: Time to wait for a write lock to be released (seconds)

        Raises:
            TimeoutError: If we couldn't acquire the lock within the timeout period
        """
        # Check if any process has a write lock
        wait_start = time.time()
        while os.path.exists(WRITE_LOCK_FILE):
            # Check if we've waited too long
            if time.time() - wait_start > wait_timeout:
                raise TimeoutError("Timed out waiting for write lock to be released")

            # Wait for the write lock to be released
            logger.info("Waiting for database write lock to be released...")
            time.sleep(0.5)

        # Now we can acquire a read lock
        try:
            with file_lock(self.read_lock_file, "w"):
                # We have the read lock
                logger.info(f"Acquired read lock for process {self.process_id}")
                yield
        finally:
            logger.info(f"Released read lock for process {self.process_id}")
            # Clean up the lock file
            if os.path.exists(self.read_lock_file):
                try:
                    os.remove(self.read_lock_file)
                except OSError:
                    pass

    @contextmanager
    def acquire_write_lock(self, wait_timeout: float = 10.0) -> None:
        """Acquire a write lock on the database.

        This will wait for any read locks to be released before acquiring
        the write lock. Only one process can hold a write lock at a time.

        Args:
            wait_timeout: Time to wait for read locks to be released (seconds)

        Raises:
            TimeoutError: If we couldn't acquire the lock within the timeout period
        """
        # First, try to acquire the global write lock
        try:
            # If we can't acquire the write lock, it means another process has it
            with file_lock(WRITE_LOCK_FILE, "w"):
                # Now check if any processes have read locks
                wait_start = time.time()
                while os.listdir(READ_LOCKS_DIR):
                    # Check if we've waited too long
                    if time.time() - wait_start > wait_timeout:
                        raise TimeoutError("Timed out waiting for read locks to be released")

                    # Wait for read locks to be released
                    logger.info("Waiting for database read locks to be released...")
                    time.sleep(0.5)

                # We have the write lock and no read locks exist
                logger.info(f"Acquired write lock for process {self.process_id}")
                yield
        finally:
            logger.info(f"Released write lock for process {self.process_id}")


# Async versions of the lock manager methods for use with asyncio


class AsyncDatabaseLockManager:
    """Async version of the DatabaseLockManager.

    This provides the same functionality but with async/await syntax
    for use with asyncio-based applications.
    """

    def __init__(self) -> None:
        """Initialize the async database lock manager."""
        self.lock_manager = DatabaseLockManager()

    @asynccontextmanager
    async def acquire_read_lock(self, wait_timeout: float = 5.0) -> None:
        """Acquire a read lock asynchronously.

        Args:
            wait_timeout: Time to wait for a write lock to be released (seconds)
        """
        loop = asyncio.get_event_loop()

        # Run the blocking lock acquisition in a thread pool
        await loop.run_in_executor(
            None, lambda: self.lock_manager.acquire_read_lock(wait_timeout).__enter__()
        )

        try:
            yield
        finally:
            # Release the lock in a thread pool
            await loop.run_in_executor(
                None, lambda: self.lock_manager.acquire_read_lock().__exit__(None, None, None)
            )

    @asynccontextmanager
    async def acquire_write_lock(self, wait_timeout: float = 10.0) -> None:
        """Acquire a write lock asynchronously.

        Args:
            wait_timeout: Time to wait for read locks to be released (seconds)
        """
        loop = asyncio.get_event_loop()

        # Run the blocking lock acquisition in a thread pool
        await loop.run_in_executor(
            None, lambda: self.lock_manager.acquire_write_lock(wait_timeout).__enter__()
        )

        try:
            yield
        finally:
            # Release the lock in a thread pool
            await loop.run_in_executor(
                None, lambda: self.lock_manager.acquire_write_lock().__exit__(None, None, None)
            )


# Global instance
_lock_manager: Optional[DatabaseLockManager] = None
_async_lock_manager: Optional[AsyncDatabaseLockManager] = None


def get_lock_manager() -> DatabaseLockManager:
    """Get the global lock manager instance."""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = DatabaseLockManager()
    return _lock_manager


def get_async_lock_manager() -> AsyncDatabaseLockManager:
    """Get the global async lock manager instance."""
    global _async_lock_manager
    if _async_lock_manager is None:
        _async_lock_manager = AsyncDatabaseLockManager()
    return _async_lock_manager
