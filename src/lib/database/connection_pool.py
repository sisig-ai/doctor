"""Connection pooling implementation for DuckDB.

This module provides a connection pool for DuckDB, optimized for the Doctor project's
needs with proper handling of read-only vs read-write connections and transaction management.
"""

import asyncio
import pathlib
import time
import redis
from typing import Dict, List, Optional, Tuple


from src.common.config import DUCKDB_PATH, REDIS_URI
from src.common.logger import get_logger
from .connection import DuckDBConnectionManager

logger = get_logger(__name__)

# Redis lock keys
DB_READ_LOCK_KEY = "doctor:db:read_lock"
DB_WRITE_LOCK_KEY = "doctor:db:write_lock"
DB_LOCK_TIMEOUT = 60  # seconds


class DatabaseLockManager:
    """Manages database locks for coordinating read/write access.

    Uses Redis for distributed locking between services.
    This ensures the web service and crawler don't try to access
    the database with conflicting access modes simultaneously.
    """

    def __init__(self) -> None:
        """Initialize the database lock manager."""
        self.redis_client = redis.from_url(REDIS_URI)

    async def acquire_read_lock(self, timeout: int = 5) -> bool:
        """Try to acquire a read lock with timeout.

        Args:
            timeout: Seconds to wait for write lock to clear

        Returns:
            bool: True if lock acquired, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if a write lock exists
            if self.redis_client.exists(DB_WRITE_LOCK_KEY):
                logger.debug("Write lock exists, waiting to acquire read lock...")
                await asyncio.sleep(0.5)
                continue

            # Increment the read lock counter
            self.redis_client.incr(DB_READ_LOCK_KEY)
            # Set expiration to prevent orphaned locks
            self.redis_client.expire(DB_READ_LOCK_KEY, DB_LOCK_TIMEOUT)
            logger.debug("Read lock acquired")
            return True

        logger.warning(f"Failed to acquire read lock after {timeout} seconds")
        return False

    def release_read_lock(self) -> None:
        """Release a read lock by decrementing the counter."""
        try:
            # Decrement the read lock counter
            count = self.redis_client.decr(DB_READ_LOCK_KEY)
            # If counter reaches 0, delete the key
            if count <= 0:
                self.redis_client.delete(DB_READ_LOCK_KEY)
            logger.debug("Read lock released")
        except Exception as e:
            logger.error(f"Error releasing read lock: {e}")

    async def acquire_write_lock(self, timeout: int = 10) -> bool:
        """Try to acquire a write lock with timeout.

        Args:
            timeout: Seconds to wait for read locks to clear

        Returns:
            bool: True if lock acquired, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if read locks exist
            read_count = self.redis_client.get(DB_READ_LOCK_KEY)
            if read_count and int(read_count) > 0:
                logger.debug(f"{read_count} read locks exist, waiting to acquire write lock...")
                await asyncio.sleep(0.5)
                continue

            # Try to set the write lock
            lock_acquired = self.redis_client.set(
                DB_WRITE_LOCK_KEY,
                "1",
                nx=True,  # Only set if it doesn't exist
                ex=DB_LOCK_TIMEOUT,  # Set expiration
            )

            if lock_acquired:
                logger.debug("Write lock acquired")
                return True

            # Someone else got the write lock, wait
            logger.debug("Another process has the write lock, waiting...")
            await asyncio.sleep(0.5)

        logger.warning(f"Failed to acquire write lock after {timeout} seconds")
        return False

    def release_write_lock(self) -> None:
        """Release the write lock."""
        try:
            self.redis_client.delete(DB_WRITE_LOCK_KEY)
            logger.debug("Write lock released")
        except Exception as e:
            logger.error(f"Error releasing write lock: {e}")


class DuckDBConnectionPool:
    """Provides a pool of DuckDB connections for efficient resource management.

    Handles connection pooling according to DuckDB's concurrency model.
    Important: DuckDB only allows either:
    1. One process with read-write access
    2. Multiple processes with read-only access
    It's not possible to mix these models across processes.

    This implementation creates connections on-demand with the requested
    read-only setting rather than maintaining separate pools.
    """

    def __init__(
        self,
        min_connections: int = 1,  # Reduced minimum connections to avoid keeping too many open
        max_connections: int = 5,  # Reduced maximum to avoid too many connections
        idle_timeout: int = 30,  # Reduced idle timeout to close connections quicker
    ) -> None:
        """Initialize the connection pool.

        Args:
            min_connections: Minimum number of connections to keep in the pool. Defaults to 1.
            max_connections: Maximum number of connections to allow in the pool. Defaults to 5.
            idle_timeout: Time in seconds after which an idle connection will be closed. Defaults to 30.
        """
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout

        # Pool as dict of read_only -> [(connection, last_used_timestamp)]
        # This lets us store both read-only and read-write connections in the same pool
        self._connection_pool: Dict[bool, List[Tuple[DuckDBConnectionManager, float]]] = {
            True: [],  # Read-only connections
            False: [],  # Read-write connections
        }

        # Lock to protect pool access
        self._pool_lock = asyncio.Lock()

        # Track total connections created
        self._total_connections = 0

        # Use a task to periodically check for connections to close
        self._cleanup_task: Optional[asyncio.Task] = None

        # Create a database lock manager for coordinating access
        self.lock_manager = DatabaseLockManager()

        logger.info(
            f"DuckDB connection pool initialized with min={min_connections}, "
            f"max={max_connections}, timeout={idle_timeout}s"
        )

    async def start(self) -> None:
        """Start the connection pool, preparing for just-in-time connections.

        This starts the cleanup task but does not pre-create any connections.
        Connections will be created only when needed by get_connection().
        """
        logger.info("Starting DuckDB connection pool (lazy initialization)")

        # Check if database exists - just for logging
        db_path = pathlib.Path(DUCKDB_PATH)
        db_exists = db_path.exists()

        if not db_exists:
            # Warn that read-only operations will fail until database is created
            logger.warning("Database file does not exist")
            logger.warning("Read-only operations will fail until database is created")

        # Start the cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            f"Connection pool ready for just-in-time connections (min={self.min_connections}, max={self.max_connections})"
        )

    async def stop(self) -> None:
        """Stop the connection pool, closing all connections."""
        logger.info("Stopping DuckDB connection pool")

        # Cancel the cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections forcibly
        async with self._pool_lock:
            for read_only, connections in self._connection_pool.items():
                for conn_manager, _ in connections:
                    try:
                        conn_manager.close()
                    except Exception as e:
                        logger.warning(f"Error closing connection during pool shutdown: {e}")
            self._connection_pool = {True: [], False: []}
            self._total_connections = 0

        logger.info("DuckDB connection pool stopped")

    async def get_connection(self, read_only: bool = True) -> DuckDBConnectionManager:
        """Get a connection from the pool with the specified read-only mode.

        Creates a new connection if needed, within the max_connections limit.
        Implements just-in-time connection creation with the requested access mode.
        Uses distributed locking to coordinate between services.

        Args:
            read_only: Whether to get a read-only connection. Defaults to True.

        Returns:
            A DuckDBConnectionManager instance with the specified read-only setting.

        Raises:
            RuntimeError: If the pool is at capacity and no connections are available.
        """
        try:
            async with asyncio.timeout(10):  # Increase timeout for lock acquisition
                # First acquire the appropriate lock based on read_only mode
                if read_only:
                    lock_acquired = await self.lock_manager.acquire_read_lock()
                else:
                    lock_acquired = await self.lock_manager.acquire_write_lock()

                if not lock_acquired:
                    if read_only:
                        raise RuntimeError(
                            "Failed to acquire database read lock - a write operation is in progress"
                        )
                    else:
                        raise RuntimeError(
                            "Failed to acquire database write lock - read operations are in progress"
                        )

                # Now get the connection with the pool lock
                async with self._pool_lock:
                    # Try to get a connection of the requested type from the pool
                    if self._connection_pool[read_only]:
                        # Get a connection from the pool
                        conn_manager, _ = self._connection_pool[read_only].pop()

                        # Ensure connection is still usable
                        try:
                            # Use async connection check
                            await conn_manager.async_ensure_connection()
                            return conn_manager
                        except Exception as e:
                            logger.warning(f"Discarding unhealthy connection: {e}")
                            try:
                                conn_manager.close()
                            except Exception:
                                pass  # Already broken, ignore further errors
                            self._total_connections -= 1

                            # Release the lock if we can't get a connection
                            if read_only:
                                self.lock_manager.release_read_lock()
                            else:
                                self.lock_manager.release_write_lock()

                    # Need to create a new connection
                    if self._total_connections < self.max_connections:
                        logger.debug(
                            f"Creating new {'read-only' if read_only else 'read-write'} connection"
                        )
                        try:
                            conn_manager = DuckDBConnectionManager(read_only=read_only)

                            # Check if database exists
                            db_path = pathlib.Path(DUCKDB_PATH)
                            if db_path.exists() or not read_only:
                                # Use async initialization
                                await conn_manager.async_ensure_connection()
                            else:
                                logger.warning(
                                    "Database file doesn't exist, connection will fail until created"
                                )

                            self._total_connections += 1
                            # Set the lock manager on the connection manager so it can release locks
                            conn_manager._lock_manager = self.lock_manager
                            conn_manager._is_read_only = read_only
                            return conn_manager
                        except Exception as e:
                            logger.warning(f"Failed to create new connection: {e}")

                            # Release the lock if we can't create a connection
                            if read_only:
                                self.lock_manager.release_read_lock()
                            else:
                                self.lock_manager.release_write_lock()

                            # Try to get an existing connection of any type as fallback
                            for pool_read_only in [read_only, not read_only]:
                                if self._connection_pool[pool_read_only]:
                                    conn_manager, _ = self._connection_pool[pool_read_only].pop()
                                    return conn_manager
                            raise RuntimeError(f"Could not create connection: {e}")
                    else:
                        # We're at capacity, fail
                        msg = f"Connection pool at capacity ({self.max_connections})"
                        logger.error(msg)

                        # Release the lock since we couldn't get a connection
                        if read_only:
                            self.lock_manager.release_read_lock()
                        else:
                            self.lock_manager.release_write_lock()

                        raise RuntimeError(msg)
        except TimeoutError:
            logger.error("Timeout while trying to get a connection from the pool")
            raise RuntimeError("Timeout while getting a database connection")

    async def release_connection(self, conn_manager: DuckDBConnectionManager) -> None:
        """Release a connection back to the pool.

        Args:
            conn_manager: The connection manager to return to the pool.
        """
        # Get read-only status
        read_only = conn_manager.read_only

        # If the connection has an active transaction, roll it back
        try:
            if conn_manager.transaction_active:
                logger.warning("Connection returned to pool with active transaction, rolling back")
                conn_manager.rollback()
        except Exception as e:
            logger.warning(f"Error checking transaction state: {e}")

        # Verify connection is actually healthy before returning it to the pool
        try:
            # Test if connection is still good using async method
            await conn_manager.async_ensure_connection()

            # Add connection back to pool according to its read_only setting
            async with self._pool_lock:
                # Don't keep more connections than the minimum when releasing
                if len(self._connection_pool[read_only]) < self.min_connections:
                    self._connection_pool[read_only].append((conn_manager, time.time()))
                else:
                    # We already have enough connections in the pool, close this one
                    conn_manager.close()
                    self._total_connections -= 1
        except Exception as e:
            logger.warning(f"Not returning bad connection to pool: {e}")
            try:
                conn_manager.close()
            except Exception:
                pass  # Already broken, ignore further errors
            self._total_connections -= 1
        finally:
            # Always release the distributed lock
            try:
                # Release the appropriate lock based on connection type
                if hasattr(conn_manager, "_is_read_only"):
                    if conn_manager._is_read_only:
                        self.lock_manager.release_read_lock()
                    else:
                        self.lock_manager.release_write_lock()
                else:
                    # Fall back to the connection's read-only status
                    if read_only:
                        self.lock_manager.release_read_lock()
                    else:
                        self.lock_manager.release_write_lock()
            except Exception as e:
                logger.error(f"Error releasing distributed lock: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up idle connections."""
        try:
            while True:
                await asyncio.sleep(self.idle_timeout / 2)
                await self._cleanup_idle_connections()
        except asyncio.CancelledError:
            logger.debug("Connection pool cleanup task cancelled")
            raise

    async def _cleanup_idle_connections(self) -> None:
        """Close idle connections that exceed the idle timeout."""
        current_time = time.time()
        cutoff_time = current_time - self.idle_timeout

        async with self._pool_lock:
            for read_only, connections in self._connection_pool.items():
                active_connections = []
                for conn_manager, last_used in connections:
                    if last_used < cutoff_time and len(connections) > self.min_connections:
                        logger.debug(
                            f"Closing idle {'read-only' if read_only else 'read-write'} connection"
                        )
                        try:
                            conn_manager.close()
                        except Exception as e:
                            logger.warning(f"Error closing idle connection: {e}")
                        self._total_connections -= 1
                    else:
                        active_connections.append((conn_manager, last_used))
                self._connection_pool[read_only] = active_connections


class PooledConnectionContext:
    """A context manager for database connections from the connection pool.

    Provides automatic connection acquisition and release with 'async with' syntax.
    """

    def __init__(
        self,
        pool: DuckDBConnectionPool,
        read_only: bool = True,
    ) -> None:
        """Initialize the connection context.

        Args:
            pool: The connection pool to get connections from.
            read_only: Whether to request a read-only connection. Defaults to True.
        """
        self.pool = pool
        self.read_only = read_only
        self.conn_manager: Optional[DuckDBConnectionManager] = None

    async def __aenter__(self) -> DuckDBConnectionManager:
        """Enter the async context, acquiring a connection from the pool.

        Returns:
            The acquired DuckDBConnectionManager instance.
        """
        self.conn_manager = await self.pool.get_connection(read_only=self.read_only)
        return self.conn_manager

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context, releasing the connection back to the pool."""
        if self.conn_manager:
            try:
                await self.pool.release_connection(self.conn_manager)
            except Exception as e:
                logger.warning(f"Error releasing connection: {e}")
                # Forcibly close if release fails
                try:
                    self.conn_manager.close()
                except Exception:
                    pass  # Already broken, ignore further errors


# Global singleton connection pool instance
_connection_pool: Optional[DuckDBConnectionPool] = None


async def get_connection_pool() -> DuckDBConnectionPool:
    """Get the global connection pool instance, initializing it if necessary.

    IMPORTANT: This function creates a pool instance but doesn't establish actual
    connections until they're needed for operations. This follows DuckDB's
    concurrency model best practices:

    1. No process holding connections long-term
    2. Creating connections just-in-time when needed
    3. Releasing connections immediately after use

    Returns:
        The global DuckDBConnectionPool instance.
    """
    global _connection_pool

    if _connection_pool is None:
        _connection_pool = DuckDBConnectionPool()
        await _connection_pool.start()
    return _connection_pool


async def get_connection(read_only: bool = True) -> PooledConnectionContext:
    """Get a database connection context from the global pool.

    Creates a just-in-time connection that will be automatically returned to
    the pool when the context exits. This is the recommended way to get
    connections for all operations.

    Usage:
        async with await get_connection(read_only=True) as conn_manager:
            conn = await conn_manager.async_ensure_connection()
            # Use conn for database operations
            result = conn.execute("SELECT * FROM table")

    Args:
        read_only: Whether to request a read-only connection. Defaults to True.

    Returns:
        A PooledConnectionContext that can be used with 'async with'.
    """
    pool = await get_connection_pool()
    return PooledConnectionContext(pool, read_only=read_only)
