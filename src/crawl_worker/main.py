"""Main module for the Doctor Crawl Worker."""

import asyncio
import pathlib
import redis
from rq import Worker

from src.common.config import DUCKDB_PATH, DUCKDB_WRITE_PATH, REDIS_URI, check_config
from src.common.logger import get_logger
from src.lib.database import get_connection
from src.lib.database.sync import start_sync_service

# Get logger for this module
logger = get_logger(__name__)


async def initialize_database() -> bool:
    """Initialize the database with write access.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        logger.info("Initializing databases for the crawl worker...")

        # First, check if web service has the database locked in read-only mode
        db_path = pathlib.Path(DUCKDB_WRITE_PATH)

        # If old database exists but write DB doesn't, initialize from old DB
        if not db_path.exists() and pathlib.Path(DUCKDB_PATH).exists():
            original_path = pathlib.Path(DUCKDB_PATH)
            logger.info(f"Initializing write database from existing DB at {original_path}")

            # Make sure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Import here to avoid circular imports
            import shutil

            # Copy the original database
            shutil.copy2(original_path, db_path)
            logger.info(f"Copied original database to write database at {db_path}")

            # Also copy any associated WAL files if they exist
            original_wal_path = original_path.with_suffix(".wal")
            write_wal_path = db_path.with_suffix(".wal")
            if original_wal_path.exists():
                logger.info("Copying original WAL file to write database")
                shutil.copy2(original_wal_path, write_wal_path)

        # Initialize with write access
        async with await get_connection(read_only=False) as conn_manager:
            # Force a checkpoint first to consolidate any WAL files
            conn = await conn_manager.async_ensure_connection()
            try:
                logger.info("Forcing initial checkpoint on write database")
                conn.execute("PRAGMA force_checkpoint")
                logger.info("Initial checkpoint completed successfully")
            except Exception as e:
                logger.warning(f"Initial checkpoint failed, continuing anyway: {e}")

            # Now initialize the database
            await conn_manager.async_initialize()

            # Test embeddings table exists
            conn = await conn_manager.async_ensure_connection()
            res = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'document_embeddings'"
            ).fetchone()

            if res[0] == 0:
                logger.warning("document_embeddings table doesn't exist, creating it...")
                conn_manager.ensure_vss_extension()
            else:
                logger.info("Verified document_embeddings table exists")

        # Start the database sync service
        await start_sync_service()

        return True
    except Exception as e:
        logger.exception(f"Failed to initialize database: {e}")
        return False


async def main() -> None:
    """Main entry point for the Doctor Crawl Worker.

    Initializes required resources and starts the worker.
    """
    logger.info("Starting Doctor Crawl Worker")

    # Check configuration
    if not check_config():
        logger.error("Invalid configuration. Check logs for details.")
        return

    # Initialize database
    if not await initialize_database():
        logger.error("Failed to initialize database. Exiting.")
        return

    # Connect to Redis
    try:
        logger.info(f"Connecting to Redis at {REDIS_URI}")
        redis_conn = redis.from_url(REDIS_URI)

        # Set up queues
        queue_name = "worker"

        # Start worker
        logger.info(f"Starting worker, listening on queue: {queue_name}")
        worker = Worker([queue_name], connection=redis_conn)
        worker.work()
    except Exception as e:
        logger.exception(f"Error in worker main loop: {e}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
