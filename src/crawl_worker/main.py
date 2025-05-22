"""Main module for the Doctor Crawl Worker."""

import redis
from rq import Worker

from src.common.config import REDIS_URI, check_config
from src.common.logger import get_logger
from src.lib.database import DatabaseOperations

# Get logger for this module
logger = get_logger(__name__)


def main() -> int:
    """Main entry point for the Crawl Worker.

    Returns:
        int: The exit code (0 for success, 1 for failure).

    """
    # Validate configuration
    if not check_config():
        logger.error("Invalid configuration. Exiting.")
        return 1

    # Initialize databases with write access
    try:
        logger.info("Initializing databases for the crawl worker...")
        db = DatabaseOperations(read_only=False)
        db.db.initialize()
        logger.info("Database initialization completed successfully")

        # Double-check that the document_embeddings table exists
        # Reuse the connection from db
        conn = db.db.conn
        if conn is None:
            logger.error("Failed to get DuckDB connection")
            return 1

        result = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'document_embeddings'",
        ).fetchone()

        if result is None:
            logger.error("Failed to execute query to check for document_embeddings table")
            return 1

        table_count = result[0]

        if table_count == 0:
            logger.exception("document_embeddings table is still missing after initialization!")
            return 1
        logger.info("Verified document_embeddings table exists")

        db.db.close()
    except Exception as e:
        logger.error(f"Database initialization failed: {e!s}")
        return 1

    # Connect to Redis
    try:
        logger.info(f"Connecting to Redis at {REDIS_URI}")
        redis_conn = redis.from_url(REDIS_URI)

        # Start worker
        logger.info("Starting worker, listening on queue: worker")
        worker = Worker(["worker"], connection=redis_conn)
        worker.work(with_scheduler=True)
        return 0  # Return success if worker completes normally
    except Exception as redis_error:
        logger.error(f"Redis worker error: {redis_error!s}")
        return 1


if __name__ == "__main__":
    main()
