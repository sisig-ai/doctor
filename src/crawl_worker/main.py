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
        # DatabaseOperations() constructor now calls initialize() internally using a context manager.
        db_ops = DatabaseOperations()
        logger.info(
            "Database initialization (via DatabaseOperations constructor) completed successfully."
        )

        # Double-check that the document_embeddings table exists
        # Use a new context-managed connection for this check
        with db_ops.db as conn_manager:
            actual_conn = conn_manager.conn
            if not actual_conn:
                logger.error("Failed to get DuckDB connection for table verification.")
                return 1

            result = actual_conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'document_embeddings'",
            ).fetchone()

            if result is None:
                logger.error("Failed to execute query to check for document_embeddings table")
                return 1  # conn_manager will close connection

            table_count = result[0]

            if table_count == 0:
                logger.error("document_embeddings table is still missing after initialization!")
                return 1  # conn_manager will close connection
            logger.info("Verified document_embeddings table exists")
        # Connection is closed by conn_manager context exit

    except Exception as e:
        logger.error(f"Database initialization or verification failed: {e!s}")
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
