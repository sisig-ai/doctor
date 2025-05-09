"""Main module for the Doctor Crawl Worker."""

import redis
from rq import Worker

from src.common.config import REDIS_URI, check_config
from src.common.db_setup import init_databases, get_duckdb_connection
from src.common.logger import get_logger

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
        init_databases(read_only=False)
        logger.info("Database initialization completed successfully")

        # Double-check that the document_embeddings table exists
        conn = get_duckdb_connection()
        table_count = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'document_embeddings'"
        ).fetchone()[0]

        if table_count == 0:
            logger.error("document_embeddings table is still missing after initialization!")
            return 1
        else:
            logger.info("Verified document_embeddings table exists")

        conn.close()
    except Exception as db_error:
        logger.error(f"Database initialization failed: {str(db_error)}")
        return 1

    # Connect to Redis
    logger.info(f"Connecting to Redis at {REDIS_URI}")
    redis_conn = redis.from_url(REDIS_URI)

    # Start worker
    logger.info("Starting worker, listening on queue: worker")
    worker = Worker(["worker"], connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
