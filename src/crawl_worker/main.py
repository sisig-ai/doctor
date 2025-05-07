"""Main module for the Doctor Crawl Worker."""

import redis
from rq import Worker

from src.common.config import REDIS_URI, check_config
from src.common.db_setup import init_databases
from src.lib.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


def main():
    """Main entry point for the Crawl Worker."""
    # Validate configuration
    if not check_config():
        logger.error("Invalid configuration. Exiting.")
        return 1

    # Initialize databases with write access
    init_databases(read_only=False)

    # Connect to Redis
    logger.info(f"Connecting to Redis at {REDIS_URI}")
    redis_conn = redis.from_url(REDIS_URI)

    # Start worker
    logger.info("Starting worker, listening on queue: worker")
    worker = Worker(["worker"], connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
