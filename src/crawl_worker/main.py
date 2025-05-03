"""Main module for the Doctor Crawl Worker."""

import logging
import redis
from rq import Worker, Queue

from src.common.config import REDIS_URI, check_config
from src.common.db_setup import init_databases

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define queue names
DEFAULT_QUEUE = "default"
HIGH_QUEUE = "high"
LOW_QUEUE = "low"


def main():
    """Main entry point for the Crawl Worker."""
    # Validate configuration
    if not check_config():
        logger.error("Invalid configuration. Exiting.")
        return 1

    # Initialize databases
    init_databases()

    # Connect to Redis
    logger.info(f"Connecting to Redis at {REDIS_URI}")
    redis_conn = redis.from_url(REDIS_URI)

    # Define queues to listen on
    listen = [HIGH_QUEUE, DEFAULT_QUEUE, LOW_QUEUE]
    queues = list(map(lambda name: Queue(name, connection=redis_conn), listen))

    # Start worker
    logger.info(f"Starting worker, listening on queues: {', '.join(listen)}")
    worker = Worker(queues, connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
