"""Admin service for the web service."""

import uuid
from typing import List, Optional

from rq import Queue
from src.lib.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


async def delete_docs(
    queue: Queue,
    tags: Optional[List[str]] = None,
    domain: Optional[str] = None,
    page_ids: Optional[List[str]] = None,
) -> str:
    """
    Delete documents from the database based on filters.

    Args:
        queue: Redis queue for job processing
        tags: Optional list of tags to filter by
        domain: Optional domain substring to filter by
        page_ids: Optional list of specific page IDs to delete

    Returns:
        str: The task ID for tracking
    """
    logger.info(
        f"Enqueueing delete task with filters: tags={tags}, domain={domain}, page_ids={page_ids}"
    )

    # Generate a task ID for tracking logs
    task_id = str(uuid.uuid4())

    # Enqueue the delete task
    queue.enqueue(
        "src.crawl_worker.tasks.delete_docs",
        task_id,
        tags,
        domain,
        page_ids,
    )

    logger.info(f"Enqueued delete task with ID: {task_id}")

    return task_id
