"""Admin API routes for the web service."""

from fastapi import APIRouter, Depends, status
from rq import Queue
import redis

from src.common.config import REDIS_URI
from src.common.models import (
    DeleteDocsRequest,
)
from src.common.logger import get_logger
from src.web_service.services.admin_service import (
    delete_docs,
)

# Get logger for this module
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["admin"])


@router.post("/delete_docs", status_code=status.HTTP_204_NO_CONTENT, operation_id="delete_docs")
async def delete_docs_endpoint(
    request: DeleteDocsRequest,
    queue: Queue = Depends(lambda: Queue("worker", connection=redis.from_url(REDIS_URI))),
) -> None:
    """Deletes documents from the database based on filters.

    Args:
        request: The delete request with optional filters.
        queue: The RQ queue for enqueueing the delete task.

    Returns:
        None: Returns a 204 No Content response upon successful enqueueing.
    """
    logger.info(
        f"API: Deleting docs with filters: tags={request.tags}, domain={request.domain}, page_ids={request.page_ids}"
    )

    try:
        # Call the service function
        await delete_docs(
            queue=queue,
            tags=request.tags,
            domain=request.domain,
            page_ids=request.page_ids,
        )

        # Return 204 No Content
        return None
    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        # Since this is an asynchronous operation, we still return 204
        # The actual deletion happens in the background
        return None
