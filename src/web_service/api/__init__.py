"""API routes package for the web service."""

from fastapi import APIRouter

from src.web_service.api.documents import router as documents_router
from src.web_service.api.jobs import router as jobs_router
from src.web_service.api.admin import router as admin_router

# Create a main router that includes all the other routers
api_router = APIRouter()

# Include the routers
api_router.include_router(documents_router)
api_router.include_router(jobs_router)
api_router.include_router(admin_router)
