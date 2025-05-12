"""Configuration settings for the Doctor project."""

import os
from typing import List

from src.common.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Vector settings
VECTOR_SIZE = 1536  # OpenAI ada-002 embedding size

# Redis settings
REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379")

# DuckDB settings
DATA_DIR = os.getenv("DATA_DIR", "data")
DUCKDB_PATH = os.path.join(DATA_DIR, "doctor.duckdb")
DUCKDB_EMBEDDINGS_TABLE = "document_embeddings"
DB_RETRY_ATTEMPTS = int(os.getenv("DB_RETRY_ATTEMPTS", "5"))
DB_RETRY_DELAY_SEC = float(os.getenv("DB_RETRY_DELAY_SEC", "0.5"))

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "openai/text-embedding-ada-002"

# Web service settings
WEB_SERVICE_HOST = os.getenv("WEB_SERVICE_HOST", "0.0.0.0")
WEB_SERVICE_PORT = int(os.getenv("WEB_SERVICE_PORT", "9111"))

# Crawl settings
DEFAULT_MAX_PAGES = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# MCP Server settings
DOCTOR_BASE_URL = os.getenv("DOCTOR_BASE_URL", "http://localhost:9111")


def validate_config() -> List[str]:
    """Validate the configuration and return a list of any issues."""
    issues = []

    # Only check for valid API key in production
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY environment variable is not set")

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    return issues


def check_config() -> bool:
    """Check if the configuration is valid and log any issues."""
    issues = validate_config()

    if issues:
        for issue in issues:
            logger.error(f"Configuration error: {issue}")
        return False

    return True


if __name__ == "__main__":
    # When run directly, validate the configuration
    check_config()
