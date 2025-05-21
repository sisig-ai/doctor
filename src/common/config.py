"""Configuration settings for the Doctor project."""

import os

from src.common.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Vector settings
VECTOR_SIZE = 3072  # OpenAI text-embedding-3-large embedding size

# Redis settings
REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379")

# DuckDB settings
DATA_DIR = os.getenv("DATA_DIR", "data")
DUCKDB_PATH = os.path.join(DATA_DIR, "doctor.duckdb")
DUCKDB_READ_PATH = os.path.join(DATA_DIR, "doctor.read.duckdb")  # Read-only copy for web service
DUCKDB_WRITE_PATH = os.path.join(DATA_DIR, "doctor.write.duckdb")  # Read-write copy for crawler
DUCKDB_EMBEDDINGS_TABLE = "document_embeddings"
DB_RETRY_ATTEMPTS = int(os.getenv("DB_RETRY_ATTEMPTS", "5"))
DB_RETRY_DELAY_SEC = float(os.getenv("DB_RETRY_DELAY_SEC", "0.5"))
DB_SYNC_INTERVAL_SEC = int(os.getenv("DB_SYNC_INTERVAL_SEC", "10"))  # Seconds between syncs

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOC_EMBEDDING_MODEL = "openai/text-embedding-3-large"
QUERY_EMBEDDING_MODEL = "openai/text-embedding-3-large"

# Web service settings
WEB_SERVICE_HOST = os.getenv("WEB_SERVICE_HOST", "0.0.0.0")
WEB_SERVICE_PORT = int(os.getenv("WEB_SERVICE_PORT", "9111"))

# Crawl settings
DEFAULT_MAX_PAGES = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Search settings
RETURN_FULL_DOCUMENT_TEXT = True

# MCP Server settings
DOCTOR_BASE_URL = os.getenv("DOCTOR_BASE_URL", "http://localhost:9111")


def validate_config() -> list[str]:
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
