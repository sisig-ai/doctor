"""Initializes the database package, exposing key components for database interaction.

This package provides a modular approach to database management for the Doctor project,
separating connection handling, high-level operations, schema definitions, and utilities.

The main interface for database operations is the `DatabaseOperations` class.
"""

from .connection import DuckDBConnectionManager
from .operations import DatabaseOperations
from .schema import (
    CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL,
    CREATE_JOBS_TABLE_SQL,
    CREATE_PAGES_TABLE_SQL,
)
from .utils import deserialize_tags, serialize_tags

__all__ = [
    "CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL",
    "CREATE_JOBS_TABLE_SQL",
    "CREATE_PAGES_TABLE_SQL",
    "DatabaseOperations",
    "DuckDBConnectionManager",
    "deserialize_tags",
    "serialize_tags",
]
