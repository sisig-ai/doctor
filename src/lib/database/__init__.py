"""Initializes the database package, exposing key components for database interaction.

This package provides a modular approach to database management for the Doctor project,
separating connection handling, high-level operations, schema definitions, and utilities.

The main interface for database operations is the `DatabaseOperations` class, also aliased
as `Database` for convenience and backward compatibility.

For new code, more efficient approaches are available:
- Use connection_pool for proper read-only/read-write connection management
- Use batch operations for efficient bulk operations
"""

from .connection import DuckDBConnectionManager, connect_with_retry, async_connect_with_retry
from .connection_pool import (
    DuckDBConnectionPool,
    PooledConnectionContext,
    get_connection,
    get_connection_pool,
)
from .operations import DatabaseOperations
from .schema import (
    CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL,
    CREATE_JOBS_TABLE_SQL,
    CREATE_PAGES_TABLE_SQL,
)
from .utils import deserialize_tags, serialize_tags
from .batch import (
    BatchOperation,
    BatchPageInsert,
    BatchJobUpdate,
    BatchExecutor,
    batch_store_pages,
)

Database = DatabaseOperations

__all__ = [
    # Schema definitions
    "CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL",
    "CREATE_JOBS_TABLE_SQL",
    "CREATE_PAGES_TABLE_SQL",
    # Core database classes
    "Database",  # Alias for DatabaseOperations
    "DatabaseOperations",
    "DuckDBConnectionManager",
    "connect_with_retry",
    "async_connect_with_retry",
    # Connection pooling
    "DuckDBConnectionPool",
    "PooledConnectionContext",
    "get_connection",
    "get_connection_pool",
    # Batch operations
    "BatchOperation",
    "BatchPageInsert",
    "BatchJobUpdate",
    "BatchExecutor",
    "batch_store_pages",
    # Utilities
    "deserialize_tags",
    "serialize_tags",
]
