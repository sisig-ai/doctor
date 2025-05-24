"""Database migration management for the Doctor project.

This module handles applying database migrations to ensure the schema is up-to-date.
"""

import pathlib

import duckdb

from src.common.logger import get_logger

logger = get_logger(__name__)


class MigrationRunner:
    """Manages and executes database migrations."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Initialize the migration runner.

        Args:
            conn: Active DuckDB connection.

        Returns:
            None.
        """
        self.conn = conn
        self.migrations_dir = pathlib.Path(__file__).parent / "migrations"

    def _ensure_migration_table(self) -> None:
        """Create the migrations tracking table if it doesn't exist.

        Args:
            None.

        Returns:
            None.
        """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                migration_name VARCHAR PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _get_applied_migrations(self) -> set[str]:
        """Get the set of already applied migrations.

        Args:
            None.

        Returns:
            set[str]: Set of migration names that have been applied.
        """
        result = self.conn.execute("SELECT migration_name FROM _migrations").fetchall()
        return {row[0] for row in result}

    def _apply_migration(self, migration_path: pathlib.Path) -> None:
        """Apply a single migration file.

        Args:
            migration_path: Path to the migration SQL file.

        Returns:
            None.

        Raises:
            Exception: If migration fails.
        """
        migration_name = migration_path.name
        logger.info(f"Applying migration: {migration_name}")

        try:
            # Read and execute the migration
            sql_content = migration_path.read_text()

            # Split by semicolons and execute each statement
            # Filter out empty statements
            statements = [s.strip() for s in sql_content.split(";") if s.strip()]

            for statement in statements:
                self.conn.execute(statement)

            # Record the migration as applied
            self.conn.execute(
                "INSERT INTO _migrations (migration_name) VALUES (?)", [migration_name]
            )

            logger.info(f"Successfully applied migration: {migration_name}")

        except Exception as e:
            logger.error(f"Failed to apply migration {migration_name}: {e}")
            raise

    def run_migrations(self) -> None:
        """Run all pending migrations in order.

        Args:
            None.

        Returns:
            None.
        """
        # Ensure migration tracking table exists
        self._ensure_migration_table()

        # Get already applied migrations
        applied = self._get_applied_migrations()

        # Find all migration files
        if not self.migrations_dir.exists():
            logger.debug("No migrations directory found")
            return

        migration_files = sorted(self.migrations_dir.glob("*.sql"))

        # Apply pending migrations
        pending_count = 0
        for migration_file in migration_files:
            if migration_file.name not in applied:
                self._apply_migration(migration_file)
                pending_count += 1

        if pending_count == 0:
            logger.debug("No pending migrations to apply")
        else:
            logger.info(f"Applied {pending_count} migration(s)")
