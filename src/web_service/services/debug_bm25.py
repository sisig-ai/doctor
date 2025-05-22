"""BM25 diagnostic function."""

import duckdb

from src.common.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


async def debug_bm25_search(conn: duckdb.DuckDBPyConnection, query: str) -> dict:
    """Diagnose BM25/FTS search issues by always attempting to query the FTS index using DuckDB's FTS extension.
    Reports the result of the FTS query attempt, any errors, and explains FTS index visibility.
    """
    logger.info(f"Running BM25/FTS search diagnostics for query: '{query}' (DuckDB-specific)")
    results = {}

    try:
        # Check if FTS extension is loaded
        fts_loaded = conn.execute("""
        SELECT * FROM duckdb_extensions()
        WHERE extension_name = 'fts' AND loaded = true
        """).fetchall()
        results["fts_extension_loaded"] = bool(fts_loaded)
        logger.info(f"FTS extension loaded: {results['fts_extension_loaded']}")

        # Count records in pages table
        try:
            pages_count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            results["pages_count"] = pages_count
            logger.info(f"Pages table record count: {pages_count}")
        except Exception as e:
            results["pages_count"] = None
            results["pages_count_error"] = str(e)
            logger.warning(f"Could not count pages: {e}")

        # Always attempt FTS/BM25 search using DuckDB FTS extension
        try:
            escaped_query = query.replace("'", "''")
            bm25_sql = f"""
            SELECT p.id, p.url, fts_main_pages.match_bm25(p.id, '{escaped_query}') AS score
            FROM pages p
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT 5
            """
            logger.info(f"Attempting FTS/BM25 search: {bm25_sql}")
            bm25_results = conn.execute(bm25_sql).fetchall()
            results["fts_bm25_search"] = {
                "success": True,
                "count": len(bm25_results),
                "samples": [
                    {"id": row[0], "url": row[1], "score": row[2]} for row in bm25_results[:3]
                ],
                "error": None,
            }
            logger.info(f"FTS/BM25 search returned {len(bm25_results)} results")
        except Exception as e_bm25:
            results["fts_bm25_search"] = {
                "success": False,
                "count": 0,
                "samples": [],
                "error": str(e_bm25),
            }
            logger.warning(f"FTS/BM25 search failed: {e_bm25}")

        # Add a note about FTS index visibility in DuckDB
        results["fts_index_visibility_note"] = (
            "DuckDB FTS indexes are not visible in sqlite_master or information_schema.tables. "
            "If FTS index creation returned an 'already exists' error, the index is present and can be queried "
            "using fts_main_<table>.match_bm25 or similar functions, even if not listed in system tables."
        )

        # Add table/column info for clarity
        try:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [row[0] for row in tables]
            results["tables"] = table_names
            if "pages" in table_names:
                columns = conn.execute("PRAGMA table_info('pages')").fetchall()
                col_names = [row[1] for row in columns]
                results["pages_columns"] = col_names
        except Exception as e_schema:
            results["schema_check_error"] = str(e_schema)

    except Exception as e:
        logger.error(f"BM25/FTS diagnostics failed: {e}")
        results["error"] = str(e)

    return results
