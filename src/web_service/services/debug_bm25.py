"""BM25 diagnostic function."""

import duckdb

from src.common.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


async def debug_bm25_search(conn: duckdb.DuckDBPyConnection, query: str) -> dict:
    """Diagnose BM25 search issues by executing tests and gathering debug information.

    Args:
        conn: Connected DuckDB connection
        query: Search query to test

    Returns:
        dict: Diagnostic information

    """
    logger.info(f"Running BM25 search diagnostics for query: '{query}'")
    results = {}

    try:
        # Check if FTS extension is loaded
        fts_loaded = conn.execute("""
        SELECT * FROM duckdb_extensions()
        WHERE extension_name = 'fts' AND loaded = true
        """).fetchall()
        results["fts_extension_loaded"] = bool(fts_loaded)
        logger.info(f"FTS extension loaded: {results['fts_extension_loaded']}")

        # Check for FTS tables
        fts_tables = conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name LIKE 'fts%'
        """).fetchall()
        results["fts_tables"] = [row[0] for row in fts_tables]
        logger.info(f"FTS tables: {results['fts_tables']}")

        # Count records in pages table
        pages_count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        results["pages_count"] = pages_count
        logger.info(f"Pages table record count: {pages_count}")

        # Test direct access to the fts_main_pages table
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM fts_main_pages").fetchone()[0]
            results["fts_main_pages_count"] = fts_count
            logger.info(f"FTS table record count: {fts_count}")

            if fts_count > 0:
                # Sample a record to see what's in the FTS table
                sample = conn.execute("""
                SELECT id, substr(raw_text, 1, 100) as sample
                FROM fts_main_pages LIMIT 1
                """).fetchone()
                if sample:
                    results["fts_sample"] = {"id": sample[0], "text_sample": sample[1] + "..."}
                    logger.info(f"FTS sample record: ID={sample[0]}, Text={sample[1]}...")
        except Exception as e:
            results["fts_main_pages_access"] = {"success": False, "error": str(e)}
            logger.warning(f"Could not access fts_main_pages table: {e}")

        # Try direct BM25 search
        try:
            # Skip the namespace/function lookup that's causing errors
            namespace = "fts_main_pages"  # Use the known namespace
            escaped_query = query.replace("'", "''")

            direct_sql = f"""
            SELECT p.id, p.url, {namespace}.match_bm25(p.id, '{escaped_query}') AS score
            FROM pages p
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT 5
            """

            results["bm25_sql"] = direct_sql
            logger.info(f"Executing direct BM25 search: {direct_sql}")

            bm25_results = conn.execute(direct_sql).fetchall()

            results["direct_bm25_search"] = {
                "success": True,
                "count": len(bm25_results),
                "samples": [
                    {"id": row[0], "url": row[1], "score": row[2]} for row in bm25_results[:3]
                ],  # Include up to 3 samples
            }
            logger.info(f"Direct BM25 search results: {len(bm25_results)}")
        except Exception as e:
            results["direct_bm25_search"] = {"success": False, "error": str(e)}
            logger.warning(f"Direct BM25 search failed: {e}")

        # Try fallback simple text search
        try:
            escaped_query = query.replace("'", "''")
            simple_sql = f"""
            SELECT id, url, substr(raw_text, 1, 100) as text_sample
            FROM pages
            WHERE raw_text LIKE '%{escaped_query}%'
            LIMIT 5
            """

            logger.info(f"Executing simple text search: {simple_sql}")
            simple_results = conn.execute(simple_sql).fetchall()

            results["simple_text_search"] = {
                "success": True,
                "count": len(simple_results),
                "samples": [
                    {"id": row[0], "url": row[1], "text_sample": row[2]}
                    for row in simple_results[:3]
                ],  # Include up to 3 samples
            }
            logger.info(f"Simple text search results: {len(simple_results)}")
        except Exception as e:
            results["simple_text_search"] = {"success": False, "error": str(e)}
            logger.warning(f"Simple text search failed: {e}")

    except Exception as e:
        logger.error(f"BM25 diagnostics failed: {e}")
        results["error"] = str(e)

    return results
