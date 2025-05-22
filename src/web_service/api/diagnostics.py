"""Diagnostic API routes for debugging purposes.
Important: Should be disabled in production.
"""

from fastapi import APIRouter, HTTPException, Query

from src.common.logger import get_logger
from src.lib.database import DatabaseOperations
from src.web_service.services.debug_bm25 import debug_bm25_search

# Get logger for this module
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["diagnostics"])


@router.get("/bm25_diagnostics", operation_id="bm25_diagnostics")
async def bm25_diagnostics_endpoint(
    query: str = Query("test", description="The search query to test with"),
    initialize: bool = Query(
        True, description="If true, attempt to initialize FTS if in write mode."
    ),
):
    """
    Combined endpoint for DuckDB FTS initialization and diagnostics.
    - Checks for the existence of the 'pages' table and 'raw_text' column before attempting FTS index creation.
    - Always includes the full error message from FTS index creation in the response.
    - Adds verbose logging and schema checks to the diagnostics output.
    """
    logger.info(
        "API: Running combined BM25 diagnostics and initialization endpoint (DuckDB-specific)"
    )

    db = DatabaseOperations(read_only=not initialize)
    conn = db.db.ensure_connection()
    read_only = db.db.read_only
    initialization_summary = {}
    schema_info = {}

    # Check for 'pages' table and 'raw_text' column existence
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [row[0] for row in tables]
        schema_info["tables"] = table_names
        pages_table_exists = "pages" in table_names
        initialization_summary["pages_table_exists"] = pages_table_exists
        if pages_table_exists:
            columns = conn.execute("PRAGMA table_info('pages')").fetchall()
            col_names = [row[1] for row in columns]
            schema_info["pages_columns"] = col_names
            initialization_summary["pages_columns"] = col_names
            raw_text_exists = "raw_text" in col_names
            initialization_summary["raw_text_column_exists"] = raw_text_exists
        else:
            initialization_summary["pages_columns"] = []
            initialization_summary["raw_text_column_exists"] = False
    except Exception as e_schema:
        initialization_summary["schema_check_error"] = str(e_schema)
        schema_info["schema_check_error"] = str(e_schema)

    try:
        if initialize and not read_only:
            logger.info("Attempting FTS initialization in write mode...")
            # 1. Load FTS extension
            try:
                conn.execute("INSTALL fts;")
                conn.execute("LOAD fts;")
                initialization_summary["fts_extension_loaded"] = True
            except Exception as e_fts:
                logger.warning(f"FTS extension loading during initialization: {e_fts}")
                initialization_summary["fts_extension_loaded"] = False
                initialization_summary["fts_extension_error"] = str(e_fts)
            # 2. Create FTS index (only if table and column exist)
            if initialization_summary.get("pages_table_exists") and initialization_summary.get(
                "raw_text_column_exists"
            ):
                try:
                    conn.execute("PRAGMA create_fts_index('pages', 'id', 'raw_text');")
                    initialization_summary["fts_index_created"] = True
                    initialization_summary["fts_index_creation_error"] = None
                except Exception as e_index:
                    logger.warning(f"FTS index creation error: {e_index}")
                    initialization_summary["fts_index_created"] = False
                    initialization_summary["fts_index_creation_error"] = str(e_index)
                    if "already exists" in str(e_index):
                        initialization_summary["fts_index_exists"] = True
            else:
                initialization_summary["fts_index_created"] = False
                initialization_summary["fts_index_creation_error"] = (
                    "'pages' table or 'raw_text' column does not exist."
                )
            # 2b. Check for FTS index existence using DuckDB's sqlite_master and information_schema.tables
            try:
                fts_indexes = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fts_idx_%'"
                ).fetchall()
                initialization_summary["fts_indexes_sqlite_master"] = (
                    [row[0] for row in fts_indexes] if fts_indexes else []
                )
                initialization_summary["fts_index_exists_sqlite_master"] = bool(fts_indexes)
                fts_idx_tables = conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'fts_idx_%'"
                ).fetchall()
                initialization_summary["fts_indexes_information_schema"] = (
                    [row[0] for row in fts_idx_tables] if fts_idx_tables else []
                )
                initialization_summary["fts_index_exists_information_schema"] = bool(fts_idx_tables)
            except Exception as e_fts_idx:
                initialization_summary["fts_index_check_error"] = str(e_fts_idx)
            # 3. Drop legacy table
            try:
                table_exists_result = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fts_main_pages'"
                ).fetchone()
                table_exists = table_exists_result[0] > 0 if table_exists_result else False
                if table_exists:
                    conn.execute("DROP TABLE IF EXISTS fts_main_pages;")
                    initialization_summary["legacy_fts_main_pages_dropped"] = True
                else:
                    initialization_summary["legacy_fts_main_pages_dropped"] = False
            except Exception as e_drop:
                initialization_summary["legacy_fts_main_pages_drop_error"] = str(e_drop)
        else:
            initialization_summary["initialization_skipped"] = True
            initialization_summary["read_only_mode"] = read_only
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        initialization_summary["initialization_error"] = str(e)

    # Always run diagnostics
    try:
        diagnostics = await debug_bm25_search(conn, query)
        # Add FTS index check and schema info to diagnostics as well
        try:
            fts_indexes_diag = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fts_idx_%'"
            ).fetchall()
            diagnostics["fts_indexes_sqlite_master"] = (
                [row[0] for row in fts_indexes_diag] if fts_indexes_diag else []
            )
            diagnostics["fts_index_exists_sqlite_master"] = bool(fts_indexes_diag)
            fts_idx_tables_diag = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'fts_idx_%'"
            ).fetchall()
            diagnostics["fts_indexes_information_schema"] = (
                [row[0] for row in fts_idx_tables_diag] if fts_idx_tables_diag else []
            )
            diagnostics["fts_index_exists_information_schema"] = bool(fts_idx_tables_diag)
            # Add schema info
            diagnostics["schema_info"] = schema_info
        except Exception as e_fts_idx_diag:
            diagnostics["fts_index_check_error"] = str(e_fts_idx_diag)
        # Remove BM25 function checks and errors from diagnostics if present
        for k in list(diagnostics.keys()):
            if "bm25" in k or "BM25" in k:
                diagnostics.pop(k)
        # Remove recommendations about BM25 function
        recs = diagnostics.get("recommendations", [])
        diagnostics["recommendations"] = [rec for rec in recs if "match_bm25 function" not in rec]
    except Exception as e:
        logger.error(f"Error during BM25 diagnostics: {e!s}")
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {e!s}")
    finally:
        db.db.close()

    # Compose response
    response = {
        "initialization": initialization_summary,
        "diagnostics": diagnostics,
    }
    # Recommendations: only mention /bm25_diagnostics?initialize=true for FTS index creation if no FTS index is found
    fts_index_exists = response["diagnostics"].get("fts_index_exists_sqlite_master") or response[
        "diagnostics"
    ].get("fts_index_exists_information_schema")
    recs = diagnostics.get("recommendations", [])
    new_recs = [
        rec for rec in recs if "/initialize_bm25" not in rec and "match_bm25 function" not in rec
    ]
    if not fts_index_exists:
        new_recs.append(
            "No FTS indexes found. Run /bm25_diagnostics?initialize=true with a write connection."
        )
    response["recommendations"] = new_recs
    if "status" in diagnostics:
        response["status"] = diagnostics["status"]
    return response
