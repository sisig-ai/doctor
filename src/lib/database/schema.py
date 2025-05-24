"""Schema definitions for Doctor project database tables."""
# Table creation SQLs and schema helpers for Doctor DB

from src.common.config import VECTOR_SIZE

# Table creation SQL statements
# These constants define the SQL for creating the main tables in the Doctor database.
CREATE_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id VARCHAR PRIMARY KEY,
    start_url VARCHAR,
    status VARCHAR,
    pages_discovered INTEGER DEFAULT 0,
    pages_crawled INTEGER DEFAULT 0,
    max_pages INTEGER,
    tags VARCHAR, -- JSON string array
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    error_message VARCHAR
)
"""

CREATE_PAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pages (
    id VARCHAR PRIMARY KEY,
    url VARCHAR,
    domain VARCHAR,
    raw_text TEXT,
    crawl_date TIMESTAMP,
    tags VARCHAR,  -- JSON string array
    job_id VARCHAR,  -- Reference to the job that crawled this page
    parent_page_id VARCHAR,  -- Reference to parent page for hierarchy
    root_page_id VARCHAR,  -- Reference to root page of the site
    depth INTEGER DEFAULT 0,  -- Distance from root page
    path TEXT,  -- Relative path from root page
    title TEXT  -- Extracted page title
)
"""

CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS document_embeddings (
    id VARCHAR PRIMARY KEY,
    embedding FLOAT[{VECTOR_SIZE}] NOT NULL,
    text_chunk VARCHAR,
    page_id VARCHAR,
    url VARCHAR,
    domain VARCHAR,
    tags VARCHAR[],
    job_id VARCHAR
);
"""

# Extension management SQL
# These constants are used to manage DuckDB extensions.
CHECK_EXTENSION_LOADED_SQL = "SELECT * FROM duckdb_loaded_extensions() WHERE name = '{0}';"
INSTALL_EXTENSION_SQL = "INSTALL {0};"
LOAD_EXTENSION_SQL = "LOAD {0};"

# FTS (Full-Text Search) related SQL
# These constants are used for managing FTS indexes and tables.
CREATE_FTS_INDEX_SQL = "PRAGMA create_fts_index('pages', 'id', 'raw_text', overwrite=1);"
CHECK_FTS_INDEXES_SQL = (
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fts_idx_%'"
)
CHECK_FTS_MAIN_PAGES_TABLE_SQL = (
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fts_main_pages'"
)
DROP_FTS_MAIN_PAGES_TABLE_SQL = "DROP TABLE IF EXISTS fts_main_pages;"

# HNSW (Vector Search) related SQL
# These constants are used for managing HNSW vector search indexes.
SET_HNSW_PERSISTENCE_SQL = "SET hnsw_enable_experimental_persistence = true;"
CHECK_HNSW_INDEX_SQL = (
    "SELECT count(*) FROM duckdb_indexes() WHERE index_name = 'hnsw_index_on_embeddings'"
)
CREATE_HNSW_INDEX_SQL = """
CREATE INDEX hnsw_index_on_embeddings
ON document_embeddings
USING HNSW (embedding)
WITH (metric = 'cosine');
"""

# VSS (Vector Similarity Search) verification SQL
# These constants are used to verify VSS extension functionality.
VSS_ARRAY_TO_STRING_TEST_SQL = "SELECT array_to_string([0.1, 0.2]::FLOAT[], ', ');"
VSS_COSINE_SIMILARITY_TEST_SQL = "SELECT list_cosine_similarity([0.1,0.2],[0.2,0.3]);"

# Table existence check SQL
# Used to check if a table exists in the database.
CHECK_TABLE_EXISTS_SQL = "SELECT count(*) FROM information_schema.tables WHERE table_name = '{0}'"

# Transaction management SQL
# Used for managing transactions and checkpoints.
BEGIN_TRANSACTION_SQL = "BEGIN TRANSACTION"
CHECKPOINT_SQL = "CHECKPOINT"
TEST_CONNECTION_SQL = "SELECT 1"

# DML (Data Manipulation Language) SQL
# Used for inserting and updating data in tables.
INSERT_PAGE_SQL = """
INSERT INTO pages (id, url, domain, raw_text, crawl_date, tags, job_id,
                   parent_page_id, root_page_id, depth, path, title)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Base for dynamic UPDATE job query
UPDATE_JOB_STATUS_BASE_SQL = "UPDATE jobs SET status = ?, updated_at = ?"
