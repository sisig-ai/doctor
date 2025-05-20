"""Schema definitions for Doctor project database tables."""
# Table creation SQLs and schema helpers for Doctor DB

from src.common.config import VECTOR_SIZE

CREATE_JOBS_TABLE_SQL = """
CREATE OR REPLACE TABLE jobs (
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
CREATE OR REPLACE TABLE pages (
    id VARCHAR PRIMARY KEY,
    url VARCHAR,
    domain VARCHAR,
    raw_text TEXT,
    crawl_date TIMESTAMP,
    tags VARCHAR,  -- JSON string array
    job_id VARCHAR  -- Reference to the job that crawled this page
)
"""

CREATE_DOCUMENT_EMBEDDINGS_TABLE_SQL = f"""
CREATE OR REPLACE TABLE document_embeddings (
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
