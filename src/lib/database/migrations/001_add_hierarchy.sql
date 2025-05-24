-- Migration: Add hierarchy columns to pages table
-- This migration adds columns to track page relationships for the maps feature

-- Add hierarchy columns to pages table
ALTER TABLE pages ADD COLUMN IF NOT EXISTS parent_page_id VARCHAR;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS root_page_id VARCHAR;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS depth INTEGER DEFAULT 0;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS path TEXT;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS title TEXT;

-- Add foreign key constraints (DuckDB doesn't enforce these, but good for documentation)
-- ALTER TABLE pages ADD CONSTRAINT fk_parent_page FOREIGN KEY (parent_page_id) REFERENCES pages(id);
-- ALTER TABLE pages ADD CONSTRAINT fk_root_page FOREIGN KEY (root_page_id) REFERENCES pages(id);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pages_parent_page_id ON pages(parent_page_id);
CREATE INDEX IF NOT EXISTS idx_pages_root_page_id ON pages(root_page_id);
CREATE INDEX IF NOT EXISTS idx_pages_depth ON pages(depth);

-- Update existing pages to have sensible defaults
-- Set root_page_id to self for existing pages (they become roots)
UPDATE pages
SET root_page_id = id
WHERE root_page_id IS NULL;

-- Set depth to 0 for existing pages
UPDATE pages
SET depth = 0
WHERE depth IS NULL;
