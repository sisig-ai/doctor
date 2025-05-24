# Maps Feature API Examples

This document provides examples of using the Maps feature API endpoints.

## Overview

The Maps feature tracks page hierarchies during crawling, allowing you to navigate crawled sites as they were originally structured. This makes it easy for LLMs to understand site organization and find related content.

## API Endpoints

### 1. Get All Sites - `/map`

Lists all crawled root pages (sites).

**Request:**
```http
GET /map
```

**Response:**
HTML page listing all sites with links to their tree views.

**Example Usage:**
```bash
curl http://localhost:9111/map
```

### 2. View Site Tree - `/map/site/{root_page_id}`

Shows the hierarchical structure of a specific site.

**Request:**
```http
GET /map/site/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
HTML page with collapsible tree view of all pages in the site.

**Example Usage:**
```bash
curl http://localhost:9111/map/site/550e8400-e29b-41d4-a716-446655440000
```

### 3. View Page - `/map/page/{page_id}`

Displays a specific page with navigation links.

**Request:**
```http
GET /map/page/660e8400-e29b-41d4-a716-446655440001
```

**Response:**
HTML page with:
- Breadcrumb navigation
- Rendered markdown content
- Links to parent, siblings, and child pages
- Link back to site map

**Example Usage:**
```bash
curl http://localhost:9111/map/page/660e8400-e29b-41d4-a716-446655440001
```

### 4. Get Raw Content - `/map/page/{page_id}/raw`

Returns the raw markdown content of a page.

**Request:**
```http
GET /map/page/660e8400-e29b-41d4-a716-446655440001/raw
```

**Response:**
```markdown
# Page Title

Raw markdown content of the page...
```

**Example Usage:**
```bash
curl http://localhost:9111/map/page/660e8400-e29b-41d4-a716-446655440001/raw > page.md
```

## Workflow Example

### 1. Crawl a Website

First, crawl a website to populate the hierarchy:

```bash
curl -X POST http://localhost:9111/fetch_url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.example.com",
    "max_pages": 100,
    "tags": ["documentation"]
  }'
```

### 2. View All Sites

Visit the map index to see all crawled sites:

```bash
curl http://localhost:9111/map
```

### 3. Explore Site Structure

Click on a site or use its ID to view the tree structure:

```bash
curl http://localhost:9111/map/site/{root_page_id}
```

### 4. Navigate Pages

Click through pages to explore content with full navigation context.

## Database Schema

The hierarchy is tracked using these fields in the `pages` table:

- `parent_page_id`: ID of the parent page (NULL for root pages)
- `root_page_id`: ID of the root page for this site
- `depth`: Distance from the root (0 for root pages)
- `path`: Relative path from the root
- `title`: Extracted page title

## Integration with LLMs

The Maps feature is particularly useful for LLMs because:

1. **Contextual Understanding**: LLMs can understand how pages relate to each other
2. **Efficient Navigation**: Direct links to related content without searching
3. **Site Structure**: Understanding the organization helps with better responses
4. **No JavaScript**: Clean HTML that's easy for LLMs to parse

## Python Client Example

```python
import httpx
import asyncio

async def explore_site_map():
    async with httpx.AsyncClient() as client:
        # Get all sites
        sites_response = await client.get("http://localhost:9111/map")

        # Parse the HTML to extract site IDs (simplified example)
        # In practice, use an HTML parser like BeautifulSoup

        # Get a specific site's tree
        tree_response = await client.get(
            "http://localhost:9111/map/site/550e8400-e29b-41d4-a716-446655440000"
        )

        # Get raw content of a page
        raw_response = await client.get(
            "http://localhost:9111/map/page/660e8400-e29b-41d4-a716-446655440001/raw"
        )

        return raw_response.text

# Run the example
content = asyncio.run(explore_site_map())
print(content)
```

## Notes

- The Maps feature automatically tracks hierarchy during crawling
- No additional configuration is needed - it works with existing crawl jobs
- Pages maintain their relationships even after crawling is complete
- The UI requires no JavaScript, making it accessible and fast
