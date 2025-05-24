"""Service for handling site map functionality."""

import html
from typing import Any

import markdown

from src.common.logger import get_logger
from src.lib.database import DatabaseOperations

logger = get_logger(__name__)


class MapService:
    """Service for handling site map operations."""

    def __init__(self) -> None:
        """Initialize the map service.

        Args:
            None.

        Returns:
            None.
        """
        self.db_ops = DatabaseOperations()

    async def get_all_sites(self) -> list[dict[str, Any]]:
        """Get all root pages (sites) in the database, including legacy pages grouped by domain.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: List of root page information.
        """
        # Get all pages that are roots (have no parent)
        root_pages = await self.db_ops.get_root_pages()

        # Group root pages by domain if there are multiple from the same domain
        domain_grouped_roots = await self._group_root_pages_by_domain(root_pages)

        # Also get legacy pages (pages without hierarchy info) and group by domain
        legacy_groups = await self._get_legacy_domain_groups()

        # Combine both lists
        all_sites = domain_grouped_roots + legacy_groups

        # Sort by crawl_date descending
        all_sites.sort(key=lambda x: x.get("crawl_date", ""), reverse=True)

        return all_sites

    async def build_page_tree(self, root_page_id: str) -> dict[str, Any]:
        """Build a hierarchical tree structure for a site.

        Args:
            root_page_id: The ID of the root page.

        Returns:
            dict[str, Any]: Tree structure with nested children.
        """
        # Check if this is a domain group
        if root_page_id.startswith("domain-") or root_page_id.startswith("legacy-domain-"):
            return await self._build_domain_tree(root_page_id)

        # Get all pages in the hierarchy
        pages = await self.db_ops.get_page_hierarchy(root_page_id)

        # Build a map for quick lookup
        page_map = {page["id"]: page for page in pages}

        # Add children list to each page
        for page in pages:
            page["children"] = []

        # Build the tree structure
        root = None
        for page in pages:
            if page["parent_page_id"]:
                # Add to parent's children
                parent = page_map.get(page["parent_page_id"])
                if parent:
                    parent["children"].append(page)
            else:
                # This is the root
                root = page

        return root or {}

    async def _group_root_pages_by_domain(
        self, root_pages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Group root pages by domain if there are multiple from the same domain.

        Args:
            root_pages: List of root pages from get_root_pages.

        Returns:
            list[dict[str, Any]]: List with individual pages and domain groups.
        """
        # Count pages per domain
        domain_counts = {}
        domain_pages = {}

        for page in root_pages:
            domain = page.get("domain", "unknown")
            if domain not in domain_counts:
                domain_counts[domain] = 0
                domain_pages[domain] = []
            domain_counts[domain] += 1
            domain_pages[domain].append(page)

        # Build result list
        result = []

        for domain, count in domain_counts.items():
            if count > 1:
                # Multiple pages from this domain - create a group
                pages = domain_pages[domain]
                oldest_page = min(pages, key=lambda p: p.get("crawl_date", ""))

                synthetic_root = {
                    "id": f"domain-{domain}",
                    "url": f"https://{domain}",
                    "domain": domain,
                    "title": f"{domain} (Multiple Pages)",
                    "crawl_date": oldest_page.get("crawl_date"),
                    "parent_page_id": None,
                    "root_page_id": f"domain-{domain}",
                    "is_synthetic": True,
                    "page_count": count,
                }
                result.append(synthetic_root)
            else:
                # Single page from this domain - keep as is
                result.extend(domain_pages[domain])

        return result

    async def _get_legacy_domain_groups(self) -> list[dict[str, Any]]:
        """Get legacy pages grouped by domain.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: List of synthetic root pages for each domain with legacy pages.
        """
        # Get all legacy pages (pages where root_page_id IS NULL and parent_page_id IS NULL)
        # but exclude pages that are already roots (where root_page_id = id)
        legacy_pages = await self.db_ops.get_legacy_pages()

        # Group by domain
        domain_groups = {}
        for page in legacy_pages:
            domain = page.get("domain", "unknown")
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(page)

        # Create synthetic root pages for each domain only if there are actually legacy pages
        synthetic_roots = []
        for domain, pages in domain_groups.items():
            if len(pages) > 0:  # Only create group if there are legacy pages
                # Find the oldest page in this domain to use as representative
                oldest_page = min(pages, key=lambda p: p.get("crawl_date", ""))

                # Create a synthetic root page
                synthetic_root = {
                    "id": f"legacy-domain-{domain}",  # Different ID format for legacy groups
                    "url": f"https://{domain}",
                    "domain": domain,
                    "title": f"{domain} (Legacy Pages)",
                    "crawl_date": oldest_page.get("crawl_date"),
                    "parent_page_id": None,
                    "root_page_id": f"legacy-domain-{domain}",
                    "is_synthetic": True,  # Flag to indicate this is a synthetic entry
                    "page_count": len(pages),
                }
                synthetic_roots.append(synthetic_root)

        return synthetic_roots

    async def _build_domain_tree(self, domain_group_id: str) -> dict[str, Any]:
        """Build a tree for a domain group.

        Args:
            domain_group_id: The synthetic domain group ID (e.g., 'domain-example.com').

        Returns:
            dict[str, Any]: Tree structure with all pages from that domain.
        """
        # Extract domain from the ID
        domain = domain_group_id.replace("domain-", "").replace("legacy-", "")

        # Get all pages for this domain
        all_domain_pages = await self.db_ops.get_pages_by_domain(domain)

        # Separate pages into groups
        root_pages = []
        legacy_pages = []

        for p in all_domain_pages:
            # Check if it's a proper root page
            is_proper_root = (
                p.get("parent_page_id") is None
                and p.get("root_page_id") is not None
                and p.get("root_page_id") == p["id"]
                and p.get("depth") is not None
                and p.get("depth") >= 0
            )
            if is_proper_root:
                root_pages.append(p)
            else:
                legacy_pages.append(p)

        # Create the root node
        root = {
            "id": domain_group_id,
            "title": f"{domain} ({len(all_domain_pages)} Pages)",
            "url": f"https://{domain}",
            "domain": domain,
            "is_synthetic": True,
            "children": [],
        }

        # Add all pages as direct children (flat structure)
        # First add root pages, then legacy pages
        all_pages = root_pages + legacy_pages
        for page in all_pages:
            page["children"] = []  # Ensure each page has a children list
            root["children"].append(page)

        # Sort children by URL for better organization
        root["children"].sort(key=lambda p: p.get("url", ""))

        return root

    async def get_navigation_context(self, page_id: str) -> dict[str, Any]:
        """Get navigation context for a page (parent, siblings).

        Args:
            page_id: The ID of the page.

        Returns:
            dict[str, Any]: Navigation context with parent and siblings.
        """
        page = await self.db_ops.get_page_by_id(page_id)
        if not page:
            return {}

        context = {
            "current_page": page,
            "parent": None,
            "siblings": [],
            "children": [],
            "root": None,
            "domain_group": None,
        }

        # Get parent page
        if page["parent_page_id"]:
            context["parent"] = await self.db_ops.get_page_by_id(page["parent_page_id"])

        # If no parent page, check if this page belongs to a domain group
        if not context["parent"] and page.get("domain"):
            # Check if there are multiple pages from this domain
            domain_pages = await self.db_ops.get_pages_by_domain(page["domain"])

            # Count root pages from this domain
            root_count = sum(
                1
                for p in domain_pages
                if (
                    p.get("parent_page_id") is None
                    and p.get("root_page_id") is not None
                    and p.get("root_page_id") == p["id"]
                    and p.get("depth") is not None
                    and p.get("depth") >= 0
                )
            )

            if root_count > 1:
                # This page belongs to a domain group
                context["domain_group"] = {
                    "id": f"domain-{page['domain']}",
                    "title": f"{page['domain']} (Multiple Pages)",
                    "url": f"https://{page['domain']}",
                    "is_synthetic": True,
                }

        # Get siblings
        context["siblings"] = await self.db_ops.get_sibling_pages(page_id)

        # Get children
        context["children"] = await self.db_ops.get_child_pages(page_id)

        # Get root page
        if page["root_page_id"]:
            context["root"] = await self.db_ops.get_page_by_id(page["root_page_id"])

        return context

    def render_page_html(self, page: dict[str, Any], navigation: dict[str, Any]) -> str:
        """Render a page's markdown content as HTML with navigation.

        Args:
            page: The page data including raw_text.
            navigation: Navigation context from get_navigation_context.

        Returns:
            str: HTML content with navigation.
        """
        # Convert markdown to HTML
        md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite"])
        content_html = md.convert(page.get("raw_text", ""))

        # Build navigation HTML
        nav_html = self._build_navigation_html(navigation)

        # Build the complete HTML page
        page_title = html.escape(page.get("title") or "Untitled")

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        nav {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 30px;
        }}
        nav a {{
            color: #0066cc;
            text-decoration: none;
            margin-right: 15px;
        }}
        nav a:hover {{
            text-decoration: underline;
        }}
        .breadcrumb {{
            color: #666;
            margin-bottom: 10px;
            font-size: 0.9em;
        }}
        .breadcrumb a {{
            color: #0066cc;
            text-decoration: none;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            overflow-x: auto;
            border-radius: 4px;
        }}
        code {{
            background-color: #f6f8fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        .children-list {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
        }}
        .children-list ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .children-list li {{
            margin: 8px 0;
        }}
        .children-list a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .children-list a:hover {{
            text-decoration: underline;
        }}
        .back-link {{
            margin-bottom: 20px;
            font-size: 1.1em;
        }}
        .back-link a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .back-link a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        {nav_html}
        <h1>{page_title}</h1>
        <article>
            {content_html}
        </article>
        {self._build_children_list_html(navigation.get("children", []))}
    </div>
</body>
</html>"""

        return html_template

    def _build_navigation_html(self, navigation: dict[str, Any]) -> str:
        """Build navigation HTML from navigation context.

        Args:
            navigation: Navigation context.

        Returns:
            str: Navigation HTML.
        """
        nav_parts = []

        # Add Back to Parent link at the very top
        parent = navigation.get("parent")
        domain_group = navigation.get("domain_group")

        if parent:
            # Regular parent page
            nav_parts.append(
                f'<div class="back-link"><a href="/map/page/{parent["id"]}">← Back to {html.escape(parent.get("title") or "Parent")}</a></div>'
            )
        elif domain_group:
            # Parent is a domain group
            nav_parts.append(
                f'<div class="back-link"><a href="/map/site/{domain_group["id"]}">← Back to {html.escape(domain_group.get("title") or "Domain Group")}</a></div>'
            )
        else:
            # No parent - link back to all sites
            nav_parts.append('<div class="back-link"><a href="/map">← Back to All Sites</a></div>')

        # Build breadcrumb
        current = navigation.get("current_page", {})
        root = navigation.get("root")

        breadcrumb_parts = []
        if root and root["id"] != current.get("id"):
            breadcrumb_parts.append(
                f'<a href="/map/page/{root["id"]}">{html.escape(root.get("title") or "Home")}</a>'
            )

        if parent and parent["id"] != root.get("id"):
            breadcrumb_parts.append(
                f'<a href="/map/page/{parent["id"]}">{html.escape(parent.get("title") or "Parent")}</a>'
            )

        breadcrumb_parts.append(
            f"<strong>{html.escape(current.get('title') or 'Current')}</strong>"
        )

        if breadcrumb_parts:
            nav_parts.append(f'<div class="breadcrumb">{" → ".join(breadcrumb_parts)}</div>')

        # Build main navigation
        nav_links = []
        if root and root["id"] != current.get("id"):
            nav_links.append(f'<a href="/map/site/{root["id"]}">Site Map</a>')

        if parent:
            nav_links.append(f'<a href="/map/page/{parent["id"]}">↑ Parent</a>')

        # Add sibling navigation
        siblings = navigation.get("siblings", [])
        if siblings:
            nav_links.append("<span>Siblings:</span>")
            for sibling in siblings[:5]:  # Limit to 5 siblings
                nav_links.append(
                    f'<a href="/map/page/{sibling["id"]}">{html.escape(sibling.get("title") or "Untitled")}</a>'
                )

        if nav_links:
            nav_parts.append(f"<nav>{''.join(nav_links)}</nav>")

        return "\n".join(nav_parts)

    def _build_children_list_html(self, children: list[dict[str, Any]]) -> str:
        """Build HTML list of child pages.

        Args:
            children: List of child pages.

        Returns:
            str: HTML for children list.
        """
        if not children:
            return ""

        items = []
        for child in children:
            title = html.escape(child.get("title") or "Untitled")
            items.append(f'<li><a href="/map/page/{child["id"]}">{title}</a></li>')

        return f"""
        <div class="children-list">
            <h3>Pages in this section:</h3>
            <ul>
                {"".join(items)}
            </ul>
        </div>
        """

    def format_site_list(self, sites: list[dict[str, Any]]) -> str:
        """Format a list of sites as HTML.

        Args:
            sites: List of root pages.

        Returns:
            str: HTML content for site list.
        """
        if not sites:
            return self._empty_sites_html()

        site_items = []
        for site in sites:
            title = html.escape(site.get("title") or "Untitled")
            url = html.escape(site.get("url") or "")
            crawl_date = site.get("crawl_date") or "Unknown"

            # Check if this is a synthetic domain group
            if site.get("is_synthetic"):
                page_count = site.get("page_count", 0)
                site_items.append(f"""
                <div class="site-item">
                    <h3><a href="/map/site/{site["id"]}">{title}</a></h3>
                    <p class="site-url">{url}</p>
                    <p class="site-meta">Domain Group • {page_count} pages</p>
                    <p class="site-meta">First crawled: {crawl_date}</p>
                </div>
                """)
            else:
                site_items.append(f"""
                <div class="site-item">
                    <h3><a href="/map/site/{site["id"]}">{title}</a></h3>
                    <p class="site-url">{url}</p>
                    <p class="site-meta">Crawled: {crawl_date}</p>
                </div>
                """)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Site Map - All Sites</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        .site-item {{
            border: 1px solid #e9ecef;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 4px;
            background-color: #f8f9fa;
        }}
        .site-item h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .site-item a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .site-item a:hover {{
            text-decoration: underline;
        }}
        .site-url {{
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
        }}
        .site-meta {{
            color: #999;
            font-size: 0.85em;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Site Map - All Crawled Sites</h1>
        <p>Select a site to explore its page hierarchy:</p>
        {"".join(site_items)}
    </div>
</body>
</html>"""

        return html_content

    def _empty_sites_html(self) -> str:
        """Return HTML for when no sites are found.

        Args:
            None.

        Returns:
            str: HTML content for empty state.
        """
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Site Map - No Sites</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        h1 {
            color: #2c3e50;
        }
        p {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>No Sites Found</h1>
        <p>No crawled sites are available. Start by crawling a website using the API.</p>
    </div>
</body>
</html>"""

    def format_site_tree(self, tree: dict[str, Any]) -> str:
        """Format a site tree as HTML.

        Args:
            tree: Tree structure from build_page_tree.

        Returns:
            str: HTML content for site tree view.
        """
        if not tree:
            return self._empty_tree_html()

        tree_html = self._build_tree_html(tree, is_root=True, is_last=True, ancestors_last=[])
        site_title = html.escape(tree.get("title") or "Site Map")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{site_title} - Site Map</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        .tree {{
            list-style-type: none;
            padding-left: 0;
        }}
        .tree ul {{
            list-style-type: none;
            padding-left: 0;
            margin-left: 20px;
        }}
        .tree li {{
            display: block;
            margin: 8px 0;
            padding-left: 0;
        }}
        .tree-line {{
            color: #999;
            font-family: inherit;
            white-space: pre;
            display: inline;
        }}
        .tree-node {{
            display: inline-block;
        }}
        .tree a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .tree a:hover {{
            text-decoration: underline;
        }}
        .tree details {{
            display: block;
        }}
        .tree summary {{
            cursor: pointer;
            list-style: none;
            display: inline;
        }}
        .tree summary::-webkit-details-marker {{
            display: none;
        }}
        .tree summary .tree-node::before {{
            content: "▶ ";
            color: #999;
            display: inline-block;
            width: 15px;
            margin-left: 5px;
        }}
        .tree details[open] > summary .tree-node::before {{
            content: "▼ ";
        }}
        .back-link {{
            margin-bottom: 20px;
        }}
        .back-link a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .back-link a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/map">← Back to all sites</a>
        </div>
        <h1>{site_title} - Site Map</h1>
        <ul class="tree">
            {tree_html}
        </ul>
    </div>
</body>
</html>"""

        return html_content

    def _build_tree_html(
        self,
        node: dict[str, Any],
        is_root: bool = True,
        is_last: bool = True,
        ancestors_last: list[bool] = None,
    ) -> str:
        """Recursively build HTML for a tree node with classic tree lines and vertical connectors.

        Args:
            node: Tree node.
            is_root: Whether this is the root node.
            is_last: Whether this node is the last child of its parent.
            ancestors_last: List of booleans indicating if each ancestor was the last child.

        Returns:
            str: HTML for the node and its children.
        """
        if ancestors_last is None:
            ancestors_last = []
        title = html.escape(node.get("title") or "Untitled")
        page_id = node["id"]
        children = node.get("children", [])

        # Build the prefix for the tree line
        prefix = ""
        for is_ancestor_last in ancestors_last:
            prefix += "&nbsp;&nbsp;&nbsp;" if is_ancestor_last else "│&nbsp;&nbsp;"
        if not is_root:
            prefix += "└── " if is_last else "├── "

        if not children:
            # Leaf node - just a link
            return f'<li><span class="tree-line">{prefix}</span><span class="tree-node"><a href="/map/page/{page_id}">{title}</a></span></li>'
        else:
            # Node with children - use collapsible details
            children_html = "".join(
                self._build_tree_html(
                    child,
                    is_root=False,
                    is_last=(i == len(children) - 1),
                    ancestors_last=ancestors_last + [is_last],
                )
                for i, child in enumerate(children)
            )

            if is_root:
                # Root node is always expanded
                return f"""<li>
                    <span class="tree-line"></span><span class="tree-node"><a href="/map/page/{page_id}">{title}</a></span>
                    <ul>{children_html}</ul>
                </li>"""
            else:
                # Non-root nodes are collapsible
                return f"""<li>
                    <details>
                        <summary><span class="tree-line">{prefix}</span><span class="tree-node"><a href="/map/page/{page_id}">{title}</a></span></summary>
                        <ul>{children_html}</ul>
                    </details>
                </li>"""

    def _empty_tree_html(self) -> str:
        """Return HTML for when tree is empty.

        Args:
            None.

        Returns:
            str: HTML content for empty tree.
        """
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Site Map - Empty</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        h1 {
            color: #2c3e50;
        }
        p {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Site Not Found</h1>
        <p>The requested site could not be found.</p>
        <p><a href="/map">Back to all sites</a></p>
    </div>
</body>
</html>"""
