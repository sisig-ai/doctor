"""API endpoints for the site map feature."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from src.common.logger import get_logger
from src.web_service.services.map_service import MapService

logger = get_logger(__name__)

router = APIRouter(tags=["map"])


@router.get("/map", response_class=HTMLResponse)
async def get_site_index() -> str:
    """Get an index of all crawled sites.

    Returns:
        HTML page listing all root pages/sites.
    """
    try:
        service = MapService()
        sites = await service.get_all_sites()
        # Log the sites data for debugging
        logger.debug(f"Retrieved {len(sites)} sites from database")
        if sites:
            logger.debug(f"First site data: {sites[0]}")
        return service.format_site_list(sites)
    except Exception as e:
        logger.error(f"Error getting site index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/map/site/{root_page_id}", response_class=HTMLResponse)
async def get_site_tree(root_page_id: str) -> str:
    """Get the hierarchical tree view for a specific site.

    Args:
        root_page_id: The ID of the root page.

    Returns:
        HTML page showing the site's page hierarchy.
    """
    try:
        service = MapService()
        tree = await service.build_page_tree(root_page_id)
        logger.debug(f"Built tree for {root_page_id}: {tree}")
        return service.format_site_tree(tree)
    except Exception as e:
        logger.error(f"Error getting site tree for {root_page_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/map/page/{page_id}", response_class=HTMLResponse)
async def view_page(page_id: str) -> str:
    """View a specific page with navigation.

    Args:
        page_id: The ID of the page to view.

    Returns:
        HTML page with the page content and navigation.
    """
    try:
        service = MapService()

        # Get the page
        page = await service.db_ops.get_page_by_id(page_id)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        # Get navigation context
        navigation = await service.get_navigation_context(page_id)

        # Render the page
        return service.render_page_html(page, navigation)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error viewing page {page_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/map/page/{page_id}/raw", response_class=Response)
async def get_page_raw(page_id: str) -> Response:
    """Get the raw markdown content of a page.

    Args:
        page_id: The ID of the page.

    Returns:
        Raw markdown content.
    """
    try:
        service = MapService()

        # Get the page
        page = await service.db_ops.get_page_by_id(page_id)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        # Return raw text as markdown
        return Response(
            content=page.get("raw_text", ""),
            media_type="text/markdown",
            headers={"Content-Disposition": f'inline; filename="{page.get("title", "page")}.md"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting raw content for page {page_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
