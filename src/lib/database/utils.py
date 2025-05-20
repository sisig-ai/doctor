"""Utility functions for the database module.

This module contains helper functions for common data transformations or
operations used within the database package, such as serializing and
deserializing tags.
"""

import json

from src.common.logger import get_logger

logger = get_logger(__name__)


def serialize_tags(tags: list[str] | None) -> str:
    """Serialize a list of tags to a JSON string.

    Args:
        tags: A list of tag strings, or None.

    Returns:
        A JSON string representation of the tags list.
        Returns an empty list '[]' if tags is None.

    """
    if tags is None:
        return json.dumps([])
    return json.dumps(tags)


def deserialize_tags(tags_json: str | None) -> list[str]:
    """Deserialize a tags JSON string to a list of strings.

    Args:
        tags_json: The JSON string containing the tags, or None.

    Returns:
        A list of tag strings. Returns an empty list if input is None, empty,
        or invalid JSON.

    """
    if not tags_json:
        return []
    try:
        return json.loads(tags_json)
    except json.JSONDecodeError:
        logger.warning(f"Could not decode tags JSON: {tags_json!r}")  # Added !r for better logging
        return []
