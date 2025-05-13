#!/usr/bin/env python3
"""
Script to bump the project version across all relevant files.
Usage: python scripts/bump_version.py <new_version>
Example: python scripts/bump_version.py 0.3.0
"""

import os
import re
import sys


def bump_version(new_version: str):
    """
    Bumps the version in all relevant files.

    Args:
        new_version: The new version to set
    """
    # Make sure new_version is in the correct format (e.g., 0.2.0)
    if not re.match(r"^\d+\.\d+\.\d+$", new_version):
        print(f"Error: Version {new_version} is not in the format X.Y.Z")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Files to update with their patterns
    files_to_update = [
        {
            "path": os.path.join(base_dir, "pyproject.toml"),
            "pattern": r'version\s*=\s*"([^"]+)"',
            "replacement": f'version = "{new_version}"',
        },
        {
            "path": os.path.join(base_dir, "src/web_service/main.py"),
            "pattern": r'version="([^"]+)"',
            "replacement": f'version="{new_version}"',
        },
    ]

    for file_info in files_to_update:
        file_path = file_info["path"]
        pattern = file_info["pattern"]
        replacement = file_info["replacement"]

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping")
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Find all matches to report what's being changed
        matches = re.findall(pattern, content)
        if not matches:
            print(f"Warning: No version pattern found in {file_path}, skipping")
            continue

        # Replace all occurrences
        old_version = matches[0]
        new_content = re.sub(pattern, replacement, content)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(new_content)

        print(f"Updated {file_path}: {old_version} → {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <new_version>")
        print("Example: python scripts/bump_version.py 0.3.0")
        sys.exit(1)

    bump_version(sys.argv[1])
    print("Version bump complete!")
