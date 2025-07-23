import os
import json
import re
import logging
from pathlib import Path
from config import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


def ensure_example_settings():
    """Synchronize settings.example.json with DEFAULT_SETTINGS."""
    example_path = "settings.example.json"
    template = {"_comment": "Copy this file to settings.json and modify as needed"}
    template.update(DEFAULT_SETTINGS)
    # _deprecated_embedding_dim was removed; mention it here for documentation only

    current = {}
    if os.path.exists(example_path):
        try:
            with open(example_path, "r", encoding="utf-8") as f:
                current = json.load(f)
        except (json.JSONDecodeError, IOError):
            current = {}

    if current != template:
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
        logger.info("Updated %s with default settings", example_path)


def fix_paths(content: str) -> str:
    path_pattern = re.compile(r'("(?:[a-zA-Z0-9_]*_)?(?:path|dir|root|model|output|folder|file|location|destination)(?:_[a-zA-Z0-9_]*)?"\s*:\s*")([^"]*)(")')

    def path_replacer(match):
        pre_value, path_value, post_value = match.groups()
        normalized = path_value.replace('\\\\', '/').replace('\\', '/')
        fixed_path = normalized.replace('/', '\\\\')
        return f"{pre_value}{fixed_path}{post_value}"

    return path_pattern.sub(path_replacer, content)
