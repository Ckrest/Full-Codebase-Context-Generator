# Configuration module for LLM Extreme Context
import os
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration used by all tools
DEFAULT_SETTINGS = {
    "LLM_model": {
        "api_key": "",
        "api_type": "gemini",
        "local_path": "",
        "model_type": "auto",
        "device": "auto",
    },
    "api_settings": {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_output_tokens": 5000,
    },
    "paths": {
        "output_dir": "output",
    },
    "query": {
        "top_k_results": 5,
        "use_spellcheck": False,
        "sub_question_count": 0,
    },
    "context": {
        "context_hops": 1,
        "max_neighbors": 5,
        "bidirectional": True,
        "outbound_weight": 1.0,
        "inbound_weight": 1.0,
    },
    "extraction": {
        "allowed_extensions": [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".mjs",
            ".cjs",
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".txt",
            ".css",
            ".scss",
            ".less",
            ".vue",
            ".svelte",
            ".svg",
            ".html",
            ".htm",
        ],
        "exclude_dirs": [
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            ".idea",
            ".vscode",
            ".pytest_cache",
        ],
        "comment_lookback_lines": 3,
        "token_estimate_ratio": 0.75,
        "minified_js_detection": {
            "max_lines_to_check": 50,
            "single_line_threshold": 2000,
            "required_long_lines": 2,
        },
    },
    "visualization": {
        "figsize": [12, 10],
        "spring_layout_k": 0.5,
        "spring_layout_iterations": 20,
        "node_size": 1500,
        "font_size": 8,
        "node_color": "skyblue",
        "auto_visualize": False,
    },
    "embedding": {
        "encoder_model_path": "",
    },
    "logging": {
        "log_markdown": True,
        "log_json": True,
        "max_functions_to_log": 100,
    },
}


SETTINGS = {}


from settings_setup import ensure_example_settings, fix_paths


def load_settings() -> dict:
    """Load settings from ``settings.json`` or create defaults."""
    ensure_example_settings()
    settings_path = "settings.json"
    settings = {}

    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                logger.info("Successfully loaded settings.json")
        except (json.JSONDecodeError, IOError):
            try:
                with open(settings_path, "r", encoding="utf-8") as f_read:
                    content = f_read.read()
                fixed_content = fix_paths(content)
                settings = json.loads(fixed_content)
                logger.info("Successfully loaded settings.json after auto-fixing paths.")
            except (json.JSONDecodeError, IOError) as e2:
                logger.error(
                    "Failed to load settings.json even after attempting to fix paths: %s. Using defaults.",
                    e2,
                )
                settings = {}
    else:
        logger.info("settings.json not found. Creating one with default settings.")
        settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
                f.write("\n")
        except IOError as e:
            logger.warning("Failed to write settings.json: %s", e)

    def ensure_keys(defaults, target):
        changed = False
        for k, v in defaults.items():
            if k not in target:
                target[k] = v
                changed = True
            elif isinstance(v, dict) and isinstance(target[k], dict):
                if ensure_keys(v, target[k]):
                    changed = True
        return changed

    ensure_keys(DEFAULT_SETTINGS, settings)

    def deep_merge(defaults, overrides):
        result = defaults.copy()
        for k, v in overrides.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    return deep_merge(DEFAULT_SETTINGS, settings)


def reload_settings() -> None:
    """Reload settings from disk into the ``SETTINGS`` dict."""
    SETTINGS.clear()
    SETTINGS.update(load_settings())


# Load settings on import
reload_settings()
