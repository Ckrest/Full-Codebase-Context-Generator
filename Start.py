import os
import re
import json
import argparse
import logging
from pathlib import Path

from user_interaction import start_event, after_generation_event, _load_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration used by all tools
DEFAULT_SETTINGS = {
    "LLM_model": {
        "api_key": "",
        "api_type": "gemini",
        "local_path": "",
    },
    "api_settings": {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_output_tokens": 5000,
    },
    "paths": {
        "output_dir": "extracted",
    },
    "query": {
        "top_k_results": 5,
        "use_spellcheck": False,
        "sub_question_count": 0,
        "prompt_suggestion_count": 0,
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
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".txt",
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
    },
    "embedding": {
        "embedding_dim": 384,
        "encoder_model_path": "",
    },
}



def ensure_example_settings():
    """Synchronize settings.example.json with DEFAULT_SETTINGS."""

    example_path = "settings.example.json"
    template = {"_comment": "Copy this file to settings.json and modify as needed"}
    template.update(DEFAULT_SETTINGS)

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


def load_settings():
    """
    Load settings from `settings.json`, fixing common path formatting errors if needed.

    If the file is missing, it's created with defaults. If it fails to load due
    to a JSON error (often from unescaped backslashes in Windows paths), this
    function will attempt to fix the paths in the raw text and reload.
    """
    ensure_example_settings()

    settings_path = "settings.json"
    settings = {}

    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                logger.info("Successfully loaded settings.json")
        except (json.JSONDecodeError, IOError) as e:
            try:
                with open(settings_path, "r", encoding="utf-8") as f_read:
                    content = f_read.read()

                # This regex finds key-value pairs where the key suggests a path,
                # allowing us to surgically fix path strings without corrupting other data.
                path_pattern = re.compile(
                    r'('
                    # Group 1: The key and opening quote of the value, e.g., '"output_dir": "'
                    r'"(?:[a-zA-Z0-9_]*_)?(?:path|dir|root|model|output|folder|file|location|destination)(?:_[a-zA-Z0-9_]*)?"'
                    r'\s*:\s*"'
                    r')'
                    # Group 2: The value itself, containing the path.
                    r'([^"]*)'
                    # Group 3: The closing quote.
                    r'(")'
                )

                def path_replacer(match):
                    """Normalizes path separators to be valid in JSON."""
                    pre_value, path_value, post_value = match.groups()
                    # Normalize all path separators to a single forward slash first.
                    normalized = path_value.replace('\\\\', '/').replace('\\', '/')
                    # Convert all forward slashes to double backslashes for JSON.
                    fixed_path = normalized.replace('/', '\\\\')
                    return f'{pre_value}{fixed_path}{post_value}'

                fixed_content = path_pattern.sub(path_replacer, content)

                settings = json.loads(fixed_content)
                logger.info("Successfully loaded settings.json after auto-fixing paths.")

            except (json.JSONDecodeError, IOError) as e2:
                logger.error(
                    f"Failed to load settings.json even after attempting to fix paths: {e2}. "
                    "Please check the file for syntax errors. Using defaults."
                )
                settings = {}  # Fallback to empty dict
    else:
        logger.info("settings.json not found. Creating one with default settings.")
        settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
                f.write("\n")
        except IOError as e:
            logger.warning(f"Failed to write settings.json: {e}")

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


SETTINGS = load_settings()


def run_extract(project_path: Path, project_name: str):
    """Extract functions from the project and build the call graph."""
    from LLM_Extreme_Context import (
        crawl_directory,
        build_call_graph,
        save_graph_json,
    )

    logger.info("Crawling source files in %s", project_path)
    entries = crawl_directory(str(project_path), respect_gitignore=True)
    logger.info("Building call graph...")
    graph = build_call_graph(entries)
    out_dir = Path(SETTINGS["paths"]["output_dir"]) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "call_graph.json"
    save_graph_json(graph, graph_path)
    logger.info("Saved call graph to %s", graph_path)


def run_generate_embeddings(project_name):
    from generate_embeddings import main as gen_main
    logger.info("Generating embeddings for %s", project_name)
    gen_main(project_name)


def run_query(project_name, problem, prompt):
    from query_sniper import main as query_main
    logger.info("Launching query tool...")
    query_main(project_name, problem, initial_query=prompt)


def main():
    parser = argparse.ArgumentParser(description="LLM Extreme Context")
    parser.add_argument("path", nargs="?", help="Project directory to analyze")
    args = parser.parse_args()

    _load_history()

    while True:
        if args.path:
            start_values = Path(args.path), None, None
            args.path = None
        else:
            start_values = start_event()

        project_path, problem, prompt = start_values
        project_name = project_path.name
        SETTINGS["default_project"] = project_name
        SETTINGS["project_root"] = str(project_path.resolve())

        out_dir = Path(SETTINGS["paths"]["output_dir"]) / project_name
        call_graph_path = out_dir / "call_graph.json"
        embeddings_path = out_dir / "embeddings.npy"
        index_path = out_dir / "faiss.index"

        if not call_graph_path.exists():
            logger.info("Extracting project and building call graph...")
            run_extract(project_path, project_name)
        else:
            logger.info("Using existing call graph at %s", call_graph_path)

        if not (embeddings_path.exists() and index_path.exists()):
            logger.info("Generating embeddings...")
            run_generate_embeddings(project_name)
        else:
            logger.info("Using existing embeddings at %s", embeddings_path)

        run_query(project_name, problem, prompt)

        if not after_generation_event():
            break


if __name__ == "__main__":
    main()
