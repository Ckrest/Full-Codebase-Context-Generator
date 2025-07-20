import os
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_paths(data: dict) -> None:
    """Replace backslashes in any path-like string values."""
    for key, value in data.items():
        if isinstance(value, dict):
            sanitize_paths(value)
        elif isinstance(value, str) and (
            "path" in key.lower() or "dir" in key.lower() or key.lower().endswith("_root")
        ):
            data[key] = value.replace("\\", "/")


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
        sanitize_paths(template)
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
        logger.info("Updated %s with default settings", example_path)


def load_settings():
    """Load settings from ``settings.json`` and ensure all keys are present.

    If the file is missing it will be created using ``DEFAULT_SETTINGS``. When
    an existing file is missing keys, those defaults are added only in memory –
    the file on disk is left untouched. The ``settings.example.json`` file is
    always kept in sync with ``DEFAULT_SETTINGS``.
    """

    ensure_example_settings()

    settings_path = "settings.json"
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"Could not load settings.json: {e}. Using defaults."
            )
            settings = {}
    else:
        logger.info("settings.json not found. Creating one with default settings.")
        settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        sanitize_paths(settings)
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
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


def run_query(project_name):
    from query_sniper import main as query_main
    logger.info("Launching interactive query tool...")
    query_main(project_name)


def main():
    parser = argparse.ArgumentParser(description="LLM Extreme Context")
    parser.add_argument("path", nargs="?", help="Project directory to analyze")
    args = parser.parse_args()

    project_path = Path(args.path) if args.path else None
    if not project_path:
        user = input("Enter path to project directory: ")
        project_path = Path(user.strip())

    if not project_path.exists():
        parser.error(f"Path '{project_path}' does not exist")

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

    run_query(project_name)


if __name__ == "__main__":
    main()
