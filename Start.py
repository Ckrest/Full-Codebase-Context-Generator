import os
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_settings():
    """Load settings from settings.json and ensure all keys are present.

    If the file is missing it will be created with the default values. If it
    exists but is missing keys, those keys will be added and the file updated.
    """

    default_settings = {
        "llm_model": "BAAI/bge-small-en",  # example local model
        "local_model_path": "",
        "output_dir": "extracted",
        "default_project": "ComfyUI",
        "embedding_dim": 384,
        "top_k_results": 20,
        "chunk_size": 1000,
        "context_hops": 1,
        "max_neighbors": 5,
        "allowed_extensions": [
            ".py", ".js", ".ts", ".json", ".yaml", ".yml",
            ".md", ".txt", ".html", ".htm"
        ],
        "exclude_dirs": [
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            "dist", "build", ".idea", ".vscode", ".pytest_cache"
        ],
    }

    settings_path = "settings.json"
    settings = {}

    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"Could not load settings.json: {e}. Using defaults and recreating file."
            )
            settings = {}
    else:
        logger.info("settings.json not found. Creating one with default settings.")

    updated = False
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
            updated = True

    if updated or not os.path.exists(settings_path):
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to write settings.json: {e}")

    merged = default_settings.copy()
    merged.update(settings)
    return merged


SETTINGS = load_settings()


def run_extract(project_path: Path, project_name: str):
    """Extract functions from the project and build the call graph."""
    from LLM_Extreme_Context import (
        extract_from_python,
        extract_from_html,
        extract_from_markdown,
        build_call_graph,
        save_graph_json,
    )

    entries = []
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SETTINGS["exclude_dirs"]]
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in SETTINGS["allowed_extensions"]:
                continue
            fpath = Path(root) / fname
            if ext == ".py":
                entries.extend(extract_from_python(str(fpath)))
            elif ext in {".html", ".htm"}:
                entries.extend(extract_from_html(str(fpath)))
            elif ext in {".md", ".markdown"}:
                entries.extend(extract_from_markdown(str(fpath)))

    graph = build_call_graph(entries)
    out_dir = Path(SETTINGS["output_dir"]) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "call_graph.json"
    save_graph_json(graph, graph_path)
    logger.info("Saved call graph to %s", graph_path)


def run_generate_embeddings():
    from generate_embeddings import main as gen_main
    gen_main()


def run_query():
    from query_sniper import main as query_main
    query_main()


def run_inspect():
    from inspect_graph import main as inspect_main
    inspect_main()


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

    out_dir = Path(SETTINGS["output_dir"]) / project_name
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
        run_generate_embeddings()
    else:
        logger.info("Using existing embeddings at %s", embeddings_path)

    run_query()


if __name__ == "__main__":
    main()
