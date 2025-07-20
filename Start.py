import os
import json
import argparse
import logging

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


def run_generate_embeddings(project_folder):
    from generate_embeddings import main as gen_main
    gen_main(project_folder)


def run_query(project_folder):
    from query_sniper import main as query_main
    query_main(project_folder)


def run_inspect(project_folder):
    from inspect_graph import main as inspect_main
    inspect_main(project_folder)


def main():
    parser = argparse.ArgumentParser(description="LLM Extreme Context")
    sub = parser.add_subparsers(dest="command")

    generate_parser = sub.add_parser("generate", help="Generate embeddings from call graph")
    generate_parser.add_argument("--project", type=str, help="Project folder to analyze")

    query_parser = sub.add_parser("query", help="Run interactive query")
    query_parser.add_argument("--project", type=str, help="Project folder to analyze")

    inspect_parser = sub.add_parser("inspect", help="Inspect call graph")
    inspect_parser.add_argument("--project", type=str, help="Project folder to analyze")

    args = parser.parse_args()

    project_folder = getattr(args, "project", None)
    if not project_folder:
        project_folder = input("Enter the project folder to analyze (relative to output_dir): ").strip()
        if not project_folder:
            print("No project folder provided. Exiting.")
            return

    if args.command == "generate":
        run_generate_embeddings(project_folder)
    elif args.command == "query":
        run_query(project_folder)
    elif args.command == "inspect":
        run_inspect(project_folder)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
