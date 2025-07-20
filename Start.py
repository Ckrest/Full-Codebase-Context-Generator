import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_settings():
    """Load settings from settings.json with fallback defaults."""
    default_settings = {
        "llm_model": "BAAI/bge-small-en",
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
        "local_model_path": ""
    }

    settings_path = "settings.json"
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                default_settings.update(settings)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load settings.json: {e}. Using defaults.")
    else:
        logger.info("settings.json not found. Using default settings.")

    return default_settings


SETTINGS = load_settings()


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
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate embeddings from call graph")
    sub.add_parser("query", help="Run interactive query")
    sub.add_parser("inspect", help="Inspect call graph")

    args = parser.parse_args()

    if args.command == "generate":
        run_generate_embeddings()
    elif args.command == "query":
        run_query()
    elif args.command == "inspect":
        run_inspect()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
