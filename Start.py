import argparse
import logging
from pathlib import Path

from user_interaction import start_event, after_generation_event, _load_history
from config import (
    SETTINGS,
    reload_settings,
    DEFAULT_SETTINGS,
    ensure_example_settings,
    load_settings,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_extract(project_path: Path, project_name: str) -> None:
    """Extract functions and build the call graph."""
    from LLM_Extreme_Context import crawl_directory, build_call_graph, save_graph_json

    logger.info("Crawling source files in %s", project_path)
    entries = crawl_directory(str(project_path), respect_gitignore=True)
    logger.info("Building call graph...")
    graph = build_call_graph(entries)
    out_dir = Path(SETTINGS["paths"]["output_dir"]) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "call_graph.json"
    save_graph_json(graph, graph_path)
    logger.info("Saved call graph to %s", graph_path)


def run_generate_embeddings(project_name: str) -> None:
    from generate_embeddings import main as gen_main

    logger.info("Generating embeddings for %s", project_name)
    gen_main(project_name)


def run_query(project_name: str, problem: str | None, prompt: str | None) -> None:
    from query_sniper import main as query_main

    logger.info("Launching query tool...")
    query_main(project_name, problem, initial_query=prompt)


def run_inspect(project_name: str) -> None:
    from inspect_graph import main as inspect_main

    inspect_main(project_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Extreme Context")
    sub = parser.add_subparsers(dest="cmd")

    p_extract = sub.add_parser("extract", help="Extract source and build call graph")
    p_extract.add_argument("path")

    p_embed = sub.add_parser("embed", help="Generate embeddings for project")
    p_embed.add_argument("project")

    p_query = sub.add_parser("query", help="Search embeddings")
    p_query.add_argument("project")
    p_query.add_argument("--problem")
    p_query.add_argument("--prompt")

    p_inspect = sub.add_parser("inspect", help="Inspect call graph")
    p_inspect.add_argument("project")

    parser.add_argument("path", nargs="?", help="Project directory for interactive mode")

    args = parser.parse_args()

    reload_settings()
    _load_history()

    if args.cmd == "extract":
        path = Path(args.path).resolve()
        run_extract(path, path.name)
        return
    if args.cmd == "embed":
        run_generate_embeddings(args.project)
        return
    if args.cmd == "query":
        run_query(args.project, args.problem, args.prompt)
        return
    if args.cmd == "inspect":
        run_inspect(args.project)
        return

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
