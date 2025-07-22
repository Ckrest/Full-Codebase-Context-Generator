import argparse
import logging
from pathlib import Path
import json

from config import SETTINGS, reload_settings
from lazy_loader import lazy_import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_extract(project_path: Path, project_name: str, visualize: bool = False) -> None:
    graph_mod = lazy_import("graph")
    logger.info("Crawling source files in %s", project_path)
    entries = graph_mod.crawl_directory(str(project_path), respect_gitignore=True)
    logger.info("Building call graph...")
    graph = graph_mod.build_call_graph(entries)
    out_dir = Path(SETTINGS["paths"]["output_dir"]) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "call_graph.json"
    graph_mod.save_graph_json(graph, graph_path)
    logger.info("Saved call graph to %s", graph_path)
    if visualize:
        img_path = graph_path.with_suffix(".png")
        data = json.loads(graph_path.read_text())
        graph_mod.visualize_call_graph(data, str(img_path))
        logger.info("Saved visualization to %s", img_path)


def run_generate_embeddings(project_name: str) -> None:
    logger.info("Generating embeddings for %s", project_name)
    embedding = lazy_import("embedding")
    embedding.generate_embeddings(project_name)


def run_query(project_name: str, problem: str | None, prompt: str | None) -> None:
    logger.info("Launching query tool...")
    query_mod = lazy_import("query")
    query_mod.main(project_name, problem, initial_query=prompt)


def run_inspect(project_name: str) -> None:
    graph_mod = lazy_import("graph")
    extracted_root = Path(SETTINGS["paths"]["output_dir"])
    selected = extracted_root / project_name
    call_graph_path = selected / "call_graph.json"
    if not call_graph_path.exists():
        print(f"No call_graph.json found in {selected}.")
        return
    data = json.loads(call_graph_path.read_text())
    graph_mod.analyze_graph(data)


def run_visualize(project_name: str) -> None:
    graph_mod = lazy_import("graph")
    out_dir = Path(SETTINGS["paths"]["output_dir"]) / project_name
    call_graph_path = out_dir / "call_graph.json"
    if not call_graph_path.exists():
        print(f"No call_graph.json found in {out_dir}.")
        return
    data = json.loads(call_graph_path.read_text())
    img_path = out_dir / "call_graph.png"
    graph_mod.visualize_call_graph(data, str(img_path))
    print(f"Saved visualization to {img_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Extreme Context")
    sub = parser.add_subparsers(dest="cmd")

    p_extract = sub.add_parser("extract", help="Extract source and build call graph")
    p_extract.add_argument("path")
    p_extract.add_argument("--visualize", action="store_true", help="Save graph image after extraction")

    p_embed = sub.add_parser("embed", help="Generate embeddings for project")
    p_embed.add_argument("project")

    p_query = sub.add_parser("query", help="Search embeddings")
    p_query.add_argument("project")
    p_query.add_argument("--problem")
    p_query.add_argument("--prompt")

    p_inspect = sub.add_parser("inspect", help="Inspect call graph")
    p_inspect.add_argument("project")

    p_vis = sub.add_parser("visualize", help="Visualize existing call graph")
    p_vis.add_argument("project")

    parser.add_argument("path", nargs="?", help="Project directory for interactive mode")

    args = parser.parse_args()

    reload_settings()

    visualize_cfg = SETTINGS.get("visualization", {}).get("auto_visualize", False)

    if args.cmd == "extract":
        path = Path(args.path).resolve()
        run_extract(path, path.name, visualize=args.visualize or visualize_cfg)
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
    if args.cmd == "visualize":
        run_visualize(args.project)
        return

    while True:
        if args.path:
            start_values = Path(args.path), None, None
            args.path = None
        else:
            cli = lazy_import("interactive_cli")
            start_values = cli.start_event()

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
            run_extract(project_path, project_name, visualize=visualize_cfg)
        else:
            logger.info("Using existing call graph at %s", call_graph_path)

        if not (embeddings_path.exists() and index_path.exists()):
            logger.info("Generating embeddings...")
            run_generate_embeddings(project_name)
        else:
            logger.info("Using existing embeddings at %s", embeddings_path)

        run_query(project_name, problem, prompt)
        cli = lazy_import("interactive_cli")
        if not cli.after_generation_event():
            break


if __name__ == "__main__":
    main()
