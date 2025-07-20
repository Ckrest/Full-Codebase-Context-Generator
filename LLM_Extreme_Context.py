import os
import ast
import json
import hashlib
import logging
import yaml
import re
from pathlib import Path
from typing import Dict, List

import networkx as nx
from bs4 import BeautifulSoup, Comment
from tree_sitter import Parser
from tree_sitter_language_pack import get_parser
import pathspec
from tqdm import tqdm
import matplotlib.pyplot as plt

from Start import SETTINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
ALLOWED_EXTENSIONS = set(SETTINGS["extraction"]["allowed_extensions"])
EXCLUDE_DIRS = set(SETTINGS["extraction"]["exclude_dirs"])
COMMENT_LOOKBACK = SETTINGS["extraction"].get("comment_lookback_lines", 3)
TOKEN_ESTIMATE_RATIO = SETTINGS["extraction"].get("token_estimate_ratio", 0.75)
MINIFIED_JS = SETTINGS["extraction"].get("minified_js_detection", {})
VIS_SETTINGS = SETTINGS.get("visualization", {})

# === Utilities ===

def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def estimate_tokens(content: str) -> int:
    return int(len(content.split()) * TOKEN_ESTIMATE_RATIO)

# === Tree-sitter setup ===
JS_PARSER: Parser = get_parser("javascript")

# === Minified JS detection ===
def looks_minified_js(
    filepath: str,
    max_lines_to_check: int = MINIFIED_JS.get("max_lines_to_check", 50),
    single_line_threshold: int = MINIFIED_JS.get("single_line_threshold", 2000),
    required_long_lines: int = MINIFIED_JS.get("required_long_lines", 2),
) -> bool:
    long_lines = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for _, line in zip(range(max_lines_to_check), f):
                if len(line.strip()) > single_line_threshold:
                    long_lines += 1
                    if long_lines >= required_long_lines:
                        return True
    except Exception:
        return False
    return False

# === Extractors ===

def extract_from_python(filepath: str) -> List[Dict]:
    results = []
    try:
        lines = Path(filepath).read_text(encoding="utf-8").splitlines(keepends=True)
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return results
    source = "".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning(f"Failed to parse {filepath}")
        return results
    import_map: Dict[str, str] = {}
    for n in tree.body:
        if isinstance(n, ast.Import):
            for alias in n.names:
                import_map[alias.asname or alias.name] = alias.name
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            for alias in n.names:
                if alias.name == "*":
                    continue
                import_map[alias.asname or alias.name] = f"{mod}.{alias.name}" if mod else alias.name

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            end = node.body[-1].end_lineno if hasattr(node.body[-1], "end_lineno") else node.body[-1].lineno
            code = "".join(lines[start:end])
            comments = [
                lines[i].strip()
                for i in range(max(0, start - COMMENT_LOOKBACK), start)
                if lines[i].strip().startswith("#")
            ]
            calls: List[str] = []
            for n in ast.walk(node):
                if not isinstance(n, ast.Call):
                    continue
                func = n.func
                if isinstance(func, ast.Name):
                    calls.append(func.id)
                elif isinstance(func, ast.Attribute):
                    base = func.value
                    if isinstance(base, ast.Name) and base.id not in {"self", "cls"}:
                        calls.append(f"{base.id}.{func.attr}")
                    else:
                        calls.append(func.attr)
            results.append({
                "file_path": filepath,
                "language": "python",
                "type": "function",
                "name": node.name,
                "code": code,
                "comments": comments,
                "called_functions": calls,
                "imports": import_map,
                "hash": hash_content(code),
                "estimated_tokens": estimate_tokens(code)
            })
    return results


def extract_from_html(filepath: str) -> List[Dict]:
    results = []
    try:
        content = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read HTML {filepath}: {e}")
        return results
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup.find_all(["head", "body", "script", "style"]):
        section = str(tag)
        results.append({
            "file_path": filepath,
            "language": "html",
            "type": "html_section",
            "name": tag.name,
            "code": section,
            "comments": [],
            "called_functions": [],
            "hash": hash_content(section),
            "estimated_tokens": estimate_tokens(section)
        })
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        text = str(c).strip()
        if text:
            results.append({
                "file_path": filepath,
                "language": "html",
                "type": "html_comment",
                "name": None,
                "code": text,
                "comments": [],
                "called_functions": [],
                "hash": hash_content(text),
                "estimated_tokens": estimate_tokens(text)
            })
    return results


def extract_from_javascript(filepath: str) -> List[Dict]:
    results = []
    if "min." in filepath.lower() or looks_minified_js(filepath):
        logger.info(f"Skipping minified JS file: {filepath}")
        return results
    try:
        lines = Path(filepath).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        logger.warning(f"Failed to read JS {filepath}: {e}")
        return results
    pattern = re.compile(r"(function\s+([A-Za-z_]\w+)|const\s+([A-Za-z_]\w+)\s*=\s*function|const\s+([A-Za-z_]\w+)\s*=\s*\([^\)]*\)\s*=>)")
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if not m:
            continue
        name = m.group(2) or m.group(3) or m.group(4) or "anonymous"
        start = i
        end = i + 1
        while end < len(lines) and not pattern.search(lines[end]):
            end += 1
        code = "\n".join(lines[start:end])
        comments = [
            lines[j].strip()
            for j in range(max(0, start - COMMENT_LOOKBACK), start)
            if lines[j].strip().startswith("//")
        ]
        calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
        results.append({
            "file_path": filepath,
            "language": "javascript",
            "type": "function",
            "name": name,
            "code": code,
            "comments": comments,
            "called_functions": calls,
            "hash": hash_content(code),
            "estimated_tokens": estimate_tokens(code)
        })
    return results


def extract_from_markdown(filepath: str) -> List[Dict]:
    results = []
    try:
        lines = Path(filepath).read_text(encoding="utf-8").splitlines(keepends=True)
    except Exception as e:
        logger.warning(f"Failed to read Markdown {filepath}: {e}")
        return results
    current = []
    title = "untitled"
    for line in lines:
        if line.strip().startswith("#"):
            if current:
                content = "".join(current)
                results.append({
                    "file_path": filepath,
                    "language": "markdown",
                    "type": "section",
                    "name": title,
                    "code": content,
                    "comments": [],
                    "called_functions": [],
                    "hash": hash_content(content),
                    "estimated_tokens": estimate_tokens(content)
                })
                current = []
            title = line.strip().lstrip("#").strip()
        current.append(line)
    if current:
        content = "".join(current)
        results.append({
                    "file_path": filepath,
                    "language": "markdown",
                    "type": "section",
                    "name": title,
                    "code": content,
                    "comments": [],
                    "called_functions": [],
                    "hash": hash_content(content),
                    "estimated_tokens": estimate_tokens(content)
        })
    return results


def extract_from_json(filepath: str) -> List[Dict]:
    results = []
    try:
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to parse JSON {filepath}: {e}")
        return results
    flat = json.dumps(data, indent=2)
    results.append({
        "file_path": filepath,
        "language": "json",
        "type": "config_entry",
        "name": None,
        "code": flat,
        "comments": [],
        "called_functions": [],
        "hash": hash_content(flat),
        "estimated_tokens": estimate_tokens(flat)
    })
    return results


def extract_from_yaml(filepath: str) -> List[Dict]:
    results = []
    try:
        data = yaml.safe_load(Path(filepath).read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to parse YAML {filepath}: {e}")
        return results
    flat = json.dumps(data, indent=2)
    results.append({
        "file_path": filepath,
        "language": "yaml",
        "type": "config_entry",
        "name": None,
        "code": flat,
        "comments": [],
        "called_functions": [],
        "hash": hash_content(flat),
        "estimated_tokens": estimate_tokens(flat)
    })
    return results


def extract_from_txt(filepath: str) -> List[Dict]:
    try:
        content = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read text {filepath}: {e}")
        return []
    return [{
        "file_path": filepath,
        "language": "text",
        "type": "raw_text",
        "name": None,
        "code": content,
        "comments": [],
        "called_functions": [],
        "hash": hash_content(content),
        "estimated_tokens": estimate_tokens(content)
    }]

# === File routing & crawler ===
EXTENSION_MAP = {
    ".py": extract_from_python,
    ".js": extract_from_javascript,
    ".html": extract_from_html,
    ".htm": extract_from_html,
    ".md": extract_from_markdown,
    ".json": extract_from_json,
    ".yaml": extract_from_yaml,
    ".yml": extract_from_yaml,
    ".txt": extract_from_txt,
}
SKIPPED_LOG: List[Dict] = []

def load_gitignore(root: str):
    gi = Path(root) / ".gitignore"
    if not gi.exists():
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text().splitlines())


def crawl_directory(root: str, respect_gitignore: bool = True) -> List[Dict]:
    all_results: List[Dict] = []
    spec = load_gitignore(root) if respect_gitignore else None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            rel = os.path.relpath(os.path.join(dirpath, filename), root)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                SKIPPED_LOG.append({"file": rel, "reason": "extension not allowed"})
                continue
            if spec and spec.match_file(rel):
                SKIPPED_LOG.append({"file": rel, "reason": ".gitignore"})
                continue
            extractor = EXTENSION_MAP.get(ext)
            if extractor:
                entries = extractor(os.path.join(dirpath, filename))
            else:
                SKIPPED_LOG.append({"file": rel, "reason": "no extractor"})
                continue
            all_results.extend(entries)
    logger.info(f"Parsed {len(all_results)} entries, skipped {len(SKIPPED_LOG)} files")
    return all_results


def save_to_jsonl(data: List[Dict], outfile: Path):
    with outfile.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Wrote {len(data)} entries to {outfile}")

# === Call graph utilities ===
def build_call_graph(entries: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    name_to_ids_global: Dict[str, List[str]] = {}
    name_to_ids_by_file: Dict[str, Dict[str, str]] = {}
    module_to_file: Dict[str, str] = {}

    for entry in entries:
        if entry.get("type") != "function":
            continue
        func_id = f"{entry['file_path']}::{entry['name']}"
        G.add_node(func_id, **entry)
        name_to_ids_global.setdefault(entry["name"], []).append(func_id)
        name_to_ids_by_file.setdefault(entry["file_path"], {})[entry["name"]] = func_id
        module_to_file[Path(entry["file_path"]).stem] = entry["file_path"]

    for entry in entries:
        if entry.get("type") != "function":
            continue
        caller_id = f"{entry['file_path']}::{entry['name']}"
        imports = entry.get("imports", {})
        callee_counts: Dict[str, int] = {}
        for callee in entry.get("called_functions", []):
            target_file = entry["file_path"]
            func_name = callee
            if "." in callee:
                base, func_name = callee.split(".", 1)
                imported = imports.get(base)
                if imported:
                    mod = imported.split(".")[0]
                    target_file = module_to_file.get(mod, target_file)
                else:
                    target_file = module_to_file.get(base, target_file)

            cid = name_to_ids_by_file.get(target_file, {}).get(func_name)
            if not cid:
                # fallback to any file containing function name
                ids = name_to_ids_global.get(func_name, [])
            else:
                ids = [cid]

            for id_ in ids:
                callee_counts[id_] = callee_counts.get(id_, 0) + 1

        for cid, count in callee_counts.items():
            if G.has_edge(caller_id, cid):
                G[caller_id][cid]["weight"] += count
            else:
                G.add_edge(caller_id, cid, weight=count)

    return G


def save_graph_json(graph: nx.DiGraph, path: Path):
    data = {"nodes": [], "edges": []}
    for node in graph.nodes:
        calls = [t for _, t in graph.out_edges(node)]
        called_by = [s for s, _ in graph.in_edges(node)]
        data["nodes"].append({"id": node, **graph.nodes[node], "calls": calls, "called_by": called_by})
    for u, v, d in graph.edges(data=True):
        data["edges"].append({"from": u, "to": v, "weight": d.get("weight", 1)})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_call_graph_image(graph: nx.DiGraph, path: Path):
    plt.figure(figsize=tuple(VIS_SETTINGS.get("figsize", [12, 10])))
    pos = nx.spring_layout(
        graph,
        k=VIS_SETTINGS.get("spring_layout_k", 0.5),
        iterations=VIS_SETTINGS.get("spring_layout_iterations", 20),
    )
    nx.draw(
        graph,
        pos,
        labels={n: n.split("::")[-1] for n in graph.nodes},
        with_labels=True,
        node_size=VIS_SETTINGS.get("node_size", 1500),
        node_color=VIS_SETTINGS.get("node_color", "skyblue"),
        font_size=VIS_SETTINGS.get("font_size", 8),
        arrows=True,
    )
    plt.savefig(str(path), format="PNG", bbox_inches="tight")
    plt.close()

# === Main ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python LLM_Extream_Compatible.py /path/to/codebase")
        sys.exit(1)
    root = sys.argv[1]
    project = os.path.basename(os.path.abspath(root))
    out_dir = Path(SETTINGS["paths"]["output_dir"]) / project
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = crawl_directory(root)
    save_to_jsonl(entries, out_dir / "function_index.jsonl")

    graph = build_call_graph(entries)
    save_graph_json(graph, out_dir / "call_graph.json")
    render_call_graph_image(graph, out_dir / "call_graph.png")

    if SKIPPED_LOG:
        save_to_jsonl(SKIPPED_LOG, out_dir / "skipped_files.jsonl")
