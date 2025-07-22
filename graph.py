import os
import ast
import json
import hashlib
import logging
import yaml
import re
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List

import networkx as nx
from bs4 import BeautifulSoup, Comment
from tree_sitter import Parser
from tree_sitter_language_pack import get_parser
import pathspec
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import SETTINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = set(SETTINGS["extraction"]["allowed_extensions"])
EXCLUDE_DIRS = set(SETTINGS["extraction"]["exclude_dirs"])
COMMENT_LOOKBACK = SETTINGS["extraction"].get("comment_lookback_lines", 3)
TOKEN_ESTIMATE_RATIO = SETTINGS["extraction"].get("token_estimate_ratio", 0.75)
MINIFIED_JS = SETTINGS["extraction"].get("minified_js_detection", {})
VIS_SETTINGS = SETTINGS.get("visualization", {})


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def estimate_tokens(content: str) -> int:
    return int(len(content.split()) * TOKEN_ESTIMATE_RATIO)


JS_PARSER: Parser = get_parser("javascript")
TS_PARSER: Parser = get_parser("typescript")


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

    class FuncVisitor(ast.NodeVisitor):
        def __init__(self, lines: List[str]):
            self.lines = lines
            self.class_stack: list[str] = []
            self.out: List[Dict] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            start = node.lineno - 1
            end_node = node.body[-1]
            end = getattr(end_node, "end_lineno", getattr(end_node, "lineno", start))
            code = "".join(self.lines[start:end])
            comments = [
                self.lines[i].strip()
                for i in range(max(0, start - COMMENT_LOOKBACK), start)
                if self.lines[i].strip().startswith("#")
            ]
            calls: List[str] = []
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    func = n.func
                    if isinstance(func, ast.Name):
                        calls.append(func.id)
                    elif isinstance(func, ast.Attribute):
                        base = func.value
                        if isinstance(base, ast.Name) and base.id not in {"self", "cls"}:
                            calls.append(f"{base.id}.{func.attr}")
                        else:
                            calls.append(func.attr)

            params: Dict[str, str] = {}
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg in {"self", "cls"}:
                    continue
                ann = ast.unparse(arg.annotation).strip() if arg.annotation else ""
                params[arg.arg] = ann
            if node.args.vararg:
                ann = ast.unparse(node.args.vararg.annotation).strip() if node.args.vararg.annotation else ""
                params[f"*{node.args.vararg.arg}"] = ann
            if node.args.kwarg:
                ann = ast.unparse(node.args.kwarg.annotation).strip() if node.args.kwarg.annotation else ""
                params[f"**{node.args.kwarg.arg}"] = ann

            entry = {
                "file_path": filepath,
                "language": "python",
                "type": "function",
                "name": node.name,
                "class": self.class_stack[-1] if self.class_stack else None,
                "code": code,
                "comments": comments,
                "docstring": ast.get_docstring(node) or "",
                "parameters": params,
                "called_functions": calls,
                "imports": import_map,
                "hash": hash_content(code),
                "estimated_tokens": estimate_tokens(code),
            }
            self.out.append(entry)

    visitor = FuncVisitor(lines)
    visitor.visit(tree)
    return visitor.out


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
    """Parse JavaScript/TypeScript files using tree-sitter."""
    results = []
    if "min." in filepath.lower() or looks_minified_js(filepath):
        logger.info(f"Skipping minified JS file: {filepath}")
        return results

    try:
        data = Path(filepath).read_bytes()
    except Exception as e:
        logger.warning(f"Failed to read JS {filepath}: {e}")
        return results

    source = data.decode("utf-8", errors="ignore")
    lines = source.splitlines()
    tree = JS_PARSER.parse(data)

    def get_code(start_byte: int, end_byte: int) -> str:
        return data[start_byte:end_byte].decode("utf-8", errors="ignore")

    def gather_comments(start_line: int) -> List[str]:
        return [
            lines[i].strip()
            for i in range(max(0, start_line - COMMENT_LOOKBACK), start_line)
            if lines[i].strip().startswith("//")
        ]

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children)

        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8", "ignore") if name_node and name_node.text else "anonymous"
            start_line = node.start_point[0]
            code = get_code(node.start_byte, node.end_byte)
            comments = gather_comments(start_line)
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
                "estimated_tokens": estimate_tokens(code),
            })

        elif node.type == "variable_declarator":
            value = node.child_by_field_name("value")
            if value and value.type in {"arrow_function", "function_expression"}:
                name_node = node.child_by_field_name("name")
                name = (
                    name_node.text.decode("utf-8", "ignore") if name_node and name_node.text else "anonymous"
                )
                start_line = value.start_point[0]
                code = get_code(value.start_byte, value.end_byte)
                comments = gather_comments(start_line)
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
                    "estimated_tokens": estimate_tokens(code),
                })

    return results


def extract_from_typescript(filepath: str) -> List[Dict]:
    """Parse TypeScript files using tree-sitter."""
    results = []
    if "min." in filepath.lower() or looks_minified_js(filepath):
        logger.info(f"Skipping minified TypeScript file: {filepath}")
        return results

    try:
        data = Path(filepath).read_bytes()
    except Exception as e:
        logger.warning(f"Failed to read TS {filepath}: {e}")
        return results

    source = data.decode("utf-8", errors="ignore")
    lines = source.splitlines()
    tree = TS_PARSER.parse(data)

    def get_code(start_byte: int, end_byte: int) -> str:
        return data[start_byte:end_byte].decode("utf-8", errors="ignore")

    def gather_comments(start_line: int) -> List[str]:
        return [
            lines[i].strip()
            for i in range(max(0, start_line - COMMENT_LOOKBACK), start_line)
            if lines[i].strip().startswith("//")
        ]

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children)

        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            name = (
                name_node.text.decode("utf-8", "ignore")
                if name_node and name_node.text
                else "anonymous"
            )
            start_line = node.start_point[0]
            code = get_code(node.start_byte, node.end_byte)
            comments = gather_comments(start_line)
            calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
            results.append(
                {
                    "file_path": filepath,
                    "language": "typescript",
                    "type": "function",
                    "name": name,
                    "code": code,
                    "comments": comments,
                    "called_functions": calls,
                    "hash": hash_content(code),
                    "estimated_tokens": estimate_tokens(code),
                }
            )

        elif node.type == "variable_declarator":
            value = node.child_by_field_name("value")
            if value and value.type in {"arrow_function", "function_expression"}:
                name_node = node.child_by_field_name("name")
                name = (
                    name_node.text.decode("utf-8", "ignore")
                    if name_node and name_node.text
                    else "anonymous"
                )
                start_line = value.start_point[0]
                code = get_code(value.start_byte, value.end_byte)
                comments = gather_comments(start_line)
                calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
                results.append(
                    {
                        "file_path": filepath,
                        "language": "typescript",
                        "type": "function",
                        "name": name,
                        "code": code,
                        "comments": comments,
                        "called_functions": calls,
                        "hash": hash_content(code),
                        "estimated_tokens": estimate_tokens(code),
                    }
                )

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


EXTENSION_MAP = {
    ".py": extract_from_python,
    ".js": extract_from_javascript,
    ".mjs": extract_from_javascript,
    ".cjs": extract_from_javascript,
    ".ts": extract_from_typescript,
    ".tsx": extract_from_typescript,
    ".html": extract_from_html,
    ".htm": extract_from_html,
    ".md": extract_from_markdown,
    ".json": extract_from_json,
    ".yaml": extract_from_yaml,
    ".yml": extract_from_yaml,
    ".css": extract_from_txt,
    ".scss": extract_from_txt,
    ".less": extract_from_txt,
    ".vue": extract_from_txt,
    ".svelte": extract_from_txt,
    ".svg": extract_from_txt,
    ".txt": extract_from_txt,
}
SKIPPED_LOG: List[Dict] = []


def load_gitignore(root: str):
    gi = Path(root) / ".gitignore"
    if not gi.exists():
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text().splitlines())


def crawl_directory(root: str, respect_gitignore: bool = True) -> List[Dict]:
    """Walk ``root`` and run extractors on all matching files."""
    all_results: List[Dict] = []
    spec = load_gitignore(root) if respect_gitignore else None

    paths: List[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            paths.append((dirpath, filename))

    for dirpath, filename in tqdm(paths, desc="Scanning files", unit="file"):
        full_path = os.path.join(dirpath, filename)
        rel = os.path.relpath(full_path, root)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.debug("Skipping %s: extension not allowed", rel)
            SKIPPED_LOG.append({"file": rel, "reason": "extension not allowed"})
            continue
        if spec and spec.match_file(rel):
            logger.debug("Skipping %s: ignored by .gitignore", rel)
            SKIPPED_LOG.append({"file": rel, "reason": ".gitignore"})
            continue
        extractor = EXTENSION_MAP.get(ext)
        if extractor:
            entries = extractor(full_path)
            for entry in entries:
                entry["file_path"] = rel
        else:
            logger.debug("Skipping %s: no extractor", rel)
            SKIPPED_LOG.append({"file": rel, "reason": "no extractor"})
            continue
        all_results.extend(entries)
    logger.info("Parsed %s entries, skipped %s files", len(all_results), len(SKIPPED_LOG))
    return all_results


def save_to_jsonl(data: List[Dict], outfile: Path):
    with outfile.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Wrote {len(data)} entries to {outfile}")


def infer_call_graph_roles(G: nx.DiGraph) -> None:
    """Annotate nodes in ``G`` with a call_graph_role."""
    for node in G.nodes:
        calls = list(G.successors(node))
        called_by = list(G.predecessors(node))

        if not called_by and calls:
            role = "entrypoint"
        elif called_by and calls:
            role = "middleware"
        elif called_by and not calls:
            role = "leaf"
        else:
            role = "orphan"

        if len(set(called_by)) >= 3 or Path(G.nodes[node].get("file_path", "")).name == "utils.py":
            role = "utility"

        G.nodes[node]["call_graph_role"] = role


def build_call_graph(entries: List[Dict]) -> nx.DiGraph:
    """Build a call graph using only function definitions."""

    code_entries = [
        e
        for e in entries
        if e.get("language") in {"python", "javascript", "typescript"}
        and e.get("type") == "function"
    ]

    G = nx.DiGraph()
    name_to_ids_global: Dict[str, List[str]] = {}
    name_to_ids_by_file: Dict[str, Dict[str, str]] = {}
    module_to_file: Dict[str, str] = {}

    for entry in code_entries:
        if entry.get("name"):
            node_id = f"{entry['file_path']}::{entry['name']}"
        else:
            node_id = f"{entry['file_path']}::{entry.get('type','item')}::{entry.get('hash','')[:8]}"
        G.add_node(node_id, **entry)
        name_to_ids_global.setdefault(entry["name"], []).append(node_id)
        name_to_ids_by_file.setdefault(entry["file_path"], {})[entry["name"]] = node_id
        module_to_file[Path(entry["file_path"]).stem] = entry["file_path"]

    for entry in code_entries:
        caller_id = f"{entry['file_path']}::{entry['name']}"
        imports = entry.get("imports", {})
        callee_counts: Dict[str, int] = {}
        for callee in entry.get("called_functions", []):
            target_file: str = entry["file_path"]
            func_name: str = callee
            if "." in callee:
                base, func_name = callee.split(".", 1)
                imported = imports.get(base)
                if imported:
                    mod = imported.split(".")[0]
                    target_file = module_to_file.get(mod, target_file)
                else:
                    target_file = module_to_file.get(base, target_file)

            file_funcs = name_to_ids_by_file.get(target_file)
            if file_funcs and func_name in file_funcs:
                cid = file_funcs[func_name]
            else:
                cid = None
            if not cid:
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

    infer_call_graph_roles(G)

    return G


def save_graph_json(graph: nx.DiGraph, path: Path):
    data = {"nodes": [], "edges": []}
    for node in graph.nodes:
        calls = [t for _, t in graph.out_edges(node)]
        called_by = [s for s, _ in graph.in_edges(node)]
        data["nodes"].append({"id": node, **graph.nodes[node], "calls": calls, "called_by": called_by})
    for u, v, d in graph.edges(data=True):
        data["edges"].append({"from": u, "to": v, "weight": d.get("weight", 1)})

    base_data = {"nodes": data["nodes"], "edges": data["edges"]}
    checksum = hashlib.sha256(
        json.dumps(base_data, sort_keys=True).encode("utf-8")
    ).hexdigest()
    data["checksum"] = checksum

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"🗃 Output saved to: {path}")


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


def visualize_call_graph(data: dict, out_path: str) -> None:
    """Render call graph JSON data to an image file."""
    G = nx.DiGraph()
    for node in data.get("nodes", []):
        G.add_node(node["id"])
    for edge in data.get("edges", []):
        G.add_edge(edge.get("from"), edge.get("to"))

    plt.figure(figsize=tuple(VIS_SETTINGS.get("figsize", [12, 10])))
    pos = nx.spring_layout(
        G,
        k=VIS_SETTINGS.get("spring_layout_k", 0.5),
        iterations=VIS_SETTINGS.get("spring_layout_iterations", 20),
    )
    nx.draw(
        G,
        pos,
        labels={n: n.split("::")[-1] for n in G.nodes},
        with_labels=True,
        node_size=VIS_SETTINGS.get("node_size", 1500),
        node_color=VIS_SETTINGS.get("node_color", "skyblue"),
        font_size=VIS_SETTINGS.get("font_size", 8),
        arrows=True,
    )
    out = Path(out_path)
    fmt = out.suffix.lstrip(".") or "png"
    plt.savefig(str(out), format=fmt.upper(), bbox_inches="tight")
    plt.close()


# ===== Context Gathering =====

def build_neighbor_map(graph: dict, bidirectional: bool = True) -> dict:
    neighbors: dict[str, list[tuple[str, float, str]]] = {
        n['id']: [] for n in graph.get('nodes', [])
    }
    for edge in graph.get('edges', []):
        w = float(edge.get('weight', 1))
        neighbors.setdefault(edge['from'], []).append((edge['to'], w, 'out'))
        if bidirectional:
            neighbors.setdefault(edge['to'], []).append((edge['from'], w, 'in'))
    return neighbors


def expand_graph(
    graph: dict,
    node_id: str,
    depth: int = 1,
    limit: int | None = None,
    bidirectional: bool = True,
    outbound_weight: float = 1.0,
    inbound_weight: float = 1.0,
) -> list[str]:
    neighbor_map = build_neighbor_map(graph, bidirectional=bidirectional)
    visited = {node_id}
    result = []
    queue = deque([(node_id, 0)])
    while queue:
        current, d = queue.popleft()
        if d >= depth:
            continue
        neighbors = neighbor_map.get(current, [])
        neighbors.sort(
            key=lambda x: -(x[1] * (outbound_weight if x[2] == 'out' else inbound_weight))
        )
        for nb, w, direction in neighbors:
            if nb not in visited:
                visited.add(nb)
                result.append(nb)
                if limit and len(result) >= limit:
                    return result
                queue.append((nb, d + 1))
    return result


def gather_context(
    graph: dict,
    node_id: str,
    depth: int = 1,
    limit: int | None = None,
    bidirectional: bool = True,
    outbound_weight: float = 1.0,
    inbound_weight: float = 1.0,
) -> str:
    node_map = {n['id']: n for n in graph.get('nodes', [])}
    base = node_map.get(node_id, {})
    texts = [base.get('code', '')]
    for nb_id in expand_graph(
        graph,
        node_id,
        depth=depth,
        limit=limit,
        bidirectional=bidirectional,
        outbound_weight=outbound_weight,
        inbound_weight=inbound_weight,
    ):
        nb = node_map.get(nb_id)
        if nb:
            texts.append(nb.get('code', ''))
    return '\n'.join(texts)


# ===== Inspection =====

def load_call_graph(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_graph(data: dict) -> None:
    nodes = data["nodes"]
    edges = data["edges"]
    total_nodes = len(nodes)
    total_edges = len(edges)

    file_counter = Counter()
    name_counter = Counter()

    for node in nodes:
        path = node.get("file_path", "unknown")
        name = node.get("name", "unnamed")
        file_counter[path] += 1
        name_counter[name] += 1

    print(f"Nodes: {total_nodes}")
    print(f"Edges: {total_edges}")
    print("\nTop files by function count:")
    for path, count in file_counter.most_common(10):
        print(f"{count:5}  {path}")

    print("\nMost common function names:")
    for name, count in name_counter.most_common(10):
        print(f"{count:5}  {name}")
