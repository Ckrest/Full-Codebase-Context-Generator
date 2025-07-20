import os
import ast
import json
import hashlib
import logging
import yaml
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from tree_sitter import Parser
import pathspec
from tree_sitter_language_pack import  get_parser
from tqdm import tqdm
import re


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Settings loader ===

def load_settings():
    """Load settings from settings.json with fallback defaults"""
    default_settings = {
        "llm_model": "BAAI/bge-small-en",
        "output_dir": "extracted",
        "default_project": "ComfyUI",
        "embedding_dim": 384,
        "top_k_results": 20,
        "chunk_size": 1000,
        "allowed_extensions": [".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".txt", ".html", ".htm"],
        "exclude_dirs": ["__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build", ".idea", ".vscode", ".pytest_cache"]
    }
    
    settings_path = "settings.json"
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                # Merge with defaults
                default_settings.update(settings)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load settings.json: {e}. Using defaults.")
    else:
        logger.info("settings.json not found. Using default settings.")
    
    return default_settings

# Load settings
SETTINGS = load_settings()

# === Files included and excluded in search ===

ALLOWED_EXTENSIONS = set(SETTINGS["allowed_extensions"])

EXCLUDE_DIRS = set(SETTINGS["exclude_dirs"])

# === Universal utilities ===

def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def estimate_tokens(content: str) -> int:
    return len(content.split()) * 0.75

# === Tree-sitter setup ===

JS_PARSER = get_parser("javascript")

# === Extractors ===

def extract_from_python(filepath: str) -> List[Dict]:
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        source = ''.join(lines)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning(f"Failed to parse {filepath}")
        return results

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            end = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else node.body[-1].lineno
            code = ''.join(lines[start:end])
            comments = [lines[i].strip() for i in range(max(0, start - 3), start) if lines[i].strip().startswith("#")]
            calls = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]

            results.append({
                "file_path": filepath,
                "language": "python",
                "type": "function",
                "name": node.name,
                "code": code,
                "comments": comments,
                "called_functions": calls,
                "hash": hash_content(code),
                "estimated_tokens": estimate_tokens(code)
            })
    return results

def extract_from_json(filepath: str) -> List[Dict]:
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.warning(f"Failed to parse JSON: {filepath}")
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

def extract_from_html(filepath: str) -> List[Dict]:
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        logger.warning(f"Failed to read HTML: {filepath}")
        return results

    soup = BeautifulSoup(content, "html.parser")
    sections = []

    for tag in soup.find_all(['head', 'body', 'script', 'style']):
        section = str(tag)
        sections.append({
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

    comments = soup.find_all(string=lambda text: isinstance(text, type(soup.Comment)))
    for c in comments:
        c_str = str(c).strip()
        if c_str:
            sections.append({
                "file_path": filepath,
                "language": "html",
                "type": "html_comment",
                "name": None,
                "code": c_str,
                "comments": [],
                "called_functions": [],
                "hash": hash_content(c_str),
                "estimated_tokens": estimate_tokens(c_str)
            })

    return sections

def extract_from_markdown(filepath: str) -> List[Dict]:
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current = []
    title = "untitled"
    for line in lines:
        if line.strip().startswith("#"):
            if current:
                content = ''.join(current)
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
        content = ''.join(current)
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

def extract_from_yaml(filepath: str) -> List[Dict]:
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        logger.warning(f"Failed to parse YAML: {filepath}")
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
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

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

def extract_called_function_names(code: str) -> List[str]:
    # Dumb static version — improve later with AST
    return re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)


def looks_minified_js(filepath: str, max_lines_to_check=50, single_line_threshold=2000, required_long_lines=2) -> bool:
    """
    Determines if a file looks minified by counting the number of exceedingly long lines.
    This helps avoid false positives from files that might have a single long line
    for legitimate reasons (e.g., a data URI).
    """
    long_line_count = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            # Check the first few lines
            for _, line in zip(range(max_lines_to_check), f):
                if len(line.strip()) > single_line_threshold:
                    long_line_count += 1
                    # Return true only when we've found the required number of long lines
                    if long_line_count >= required_long_lines:
                        return True
    except Exception:
        return False
    
    return False


def extract_from_javascript(filepath: str) -> List[Dict]:
    if "min." in filepath.lower() or looks_minified_js(filepath):
        logger.info(f"Skipping minified JS file: {filepath}")
        SKIPPED_LOG.append({
            "file": filepath,
            "reason": "minified javascript"
        })
        return []

    functions = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return []

    func_pattern = re.compile(r"(function\s+(\w+)|const\s+(\w+)\s*=\s*function|const\s+(\w+)\s*=\s*\([^\)]*\)\s*=>)")
    for i, line in enumerate(lines):
        match = func_pattern.search(line)
        if match:
            name = match.group(2) or match.group(3) or match.group(4) or "anonymous"
            start_line = i
            # Naive block capture (stop at next function or end)
            end_line = start_line + 1
            while end_line < len(lines) and not func_pattern.search(lines[end_line]):
                end_line += 1

            code_body = "".join(lines[start_line:end_line])
            functions.append({
                "type": "function",
                "file_path": filepath,
                "name": name,
                "start_line": start_line + 1,
                "end_line": end_line,
                "code": code_body,
                "comments": "",  # You could scan up if needed
                "called_functions": extract_called_function_names(code_body)
            })

    return functions


def extract_fallback(filepath: str) -> List[Dict]:
    logger.info(f"Using fallback extractor for {filepath}")
    return extract_from_txt(filepath)

# === File routing ===

EXTENSION_MAP = {
    ".py": extract_from_python,
    ".md": extract_from_markdown,
    ".yaml": extract_from_yaml,
    ".yml": extract_from_yaml,
    ".json": extract_from_json,
    ".txt": extract_from_txt,
    ".html": extract_from_html,
    ".htm": extract_from_html,
    ".js": extract_from_javascript,
}

def route_file(filepath: str) -> List[Dict]:
    ext = os.path.splitext(filepath)[1].lower()
    extractor = EXTENSION_MAP.get(ext)

    if extractor:
        return extractor(filepath)

    # Fallback: read a few lines only
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            preview = ''.join(f.readlines()[:20])
    except Exception:
        preview = ""

    SKIPPED_LOG.append({
        "file": filepath,
        "reason": "no extractor",
        "preview": preview[:200]
    })
    return []



# === File crawler ===


SKIPPED_LOG = []

def load_gitignore(path):
    gitignore_path = os.path.join(path, ".gitignore")
    if not os.path.exists(gitignore_path):
        return None
    with open(gitignore_path) as f:
        return pathspec.PathSpec.from_lines("gitwildmatch", f)

def crawl_directory(root: str, respect_gitignore: bool = True) -> List[Dict]:
    all_results = []
    root_abs = os.path.abspath(root)
    spec = load_gitignore(root_abs) if respect_gitignore else None

    total_files = 0
    parsed_files = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if os.path.basename(d) not in EXCLUDE_DIRS]
        for filename in tqdm(filenames, desc=f"Scanning {os.path.relpath(dirpath, root)}", leave=False):
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_abs)
            ext = os.path.splitext(filename)[1].lower()
            total_files += 1

            if ext not in ALLOWED_EXTENSIONS:
                SKIPPED_LOG.append({
                    "file": rel_path,
                    "reason": "extension not whitelisted"
                })
                continue

            if spec and spec.match_file(rel_path):
                SKIPPED_LOG.append({
                    "file": rel_path,
                    "reason": ".gitignore"
                })
                continue

            try:
                result = route_file(full_path)
                if result:
                    parsed_files += 1
                all_results.extend(result)
            except Exception as e:
                logger.warning(f"Failed on {full_path}: {e}")
                SKIPPED_LOG.append({
                    "file": rel_path,
                    "reason": f"exception: {str(e)}"
                })

    logger.info(f"Files seen: {total_files}")
    logger.info(f"Files parsed: {parsed_files}")
    logger.info(f"Files skipped: {len(SKIPPED_LOG)}")

    return all_results


# === Output ===

def save_to_jsonl(data: List[Dict], outfile: str = "function_index.jsonl"):
    with open(outfile, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Wrote {len(data)} entries to {outfile}")

# === Main ===

if __name__ == "__main__":
    import sys
    import networkx as nx
    import matplotlib.pyplot as plt
    from pathlib import Path

    def build_call_graph(entries: List[Dict]) -> nx.DiGraph:
        G = nx.DiGraph()
        name_to_ids = {}  # name → list of full IDs (for resolving calls)

        # First pass: add all nodes with unique IDs
        for entry in entries:
            if entry["type"] != "function":
                continue

            func_id = f"{entry['file_path']}::{entry['name']}"
            G.add_node(func_id, **entry)

            if entry["name"] not in name_to_ids:
                name_to_ids[entry["name"]] = []
            name_to_ids[entry["name"]].append(func_id)

        # Second pass: add edges between unique IDs
        for entry in entries:
            if entry["type"] != "function":
                continue

            caller_id = f"{entry['file_path']}::{entry['name']}"
            for callee_name in entry.get("called_functions", []):
                # Add edge to every matching callee name (fuzzy match across files)
                for callee_id in name_to_ids.get(callee_name, []):
                    G.add_edge(caller_id, callee_id)

        return G


    def save_graph_json(graph: nx.DiGraph, path: Path):
        data = {
            "nodes": [{"id": n, **graph.nodes[n]} for n in graph.nodes],
            "edges": [{"from": u, "to": v} for u, v in graph.edges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def render_call_graph_image(graph: nx.DiGraph, path: Path):
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(graph, k=0.5, iterations=20)
        nx.draw(
            graph,
            pos,
            labels={n: n.split("::")[-1] for n in graph.nodes},
            with_labels=True,
            node_size=1500,
            node_color="skyblue",
            font_size=8,
            arrows=True
        )
        plt.savefig(path, format="PNG", bbox_inches="tight")
        plt.close()

    if len(sys.argv) < 2:
        print("Usage: python extractor.py /path/to/codebase")
        exit(1)

    root_path = Path(sys.argv[1])
    project_name = root_path.name
    output_dir = Path(SETTINGS["output_dir"]) / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = crawl_directory(str(root_path))
    save_to_jsonl(results, output_dir / "function_index.jsonl")

    call_graph = build_call_graph(results)
    save_graph_json(call_graph, output_dir / "call_graph.json")
    render_call_graph_image(call_graph, output_dir / "call_graph.png")

    logger.info(f"Saved output to: {output_dir}")


    if SKIPPED_LOG:
        with open(output_dir / "skipped_files.jsonl", "w", encoding="utf-8") as f:
            for entry in SKIPPED_LOG:
                f.write(json.dumps(entry) + "\n")

