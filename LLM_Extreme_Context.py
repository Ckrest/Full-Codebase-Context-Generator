import os
import ast
import json
import hashlib
import logging
import yaml
from bs4 import BeautifulSoup, Comment
from typing import List, Dict, Optional
from tree_sitter import Parser
import pathspec
from tree_sitter_language_pack import get_parser
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from Start import SETTINGS

# === Files included and excluded in search ===
ALLOWED_EXTENSIONS = set(SETTINGS["allowed_extensions"])
EXCLUDE_DIRS = set(SETTINGS["exclude_dirs"])

# === Universal utilities ===
def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def estimate_tokens(content: str) -> int:
    return int(len(content.split()) * 0.75)

# === Tree-sitter setup ===
#JS_PARSER = get_parser("javascript")

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
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
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

# === Call graph utilities ===
import networkx as nx
from pathlib import Path


def build_call_graph(entries: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    name_to_ids = {}
    for entry in entries:
        if entry["type"] != "function":
            continue
        func_id = f"{entry['file_path']}::{entry['name']}"
        G.add_node(func_id, **entry)
        name_to_ids.setdefault(entry["name"], []).append(func_id)
    for entry in entries:
        if entry["type"] != "function":
            continue
        caller_id = f"{entry['file_path']}::{entry['name']}"
        for callee_name in entry.get("called_functions", []):
            for callee_id in name_to_ids.get(callee_name, []):
                G.add_edge(caller_id, callee_id)
    return G


def save_graph_json(graph: nx.DiGraph, path: Path):
    data = {"nodes": [], "edges": []}
    for node in graph.nodes:
        calls = [t for _, t in graph.out_edges(node)]
        called_by = [s for s, _ in graph.in_edges(node)]
        ndata = {"id": node, **graph.nodes[node], "calls": calls, "called_by": called_by}
        data["nodes"].append(ndata)
    for u, v in graph.edges:
        data["edges"].append({"from": u, "to": v})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

