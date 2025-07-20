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
        pass  # ...existing code...
    return results
