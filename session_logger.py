import json
from pathlib import Path
from datetime import datetime
import json
import re


def slugify(text: str) -> str:
    """Convert text to a simple slug usable in file names."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "query"


def get_timestamp() -> str:
    """Return a UTC timestamp suitable for file names."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H%MZ")


def format_function_entry(node: dict, relevance: dict, graph: dict) -> dict:
    """Return structured JSON for a single function node."""
    callers = []
    for cid in node.get("called_by", []):
        edge = next((e for e in graph.get("edges", []) if e["from"] == cid and e["to"] == node["id"]), None)
        count = edge.get("weight", 1) if edge else 1
        caller_node = next((n for n in graph.get("nodes", []) if n["id"] == cid), {})
        callers.append({
            "function": caller_node.get("name", cid.split("::")[-1]),
            "file": caller_node.get("file_path"),
            "count": count,
        })

    callees = []
    for cid in node.get("calls", []):
        edge = next((e for e in graph.get("edges", []) if e["from"] == node["id"] and e["to"] == cid), None)
        count = edge.get("weight", 1) if edge else 1
        callee_node = next((n for n in graph.get("nodes", []) if n["id"] == cid), {})
        callees.append({
            "function": callee_node.get("name", cid.split("::")[-1]),
            "file": callee_node.get("file_path"),
            "count": count,
        })

    return {
        "function_name": node.get("name"),
        "file": node.get("file_path"),
        "class": node.get("class"),
        "relevance_scores": relevance,
        "call_relations": {
            "callers": callers,
            "callees": callees,
        },
        "call_graph_role": node.get("call_graph_role"),
        "parameters": node.get("parameters", {}),
        "comment": (node.get("docstring") or " ").strip(),
        "code": node.get("code", ""),
    }


def log_session_to_json(data: dict, path: str) -> str:
    """Write session data to ``path`` in JSON format and return file path."""
    logs = Path(path)
    logs.mkdir(parents=True, exist_ok=True)
    slug = slugify(data.get("query", data.get("original_query", "query")))
    fname = f"{get_timestamp()}_{slug}.json"
    full_path = logs / fname
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return str(full_path)


def log_summary_to_markdown(data: dict, path: str) -> str:
    """Write a human-readable summary of ``data`` to ``path`` and return file path."""
    logs = Path(path)
    logs.mkdir(parents=True, exist_ok=True)
    slug = slugify(data.get("original_query", "query"))
    fname = f"query_{slug}_{get_timestamp()}.md"
    full_path = logs / fname

    subqueries = data.get("subqueries", [])
    functions = data.get("functions", {})

    total_sub = len(subqueries)
    unique_funcs = len(functions)
    core_hits = sum(1 for f in functions.values() if len(f.get("subqueries", [])) == total_sub)
    unique_hits = sum(1 for f in functions.values() if len(f.get("subqueries", [])) == 1)
    dup_funcs = sum(1 for f in functions.values() if f.get("duplicate_count", 0) > 1)

    lines = [
        f"# Query Summary",
        "",
        f"Original Query: {data.get('original_query','')}",
        "",
        f"Total subqueries: {total_sub}",
        f"Total unique functions: {unique_funcs}",
        f"Number of core hits: {core_hits}",
        f"Number of unique hits: {unique_hits}",
        f"Functions appearing multiple times: {dup_funcs}",
        "",
        "## Subquery Results",
    ]

    for i, sq in enumerate(subqueries, start=1):
        lines.append("")
        lines.append(f"### {i}. {sq.get('text','')}")
        for fn in sq.get("functions", []):
            meta = functions.get(fn["name"], {})
            tags = []
            if len(meta.get("subqueries", [])) == total_sub:
                tags.append("CORE")
            if meta.get("duplicate_count", 0) > 1:
                tags.append("DUP")
            tag_text = f" [{' ,'.join(tags)}]" if tags else ""
            lines.append(
                f"- {fn['name']} ({fn['file']}, score {fn['score']:.3f}, rank {fn['rank']}){tag_text}"
            )

    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(full_path)
