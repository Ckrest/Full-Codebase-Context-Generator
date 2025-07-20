from pathlib import Path
import json
from summary_formatter import _truncate_code


def build_prompt(metadata_path, graph_path, indices, user_question, base_dir=None, save_path=None):
    """
    Given FAISS result indices, metadata, and call graph, format a Gemini-ready prompt.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    node_map = {n["id"]: n for n in graph.get("nodes", [])}
    blocks = ["# QUESTION", user_question.strip(), "", f"# FUNCTION CONTEXT (Top {len(indices)} Matches)"]
    blocks_full = ["# QUESTION", user_question.strip(), "", f"# FUNCTION CONTEXT (Top {len(indices)} Matches)"]

    for i, idx in enumerate(indices):
        meta = metadata[idx]
        node = node_map.get(meta["id"], {})

        name = node.get("name", meta.get("name", "unknown"))
        file_path = node.get("file_path", meta.get("file", "unknown"))
        display_path = file_path
        if base_dir:
            try:
                display_path = str(Path(file_path).resolve().relative_to(Path(base_dir).resolve()))
            except Exception:
                pass
        comments = node.get("comments", [])[:5]
        calls = [node_map.get(cid, {}).get("name", cid) for cid in node.get("calls", [])]
        called_by = [node_map.get(cid, {}).get("name", cid) for cid in node.get("called_by", [])]
        code = _truncate_code(node.get("code", ""))

        section = [
            f"## Function {i + 1}",
            f"ID: {meta['id']}",
            f"Name: {name}",
            f"File: {display_path}",
            f"Comments:",
        ]
        section += [f"- {c}" for c in comments] if comments else ["- None"]
        section.append("Calls:")
        section += [f"- {c}" for c in calls] if calls else ["- None"]
        section.append("Called By:")
        section += [f"- {c}" for c in called_by] if called_by else ["- None"]
        section.append("Code:")
        section.append("```python")
        section.append(code)
        section.append("```")

        blocks.append("\n".join(section))

        full_section = [
            f"## Function {i + 1}",
            f"ID: {meta['id']}",
            f"Name: {name}",
            f"File: {file_path}",
            f"Comments:",
        ]
        full_section += [f"- {c}" for c in comments] if comments else ["- None"]
        full_section.append("Calls:")
        full_section += [f"- {c}" for c in calls] if calls else ["- None"]
        full_section.append("Called By:")
        full_section += [f"- {c}" for c in called_by] if called_by else ["- None"]
        full_section.append("Code:")
        full_section.append("```python")
        full_section.append(code)
        full_section.append("```")

        blocks_full.append("\n".join(full_section))

    result = "\n\n".join(blocks)
    if save_path:
        Path(save_path).write_text("\n\n".join(blocks_full), encoding="utf-8")
    return result


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python build_prompt.py <metadata_path> <graph_path> <indices_json> <question>")
        sys.exit(1)

    metadata_file = sys.argv[1]
    graph_file = sys.argv[2]
    indices = json.loads(sys.argv[3])  # e.g. "[5, 12, 8]"
    question = sys.argv[4]

    result = build_prompt(metadata_file, graph_file, indices, question)
    print(result)
