from pathlib import Path
import json
from summary_formatter import _truncate_code


def build_prompt(metadata_path, graph_path, indices, user_question):
    """
    Given FAISS result indices, metadata, and call graph, format a Gemini-ready prompt.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    node_map = {n["id"]: n for n in graph.get("nodes", [])}
    blocks = ["# QUESTION", user_question.strip(), "", f"# FUNCTION CONTEXT (Top {len(indices)} Matches)"]

    for i, idx in enumerate(indices):
        meta = metadata[idx]
        node = node_map.get(meta["id"], {})

        name = node.get("name", meta.get("name", "unknown"))
        file_path = node.get("file_path", meta.get("file", "unknown"))
        comments = node.get("comments", [])[:5]
        calls = [node_map.get(cid, {}).get("name", cid) for cid in node.get("calls", [])]
        called_by = [node_map.get(cid, {}).get("name", cid) for cid in node.get("called_by", [])]
        code = _truncate_code(node.get("code", ""))

        section = [
            f"## Function {i + 1}",
            f"ID: {meta['id']}",
            f"Name: {name}",
            f"File: {file_path}",
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

    return "\n\n".join(blocks)


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
