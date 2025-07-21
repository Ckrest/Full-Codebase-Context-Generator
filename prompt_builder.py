from pathlib import Path
import json
from typing import Iterable, Dict, Any, Optional


def _truncate_code(code: str, max_lines: int = 40) -> str:
    """Truncate code if longer than ``max_lines`` keeping the head and tail."""
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    half = max_lines // 2
    return "\n".join(lines[:half] + ["..."] + lines[-half:])


def format_summary(
    indices: Iterable[int],
    metadata: Iterable[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
    *,
    save_path: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> str:
    """Return formatted summaries for the selected functions."""
    idx_list = list(indices)
    total = len(idx_list)
    blocks: list[str] = []
    blocks_full: list[str] = []
    for pos, idx in enumerate(idx_list, start=1):
        meta = metadata[idx]
        node = node_map.get(meta.get("id"), {})

        name = node.get("name", meta.get("name"))
        file_path = node.get("file_path", meta.get("file"))
        display_path = file_path
        if base_dir:
            try:
                display_path = str(Path(file_path).resolve().relative_to(Path(base_dir).resolve()))
            except Exception:
                pass
        lang = node.get("language", "unknown")
        tokens = node.get("estimated_tokens", 0)

        comments = node.get("comments", [])[:5]
        calls = [node_map.get(cid, {}).get("name", cid) for cid in node.get("calls", [])]
        called_by = [node_map.get(cid, {}).get("name", cid) for cid in node.get("called_by", [])]
        full_code = node.get("code", "")
        lines = full_code.splitlines()
        truncated = len(lines) > 40
        code = _truncate_code(full_code)

        block_lines = [
            f"======= [{pos} of {total}] =======",
            f"Function: {name} | File: {display_path} | Calls: {len(calls)} | Called By: {len(called_by)}",
        ]
        if tokens > 1000:
            block_lines.append(
                f"⚠️ Warning: function may be too large to analyze effectively (est. {tokens})"
            )
        block_lines += ["", "Comments:"]
        for c in comments:
            block_lines.append(f"  - {c}")
        if not comments:
            block_lines.append("  - None")

        block_lines.append("")
        block_lines.append("Calls:")
        for c in calls:
            block_lines.append(f"  - {c}")
        if not calls:
            block_lines.append("  - None")

        block_lines.append("")
        block_lines.append("Called By:")
        for c in called_by:
            block_lines.append(f"  - {c}")
        if not called_by:
            block_lines.append("  - None")

        block_lines.append("")
        block_lines.append("Code:")
        block_lines.append(code)
        if truncated:
            block_lines.append("📏 Code truncated to first/last 20 lines")
        block_lines.append("=" * 40)
        blocks.append("\n".join(block_lines))

        block_full = [
            f"======= [{pos} of {total}] =======",
            f"Function: {name} | File: {file_path} | Calls: {len(calls)} | Called By: {len(called_by)}",
        ]
        if tokens > 1000:
            block_full.append(
                f"⚠️ Warning: function may be too large to analyze effectively (est. {tokens})"
            )
        block_full += ["", "Comments:"]
        for c in comments:
            block_full.append(f"  - {c}")
        if not comments:
            block_full.append("  - None")
        block_full.append("")
        block_full.append("Calls:")
        for c in calls:
            block_full.append(f"  - {c}")
        if not calls:
            block_full.append("  - None")
        block_full.append("")
        block_full.append("Called By:")
        for c in called_by:
            block_full.append(f"  - {c}")
        if not called_by:
            block_full.append("  - None")
        block_full.append("")
        block_full.append("Code:")
        block_full.append(code)
        if truncated:
            block_full.append("📏 Code truncated to first/last 20 lines")
        block_full.append("=" * 40)
        blocks_full.append("\n".join(block_full))

    output = "\n".join(blocks)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(blocks_full))
        print(f"🗃 Output saved to: {save_path}")
    return output


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
        print(f"🗃 Output saved to: {save_path}")
    return result
