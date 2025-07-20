# Utilities to format search results as function summaries
from typing import Iterable, Dict, Any, Optional


def _truncate_code(code: str, max_lines: int = 40) -> str:
    """Truncate code if longer than ``max_lines`` by keeping head and tail."""
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    head = lines[: max_lines // 2]
    tail = lines[-(max_lines // 2) :]
    return "\n".join(head + ["..."] + tail)


def format_summary(
    indices: Iterable[int],
    metadata: Iterable[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
    *,
    save_path: Optional[str] = None,
) -> str:
    """Generate summary text for given result indices and metadata.

    Parameters
    ----------
    indices: Iterable[int]
        List of indices from the FAISS search results.
    metadata: Iterable[Dict[str, Any]]
        Sequence of metadata objects corresponding to those indices.
    node_map: Dict[str, Dict[str, Any]]
        Mapping from node ID to full node information loaded from the call
        graph.
    save_path: str, optional
        If provided, write the output to this file.
    """
    blocks: list[str] = []
    for idx in indices:
        meta = metadata[idx]
        node = node_map.get(meta.get("id"), {})

        name = node.get("name", meta.get("name"))
        file_path = node.get("file_path", meta.get("file"))
        lang = node.get("language", "unknown")
        tokens = node.get("estimated_tokens", 0)

        comments = node.get("comments", [])[:5]
        calls = [node_map.get(cid, {}).get("name", cid) for cid in node.get("calls", [])]
        called_by = [node_map.get(cid, {}).get("name", cid) for cid in node.get("called_by", [])]
        code = _truncate_code(node.get("code", ""))

        block_lines = [
            "=" * 40,
            f"Function: {name}",
            f"File: {file_path}",
            f"Language: {lang}",
            f"Estimated Tokens: {tokens}",
            "",
            "Comments:",
        ]
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
        block_lines.append("=" * 40)
        blocks.append("\n".join(block_lines))

    output = "\n".join(blocks)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output)
    return output
