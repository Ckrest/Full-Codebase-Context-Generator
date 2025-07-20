# Utilities to format search results as function summaries
from typing import Iterable, Dict, Any, Optional
from pathlib import Path


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
    base_dir: Optional[str] = None,
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
    blocks_full: list[str] = []
    for idx in indices:
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
        code = _truncate_code(node.get("code", ""))

        block_lines = [
            "=" * 40,
            f"Function: {name}",
            f"File: {display_path}",
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

        # Full-path version for saving
        block_full = [
            "=" * 40,
            f"Function: {name}",
            f"File: {file_path}",
            f"Language: {lang}",
            f"Estimated Tokens: {tokens}",
            "",
            "Comments:",
        ]
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
        block_full.append("=" * 40)
        blocks_full.append("\n".join(block_full))

    output = "\n".join(blocks)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(blocks_full))
    return output
