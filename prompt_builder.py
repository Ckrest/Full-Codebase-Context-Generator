from pathlib import Path
import json
from typing import Iterable, Dict, Any, Optional

from workspace import DataWorkspace


def _truncate_code(code: str, max_lines: int = 40) -> str:
    """Truncate code if longer than ``max_lines`` keeping the head and tail."""
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    half = max_lines // 2
    return "\n".join(lines[:half] + ["..."] + lines[-half:])



def format_function_list(functions: list[dict]) -> str:
    """Return minimal text summaries for ``functions``."""
    blocks = []
    for fn in functions:
        name = fn.get("name") or fn.get("function_name")
        file_path = fn.get("file_path") or fn.get("file")
        code = _truncate_code(fn.get("code", ""))
        blocks.append(f"Function: {name}\nFile: {file_path}\nCode:\n{code}")
    return "\n\n".join(blocks)


def build_context_prompt(problem: str, functions: list[dict], history: list[dict] | None = None) -> str:
    """Build prompt text for the iterative context gathering flow."""
    from llm import NEW_CONTEXT_INSTRUCT

    blocks = [NEW_CONTEXT_INSTRUCT, "", "# PROBLEM:", problem.strip(), ""]
    if history:
        for i, round in enumerate(history, start=1):
            blocks.append(f"# PREVIOUS ROUND {i} RESPONSE:")
            blocks.append(round.get("response", ""))
            blocks.append("")
    if functions:
        blocks.append("# FUNCTIONS:")
        blocks.append(format_function_list(functions))
    return "\n".join(blocks)
