"""Utilities for interactive CLI events."""

from __future__ import annotations

import json
import atexit
from pathlib import Path

try:
    import readline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    readline = None

HISTORY_FILE = Path.home() / ".full_context_history.json"
READLINE_HISTORY_FILE = Path.home() / ".full_context_readline"
_INPUT_HISTORY: dict[str, list[str]] = {}


def _load_history() -> None:
    """Load input history from ``HISTORY_FILE``."""
    global _INPUT_HISTORY
    if _INPUT_HISTORY:
        return
    if HISTORY_FILE.exists():
        try:
            _INPUT_HISTORY = json.loads(HISTORY_FILE.read_text())
        except Exception:
            _INPUT_HISTORY = {}
    if readline:
        try:
            if READLINE_HISTORY_FILE.exists():
                readline.read_history_file(str(READLINE_HISTORY_FILE))
        except Exception:
            pass
        atexit.register(lambda: _save_readline())


def _save_history() -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(_INPUT_HISTORY, indent=2))
    except Exception:
        pass
    _save_readline()


def _save_readline() -> None:
    if not readline:
        return
    try:
        readline.clear_history()
        for lst in _INPUT_HISTORY.values():
            for item in lst:
                readline.add_history(item)
        readline.write_history_file(str(READLINE_HISTORY_FILE))
    except Exception:
        pass


def ask_with_history(prompt: str, key: str) -> str:
    """Prompt the user with ``prompt`` using history keyed by ``key``."""
    _load_history()
    while True:
        if readline:
            readline.clear_history()
            for item in _INPUT_HISTORY.get(key, []):
                readline.add_history(item)
        answer = input(prompt)
        if answer.lower() == "settings":
            change_settings_event()
            continue
        if answer:
            lst = _INPUT_HISTORY.setdefault(key, [])
            if answer not in lst:
                lst.append(answer)
                if len(lst) > 20:
                    lst.pop(0)
            _save_history()
        return answer


def start_event(path: Path | None = None) -> tuple[Path, str, str]:
    """Ask for path, problem and query prompt."""

    while not path:
        p = ask_with_history("Enter path to project directory: ", "project_path")
        path = Path(p.strip())
        if not path.exists():
            print("Path does not exist. Try again or type 'settings'.")
            path = None

    problem = ask_with_history("What problem are you trying to solve?\n> ", "problem")
    prompt = ask_with_history(
        "What prompt should be used to find related functions?\n> ", "prompt"
    )
    return path, problem, prompt


def after_generation_event() -> bool:
    """Ask the user if they want to start over."""

    ans = ask_with_history("Start over? [y/N] ", "after_generation").strip().lower()
    return ans.startswith("y")


def ask_problem() -> str:
    """Ask the user for the problem statement."""

    return ask_with_history("What problem are you trying to solve?\n> ", "problem").strip()


def ask_project_folder() -> str:
    """Ask for a project folder relative to ``output_dir``."""

    return ask_with_history(
        "Enter the project folder to analyze (relative to output_dir): ",
        "project_folder",
    ).strip()


def ask_search_prompt() -> str:
    """Prompt for the query used to find related functions."""

    return ask_with_history(
        "What prompt should be used to find related functions? (type 'exit' or 'neighbors <n>')\n> ",
        "prompt",
    )


def change_settings_event() -> None:
    """Interactively change values in ``settings.json``."""

    import config

    settings_path = Path("settings.json")
    settings = config.load_settings()

    while True:
        print("\nCurrent settings:")
        print(json.dumps(settings, indent=2))
        field = input("Enter setting path to change (or press Enter to exit): ")
        if not field:
            break
        keys = field.split(".")
        ref = settings
        for k in keys[:-1]:
            if k not in ref or not isinstance(ref[k], dict):
                print("Invalid path")
                break
            ref = ref[k]
        else:
            last = keys[-1]
            if last not in ref:
                print("Invalid key")
                continue
            current = ref[last]
            new_val = input(f"New value for {field} (current: {current}): ")
            try:
                ref[last] = json.loads(new_val)
            except json.JSONDecodeError:
                ref[last] = new_val
            continue
        print("Path not found. Try again.")

    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
    except Exception as e:  # pragma: no cover - don't fail interactivity
        print(f"Failed to save settings: {e}")

    config.reload_settings()

