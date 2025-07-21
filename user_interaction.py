"""Utilities for interactive CLI events."""

from __future__ import annotations

import json
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

HISTORY_DIR = Path.home() / ".full_context_history"
HISTORY_DIR.mkdir(exist_ok=True)
_SESSIONS: dict[str, PromptSession] = {}


def _get_session(key: str) -> PromptSession:
    session = _SESSIONS.get(key)
    if session is None:
        hist_file = HISTORY_DIR / f"{key}.txt"
        session = PromptSession(history=FileHistory(str(hist_file)))
        _SESSIONS[key] = session
    return session


def ask_with_history(prompt: str, key: str) -> str:
    """Prompt the user with ``prompt`` using history keyed by ``key``."""
    session = _get_session(key)
    while True:
        answer = session.prompt(prompt)
        if answer.lower() == "settings":
            change_settings_event()
            continue
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

