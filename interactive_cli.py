from pathlib import Path
import json
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from config import SETTINGS
from lazy_loader import safe_lazy_import

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


def ask_with_history(prompt: str, key: str, max_attempts: int = 5) -> str:
    session = _get_session(key)
    attempts = 0
    while True:
        answer = session.prompt(prompt)
        lower = answer.lower().strip()
        if lower in {"exit", "quit"}:
            raise SystemExit
        if lower == "settings":
            change_settings_event()
            continue
        if answer.strip():
            return answer
        attempts += 1
        if attempts >= max_attempts:
            print("Too many invalid responses. Exiting.")
            raise SystemExit


def start_event(path: Path | None = None) -> tuple[Path, str]:
    while not path:
        p = ask_with_history(
            "Enter path to your project folder (absolute or relative). Type 'settings' to configure, or press Enter to cancel:\n> ",
            "project_path",
        )
        if not p:
            path = None
            continue
        path = Path(p.strip())
        if not path.exists():
            print(
                "❌ That path doesn't exist. Try again or type 'settings' to update configuration."
            )
            path = None

    problem = ask_with_history(
        "What technical problem are you trying to solve? (e.g., 'Find all state mutations')\n> ",
        "problem",
    )

    return path, problem


def after_generation_event() -> int:
    print("\n🔄 What would you like to do next?")
    print("  [1] Start over with a new project path")
    print("  [2] Enter a new problem statement")
    ans = ask_with_history(
        "Enter a number or press Enter to exit: ",
        "after_generation",
    ).strip()
    if ans.isdigit():
        choice = int(ans)
        if choice in (1, 2):
            return choice
    return 0


def ask_problem() -> str:
    return ask_with_history(
        "What technical problem are you trying to solve? (e.g., 'Find all state mutations')\n> ",
        "problem",
    ).strip()


def ask_project_folder() -> str:
    return ask_with_history(
        "Enter path to your project folder (absolute or relative). Type 'settings' to configure, or press Enter to cancel:\n> ",
        "project_folder",
    ).strip()



def change_settings_event() -> None:
    import config

    settings_path = Path("settings.json")
    settings = config.load_settings()

    while True:
        print("\n🔧 Current configuration loaded from settings.json:")
        print(json.dumps(settings, indent=2))
        field = input(
            "Enter the setting path you want to change (e.g., query.top_k_results), or press Enter to finish:"
        )
        if not field:
            break
        keys = field.split(".")
        ref = settings
        for k in keys[:-1]:
            if k not in ref or not isinstance(ref[k], dict):
                break
            ref = ref[k]
        else:
            last = keys[-1]
            if last in ref:
                current = ref[last]
                new_val = input(f"New value for {field} (currently: {current}): ")
                try:
                    ref[last] = json.loads(new_val)
                except json.JSONDecodeError:
                    ref[last] = new_val
                continue
        print("⚠️ That setting path isn't valid. Use dot notation like 'query.top_k_results'.")

    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        print("✔ Settings saved to settings.json")
    except Exception as e:
        print(f"💥 Failed to write settings.json: {e}")

    config.reload_settings()
