from pathlib import Path
import json
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from typing import List

from config import SETTINGS
from llm import get_llm_model
from query import generate_prompt_suggestions, generate_new_prompt

HISTORY_DIR = Path.home() / ".full_context_history"
HISTORY_DIR.mkdir(exist_ok=True)
_SESSIONS: dict[str, PromptSession] = {}
_PROMPT_SUGGESTIONS: List[str] = []


def _get_session(key: str) -> PromptSession:
    session = _SESSIONS.get(key)
    if session is None:
        hist_file = HISTORY_DIR / f"{key}.txt"
        session = PromptSession(history=FileHistory(str(hist_file)))
        _SESSIONS[key] = session
    return session


def ask_with_history(prompt: str, key: str) -> str:
    session = _get_session(key)
    while True:
        answer = session.prompt(prompt)
        if answer.lower() == "settings":
            change_settings_event()
            continue
        return answer


def get_prompt_suggestions() -> List[str]:
    return _PROMPT_SUGGESTIONS


def start_event(path: Path | None = None) -> tuple[Path, str, str]:
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

    llm_model = get_llm_model()
    count = int(SETTINGS.get("query", {}).get("prompt_suggestion_count", 0))
    if count > 0:
        print("[⏳ Working...] Generating prompt suggestions")
        try:
            _PROMPT_SUGGESTIONS[:] = generate_prompt_suggestions(problem, count, llm_model)
            print("[✔ Done]")
        except Exception:
            print("[❌ Failed]")
            _PROMPT_SUGGESTIONS[:] = []

    prompt = ask_search_prompt(_PROMPT_SUGGESTIONS, problem, llm_model)
    return path, problem, prompt


def after_generation_event() -> bool:
    ans = ask_with_history(
        "🔄 Would you like to run another search from scratch? [y/N] ",
        "after_generation",
    ).strip().lower()
    return ans.startswith("y")


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


def ask_search_prompt(suggestions: List[str], problem: str, llm_model) -> str:
    while True:
        print("\n🎯 How do you want to search the codebase? Choose an option:")
        print("  [1] Generate a new search prompt")
        print("  [2] Use your full problem statement")
        print("  [3] Use one of the suggested prompts below:")
        for i, q in enumerate(suggestions, start=3):
            print(f"    {i}) {q}")

        ans = ask_with_history(
            "Select an option or type 'exit' / 'neighbors <n>':",
            "prompt",
        )
        if ans.isdigit():
            choice = int(ans)
            if choice == 1:
                print("[⏳ Working...] Generating new prompt")
                try:
                    new_q = generate_new_prompt(problem, suggestions, llm_model)
                    if new_q:
                        print("[✔ Done]")
                        print(f"Generated prompt: {new_q}")
                        suggestions.append(new_q)
                        return new_q
                    print("[❌ Failed]")
                except Exception:
                    print("[❌ Failed]")
                continue
            if choice == 2:
                return problem
            if 3 <= choice < 3 + len(suggestions):
                return suggestions[choice - 3]
            print("⚠️ Invalid selection. Enter a number from the list, 'exit', or 'neighbors <n>'.")
            continue
        return ans


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
