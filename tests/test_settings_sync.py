import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Start import DEFAULT_SETTINGS, ensure_example_settings


def test_settings_example_matches_defaults(tmp_path, monkeypatch):
    example = tmp_path / "settings.example.json"
    monkeypatch.chdir(tmp_path)
    ensure_example_settings()
    data = json.loads(example.read_text())
    assert data == {"_comment": "Copy this file to settings.json and modify as needed", **DEFAULT_SETTINGS}


def test_api_settings_defaults():
    api = DEFAULT_SETTINGS["api_settings"]
    assert api["max_output_tokens"] == 5000
    assert api["temperature"] == 0.6

