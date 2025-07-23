import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_SETTINGS
from settings_setup import ensure_example_settings


def test_settings_example_matches_defaults(tmp_path, monkeypatch):
    example = tmp_path / "settings.example.json"
    monkeypatch.chdir(tmp_path)
    ensure_example_settings()
    data = json.loads(example.read_text())
    expected = {"_comment": "Copy this file to settings.json and modify as needed"}
    expected.update(json.loads(json.dumps(DEFAULT_SETTINGS)))
    assert data == expected


def test_api_settings_defaults():
    api = DEFAULT_SETTINGS["api_settings"]
    assert api["max_output_tokens"] == 5000
    assert api["temperature"] == 0.6


def test_visualization_auto_setting():
    vis = DEFAULT_SETTINGS["visualization"]
    assert vis["auto_visualize"] is False

