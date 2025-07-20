import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import LLM_Extreme_Context as lec


def test_extract_from_python(tmp_path):
    code = """
def foo():
    return 1


def bar(x):
    foo()
    return x
"""
    f = tmp_path / "sample.py"
    f.write_text(code)
    results = lec.extract_from_python(str(f))
    names = {r["name"] for r in results}
    assert names == {"foo", "bar"}
    bar_entry = next(r for r in results if r["name"] == "bar")
    assert "foo" in bar_entry["called_functions"]
