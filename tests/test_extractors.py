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


def test_build_call_graph_called_by(tmp_path):
    code = """
def foo():
    pass

def bar():
    foo()
"""
    f = tmp_path / "sample.py"
    f.write_text(code)
    entries = lec.extract_from_python(str(f))
    graph = lec.build_call_graph(entries)
    out = tmp_path / "graph.json"
    lec.save_graph_json(graph, out)
    data = json.loads(out.read_text())
    foo_id = f"{f}::foo"
    bar_id = f"{f}::bar"
    node_map = {n["id"]: n for n in data["nodes"]}
    assert bar_id in node_map[foo_id]["called_by"]
    assert foo_id in node_map[bar_id]["calls"]
