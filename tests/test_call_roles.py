import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import graph

CODE = """
def a():
    b()

def b():
    c()

def c():
    pass

def d():
    pass
"""

def test_call_graph_roles(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text(CODE)
    entries = graph.extract_from_python(str(f))
    G = graph.build_call_graph(entries)
    out = tmp_path / "graph.json"
    graph.save_graph_json(G, out)
    data = json.loads(out.read_text())
    roles = {n["name"]: n["call_graph_role"] for n in data["nodes"]}
    assert roles["a"] == "entrypoint"
    assert roles["b"] == "middleware"
    assert roles["c"] == "leaf"
    assert roles["d"] == "orphan"
