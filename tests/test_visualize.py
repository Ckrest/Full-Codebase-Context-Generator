import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph import visualize_call_graph


def test_visualize_call_graph(tmp_path):
    data = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [{"from": "A", "to": "B"}],
    }
    out = tmp_path / "graph.png"
    visualize_call_graph(data, str(out))
    assert out.exists()
