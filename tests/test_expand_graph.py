import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import context_utils as cu

SIMPLE_GRAPH = {
    "nodes": [
        {"id": "A"},
        {"id": "B"},
        {"id": "C"},
    ],
    "edges": [
        {"from": "A", "to": "B"},
        {"from": "B", "to": "C"},
    ],
}

def test_expand_graph_directional():
    res = cu.expand_graph(SIMPLE_GRAPH, "B", depth=1, bidirectional=False)
    assert res == ["C"]
    res = cu.expand_graph(SIMPLE_GRAPH, "C", depth=1, bidirectional=False)
    assert res == []


def test_expand_graph_bidirectional():
    res = set(cu.expand_graph(SIMPLE_GRAPH, "B", depth=1, bidirectional=True))
    assert res == {"A", "C"}


def test_expand_graph_weighted():
    graph = {
        "nodes": [
            {"id": "A"},
            {"id": "B"},
            {"id": "C"},
        ],
        "edges": [
            {"from": "A", "to": "B", "weight": 2},
            {"from": "B", "to": "C", "weight": 1},
        ],
    }
    res = cu.expand_graph(
        graph,
        "A",
        depth=2,
        bidirectional=False,
        outbound_weight=2,
        inbound_weight=1,
    )
    assert res == ["B", "C"]


