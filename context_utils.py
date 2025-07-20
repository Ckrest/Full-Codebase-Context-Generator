import json
from collections import deque
from pathlib import Path


def load_graph(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_neighbor_map(graph: dict, bidirectional: bool = True) -> dict:
    """Return adjacency map from a call graph dictionary."""
    neighbors = {n['id']: set() for n in graph.get('nodes', [])}
    for edge in graph.get('edges', []):
        neighbors.setdefault(edge['from'], set()).add(edge['to'])
        if bidirectional:
            neighbors.setdefault(edge['to'], set()).add(edge['from'])
    return neighbors


def expand_graph(
    graph: dict,
    node_id: str,
    depth: int = 1,
    limit: int | None = None,
    bidirectional: bool = True,
) -> list[str]:
    """Breadth-first expansion of a node's neighbors."""
    neighbor_map = build_neighbor_map(graph, bidirectional=bidirectional)
    visited = {node_id}
    result = []
    queue = deque([(node_id, 0)])
    while queue:
        current, d = queue.popleft()
        if d >= depth:
            continue
        for nb in neighbor_map.get(current, []):
            if nb not in visited:
                visited.add(nb)
                result.append(nb)
                if limit and len(result) >= limit:
                    return result
                queue.append((nb, d + 1))
    return result


def expand_neighborhood(
    graph: dict, node_id: str, depth: int = 1, limit: int | None = None
) -> list[str]:
    """Backward compatible wrapper for expand_graph."""
    return expand_graph(graph, node_id, depth=depth, limit=limit, bidirectional=True)


def gather_context(
    graph: dict,
    node_id: str,
    depth: int = 1,
    limit: int | None = None,
    bidirectional: bool = True,
) -> str:
    """Collect code from a node and its neighbors."""
    node_map = {n['id']: n for n in graph.get('nodes', [])}
    base = node_map.get(node_id, {})
    texts = [base.get('code', '')]
    for nb_id in expand_graph(
        graph, node_id, depth=depth, limit=limit, bidirectional=bidirectional
    ):
        nb = node_map.get(nb_id)
        if nb:
            texts.append(nb.get('code', ''))
    return '\n'.join(texts)
