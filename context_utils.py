import json
from collections import deque
from pathlib import Path


def load_graph(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_neighbor_map(graph: dict, bidirectional: bool = True) -> dict:
    """Return adjacency map with weights from a call graph dictionary."""
    neighbors: dict[str, list[tuple[str, float, str]]] = {
        n['id']: [] for n in graph.get('nodes', [])
    }
    for edge in graph.get('edges', []):
        w = float(edge.get('weight', 1))
        neighbors.setdefault(edge['from'], []).append((edge['to'], w, 'out'))
        if bidirectional:
            neighbors.setdefault(edge['to'], []).append((edge['from'], w, 'in'))
    return neighbors


def expand_graph(
    graph: dict,
    node_id: str,
    depth: int = 1,
    limit: int | None = None,
    bidirectional: bool = True,
    outbound_weight: float = 1.0,
    inbound_weight: float = 1.0,
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
        neighbors = neighbor_map.get(current, [])
        neighbors.sort(
            key=lambda x: -(x[1] * (outbound_weight if x[2] == 'out' else inbound_weight))
        )
        for nb, w, direction in neighbors:
            if nb not in visited:
                visited.add(nb)
                result.append(nb)
                if limit and len(result) >= limit:
                    return result
                queue.append((nb, d + 1))
    return result


def gather_context(
    graph: dict,
    node_id: str,
    depth: int = 1,
    limit: int | None = None,
    bidirectional: bool = True,
    outbound_weight: float = 1.0,
    inbound_weight: float = 1.0,
) -> str:
    """Collect code from a node and its neighbors."""
    node_map = {n['id']: n for n in graph.get('nodes', [])}
    base = node_map.get(node_id, {})
    texts = [base.get('code', '')]
    for nb_id in expand_graph(
        graph,
        node_id,
        depth=depth,
        limit=limit,
        bidirectional=bidirectional,
        outbound_weight=outbound_weight,
        inbound_weight=inbound_weight,
    ):
        nb = node_map.get(nb_id)
        if nb:
            texts.append(nb.get('code', ''))
    return '\n'.join(texts)
