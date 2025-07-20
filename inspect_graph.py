import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from Start import SETTINGS

def degree_plot(data):
    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"])
    for edge in data["edges"]:
        G.add_edge(edge["from"], edge["to"])
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=50)
    plt.title("Node Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

def load_call_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_graph(data):
    nodes = data["nodes"]
    edges = data["edges"]
    total_nodes = len(nodes)
    total_edges = len(edges)

    file_counter = Counter()
    name_counter = Counter()

    for node in nodes:
        path = node.get("file_path", "unknown")
        name = node.get("name", "unnamed")
        file_counter[path] += 1
        name_counter[name] += 1

    print(f"Nodes: {total_nodes}")
    print(f"Edges: {total_edges}")
    print("\nTop files by function count:")
    for path, count in file_counter.most_common(10):
        print(f"{count:5}  {path}")

    print("\nMost common function names:")
    for name, count in name_counter.most_common(10):
        print(f"{count:5}  {name}")

def main(project_folder):
    extracted_root = Path(SETTINGS["output_dir"])
    selected = extracted_root / project_folder
    call_graph_path = selected / "call_graph.json"
    if not call_graph_path.exists():
        print(f"No call_graph.json found in {selected}.")
        exit(1)

    data = load_call_graph(call_graph_path)
    analyze_graph(data)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        folder = input("Enter the project folder to analyze (relative to output_dir): ").strip()
        if folder:
            main(folder)
