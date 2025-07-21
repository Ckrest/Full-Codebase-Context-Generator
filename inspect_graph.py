import json
from collections import Counter
from pathlib import Path

from Start import SETTINGS

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
    extracted_root = Path(SETTINGS["paths"]["output_dir"])
    selected = extracted_root / project_folder
    call_graph_path = selected / "call_graph.json"
    if not call_graph_path.exists():
        print(f"No call_graph.json found in {selected}.")
        exit(1)

    data = load_call_graph(call_graph_path)
    analyze_graph(data)


if __name__ == "__main__":
    import sys
    from user_interaction import ask_project_folder

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        folder = ask_project_folder()
        if folder:
            main(folder)
