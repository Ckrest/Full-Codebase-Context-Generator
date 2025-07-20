import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from context_utils import expand_neighborhood

from Start import SETTINGS


def main(project_folder):
    """Interactive search of the generated embeddings."""
    MODEL_NAME = SETTINGS["llm_model"]
    BASE_DIR = Path(SETTINGS["output_dir"]) / project_folder
    METADATA_PATH = BASE_DIR / "embedding_metadata.json"
    INDEX_PATH = BASE_DIR / "faiss.index"
    CALL_GRAPH_PATH = BASE_DIR / "call_graph.json"

    TOP_K = SETTINGS["top_k_results"]

    print("🔧 Running... Model, context, and settings info:")
    print(f"Model: {MODEL_NAME}")
    print(f"Context source: {CALL_GRAPH_PATH}")
    print(f"Index file: {INDEX_PATH}")
    print(f"Top-K results: {TOP_K}\n")

    print("🔄 Loading model and index...")
    model_path = SETTINGS.get("local_model_path") or MODEL_NAME
    model = SentenceTransformer(model_path)
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(CALL_GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)
    node_map = {n["id"]: n for n in graph.get("nodes", [])}

    last = None
    while True:
        query = input("🧠 What is the query? (type 'exit' or 'neighbors <n>')\n> ")
        if query.strip().lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        if query.startswith("neighbors"):
            if not last:
                print("No previous search results.")
                continue
            try:
                num = int(query.split()[1]) - 1
                idx = last[num]
            except Exception:
                print("Usage: neighbors <result_number>")
                continue
            meta = metadata[idx]
            nb_ids = expand_neighborhood(
                graph,
                meta["id"],
                depth=SETTINGS.get("context_hops", 1),
                limit=SETTINGS.get("max_neighbors", 5),
            )
            print("Neighbors:")
            for nid in nb_ids:
                node = node_map.get(nid, {})
                print(f"- {node.get('name')} — {node.get('file_path')}")
            print()
            continue

        query_vec = model.encode([query], normalize_embeddings=True)
        distances, indices = index.search(np.array(query_vec).astype(np.float32), TOP_K)
        last = indices[0]

        print("\n🎯 Top Matches:")
        for rank, idx in enumerate(indices[0]):
            meta = metadata[idx]
            score = distances[0][rank]
            print(f"{rank+1}. {meta['name']} — {meta['file']} (score: {score:.3f})")
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        folder = input("Enter the project folder to analyze (relative to output_dir): ").strip()
        if folder:
            main(folder)
