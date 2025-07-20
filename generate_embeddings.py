import json
import os
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from context_utils import gather_context

from Start import SETTINGS
def main():
    CALL_GRAPH_PATH = Path(SETTINGS["output_dir"]) / SETTINGS["default_project"] / "call_graph.json"
    OUTPUT_DIR = Path(SETTINGS["output_dir"]) / SETTINGS["default_project"]
    EMBEDDING_DIM = SETTINGS["embedding_dim"]
    MODEL_NAME = SETTINGS["llm_model"]

    print("Loading embedding model...")
    model_path = SETTINGS.get("local_model_path") or MODEL_NAME
    model = SentenceTransformer(model_path)

    print("Loading call graph...")
    with open(CALL_GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    texts = []
    metadata = []

    depth = SETTINGS.get("context_hops", 1)
    limit = SETTINGS.get("max_neighbors", 5)

    print("Encoding function nodes...")
    for node in nodes:
        name = node.get("name", "")
        context = gather_context(graph, node["id"], depth=depth, limit=limit)
        full_text = f"{name}\n{context}"
        texts.append(full_text)
        metadata.append({
            "id": node["id"],
            "file": node.get("file_path", ""),
            "name": name,
        })

    print(f"Embedding {len(texts)} nodes...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    with open(OUTPUT_DIR / "embedding_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))

    print(f"✅ Saved embeddings and index to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
