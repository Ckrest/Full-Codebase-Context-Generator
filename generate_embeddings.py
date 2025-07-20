import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from context_utils import gather_context

from Start import SETTINGS
def main(project_folder):
    CALL_GRAPH_PATH = Path(SETTINGS["paths"]["output_dir"]) / project_folder / "call_graph.json"
    OUTPUT_DIR = Path(SETTINGS["paths"]["output_dir"]) / project_folder
    EMBEDDING_DIM = SETTINGS["embedding"]["embedding_dim"]
    model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")

    print("Loading embedding model...")
    if not model_path:
        print(
            "encoder_model_path is not set; downloading 'sentence-transformers/all-MiniLM-L6-v2'"
        )
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        model = SentenceTransformer(model_path)

    print(f"Loading call graph from {CALL_GRAPH_PATH} ...")
    with open(CALL_GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    texts = []
    metadata = []

    depth = SETTINGS["context"].get("context_hops", 1)
    limit = SETTINGS["context"].get("max_neighbors", 5)
    bidir = SETTINGS["context"].get("bidirectional", True)
    out_w = SETTINGS["context"].get("outbound_weight", 1.0)
    in_w = SETTINGS["context"].get("inbound_weight", 1.0)

    print("Encoding function nodes...")
    for node in tqdm(nodes, desc="Gathering context", unit="node"):
        name = node.get("name", "")
        context = gather_context(
            graph,
            node["id"],
            depth=depth,
            limit=limit,
            bidirectional=bidir,
            outbound_weight=out_w,
            inbound_weight=in_w,
        )
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
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)
    index.add(embeddings_np)
    faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))

    print(f"✅ Saved embeddings and index to {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        folder = input("Enter the project folder to analyze (relative to output_dir): ").strip()
        if folder:
            main(folder)
