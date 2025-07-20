import json
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from context_utils import expand_neighborhood

# === Settings loader ===
def load_settings():
    """Load settings from settings.json with fallback defaults"""
    default_settings = {
        "llm_model": "BAAI/bge-small-en",
        "output_dir": "extracted",
        "default_project": "ComfyUI",
        "embedding_dim": 384,
        "top_k_results": 20,
        "chunk_size": 1000,
        "context_hops": 1,
        "max_neighbors": 5
    }
    
    settings_path = "settings.json"
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                default_settings.update(settings)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings.json: {e}. Using defaults.")
    else:
        print("settings.json not found. Using default settings.")
    
    return default_settings

# Load settings
SETTINGS = load_settings()

# === CONFIG ===
MODEL_NAME = SETTINGS["llm_model"]
BASE_DIR = Path(SETTINGS["output_dir"]) / SETTINGS["default_project"]
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
METADATA_PATH = BASE_DIR / "embedding_metadata.json"
INDEX_PATH = BASE_DIR / "faiss.index"
CALL_GRAPH_PATH = BASE_DIR / "call_graph.json"

TOP_K = SETTINGS["top_k_results"]

# === INIT ===
print("🔧 Running... Model, context, and settings info:")
print(f"Model: {MODEL_NAME}")
print(f"Context source: {CALL_GRAPH_PATH}")
print(f"Index file: {INDEX_PATH}")
print(f"Top-K results: {TOP_K}\n")

print("🔄 Loading model and index...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(str(INDEX_PATH))
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
with open(CALL_GRAPH_PATH, "r", encoding="utf-8") as f:
    graph = json.load(f)
node_map = {n["id"]: n for n in graph.get("nodes", [])}

# === QUERY LOOP ===
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
        nb_ids = expand_neighborhood(graph, meta["id"], depth=SETTINGS.get("context_hops", 1), limit=SETTINGS.get("max_neighbors", 5))
        print("Neighbors:")
        for nid in nb_ids:
            node = node_map.get(nid, {})
            print(f"- {node.get('name')} — {node.get('file_path')}")
        print()
        continue

    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vec).astype(np.float32), TOP_K)
    last = I[0]

    print("\n🎯 Top Matches:")
    for rank, idx in enumerate(I[0]):
        meta = metadata[idx]
        score = D[0][rank]
        print(f"{rank+1}. {meta['name']} — {meta['file']} (score: {score:.3f})")
    print()
