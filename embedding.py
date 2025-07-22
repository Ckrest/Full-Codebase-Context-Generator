import json
from pathlib import Path
from config import SETTINGS
from lazy_loader import safe_lazy_import


def load_embedding_model(model_path: str | None):
    """Load a ``SentenceTransformer`` model or download a default."""
    model_lib = safe_lazy_import("sentence_transformers")
    SentenceTransformer = model_lib.SentenceTransformer
    if not model_path:
        print(
            "encoder_model_path is not set; downloading 'sentence-transformers/all-MiniLM-L6-v2'"
        )
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_path)


def generate_embeddings(project_folder: str) -> None:
    """Generate embeddings and FAISS index for a project."""
    call_graph_path = Path(SETTINGS["paths"]["output_dir"]) / project_folder / "call_graph.json"
    output_dir = Path(SETTINGS["paths"]["output_dir"]) / project_folder
    model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")

    print("Loading embedding model...")
    model = load_embedding_model(model_path)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Detected embedding dimension: {embedding_dim}")

    print(f"Loading call graph from {call_graph_path} ...")
    with open(call_graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    graph_checksum = graph.get("checksum")

    nodes = graph["nodes"]
    texts = []
    metadata = []

    depth = SETTINGS["context"].get("context_hops", 1)
    limit = SETTINGS["context"].get("max_neighbors", 5)
    bidir = SETTINGS["context"].get("bidirectional", True)
    out_w = SETTINGS["context"].get("outbound_weight", 1.0)
    in_w = SETTINGS["context"].get("inbound_weight", 1.0)

    graph_mod = safe_lazy_import("graph")
    np = safe_lazy_import("numpy")
    faiss = safe_lazy_import("faiss")
    tqdm_mod = safe_lazy_import("tqdm")
    tqdm = tqdm_mod.tqdm

    print("Encoding nodes...")
    for node in tqdm(nodes, desc="Gathering context", unit="node"):
        name = node.get("name") or node.get("type", "")
        context = graph_mod.gather_context(
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

    np.save(output_dir / "embeddings.npy", embeddings)

    meta_file = output_dir / "embedding_metadata.json"
    meta_out = {
        "graph_checksum": graph_checksum,
        "records": metadata,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"🗃 Output saved to: {meta_file}")

    index = faiss.IndexFlatIP(embedding_dim)
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)
    index.add(embeddings_np)
    faiss.write_index(index, str(output_dir / "faiss.index"))

    print(f"✅ Saved embeddings and index to {output_dir}")
