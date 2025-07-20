import json
from pathlib import Path
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell
from llm_utils import get_llm_model, call_llm

from context_utils import expand_graph
from summary_formatter import format_summary
from prompt_builder import build_prompt
from Start import SETTINGS

def build_symspell(metadata):
    """Builds a SymSpell dictionary for spell correction."""
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    print("Building spellcheck dictionary...")
    for item in metadata:
        name = item.get("name", "")
        for token in str(name).split():
            sym.create_dictionary_entry(token, 1)
    return sym


def correct_query(symspell, query: str) -> str:
    """Corrects the query using the SymSpell dictionary."""
    if not symspell:
        return query
    suggestions = symspell.lookup_compound(query, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return query


def average_embeddings(model, texts) -> np.ndarray:
    """Encodes texts and returns the average of their embeddings."""
    vecs = model.encode(list(texts), normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=float)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    return np.mean(vecs, axis=0, keepdims=True)


def main(project_folder):
    """Interactive search of the generated embeddings."""
    # --- Configuration Loading ---
    model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")
    BASE_DIR = Path(SETTINGS["paths"]["output_dir"]) / project_folder
    METADATA_PATH = BASE_DIR / "embedding_metadata.json"
    INDEX_PATH = BASE_DIR / "faiss.index"
    CALL_GRAPH_PATH = BASE_DIR / "call_graph.json"
    TOP_K = SETTINGS["query"]["top_k_results"]

    print("🔧 Running... Model, context, and settings info:")
    print(f"Encoder model: {model_path}")
    print(f"Context source: {CALL_GRAPH_PATH}")
    print(f"Index file: {INDEX_PATH}")
    print(f"Top-K results: {TOP_K}\n")

    # --- Model and Data Loading ---
    print("🚀 Loading models and data...")
    if not model_path:
        print(
            "encoder_model_path is not set; downloading 'sentence-transformers/all-MiniLM-L6-v2'"
        )
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        model = SentenceTransformer(model_path)
    llm_model = get_llm_model()
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(CALL_GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)
    node_map = {n["id"]: n for n in graph.get("nodes", [])}

    # --- Optional Tools Initialization ---
    symspell = None
    if SETTINGS["query"].get("use_spellcheck"):
        symspell = build_symspell(metadata)

    sub_question_count = int(SETTINGS["query"].get("sub_question_count", 0))
    use_sub_questions = sub_question_count > 0
    RESULTS_FILE = BASE_DIR / "results.txt"

    # --- Main Interactive Loop ---
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
            except (ValueError, IndexError):
                print("Usage: neighbors <result_number>")
                continue
            
            meta = metadata[idx]
            nb_ids = expand_graph(
                graph,
                meta["id"],
                depth=SETTINGS["context"].get("context_hops", 1),
                limit=SETTINGS["context"].get("max_neighbors", 5),
                bidirectional=SETTINGS["context"].get("bidirectional", True),
                outbound_weight=SETTINGS["context"].get("outbound_weight", 1.0),
                inbound_weight=SETTINGS["context"].get("inbound_weight", 1.0),
            )
            print("Neighbors:")
            for nid in nb_ids:
                node = node_map.get(nid, {})
                print(f"- {node.get('name')} — {node.get('file_path')}")
            print()
            continue

        queries = [correct_query(symspell, query)]
        sub_queries = []
        if use_sub_questions and llm_model:
            print("Generating sub-queries...")
            prompt = (
                "You are helping search a codebase. Given the following question, "
                f"break it into {sub_question_count} distinct sub-questions. Each one should represent a different aspect "
                "of the original question that might be independently searchable. Be concise, specific, "
                "and prefer technical phrasing.\n\n# Original Question:\n"
                f"{query}\n\n# Sub-Queries:"
            )
            sub_text = call_llm(llm_model, prompt)
            for line in sub_text.splitlines():
                t = line.strip()
                if not t:
                    continue
                if t[0].isdigit():
                    t = t.split(".", 1)[-1].strip()
                sub_queries.append(t)
            if sub_queries:
                sub_queries = sub_queries[:sub_question_count]
                queries.extend(correct_query(symspell, q) for q in sub_queries)
        elif use_sub_questions:
            print("Sub-question generation skipped because no LLM model was available.")

        if sub_queries or use_sub_questions:
            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(f"Original query: {query}\n")
                if sub_queries:
                    f.write("Sub-queries:\n")
                    for q in sub_queries:
                        f.write(f"- {q}\n")
                else:
                    f.write("Sub-queries: none\n")
                f.write("\n")

        query_vec = average_embeddings(model, queries)
        distances, indices = index.search(np.asarray(query_vec, dtype=np.float32), TOP_K)
        last = indices[0]

        print("\n🎯 Top Matches:")
        summary = format_summary(
            indices[0],
            metadata,
            node_map,
            save_path=str(BASE_DIR / "last_summary.txt"),
            base_dir=SETTINGS.get("project_root"),
        )
        print(summary)
        print()
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write("Functions returned:\n")
            for i in indices[0]:
                meta = metadata[i]
                node = node_map.get(meta.get("id"), {})
                name = node.get("name", meta.get("name"))
                f.write(f"- {name}\n")
            f.write("\n")

        follow = input("What problem are you trying to solve?\n> ").strip()
        prompt_text = build_prompt(
            METADATA_PATH,
            CALL_GRAPH_PATH,
            [int(i) for i in indices[0]],
            follow,
            base_dir=SETTINGS.get("project_root"),
            save_path=str(BASE_DIR / "initial_prompt.txt"),
        )
        print("\nGenerated initial prompt:\n")
        print(prompt_text)
        print()
        
        if llm_model:
            print("⏳ Querying Gemini...")
            FINAL_CONTEXT = call_llm(llm_model, prompt_text)
            print(FINAL_CONTEXT)
            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write("LLM Response:\n")
                f.write(FINAL_CONTEXT + "\n\n")
        else:
            print("Skipping LLM query as the model is not available.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        folder = input("Enter the project folder to analyze (relative to output_dir): ").strip()
        if folder:
            main(folder)
