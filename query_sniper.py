import json
from pathlib import Path
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell, Verbosity
from transformers.pipelines import pipeline
from google import genai

from context_utils import expand_graph
from summary_formatter import format_summary
from prompt_builder import build_prompt
from Start import SETTINGS

# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')


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


# REFACTORED FUNCTION
def call_llm(model_obj, prompt_text, temperature=0.6):
    """
    Calls the generative model with the provided prompt using a pre-initialized model object.
    """
    # The model object is now passed in, not created here.
    if not model_obj:
        return "❌ Generative model not initialized. Check API key in settings."
    try:
        response = model_obj.generate_content(
            prompt_text,
            generation_config={
                "temperature": temperature,
                "top_p": 1.0,
                "max_output_tokens": 1000,
            },
        )
        return response.text.strip()
    except Exception as e:
        return f"💥 Gemini query failed: {e}"


def main(project_folder):
    """Interactive search of the generated embeddings."""
    # --- Configuration Loading ---
    MODEL_NAME = SETTINGS["model"]["llm_model"]
    BASE_DIR = Path(SETTINGS["paths"]["output_dir"]) / project_folder
    METADATA_PATH = BASE_DIR / "embedding_metadata.json"
    INDEX_PATH = BASE_DIR / "faiss.index"
    CALL_GRAPH_PATH = BASE_DIR / "call_graph.json"
    TOP_K = SETTINGS["query"]["top_k_results"]

    print("🔧 Running... Model, context, and settings info:")
    print(f"Model: {MODEL_NAME}")
    print(f"Context source: {CALL_GRAPH_PATH}")
    print(f"Index file: {INDEX_PATH}")
    print(f"Top-K results: {TOP_K}\n")

    # --- Model and Data Loading ---
    print("🚀 Loading models and data...")
    model_path = SETTINGS.get("model", {}).get("local_model_path") or MODEL_NAME
    model = SentenceTransformer(model_path)
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

    paraphraser = None
    rephrase_count = int(SETTINGS["query"].get("rephrase_count", 1))
    if rephrase_count > 1:
        try:
            model_or_path = (
                SETTINGS["query"].get("rephrase_model_path") or "Vamsi/T5_Paraphrase_Paws"
            )
            paraphraser = pipeline("text2text-generation", model=model_or_path)
            print("Loaded paraphrasing model for query rephrasing.")
        except Exception as e:
            print(f"⚠️ Could not load paraphrasing model: {e}")

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
        if paraphraser and rephrase_count > 1:
            try:
                print("Rephrasing query...")
                results = paraphraser(
                    query, num_return_sequences=rephrase_count - 1, num_beams=max(4, rephrase_count)
                )
                queries.extend([r["generated_text"] for r in results])
            except Exception as e:
                print(f"Paraphrasing failed: {e}")

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
        
        # FIXED: Call the LLM using the pre-initialized model object.
        if ("insert is gemini api key here"):
            print("⏳ Querying Gemini...")
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt_text,
            )
            FINAL_CONTEXT = response.text
            print(FINAL_CONTEXT)
        else:
            print("Skipping LLM query as the model is not available.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        folder = input("Enter the project folder to analyze (relative to output_dir): ").strip()
        if folder:
            main(folder)