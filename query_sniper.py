import json
from pathlib import Path
import sys

import faiss
import numpy as np
from symspellpy import SymSpell
from llm_utils import get_llm_model, call_llm, load_embedding_model

def get_example_json(n):
    return ",\n  ".join(f'"query suggestion {i+1}"' for i in range(n))

PROMPT_GEN_TEMPLATE = """You are an expert in semantic code search.

Given the user’s problem statement below, generate {n} recommended queries that are each:
- Short (5-12 words)
- Technically focused
- Different in angle or phrasing
- Useful for embedding-based code search

Respond only with a JSON list of strings — no commentary, no markdown.

# Problem Statement
{problem}

# Output Format
[
  {get_example_json}
]
"""


PROMPT_NEW_QUERY = """You previously generated the following recommended queries for a problem. Now generate a single, new query that:
- Is different in phrasing or focus
- Still relevant to the original problem
- Is useful for code search
- Is short and specific

Respond only with the query string.

# Problem Statement
{problem}

# Existing Queries
{existing}

# New Query"""

from context_utils import expand_graph
from summary_formatter import format_summary
from prompt_builder import build_prompt
from config import SETTINGS

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


def parse_json_list(text):
    """Extract and parse a JSON list from ``text``."""
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return []


def generate_prompt_suggestions(problem, count, llm_model):
    if not llm_model or count <= 0:
        return []
    example_json = get_example_json(count)
    prompt = PROMPT_GEN_TEMPLATE.format(problem=problem, n=count, example_json=example_json)
    text = call_llm(llm_model, prompt)
    return parse_json_list(text)


def generate_new_prompt(problem, existing, llm_model):
    if not llm_model:
        return ""
    prompt = PROMPT_NEW_QUERY.format(problem=problem, existing=json.dumps(existing))
    return call_llm(llm_model, prompt).strip()


def main(project_folder, problem=None, initial_query=None):
    """Search the generated embeddings. If ``initial_query`` is provided, run
    once and exit without prompting."""
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
    model = load_embedding_model(model_path)
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

    if problem is None:
        from user_interaction import ask_problem

        problem = ask_problem()

    suggestion_count = int(SETTINGS["query"].get("prompt_suggestion_count", 0))
    suggestions = generate_prompt_suggestions(problem, suggestion_count, llm_model)

    def run_search(query, last_indices=None):
        nonlocal last
        if query.startswith("neighbors"):
            if not last_indices:
                print("No previous search results.")
                return last_indices
            try:
                num = int(query.split()[1]) - 1
                idx = last_indices[num]
            except (ValueError, IndexError):
                print("Usage: neighbors <result_number>")
                return last_indices

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
            return last_indices

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

        prompt_text = build_prompt(
            METADATA_PATH,
            CALL_GRAPH_PATH,
            [int(i) for i in indices[0]],
            problem,
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
        return last

    # --- Main Interactive Loop ---
    last = None
    if initial_query:
        run_search(initial_query, last)
        return

    while True:
        print("\nAvailable query prompts:")
        print("1) Generate a new prompt suggestion")
        print("2) Use problem statement")
        for i, q in enumerate(suggestions, start=3):
            print(f"{i}) {q}")

        from user_interaction import ask_search_prompt

        user_in = ask_search_prompt()
        if user_in.strip().lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        if user_in.isdigit():
            choice = int(user_in)
            if choice == 1:
                new_q = generate_new_prompt(problem, suggestions, llm_model)
                if not new_q:
                    print("LLM not available to generate a query.")
                    continue
                print(f"Generated prompt: {new_q}")
                suggestions.append(new_q)
                query = new_q
            elif choice == 2:
                query = problem
            elif 3 <= choice < 3 + len(suggestions):
                query = suggestions[choice - 3]
            else:
                print("Invalid selection.")
                continue
        else:
            query = user_in

        if query.strip().lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        last = run_search(query, last)


if __name__ == "__main__":
    from user_interaction import ask_project_folder

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        folder = ask_project_folder()
        if folder:
            main(folder)
