import json
from pathlib import Path
import sys

import faiss
import numpy as np

from llm import get_llm_model, call_llm, PROMPT_GEN_TEMPLATE, PROMPT_NEW_QUERY, get_example_json
from embedding import load_embedding_model
from spellcheck_utils import create_symspell_from_terms, correct_phrase
from graph import expand_graph
from prompt_builder import build_prompt, format_summary
from config import SETTINGS


def average_embeddings(model, texts) -> np.ndarray:
    vecs = model.encode(list(texts), normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=float)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    return np.mean(vecs, axis=0, keepdims=True)


def parse_json_list(text: str):
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return []


def generate_prompt_suggestions(problem: str, count: int, llm_model):
    if not llm_model or count <= 0:
        return []
    example_json = get_example_json(count)
    prompt = PROMPT_GEN_TEMPLATE.format(problem=problem, n=count, get_example_json=example_json)
    text = call_llm(llm_model, prompt)
    return parse_json_list(text)


def generate_new_prompt(problem: str, existing, llm_model):
    if not llm_model:
        return ""
    prompt = PROMPT_NEW_QUERY.format(problem=problem, existing=json.dumps(existing))
    return call_llm(llm_model, prompt).strip()


def main(project_folder: str, problem: str | None = None, initial_query: str | None = None):
    model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")
    base_dir = Path(SETTINGS["paths"]["output_dir"]) / project_folder
    metadata_path = base_dir / "embedding_metadata.json"
    index_path = base_dir / "faiss.index"
    call_graph_path = base_dir / "call_graph.json"
    top_k = SETTINGS["query"]["top_k_results"]

    print("🔧 Running... Model, context, and settings info:")
    print(f"Encoder model: {model_path}")
    print(f"Context source: {call_graph_path}")
    print(f"Index file: {index_path}")
    print(f"Top-K results: {top_k}\n")

    print("🚀 Loading models and data...")
    model = load_embedding_model(model_path)
    llm_model = get_llm_model()
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(call_graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    node_map = {n["id"]: n for n in graph.get("nodes", [])}

    symspell = None
    if SETTINGS["query"].get("use_spellcheck"):
        names = [item.get("name") for item in metadata if "name" in item]
        symspell = create_symspell_from_terms(names)

    sub_question_count = int(SETTINGS["query"].get("sub_question_count", 0))
    use_sub_questions = sub_question_count > 0
    results_file = base_dir / "results.txt"

    if problem is None:
        from interactive_cli import ask_problem

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

        queries = [correct_phrase(symspell, query)]
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
                queries.extend(correct_phrase(symspell, q) for q in sub_queries)
        elif use_sub_questions:
            print("Sub-question generation skipped because no LLM model was available.")

        if sub_queries or use_sub_questions:
            with open(results_file, "a", encoding="utf-8") as f:
                f.write(f"Original query: {query}\n")
                if sub_queries:
                    f.write("Sub-queries:\n")
                    for q in sub_queries:
                        f.write(f"- {q}\n")
                else:
                    f.write("Sub-queries: none\n")
                f.write("\n")

        query_vec = average_embeddings(model, queries)
        distances, indices = index.search(np.asarray(query_vec, dtype=np.float32), top_k)
        last = indices[0]

        print("\n🎯 Top Matches:")
        summary = format_summary(
            indices[0],
            metadata,
            node_map,
            save_path=str(base_dir / "last_summary.txt"),
            base_dir=SETTINGS.get("project_root"),
        )
        print(summary)
        print()
        with open(results_file, "a", encoding="utf-8") as f:
            f.write("Functions returned:\n")
            for i in indices[0]:
                meta = metadata[i]
                node = node_map.get(meta.get("id"), {})
                name = node.get("name", meta.get("name"))
                f.write(f"- {name}\n")
            f.write("\n")

        prompt_text = build_prompt(
            metadata_path,
            call_graph_path,
            [int(i) for i in indices[0]],
            problem,
            base_dir=SETTINGS.get("project_root"),
            save_path=str(base_dir / "initial_prompt.txt"),
        )
        print("\nGenerated initial prompt:\n")
        print(prompt_text)
        print()

        if llm_model:
            print("⏳ Querying Gemini...")
            final_context = call_llm(llm_model, prompt_text)
            print(final_context)
            with open(results_file, "a", encoding="utf-8") as f:
                f.write("LLM Response:\n")
                f.write(final_context + "\n\n")
        else:
            print("Skipping LLM query as the model is not available.")
        return last

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

        from interactive_cli import ask_search_prompt

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


