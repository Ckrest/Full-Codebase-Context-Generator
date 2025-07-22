import json
from pathlib import Path
import sys

import numpy as np

from llm import get_llm_model, call_llm, PROMPT_GEN_TEMPLATE, PROMPT_NEW_QUERY, get_example_json
from workspace import DataWorkspace, QuerySession
from spellcheck_utils import create_symspell_from_terms, correct_phrase
from graph import expand_graph
from prompt_builder import build_json_prompt, format_summary
from config import SETTINGS
from session_logger import (
    log_session_to_json,
    log_summary_to_markdown,
    format_function_entry,
)


def aggregate_scores(function_index: dict) -> dict:
    """Return aggregated relevance scores per function."""
    aggregated = {}
    for name, meta in function_index.items():
        matches = [
            m for m in meta.get("subqueries", [])
            if all(k in m for k in ("score", "rank", "index"))
        ]
        if not matches:
            continue
        scores = [m["score"] for m in matches]
        agg = {
            "avg_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "stddev_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "queries_matched": [m["text"] for m in matches],
        }
        aggregated[name] = agg
    return aggregated


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


def generate_sub_questions(query: str, count: int, llm_model) -> list[str]:
    if not llm_model or count <= 0:
        return []
    prompt = (
        "You are helping search a codebase. Given the following question, "
        f"break it into {count} distinct sub-questions. Each one should represent a different "
        "aspect of the original question that might be independently searchable. "
        "Be concise and technical.\n\n# Original Question:\n"
        f"{query}\n\n# Sub-Queries:"
    )
    text = call_llm(llm_model, prompt)
    results = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if t[0].isdigit():
            t = t.split(".", 1)[-1].strip()
        results.append(t)
    return results[:count]


def main(project_folder: str, problem: str | None = None, initial_query: str | None = None):
    workspace = DataWorkspace.load(project_folder)
    base_dir = workspace.base_dir
    model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")
    top_k = SETTINGS["query"]["top_k_results"]

    print(f"📁 Using project: {base_dir}")
    call_graph_path = base_dir / "call_graph.json"
    metadata_path = base_dir / "embedding_metadata.json"
    index_path = base_dir / "faiss.index"

    print("🔧 Running... Model, context, and settings info:")
    print(f"Encoder model: {model_path}")
    print(f"Context source: {call_graph_path}")
    print(f"Index file: {index_path}")
    print(f"Top-K results: {top_k}\n")

    print("🚀 Loading models and data...")
    model = workspace.model
    llm_model = get_llm_model()
    index = workspace.index
    metadata = workspace.metadata
    graph = workspace.graph
    node_map = workspace.node_map

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
    from interactive_cli import get_prompt_suggestions
    suggestions = get_prompt_suggestions()
    if not suggestions:
        print("[⏳ Working...] Generating prompt suggestions")
        try:
            suggestions.extend(generate_prompt_suggestions(problem, suggestion_count, llm_model))
            print("[✔ Done]")
        except Exception:
            print("[❌ Failed]")

    def run_search(query, last_session: QuerySession | None = None):
        nonlocal last
        if query.startswith("neighbors"):
            if not last_session:
                print("No previous search results.")
                return last_session
            try:
                num = int(query.split()[1]) - 1
                idx = last_session.final_indices[num]
            except (ValueError, IndexError):
                print("Usage: neighbors <result_number>")
                return last_session

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
            return last_session

        queries = [correct_phrase(symspell, query)]
        sub_queries = []
        if use_sub_questions and llm_model:
            print("[⏳ Working...] Generating sub-questions")
            try:
                sub_queries = generate_sub_questions(query, sub_question_count, llm_model)
                print("[✔ Done]")
            except Exception:
                print("[❌ Failed]")
                sub_queries = []
            if sub_queries:
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
            print(f"🗃 Output saved to: {results_file}")

        vecs = model.encode(queries, normalize_embeddings=True)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        subquery_data = [
            {"text": q, "embedding": vec.tolist(), "functions": []}
            for q, vec in zip(queries, vecs)
        ]
        function_index: dict[str, dict] = {}
        all_scores: dict[int, list[float]] = {}

        for sq_idx, vec in enumerate(vecs):
            dists, idxs = index.search(
                np.asarray(vec, dtype=np.float32).reshape(1, -1), top_k
            )
            for rank, (dist, idx) in enumerate(zip(dists[0], idxs[0]), start=1):
                meta = metadata[int(idx)]
                node = node_map.get(meta.get("id"), {})
                name = node.get("name", meta.get("name"))
                file_path = node.get("file_path", meta.get("file"))

                entry = {
                    "name": name,
                    "file": file_path,
                    "score": float(dist),
                    "rank": rank,
                }
                subquery_data[sq_idx]["functions"].append(entry)

                func_meta = function_index.setdefault(
                    name,
                    {
                        "file": file_path,
                        "id": meta.get("id"),
                        "subqueries": [],
                    },
                )
                func_meta["subqueries"].append(
                    {
                        "index": sq_idx,
                        "text": queries[sq_idx],
                        "score": float(dist),
                        "rank": rank,
                    }
                )

                all_scores.setdefault(int(idx), []).append(float(dist))

        total_sub_count = len(queries)
        for meta in function_index.values():
            scores = [s["score"] for s in meta["subqueries"]]
            times = len(scores)
            meta["value_score"] = float(np.mean(scores)) if scores else 0.0
            meta["duplicate_count"] = times if times > 1 else 0
            meta["reason"] = (
                f"matched {times} subqueries" if times > 1 else "matched a single subquery"
            )
        averaged = sorted(
            ((i, np.mean(ds)) for i, ds in all_scores.items()), key=lambda x: x[1]
        )
        final_indices = [i for i, _ in averaged[:top_k]]

        print("\n🎯 Top Matches:")
        for rank, idx in enumerate(final_indices, start=1):
            meta = metadata[idx]
            node = node_map.get(meta.get("id"), {})
            name = node.get("name", meta.get("name"))
            file_path = node.get("file_path", meta.get("file"))

            best_score = None
            best_text = ""
            for sq in subquery_data:
                for fn in sq["functions"]:
                    if fn["name"] == name and fn["file"] == file_path:
                        if best_score is None or fn["score"] > best_score:
                            best_score = fn["score"]
                            best_text = sq["text"]

            score_str = f"{best_score:.3f}" if best_score is not None else "n/a"
            print("-" * 40)
            print(f"{rank}. {name} ({file_path})")
            print(f"Match Score: {score_str}")
            if len(subquery_data) > 1:
                print(f"Matched Query: {best_text}")

        relevance = aggregate_scores(function_index)

        full_function_objects = []
        for idx in final_indices:
            meta = metadata[idx]
            node = node_map.get(meta.get("id"), {})
            name = node.get("name", meta.get("name"))
            score_info = relevance.get(name, {})

            callers = []
            for edge in graph.get("edges", []):
                if edge.get("to") == node.get("id"):
                    cid = edge.get("from")
                    caller_node = node_map.get(cid, {})
                    callers.append({
                        "function": caller_node.get("name", cid.split("::")[-1]),
                        "file": caller_node.get("file_path"),
                        "count": edge.get("weight", 1),
                    })

            callees = []
            for edge in graph.get("edges", []):
                if edge.get("from") == node.get("id"):
                    cid = edge.get("to")
                    callee_node = node_map.get(cid, {})
                    callees.append({
                        "function": callee_node.get("name", cid.split("::")[-1]),
                        "file": callee_node.get("file_path"),
                        "count": edge.get("weight", 1),
                    })

            full_function_objects.append({
                "function_name": name,
                "file": node.get("file_path", meta.get("file")),
                "class": node.get("class"),
                "relevance_scores": score_info,
                "call_relations": {
                    "callers": callers,
                    "callees": callees,
                },
                "call_graph_role": node.get("call_graph_role"),
                "parameters": node.get("parameters", {}),
                "comment": node.get("docstring") or (node.get("comments") or [""])[0],
                "code": node.get("code", ""),
            })

        print("\n🎯 Function Summary:")
        for func in full_function_objects:
            name = func["function_name"]
            score = func.get("relevance_scores", {}).get("avg_score", 0.0)
            role = func.get("call_graph_role", "unknown")
            print(f"- {name} | Score: {score:.3f} | Role: {role}")

        summary = format_summary(
            final_indices,
            metadata,
            node_map,
            save_path=str(base_dir / "last_summary.txt"),
            base_dir=SETTINGS.get("project_root"),
            workspace=workspace,
        )
        print(summary)
        print()

        with open(results_file, "a", encoding="utf-8") as f:
            f.write("Functions returned:\n")
            for i in final_indices:
                meta = metadata[i]
                node = node_map.get(meta.get("id"), {})
                name = node.get("name", meta.get("name"))
                f.write(f"- {name}\n")
            f.write("\n")
        print(f"🗃 Output saved to: {results_file}")

        print("[⏳ Working...] Generating final prompt")
        try:
            prompt_text = build_json_prompt(
                full_function_objects,
                problem,
                save_path=str(base_dir / "initial_prompt.json"),
            )
            print("[✔ Done]")
        except Exception:
            print("[❌ Failed]")
            prompt_text = ""
        print("\nGenerated initial prompt:\n")
        print(prompt_text)
        print()

        if llm_model:
            print("[⏳ Working...] Querying Gemini")
            try:
                final_context = call_llm(llm_model, prompt_text)
                print("[✔ Done]")
            except Exception:
                final_context = "💥 Gemini query failed"
                print("[❌ Failed]")
            print(final_context)
            with open(results_file, "a", encoding="utf-8") as f:
                f.write("LLM Response:\n")
                f.write(final_context + "\n\n")
            print(f"🗃 Output saved to: {results_file}")
        else:
            print("Skipping LLM query as the model is not available.")
            final_context = ""

        functions_list = []
        for name, scores in relevance.items():
            node_info = function_index.get(name)
            node = node_map.get(node_info.get("id")) if node_info else None
            if node:
                functions_list.append(
                    format_function_entry(node, scores, graph)
                )

        max_log = SETTINGS.get("logging", {}).get("max_functions_to_log", 100)
        sorted_funcs = sorted(
            function_index.items(), key=lambda x: -x[1].get("value_score", 0.0)
        )
        if max_log:
            sorted_funcs = sorted_funcs[:max_log]
        function_index = {k: v for k, v in sorted_funcs}

        if not SETTINGS.get("logging", {}).get("track_duplicates", True):
            for meta in function_index.values():
                meta.pop("duplicate_count", None)

        summary_data = {
            "total_subqueries": len(queries),
            "total_functions": len(function_index),
            "core_hits": sum(
                1
                for m in function_index.values()
                if len(m.get("subqueries", [])) == len(queries)
            ),
            "duplicate_functions": sum(
                1
                for m in function_index.values()
                if m.get("duplicate_count", 0) > 1
            ),
        }

        log_data = {
            "query": query,
            "subqueries": [sq["text"] for sq in subquery_data],
            "functions": functions_list,
        }

        if SETTINGS.get("logging", {}).get("log_markdown", True):
            md_file = log_summary_to_markdown({
                "original_query": query,
                "subqueries": subquery_data,
                "functions": function_index,
                "summary": summary_data,
                "llm_response": final_context,
            }, "logs")
            print(f"✔ Saved {md_file}")

        if SETTINGS.get("logging", {}).get("log_json", True):
            json_file = log_session_to_json(log_data, "logs")
            print(f"✔ Saved {json_file}")

        session = QuerySession(
            problem=problem or "",
            queries=queries,
            subquery_data=subquery_data,
            function_matches=function_index,
            final_indices=final_indices,
            llm_response=final_context,
        )
        last = session
        return session

    last = None
    if initial_query:
        last = run_search(initial_query, last)
        return

    from interactive_cli import ask_search_prompt

    while True:
        query = ask_search_prompt(suggestions, problem, llm_model)
        if query.strip().lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        last = run_search(query, last)


