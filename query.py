from __future__ import annotations

import json
from pathlib import Path
import sys
import logging

from lazy_loader import safe_lazy_import
from spellcheck_utils import create_symspell_from_terms, correct_phrase
from config import SETTINGS
from session_logger import (
    log_session_to_json,
    log_summary_to_markdown,
    format_function_entry,
    get_timestamp,
)

logger = logging.getLogger(__name__)


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
        np = safe_lazy_import("numpy")
        scores = [m["score"] for m in matches]
        agg = {
            "avg_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "stddev_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "queries_matched": [m["text"] for m in matches],
        }
        aggregated[name] = agg
    return aggregated


def average_embeddings(model, texts) -> object:
    np = safe_lazy_import("numpy")
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
    llm = safe_lazy_import("llm")
    example_json = llm.get_example_json(count)
    prompt = llm.PROMPT_GEN_TEMPLATE.format(problem=problem, n=count, get_example_json=example_json)
    text = llm.call_llm(llm_model, prompt)
    return parse_json_list(text)


def generate_new_prompt(problem: str, existing, llm_model):
    if not llm_model:
        return ""
    llm = safe_lazy_import("llm")
    prompt = llm.PROMPT_NEW_QUERY.format(problem=problem, existing=json.dumps(existing))
    return llm.call_llm(llm_model, prompt).strip()


def generate_sub_questions(query: str, count: int, llm_model) -> list[str]:
    if not llm_model or count <= 0:
        return []
    llm = safe_lazy_import("llm")
    prompt = (
        "You are helping search a codebase. Given the following question, "
        f"break it into {count} distinct sub-questions. Each one should represent a different "
        "aspect of the original question that might be independently searchable. "
        "Be concise and technical.\n\n# Original Question:\n"
        f"{query}\n\n# Sub-Queries:"
    )
    text = llm.call_llm(llm_model, prompt)
    results = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if t[0].isdigit():
            t = t.split(".", 1)[-1].strip()
        results.append(t)
    return results[:count]


class QueryProcessor:
    """Handle a single query run."""

    def __init__(self, workspace, problem, symspell, llm_model, suggestions):
        self.workspace = workspace
        self.problem = problem or ""
        self.symspell = symspell
        self.llm_model = llm_model
        self.suggestions = suggestions or []
        self.base_dir = workspace.base_dir
        self.top_k = SETTINGS["query"]["top_k_results"]

    def _setup_run_directory(self, query):
        run_id = get_timestamp()
        run_dir = self.base_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "query": query,
            "project": str(self.base_dir),
            "subqueries": [],
            "files": [],
            "embedding_model": SETTINGS.get("embedding", {}).get("encoder_model_path")
            or "sentence-transformers/all-MiniLM-L6-v2",
            "timestamp": run_id,
        }
        with open(run_dir / "settings_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(SETTINGS, f, indent=2)
        manifest.setdefault("files", []).append({
            "file": "settings_snapshot.json",
            "description": "Snapshot of settings used for this run.",
        })
        if self.suggestions:
            with open(run_dir / "prompt_suggestions.json", "w", encoding="utf-8") as f:
                json.dump(self.suggestions, f, indent=2)
            manifest["files"].append({
                "file": "prompt_suggestions.json",
                "description": "Possible follow-up prompts suggested by the LLM.",
            })
        return run_id, run_dir, manifest

    def _get_search_queries(self, query):
        queries = [correct_phrase(self.symspell, query)]
        sub_queries = []
        sub_question_count = int(SETTINGS["query"].get("sub_question_count", 0))
        if sub_question_count > 0:
            if self.llm_model:
                logger.info("[⏳ Working...] Generating sub-questions")
                try:
                    sub_queries = generate_sub_questions(query, sub_question_count, self.llm_model)
                    logger.info("[✔ Done]")
                except Exception as e:
                    logger.error("Failed to generate sub-questions: %s", e, exc_info=True)
            else:
                logger.info("Sub-question generation skipped because no LLM model was available.")
        if sub_queries:
            queries.extend(correct_phrase(self.symspell, q) for q in sub_queries)
        return queries, sub_queries

    def _execute_faiss_search(self, vectors):
        subquery_data = [
            {"text": q, "embedding": vec.tolist(), "functions": []}
            for q, vec in zip(self.queries, vectors)
        ]
        function_index = {}
        all_scores = {}
        for sq_idx, vec in enumerate(vectors):
            np = safe_lazy_import("numpy")
            dists, idxs = self.workspace.index.search(np.asarray(vec, dtype=np.float32).reshape(1, -1), self.top_k)
            for rank, (dist, idx) in enumerate(zip(dists[0], idxs[0]), start=1):
                meta = self.workspace.metadata[int(idx)]
                node = self.workspace.node_map.get(meta.get("id"), {})
                name = node.get("name", meta.get("name"))
                node_id = meta.get("id")
                key = name if name else node_id
                file_path = node.get("file_path", meta.get("file"))
                entry = {"name": name, "file": file_path, "score": float(dist), "rank": rank}
                subquery_data[sq_idx]["functions"].append(entry)
                func_meta = function_index.setdefault(key, {"file": file_path, "id": node_id, "subqueries": []})
                func_meta["subqueries"].append({"index": sq_idx, "text": self.queries[sq_idx], "score": float(dist), "rank": rank})
                all_scores.setdefault(int(idx), []).append(float(dist))
        return subquery_data, function_index, all_scores

    def _aggregate_search_results(self, subquery_data, function_index, all_scores):
        np = safe_lazy_import("numpy")
        for meta in function_index.values():
            scores = [s["score"] for s in meta["subqueries"]]
            times = len(scores)
            meta["value_score"] = float(np.mean(scores)) if scores else 0.0
            meta["duplicate_count"] = times if times > 1 else 0
            meta["reason"] = (
                f"matched {times} subqueries" if times > 1 else "matched a single subquery"
            )
        averaged = sorted(((i, np.mean(ds)) for i, ds in all_scores.items()), key=lambda x: x[1])
        final_indices = [i for i, _ in averaged[: self.top_k]]
        relevance = aggregate_scores(function_index)
        full_function_objects = []
        graph = self.workspace.graph
        node_map = self.workspace.node_map
        metadata = self.workspace.metadata
        for idx in final_indices:
            meta = metadata[idx]
            node = node_map.get(meta.get("id"), {})
            name = node.get("name", meta.get("name"))
            key = name if name else meta.get("id")
            score_info = relevance.get(key, {})
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
                "type": node.get("type"),
                "file": node.get("file_path", meta.get("file")),
                "class": node.get("class"),
                "relevance_scores": score_info,
                "call_relations": {"callers": callers, "callees": callees},
                "call_graph_role": node.get("call_graph_role"),
                "parameters": node.get("parameters", {}),
                "comment": node.get("docstring") or (node.get("comments") or [""])[0],
                "code": node.get("code", ""),
            })
        return final_indices, relevance, full_function_objects, subquery_data, function_index

    def _build_llm_prompt(self, run_dir, functions):
        prompt_builder = safe_lazy_import("prompt_builder")
        prompt_text = prompt_builder.build_json_prompt(
            functions,
            self.problem,
            save_path=str(run_dir / "prompt.json"),
        )
        return prompt_text

    def _log_session_artifacts(self, run_dir, manifest, queries, subquery_data, function_index, summary_data, final_context):
        if SETTINGS.get("logging", {}).get("log_markdown", True):
            md_file = log_summary_to_markdown(
                {
                    "original_query": queries[0] if queries else "",
                    "subqueries": subquery_data,
                    "functions": function_index,
                    "summary": summary_data,
                    "llm_response": final_context,
                },
                run_dir / "summary.md",
            )
            manifest["files"].append({"file": "summary.md", "description": "Human-readable summary of query results."})
            logger.info("Saved %s", md_file)

        if SETTINGS.get("logging", {}).get("log_json", True):
            log_data = {
                "query": queries[0] if queries else "",
                "subqueries": [sq["text"] for sq in subquery_data],
                "functions": [format_function_entry(self.workspace.node_map.get(function_index[k]["id"]), v, self.workspace.graph) for k, v in function_index.items()],
            }
            json_file = log_session_to_json(log_data, run_dir / "results.json")
            manifest["files"].append({"file": "results.json", "description": "Machine-readable summary of query results."})
            logger.info("Saved %s", json_file)

    def process(self, query):
        run_id, run_dir, manifest = self._setup_run_directory(query)
        self.queries, sub_queries = self._get_search_queries(query)
        model = self.workspace.model
        vectors = model.encode(self.queries, normalize_embeddings=True)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        subquery_data, function_index, all_scores = self._execute_faiss_search(vectors)
        final_indices, relevance, functions, subquery_data, function_index = self._aggregate_search_results(subquery_data, function_index, all_scores)
        prompt_text = self._build_llm_prompt(run_dir, functions)
        final_context = ""
        if self.llm_model:
            llm_mod = safe_lazy_import("llm")
            logger.info("[⏳ Working...] Querying Gemini")
            try:
                final_context = llm_mod.call_llm(self.llm_model, prompt_text)
                logger.info("[✔ Done]")
            except Exception as e:
                final_context = "💥 Gemini query failed"
                logger.error("LLM query failed: %s", e, exc_info=True)
            with open(run_dir / "raw_llm_response.txt", "w", encoding="utf-8") as f:
                f.write(final_context + "\n")
            manifest["files"].append({"file": "raw_llm_response.txt", "description": "Unprocessed output returned by the LLM."})
        else:
            logger.info("Skipping LLM query as the model is not available.")

        summary_data = {
            "total_subqueries": len(self.queries),
            "total_functions": len(function_index),
            "core_hits": sum(1 for m in function_index.values() if len(m.get("subqueries", [])) == len(self.queries)),
            "duplicate_functions": sum(1 for m in function_index.values() if m.get("duplicate_count", 0) > 1),
        }

        self._log_session_artifacts(run_dir, manifest, self.queries, subquery_data, function_index, summary_data, final_context)

        readme_lines = ["# Query Run " + run_id, "", f"Original query: {query}", "", "## Artifacts"]
        for item in manifest.get("files", []):
            if isinstance(item, dict):
                readme_lines.append(f"- **{item['file']}** - {item.get('description','')}")
            else:
                readme_lines.append(f"- **{item}**")
        readme_path = run_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(readme_lines) + "\n")
        manifest["files"].append({"file": "README.md", "description": "Overview of the run and descriptions of generated files."})

        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        workspace_mod = safe_lazy_import("workspace")
        session = workspace_mod.QuerySession(
            problem=self.problem,
            queries=self.queries,
            subquery_data=subquery_data,
            function_matches=function_index,
            final_indices=final_indices,
            llm_response=final_context,
            output_dir=run_dir,
        )
        return session


def main(project_folder: str, problem: str | None = None, initial_query: str | None = None):
    workspace_mod = safe_lazy_import("workspace")
    workspace = workspace_mod.DataWorkspace.load(project_folder)
    base_dir = workspace.base_dir
    model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")

    logger.info("\ud83d\udccb Using project: %s", base_dir)
    call_graph_path = base_dir / "call_graph.json"
    metadata_path = base_dir / "embedding_metadata.json"
    index_path = base_dir / "faiss.index"

    logger.info("\ud83d\udd27 Running... Model, context, and settings info:")
    logger.info("Encoder model: %s", model_path)
    logger.info("Context source: %s", call_graph_path)
    logger.info("Index file: %s", index_path)

    logger.info("\ud83d\ude80 Loading models and data...")
    model = workspace.model
    llm = safe_lazy_import("llm")
    llm_model = llm.get_llm_model()
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

    if problem is None:
        from interactive_cli import ask_problem

        problem = ask_problem()

    suggestion_count = int(SETTINGS["query"].get("prompt_suggestion_count", 0))
    from interactive_cli import get_prompt_suggestions
    suggestions = get_prompt_suggestions()
    if not suggestions:
        logger.info("[⏳ Working...] Generating prompt suggestions")
        try:
            suggestions.extend(generate_prompt_suggestions(problem, suggestion_count, llm_model))
            logger.info("[✔ Done]")
        except Exception as e:
            logger.error("Failed to generate suggestions: %s", e, exc_info=True)

    def run_search(query, last_session: "workspace.QuerySession" | None = None):
        nonlocal last
        if query.startswith("neighbors"):
            if not last_session:
                logger.info("No previous search results.")
                return last_session
            try:
                num = int(query.split()[1]) - 1
                idx = last_session.final_indices[num]
            except (ValueError, IndexError):
                logger.info("Usage: neighbors <result_number>")
                return last_session
            meta = metadata[idx]
            graph_mod = safe_lazy_import("graph")
            nb_ids = graph_mod.expand_graph(
                graph,
                meta["id"],
                depth=SETTINGS["context"].get("context_hops", 1),
                limit=SETTINGS["context"].get("max_neighbors", 5),
                bidirectional=SETTINGS["context"].get("bidirectional", True),
                outbound_weight=SETTINGS["context"].get("outbound_weight", 1.0),
                inbound_weight=SETTINGS["context"].get("inbound_weight", 1.0),
            )
            logger.info("Neighbors:")
            for nid in nb_ids:
                node = node_map.get(nid, {})
                logger.info("- %s — %s", node.get('name'), node.get('file_path'))
            return last_session

        processor = QueryProcessor(workspace, problem, symspell, llm_model, suggestions)
        session = processor.process(query)
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
            logger.info("👋 Exiting.")
            break
        last = run_search(query, last)

