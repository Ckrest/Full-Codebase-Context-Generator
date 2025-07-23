import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_SETTINGS
from spellcheck_utils import create_symspell_from_terms, correct_phrase
from query import average_embeddings


def test_default_query_settings():
    q = DEFAULT_SETTINGS["query"]
    assert q["use_spellcheck"] is False
    assert q["sub_question_count"] == 0
    assert q["prompt_suggestion_count"] == 3


def test_spellcheck_basic():
    terms = ["foobar"]
    sym = create_symspell_from_terms(terms)
    corrected = correct_phrase(sym, "foobzr")
    assert corrected == "foobar"


def test_average_embeddings():
    class Dummy:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    vec = average_embeddings(Dummy(), ["a", "b"])
    assert np.allclose(vec, [[2.0, 3.0]])


def test_iterative_llm_loop(monkeypatch, tmp_path):
    """QueryProcessor should loop until an info response is returned."""
    import workspace
    import query as query_mod

    ws = workspace.DataWorkspace(
        project_folder="proj",
        base_dir=tmp_path,
        metadata=[{"id": "1", "file": "foo.py", "name": "foo"}],
        graph={"nodes": [{"id": "1", "name": "foo", "file_path": "foo.py", "type": "function", "code": "def foo(): pass"}], "edges": []},
        node_map={"1": {"id": "1", "name": "foo", "file_path": "foo.py", "type": "function", "code": "def foo(): pass"}},
        index=None,
        model=type("M", (), {"encode": lambda self, q, normalize_embeddings=True: np.zeros((len(q), 2))})(),
    )

    processor = query_mod.QueryProcessor(ws, "prob", None, llm_model="x", suggestions=[])

    monkeypatch.setattr(processor, "_setup_run_directory", lambda q: ("id", tmp_path, {"files": []}))
    monkeypatch.setattr(processor, "_get_search_queries", lambda q: ([q], []))
    monkeypatch.setattr(processor, "_execute_faiss_search", lambda v: ([], {"foo": {"file": "foo.py", "id": "1", "subqueries": []}}, {0: [0.1]}))
    monkeypatch.setattr(processor, "_aggregate_search_results", lambda *a: ([0], {}, [ws.node_map["1"]], [], {"foo": {"id": "1"}}))
    monkeypatch.setattr(processor, "_build_llm_prompt", lambda *a, **k: "prompt")

    responses = [
        '{"response_type": "functions", "functions": ["foo"], "total": 1}',
        '{"response_type": "info", "summary": "done"}',
    ]

    def fake_call(client, text, instruction=None):
        return responses.pop(0)

    dummy_llm = type("L", (), {"call_llm": staticmethod(fake_call), "NEW_CONTEXT_INSTRUCT": "inst"})
    monkeypatch.setitem(sys.modules, "llm", dummy_llm)

    session = processor.process("search")
    assert session.llm_response == "done"
    assert len(session.conversation) == 2
