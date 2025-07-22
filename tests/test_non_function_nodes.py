import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import graph as lec
import embedding
from config import SETTINGS


def test_non_function_nodes(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("def foo():\n    pass\n")
    (tmp_path / "doc.md").write_text("# Title\ntext")
    (tmp_path / "page.html").write_text("<body><!-- hi --></body>")
    (tmp_path / "conf.json").write_text('{"a": 1}')
    (tmp_path / "conf.yaml").write_text('a: b')
    (tmp_path / "notes.txt").write_text('note')

    entries = lec.crawl_directory(str(tmp_path), respect_gitignore=False)
    G = lec.build_call_graph(entries)
    types = {d["type"] for _, d in G.nodes(data=True)}
    assert {"function", "section", "html_section", "config_entry", "raw_text"} <= types

    project = "proj"
    out_dir = tmp_path / "out"
    monkeypatch.setitem(SETTINGS["paths"], "output_dir", str(out_dir))
    out_proj = out_dir / project
    out_proj.mkdir(parents=True)
    lec.save_graph_json(G, out_proj / "call_graph.json")

    class DummyModel:
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=True):
            return [[0.0, 0.0] for _ in texts]

    class DummyFaiss:
        class IndexFlatIP:
            def __init__(self, dim):
                self.vecs = []
            def add(self, arr):
                self.vecs.extend(arr)
        def write_index(self, index, path):
            Path(path).write_text("index")

    monkeypatch.setattr(embedding, "load_embedding_model", lambda _: DummyModel())
    sys.modules["faiss"] = DummyFaiss()

    embedding.generate_embeddings(project)

    meta = json.loads((out_proj / "embedding_metadata.json").read_text())
    graph_data = json.loads((out_proj / "call_graph.json").read_text())
    id_to_type = {n["id"]: n["type"] for n in graph_data["nodes"]}
    meta_types = {id_to_type[m["id"]] for m in meta}
    assert {"function", "section", "html_section", "config_entry", "raw_text"} <= meta_types

