import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import graph
import workspace
import embedding
from config import SETTINGS


def test_workspace_checksum_validation(tmp_path, monkeypatch):
    project = "proj"
    out_dir = tmp_path / project
    out_dir.mkdir()
    monkeypatch.setitem(SETTINGS["paths"], "output_dir", str(tmp_path))

    # build minimal graph
    G = graph.build_call_graph([])
    node_id = "file.py::foo"
    G.add_node(node_id, file_path="file.py", type="function", name="foo")
    graph.save_graph_json(G, out_dir / "call_graph.json")
    graph_data = json.loads((out_dir / "call_graph.json").read_text())
    checksum = graph_data["checksum"]

    # dummy embedding artifacts
    np.save(out_dir / "embeddings.npy", np.zeros((1, 2)))
    (out_dir / "faiss.index").write_text("index")

    class DummyModel:
        def get_sentence_embedding_dimension(self):
            return 2

    class DummyFaiss:
        @staticmethod
        def read_index(path):
            return "index"

    sys.modules["faiss"] = DummyFaiss
    monkeypatch.setattr(embedding, "load_embedding_model", lambda _: DummyModel())

    meta = {
        "graph_checksum": checksum,
        "records": [{"id": node_id, "file": "file.py", "name": "foo"}],
    }
    (out_dir / "embedding_metadata.json").write_text(json.dumps(meta))

    ws = workspace.DataWorkspace.load(project)
    assert ws.graph["checksum"] == checksum

    # mismatch should raise
    meta["graph_checksum"] = "bad"
    (out_dir / "embedding_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):
        workspace.DataWorkspace.load(project)
