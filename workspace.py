from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import faiss

from embedding import load_embedding_model
from config import SETTINGS


@dataclass
class DataWorkspace:
    """Container for static project data like graph and embeddings."""

    project_folder: str
    base_dir: Path
    metadata: list = field(default_factory=list)
    graph: dict = field(default_factory=dict)
    node_map: dict = field(default_factory=dict)
    index: faiss.Index | None = None
    model: object | None = None

    @classmethod
    def load(cls, project_folder: str) -> "DataWorkspace":
        """Load graph, metadata, and FAISS index for ``project_folder``."""
        base_dir = Path(SETTINGS["paths"]["output_dir"]) / project_folder
        metadata_path = base_dir / "embedding_metadata.json"
        graph_path = base_dir / "call_graph.json"
        index_path = base_dir / "faiss.index"
        model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")

        model = load_embedding_model(model_path)
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        node_map = {n["id"]: n for n in graph.get("nodes", [])}

        return cls(
            project_folder=project_folder,
            base_dir=base_dir,
            metadata=metadata,
            graph=graph,
            node_map=node_map,
            index=index,
            model=model,
        )


@dataclass
class QuerySession:
    """In-memory details of a single query run."""

    problem: str
    queries: list[str]
    subquery_data: list[dict] = field(default_factory=list)
    function_matches: dict[str, dict] = field(default_factory=dict)
    final_indices: list[int] = field(default_factory=list)
    llm_response: str = ""
