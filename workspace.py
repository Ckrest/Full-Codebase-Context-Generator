from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from lazy_loader import lazy_import
from config import SETTINGS


@dataclass
class DataWorkspace:
    """Container for static project data like graph and embeddings."""

    project_folder: str
    base_dir: Path
    metadata: list = field(default_factory=list)
    graph: dict = field(default_factory=dict)
    node_map: dict = field(default_factory=dict)
    index: object | None = None
    model: object | None = None

    @classmethod
    def load(cls, project_folder: str) -> "DataWorkspace":
        """Load graph, metadata, and FAISS index for ``project_folder``."""
        base_dir = Path(SETTINGS["paths"]["output_dir"]) / project_folder
        metadata_path = base_dir / "embedding_metadata.json"
        graph_path = base_dir / "call_graph.json"
        index_path = base_dir / "faiss.index"
        model_path = SETTINGS.get("embedding", {}).get("encoder_model_path")

        if not (metadata_path.exists() and graph_path.exists() and index_path.exists()):
            raise FileNotFoundError(
                "Data files are missing. Re-run the embed step to generate them."
            )

        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        graph_checksum = graph.get("checksum")
        if not graph_checksum:
            raise RuntimeError(
                "call_graph.json is missing a checksum. Re-run the embed step."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_raw = json.load(f)
        if isinstance(meta_raw, list):
            raise RuntimeError(
                "embedding_metadata.json is outdated. Re-run the embed step."
            )
        metadata_checksum = meta_raw.get("graph_checksum")
        if metadata_checksum != graph_checksum:
            raise RuntimeError(
                "Data artifacts are out of sync. Re-run the embed step."
            )
        metadata = meta_raw.get("records", [])

        embedding = lazy_import("embedding")
        faiss = lazy_import("faiss")
        model = embedding.load_embedding_model(model_path)
        index = faiss.read_index(str(index_path))
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
    output_dir: Path | None = None
