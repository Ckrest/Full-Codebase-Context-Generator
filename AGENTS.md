# Repository Overview

This project extracts code artifacts from a source tree, builds a call graph,
creates text embeddings for each node, and provides tools to query or inspect
those embeddings. The main scripts are located in the repository root.

## Core Components

| File | Description |
| ---- | ----------- |
| `LLM_Extreme_Context.py` | Extracts functions from Python, HTML, Markdown, and JavaScript (via tree-sitter) then builds the call graph. |
| `context_utils.py` | Utility helpers for loading graphs and gathering context around nodes. |
| `generate_embeddings.py` | Generates embeddings for call graph nodes using `sentence-transformers` and FAISS. |
| `query_sniper.py` | Interactive CLI to search embeddings and explore neighboring nodes. |
| `inspect_graph.py` | Basic analysis/visualization of the generated call graph. |
| `Start.py` | Entry point providing `generate`, `query`, and `inspect` commands. |

## Function Relationships

1. **Extraction** – `LLM_Extreme_Context.extract_from_python`,
   `extract_from_html`, `extract_from_markdown`, and
   `extract_from_javascript` parse files and return a
   list of entries. `build_call_graph` then creates a `networkx` graph from
   these entries and `save_graph_json` writes it to disk.
2. **Context Gathering** – `context_utils.gather_context` uses
   `expand_neighborhood` to collect code from surrounding nodes in a graph.
3. **Embedding Generation** – `generate_embeddings.main` loads the call graph,
   gathers context for each node, encodes the text with
   `SentenceTransformer`, and saves a FAISS index.
4. **Querying** – `query_sniper.main` loads the embeddings and lets users run
   similarity searches. It can also show neighbors using
   `expand_neighborhood`.
5. **Inspection** – `inspect_graph.main` analyzes the call graph (e.g., node
   degree) and can plot distributions.

The `Start.py` module orchestrates these utilities via command line.

## Testing

Unit tests live in the `tests/` directory. Run them with:

```bash
pytest -q
```

The main test file is `tests/test_extractors.py` which verifies the extractors and call graph generation.
