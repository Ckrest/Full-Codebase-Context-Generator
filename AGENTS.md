# Repository Overview

This project extracts code artifacts from a source tree, builds a call graph,
creates text embeddings for each node, and provides tools to query or inspect
those embeddings. The main scripts are located in the repository root.

## Core Components

| File | Description |
| ---- | ----------- |
| `main.py` | Unified CLI with `extract`, `embed`, `query`, and interactive modes. |
| `graph.py` | Extraction utilities and call graph helpers. |
| `embedding.py` | Generates embeddings and FAISS indices. |
| `query.py` | Searches embeddings and builds prompts for the LLM. |
| `prompt_builder.py` | Formats search results and function summaries. |
| `llm.py` | Helper routines for calling the Gemini API. |
| `interactive_cli.py` | PromptToolkit dialogs for the interactive workflow. |
| `config.py` | Loads and stores project settings. |

## Function Relationships

1. **Extraction** – `graph.extract_from_python`,
   `extract_from_html`, `extract_from_markdown`,
   `extract_from_javascript`, and `extract_from_typescript` parse files
   and return a list of entries. `build_call_graph` adds every entry to a
   `networkx` graph (not just functions) and `save_graph_json` writes it to
   disk.
2. **Context Gathering** – `graph.gather_context` collects code from related
   nodes using `expand_graph`. Direction can be controlled with the
   `bidirectional` setting.
3. **Embedding Generation** – `embedding.generate_embeddings` loads the call
   graph, gathers context for each node, encodes the text with a
   `SentenceTransformer`, and saves a FAISS index.
4. **Querying** – `query.main` searches the embeddings, builds prompts via
   `prompt_builder.format_summary` and optionally calls an LLM.
5. **Inspection** – `graph.analyze_graph` prints basic statistics on the
   generated call graph.

The `main.py` script orchestrates these utilities via command line or interactive mode.

### Usage Notes

Run `python main.py <command>` where `<command>` is one of:

- `extract` – build a call graph for a project
- `embed` – generate embeddings from a call graph
- `query` – search the embeddings interactively
- `inspect` – print basic graph statistics

Running `main.py` without a command launches the interactive workflow. Command
history is stored in the `~/.full_context_history/` directory via
`prompt_toolkit`.
The interactive search can optionally correct queries using SymSpell when
`use_spellcheck` is set to `true` in `settings.json`.

### Debugging

Scripts print detailed progress information when run with `PYTHONLOGLEVEL=DEBUG`.
Extraction and embedding generation also show `tqdm` progress bars for visual
feedback.

## Testing

Unit tests live in the `tests/` directory. Run them with:

```bash
pytest -q
```

The main test file is `tests/test_extractors.py` which verifies the extractors and call graph generation.
Always run `pytest -q` before committing changes.
