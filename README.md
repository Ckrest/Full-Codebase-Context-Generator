# LLM Extreme Context

## Configuration

This project uses a `settings.json` file for configuration.
If the file does not exist it will be created automatically with default values.
The `settings.example.json` file is kept in sync with the defaults in `Start.py`
whenever the tools run, so copying that file is an easy way to start customising
your own settings. The extractors handle Python, HTML, Markdown, and JavaScript
source files using tree-sitter for the latter.

### Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the tests with `pytest` to ensure everything works:

```bash
pytest -q
```

### Settings Options

Settings are grouped into categories for easier navigation:

- **model** – `llm_model`, `local_model_path`
- **paths** – `output_dir`
- **embedding** – `embedding_dim`
- **query** – `top_k_results`
- **context** – `chunk_size`, `context_hops`, `max_neighbors`, `bidirectional`
- **extraction** – `allowed_extensions`, `exclude_dirs`, `comment_lookback_lines`,
  `token_estimate_ratio`, and `minified_js_detection` options
- **visualization** – parameters controlling call graph rendering

The extraction step relies on `crawl_directory` which automatically skips files
listed in `.gitignore` and only processes paths with extensions from
`allowed_extensions`.

When `bidirectional` is `True`, context expansion walks both callers and
callees. Set it to `False` to only follow outgoing calls.

### Files

 - `LLM_Extreme_Context.py`: Extracts Python, HTML, Markdown, and JavaScript using tree-sitter
- `Start.py`: Entry point for running other utilities
- `generate_embeddings.py`: Generate embeddings from call graph
- `query_sniper.py`: Interactive query interface
- `inspect_graph.py`: Graph analysis and visualization
- `settings.json`: Your local configuration (not tracked by git)
- `settings.example.json`: Template configuration file

## Usage

The `settings.json` file is automatically loaded by all scripts. If the file doesn't exist, it will be generated with defaults the first time you run the tools.
Run `python Start.py [path]` where `path` is the folder you want to analyse. If the artifacts for that project do not exist they will be created and you will then be dropped into the interactive query interface.
While in the query interface you can type `neighbors <n>` to see the graph neighbors of result `n` from the previous search.
