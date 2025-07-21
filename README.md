# LLM Extreme Context

## Configuration

This project uses a `settings.json` file for configuration.
If the file does not exist it will be created automatically with default values.
The `settings.example.json` file is kept in sync with the defaults whenever the
tools run, so copying that file is an easy way to start customizing your own
settings. The extractors handle Python, HTML, Markdown, and JavaScript source
files using tree-sitter for the latter.

### Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Workflow

1. **extract** – parse source files and build a call graph
2. **embed** – generate embeddings for each node
3. **query** – interactively search the embeddings
4. **inspect** – optional graph statistics and visualisation

If `sub_question_count` is greater than 0, an LLM call breaks your question into
that many sub-queries before searching the embeddings. This helps find relevant
code from multiple angles.

Run the tests with `pytest` to ensure everything works:

```bash
pytest -q
```

### Settings Options

- Settings are grouped into categories for easier navigation:

- **LLM_model** – `api_key`, `api_type`, `local_path`
- **api_settings** – `temperature`, `top_p`, `max_output_tokens`
 - **encoder_model_path** – location of the sentence transformer model. If left
   blank, the small `sentence-transformers/all-MiniLM-L6-v2` model will be
   downloaded automatically.
- **paths** – `output_dir`
- **query** – `top_k_results`, `use_spellcheck`, `sub_question_count`
- **context** – `context_hops`, `max_neighbors`, `bidirectional`
- **extraction** – `allowed_extensions`, `exclude_dirs`, `comment_lookback_lines`,
  `token_estimate_ratio`, and `minified_js_detection` options
- **visualization** – parameters controlling call graph rendering
- **embedding** – `embedding_dim`, `encoder_model_path`

The extraction step relies on `crawl_directory` which automatically skips files
listed in `.gitignore` and only processes paths with extensions from
`allowed_extensions`.

When `bidirectional` is `True`, context expansion walks both callers and
callees. Set it to `False` to only follow outgoing calls.

### Files

 - `LLM_Extreme_Context.py`: Extracts Python, HTML, Markdown, and JavaScript using tree-sitter
 - `Start.py`: Unified CLI with `extract`, `embed`, `query`, and `inspect` commands
- `generate_embeddings.py`: Generate embeddings from call graph
- `query_sniper.py`: Interactive query interface
- `inspect_graph.py`: Graph analysis and visualization
- `settings.json`: Your local configuration (not tracked by git)
- `settings.example.json`: Template configuration file

## Usage

The `settings.json` file is automatically loaded by all scripts. If the file doesn't exist, it will be generated with defaults the first time you run the tools.
Invoke the CLI with `python Start.py <command>`:

```bash
python Start.py extract /path/to/project
python Start.py embed my_project
python Start.py query my_project --problem "bug" --prompt "search term"
python Start.py inspect my_project
```

Running `Start.py` with just a path enters an interactive loop. While querying you can type `neighbors <n>` to see the graph neighbors of result `n` from the previous search. Interactive prompts are stored in `.full_context_history.json`.

### Spellcheck and Sub-Queries

Two optional features help refine search queries:

- **use_spellcheck** – when `true`, the query is corrected using SymSpell before searching.
  A small dictionary is built from function names automatically, but you can also
  download a larger frequency dictionary from the SymSpell repository and place it
  in the project root.
- **sub_question_count** – when greater than 0, your question is first sent to the
  LLM to produce that many sub-queries. Their embeddings plus the original
  query are averaged together before searching. The default value is `0` (off).

Enable these options in `settings.json` under the `query` section.

### API Settings

The `api_settings` section controls parameters sent to the LLM API. Adjust
`temperature`, `top_p`, and `max_output_tokens` to tune response style and length.
The default `max_output_tokens` is **5000**.

### Debug Logging

All tools output progress information using the `logging` module and `tqdm`
progress bars. Run scripts with the `PYTHONLOGLEVEL=DEBUG` environment variable
to see verbose details while files are scanned and embeddings are generated.
