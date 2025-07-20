# LLM Extreme Context

## Configuration

This project uses a `settings.json` file for configuration.
If the file does not exist it will be created automatically with default values.
You can also copy `settings.example.json` to `settings.json` and modify it as needed.

### Settings Options

- **llm_model**: The sentence transformer model to use (default: "BAAI/bge-small-en")
- **output_dir**: Directory where extracted data is stored (default: "extracted")
- **default_project**: Default project to use for querying (default: "ComfyUI")
- **embedding_dim**: Embedding dimension for the model (default: 384)
- **top_k_results**: Number of top results to return in queries (default: 20)
- **chunk_size**: Text chunk size for processing (default: 1000)
- **allowed_extensions**: File extensions to include in processing
- **exclude_dirs**: Directories to exclude from processing
- **local_model_path**: Optional path to a local embedding model

### Files

- `LLM_Extream_Context.py`: Main extraction script
- `Start.py`: Entry point for running other utilities
- `generate_embeddings.py`: Generate embeddings from call graph
- `query_sniper.py`: Interactive query interface
- `inspect_graph.py`: Graph analysis and visualization
- `settings.json`: Your local configuration (not tracked by git)
- `settings.example.json`: Template configuration file

## Usage

The `settings.json` file is automatically loaded by all scripts. If the file doesn't exist, it will be generated with defaults the first time you run the tools.
Run `python Start.py [path]` where `path` is the folder you want to analyse. If the artifacts for that project do not exist they will be created and you will then be dropped into the interactive query interface.
