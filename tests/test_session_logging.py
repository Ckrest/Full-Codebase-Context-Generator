import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_SETTINGS
from session_logger import log_session_to_json, log_summary_to_markdown


def test_logging_defaults_present():
    logging_cfg = DEFAULT_SETTINGS["logging"]
    assert logging_cfg["log_markdown"] is True
    assert logging_cfg["log_json"] is True
    assert logging_cfg["max_functions_to_log"] == 100


def test_session_logger_writes_files(tmp_path):
    data = {
        "query": "Find foo",
        "subqueries": [
            {"text": "foo", "functions": []}
        ],
        "functions": {},
    }
    json_path = log_session_to_json(data, tmp_path)
    md_path = log_summary_to_markdown({"original_query": "Find foo", "subqueries": [], "functions": {}}, tmp_path)
    assert Path(json_path).exists()
    assert Path(md_path).exists()
    assert re.search(r"find_foo\.json$", Path(json_path).name)
    assert re.search(r"query_find_foo_", Path(md_path).name)

