import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from query import parse_llm_response


def test_parse_llm_response_triple_quotes():
    text = "'''\n{\"response_type\": \"functions\", \"functions\": [\"foo\"], \"total\": 1}\n'''"
    result = parse_llm_response(text)
    assert result == {"response_type": "functions", "functions": ["foo"], "total": 1}

