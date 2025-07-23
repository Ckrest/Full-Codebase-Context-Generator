import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from query import parse_llm_response, MalformedLLMOutput


def test_parse_llm_response_triple_quotes():
    text = "'''\n{\"response_type\": \"functions\", \"functions\": [\"foo\"], \"total\": 1}\n'''"
    result = parse_llm_response(text)
    assert result == {"response_type": "functions", "functions": ["foo"], "total": 1}


def test_parse_llm_response_malformed():
    bad = "not valid json"
    with pytest.raises(MalformedLLMOutput):
        parse_llm_response(bad)

