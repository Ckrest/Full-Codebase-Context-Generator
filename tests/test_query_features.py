import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_SETTINGS
from spellcheck_utils import create_symspell_from_terms, correct_phrase
from query import average_embeddings


def test_default_query_settings():
    q = DEFAULT_SETTINGS["query"]
    assert q["use_spellcheck"] is False
    assert q["sub_question_count"] == 0
    assert q["prompt_suggestion_count"] == 0


def test_spellcheck_basic():
    terms = ["foobar"]
    sym = create_symspell_from_terms(terms)
    corrected = correct_phrase(sym, "foobzr")
    assert corrected == "foobar"


def test_average_embeddings():
    class Dummy:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    vec = average_embeddings(Dummy(), ["a", "b"])
    assert np.allclose(vec, [[2.0, 3.0]])
