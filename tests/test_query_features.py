import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Start import DEFAULT_SETTINGS
from query_sniper import build_symspell, correct_query, average_embeddings


def test_default_query_settings():
    q = DEFAULT_SETTINGS["query"]
    assert q["use_spellcheck"] is False
    assert q["rephrase_count"] == 1
    assert q["rephrase_model_path"] == ""


def test_spellcheck_basic():
    metadata = [{"name": "foobar"}]
    sym = build_symspell(metadata)
    corrected = correct_query(sym, "foobzr")
    assert corrected == "foobar"


def test_average_embeddings():
    class Dummy:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    vec = average_embeddings(Dummy(), ["a", "b"])
    assert np.allclose(vec, [[2.0, 3.0]])
