from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from query import aggregate_scores


def test_aggregate_scores_basic():
    data = {
        "foo": {
            "subqueries": [
                {"index": 0, "text": "q1", "score": 0.8, "rank": 1},
                {"index": 1, "text": "q2", "score": 0.6, "rank": 2},
            ]
        }
    }
    agg = aggregate_scores(data)
    assert "foo" in agg
    assert np.isclose(agg["foo"]["avg_score"], 0.7)
    assert np.isclose(agg["foo"]["max_score"], 0.8)
    assert np.isclose(agg["foo"]["stddev_score"], np.std([0.8,0.6]))
    assert agg["foo"]["queries_matched"] == ["q1","q2"]
