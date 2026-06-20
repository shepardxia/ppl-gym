"""Tests for the realization availability mechanism.

Covers:
  (a) is_available predicate
  (b) load_corpus / load_unavailable against the real pyro.jsonl file
"""
from __future__ import annotations

import pytest

from eval.corpus import is_available, load_corpus, load_realizations, load_unavailable

# The inference-algorithms hard-condition method demos — Pyro-unavailable.
_UNAVAILABLE_IDS = {
    "probmods2-inference-algorithms/ex1.1",
    "probmods2-inference-algorithms/ex1.2",
    "probmods2-inference-algorithms/ex1.3",
    "probmods2-inference-algorithms/ex2.4",
}


# ---------------------------------------------------------------------------
# (a) is_available — available iff (available != False) AND has a code key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rec, expected",
    [
        # code key present, no explicit available field → available
        ({"problem_id": "p", "language": "pyro", "code": "..."}, True),
        # available: false with a reason and no code → not available
        ({"problem_id": "p", "language": "pyro", "available": False,
          "reason": "some reason"}, False),
        # no code key, no available field → not available
        ({"problem_id": "p", "language": "pyro"}, False),
        # available: false and no code → not available (belt-and-suspenders)
        ({"problem_id": "p", "available": False}, False),
    ],
)
def test_is_available(rec, expected):
    assert is_available(rec) is expected


# ---------------------------------------------------------------------------
# (b) load_corpus / load_unavailable against the real pyro.jsonl
# ---------------------------------------------------------------------------


def test_load_corpus_pyro_excludes_unavailable():
    """load_corpus('pyro') must not include any documented-unavailable problem."""
    _, reals = load_corpus(language="pyro")
    real_ids = {r["problem_id"] for r in reals}
    assert real_ids.isdisjoint(_UNAVAILABLE_IDS), (
        f"unavailable ids found in corpus: {real_ids & _UNAVAILABLE_IDS}"
    )


def test_load_corpus_pyro_count_matches_available():
    """load_corpus('pyro') count equals the number of available pyro realizations
    (derived from the data, so it does not rebreak as the column evolves)."""
    _, reals = load_corpus(language="pyro")
    available = [r for r in load_realizations("pyro") if is_available(r)]
    assert len(reals) == len(available), f"corpus {len(reals)} != available {len(available)}"


def test_load_unavailable_pyro():
    """load_unavailable('pyro') returns exactly the documented gaps, and every
    record carries a non-empty reason, has no code, and fails is_available."""
    recs = load_unavailable("pyro")

    unavail_ids = {r["problem_id"] for r in recs}
    assert unavail_ids == _UNAVAILABLE_IDS, f"expected {_UNAVAILABLE_IDS}, got {unavail_ids}"

    for rec in recs:
        assert rec.get("reason"), f"missing reason on {rec['problem_id']}"
        assert "code" not in rec, f"unavailable record must not have code: {rec['problem_id']}"
        assert not is_available(rec), f"is_available returned True for {rec['problem_id']}"
