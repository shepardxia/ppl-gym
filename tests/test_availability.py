"""Tests for the realization availability mechanism.

Covers:
  (a) is_available predicate
  (b) load_corpus / load_unavailable against the real pyro.jsonl file
"""
from __future__ import annotations

from eval.corpus import is_available, load_corpus, load_realizations, load_unavailable

# ---------------------------------------------------------------------------
# (a) is_available
# ---------------------------------------------------------------------------

# The inference-algorithms hard-condition method demos — Pyro-unavailable.
_UNAVAILABLE_IDS = {
    "probmods2-inference-algorithms/ex1.1",
    "probmods2-inference-algorithms/ex1.2",
    "probmods2-inference-algorithms/ex1.3",
    "probmods2-inference-algorithms/ex2.4",
}


def test_is_available_code_record():
    """A record with a code key (and no available field) is available."""
    assert is_available({"problem_id": "p", "language": "pyro", "code": "..."})


def test_is_available_absent_available_field():
    """available absent defaults to True; code present → available."""
    assert is_available({"problem_id": "p", "language": "pyro", "code": "x = 1"})


def test_is_available_explicit_false():
    """available: false with no code → not available."""
    assert not is_available({
        "problem_id": "p", "language": "pyro",
        "available": False, "reason": "some reason",
    })


def test_is_available_no_code_key():
    """A record without a code key (but no explicit available field) is not available."""
    assert not is_available({"problem_id": "p", "language": "pyro"})


def test_is_available_false_without_code():
    """available: false and no code → not available (belt-and-suspenders)."""
    assert not is_available({"problem_id": "p", "available": False})


# ---------------------------------------------------------------------------
# (b) load_corpus / load_unavailable against the real pyro.jsonl
# ---------------------------------------------------------------------------


def test_load_corpus_pyro_excludes_unavailable():
    """load_corpus('pyro') must not include the 3 unavailable problems."""
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


def test_load_unavailable_pyro_returns_documented_gaps():
    """load_unavailable('pyro') must return exactly the documented unavailable set."""
    unavail_ids = {r["problem_id"] for r in load_unavailable("pyro")}
    assert unavail_ids == _UNAVAILABLE_IDS, (
        f"expected {_UNAVAILABLE_IDS}, got {unavail_ids}"
    )


def test_load_unavailable_pyro_has_reasons():
    """Each unavailable record must carry a non-empty reason."""
    for rec in load_unavailable("pyro"):
        assert rec.get("reason"), f"missing reason on {rec['problem_id']}"
        assert "code" not in rec, f"unavailable record must not have code: {rec['problem_id']}"


def test_load_unavailable_pyro_not_available():
    """Every record returned by load_unavailable must fail is_available."""
    for rec in load_unavailable("pyro"):
        assert not is_available(rec), f"is_available returned True for {rec['problem_id']}"
