"""Tests for the realization availability mechanism.

Covers:
  (a) is_available predicate
  (b) load_corpus / load_unavailable against the real pyro.jsonl file
"""
from __future__ import annotations

import pytest

from eval.corpus import is_available, load_corpus, load_realizations, load_unavailable


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


@pytest.mark.parametrize("language", ["webppl", "pyro", "gen", "stan", "reference"])
def test_load_corpus_excludes_unavailable(language):
    """load_corpus must not include any documented-unavailable problem (holds for
    any unavailable set, including empty — data-derived, so it never rebreaks as
    the column evolves)."""
    _, reals = load_corpus(language=language)
    real_ids = {r["problem_id"] for r in reals}
    unavail_ids = {r["problem_id"] for r in load_unavailable(language)}
    assert real_ids.isdisjoint(unavail_ids), (
        f"[{language}] unavailable ids found in corpus: {real_ids & unavail_ids}"
    )


@pytest.mark.parametrize("language", ["webppl", "pyro", "gen", "stan", "reference"])
def test_load_corpus_count_matches_available(language):
    """load_corpus count equals the number of available realizations (data-derived,
    so it does not rebreak as the column evolves)."""
    _, reals = load_corpus(language=language)
    available = [r for r in load_realizations(language) if is_available(r)]
    assert len(reals) == len(available), f"[{language}] corpus {len(reals)} != available {len(available)}"


@pytest.mark.parametrize("language", ["webppl", "pyro", "gen", "stan", "reference"])
def test_unavailable_records_wellformed(language):
    """Every unavailable record (however many) carries a non-empty reason, has no
    code, and fails is_available — the unavailable-record contract, checked
    against whatever the column currently declares."""
    for rec in load_unavailable(language):
        assert rec.get("reason"), f"missing reason on {rec['problem_id']}"
        assert "code" not in rec, f"unavailable record must not have code: {rec['problem_id']}"
        assert not is_available(rec), f"is_available returned True for {rec['problem_id']}"
