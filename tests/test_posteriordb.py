"""Tests for the posteriordb integration: bundle codec, ingestion, executors.

Three tiers:
  (a) stan_bundle codec        — pure, always runs
  (b) data-block parsing       — pure, always runs
  (c) ingestion / committed records / reference replay — needs the posteriordb
      clone (a gitignored dep) and/or the committed dataset rows
The Stan *execution* path (compile + NUTS) is slow and toolchain-dependent; it
runs only when PPL_GYM_RUN_STAN=1 is set.
"""
from __future__ import annotations

import os

import pytest

from eval import stan_bundle
from eval.corpus import BATCH_EXECUTORS
from eval.gt_cache import EXECUTOR_VERSION
from eval.io import load_jsonl
from eval.posteriordb import _PDB_ROOT, data_block_vars

_HAVE_PDB = _PDB_ROOT.exists()
_pdb = pytest.mark.skipif(not _HAVE_PDB, reason="posteriordb clone not present")


# ---------------------------------------------------------------------------
# (a) stan_bundle codec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, data, params, sampling, expect_sampling",
    [
        # full roundtrip: model/data/params survive, sampling overrides applied
        (
            "data { int N; }\nparameters { real x; }\nmodel { x ~ normal(0,1); }",
            {"N": 3, "y": [1.0, 2.0, 3.0]},
            ["x", "z[1]"],
            {"chains": 2, "iter_warmup": 500, "iter_sampling": 500},
            {"chains": 2, "iter_warmup": 500, "iter_sampling": 500},
        ),
        # omitted sampling falls back to the default config
        ("model {}", {"N": 1}, ["x"], None, stan_bundle.DEFAULT_SAMPLING),
    ],
)
def test_bundle_roundtrip(model, data, params, sampling, expect_sampling):
    b = stan_bundle.unpack(stan_bundle.pack(model, data, params, sampling))
    assert b.data == data
    assert b.params == params
    for k, v in expect_sampling.items():
        assert b.sampling[k] == v
    # the recovered model preserves source and strips the directive lines
    assert model.strip().splitlines()[0] in b.model
    assert "//@" not in b.model


@pytest.mark.parametrize(
    "bundle, missing",
    [
        ('model {}\n//@ PARAMS ["x"]\n', "DATA"),
        ("model {}\n//@ DATA {}\n", "PARAMS"),
    ],
)
def test_bundle_missing_directives_raise(bundle, missing):
    with pytest.raises(ValueError, match=missing):
        stan_bundle.unpack(bundle)


# ---------------------------------------------------------------------------
# (b) data-block parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, expected",
    [
        # constraints, arrays, matrices, and line comments all strip to the name
        (
            """
            data {
              int<lower=0> N;
              vector<lower=0, upper=200>[N] kid_score;  // bounded
              array[N] real y;
              matrix[N, 2] X;
            }
            parameters { real beta; }
            """,
            ["N", "kid_score", "y", "X"],
        ),
        # no data block at all -> empty
        ("parameters { real x; }", []),
    ],
)
def test_data_block_vars(model, expected):
    assert data_block_vars(model) == expected


# ---------------------------------------------------------------------------
# wiring: both new languages registered everywhere GT collection needs them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["stan", "reference"])
def test_language_registered(lang):
    assert lang in BATCH_EXECUTORS
    assert lang in EXECUTOR_VERSION


# ---------------------------------------------------------------------------
# (c) ingestion + committed records
# ---------------------------------------------------------------------------

@_pdb
def test_gold_posteriors_discovered():
    from eval.posteriordb import gold_posterior_names
    names = gold_posterior_names()
    assert len(names) >= 40
    assert "eight_schools-eight_schools_noncentered" in names


@_pdb
def test_stan_bundle_data_trimmed_to_declared():
    """kidiq ships extra columns; the bundle keeps only declared inputs."""
    from eval.posteriordb import stan_realization
    b = stan_bundle.unpack(stan_realization("kidiq-kidscore_momhs")["code"])
    assert set(b.data) == {"N", "kid_score", "mom_hs"}
    assert "mom_iq" not in b.data


@_pdb
def test_answer_spec_is_record_of_real_marginals():
    from eval.posteriordb import answer_spec, param_names
    name = "eight_schools-eight_schools_noncentered"
    spec = answer_spec(name)
    assert spec["kind"] == "record"
    assert set(spec["fields"]) == set(param_names(name))
    for f in spec["fields"].values():
        assert f == {"kind": "dist", "domain": "real"}


@_pdb
def test_reference_blocks_partition_chains():
    """Blocks are a disjoint partition of the gold chains (no draw lost/dup'd),
    and the reference realization replays under the `reference` language by name."""
    from eval.posteriordb import (problem_id, reference_blocks,
                                  reference_chains, reference_realization)
    name = "eight_schools-eight_schools_noncentered"
    chains = reference_chains(name)
    blocks = reference_blocks(name, 5)
    assert len(blocks) == 5
    total_chain = sum(len(c["mu"]) for c in chains)
    total_block = sum(len(b["mu"]) for b in blocks)
    assert total_block == total_chain

    r = reference_realization("arma-arma11")
    assert r["language"] == "reference"
    assert r["code"] == "arma-arma11"
    assert r["problem_id"] == problem_id("arma-arma11")


# ---------------------------------------------------------------------------
# committed dataset rows
# ---------------------------------------------------------------------------

def test_committed_records_align_and_have_statements():
    """Each committed posteriordb problem has a stan + reference realization and
    a complete (given/model/query) statement."""
    rows = load_jsonl("data/problems/posteriordb.jsonl")
    if not rows:
        pytest.skip("no posteriordb problems committed yet")
    probs = {r["problem_id"] for r in rows}
    stan = {r["problem_id"] for r in load_jsonl("data/realizations/stan.jsonl")}
    ref = {r["problem_id"] for r in load_jsonl("data/realizations/reference.jsonl")}
    assert probs <= stan, f"problems missing stan realization: {probs - stan}"
    assert probs <= ref, f"problems missing reference realization: {probs - ref}"
    for r in rows:
        s = r["statement"]
        assert s.get("given") and s.get("model") and s.get("query"), \
            f"incomplete statement: {r['problem_id']}"


def test_statements_have_no_markdown_or_stan_leakage():
    """Statements are plain prose that pin the model, not the Stan program."""
    import re
    rows = load_jsonl("data/problems/posteriordb.jsonl")
    if not rows:
        pytest.skip("no posteriordb problems committed yet")
    md = re.compile(r"##|\*\*|^\s*[-*]\s", re.M)
    kw = re.compile(r"positive_ordered|\bsimplex\b|ordered\[|\bvector\[|int<lower|"
                    r"real<lower|transformed parameters|parameters\s*\{|generated quantities")
    for r in rows:
        txt = "\n".join(r["statement"][k] for k in ("given", "model", "query"))
        assert not md.search(txt), f"markdown in statement: {r['problem_id']}"
        assert not kw.search(txt), f"Stan keyword leak in statement: {r['problem_id']}"


def test_excluded_problems_are_absent_and_documented():
    """A pruned problem (non-discriminable) is removed from every live file and
    recorded in the excluded ledger with a reason — never silently dropped."""
    from eval.posteriordb import EXCLUDED_OUT
    if not EXCLUDED_OUT.exists():
        pytest.skip("no exclusions recorded")
    excluded = load_jsonl(EXCLUDED_OUT)
    live = {r["problem_id"] for r in load_jsonl("data/problems/posteriordb.jsonl")}
    stan = {r["problem_id"] for r in load_jsonl("data/realizations/stan.jsonl")}
    ref = {r["problem_id"] for r in load_jsonl("data/realizations/reference.jsonl")}
    for rec in excluded:
        pid = rec["problem_id"]
        assert rec.get("reason"), f"excluded {pid} has no reason"
        assert pid not in live, f"excluded {pid} still in live corpus"
        assert pid not in stan and pid not in ref, f"excluded {pid} still has a realization"


# ---------------------------------------------------------------------------
# (c') Stan execution path — slow, opt-in
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_HAVE_PDB and os.environ.get("PPL_GYM_RUN_STAN") == "1"),
                    reason="set PPL_GYM_RUN_STAN=1 (and clone posteriordb) for the Stan exec test")
def test_stan_executor_reproduces_gold_mean():
    import statistics
    from eval.executor_stan import execute_stan_batch
    from eval.posteriordb import reference_blocks, stan_realization
    name = "eight_schools-eight_schools_noncentered"
    out = execute_stan_batch(stan_realization(name)["code"], [42], timeout=120, workers=4)
    assert out[0] is not None
    ref = reference_blocks(name, 1)[0]
    for p in ("mu", "tau"):
        got = statistics.mean(out[0][p])
        want = statistics.mean(ref[p])
        assert abs(got - want) < 0.5, f"{p}: stan {got:.3f} vs gold {want:.3f}"
