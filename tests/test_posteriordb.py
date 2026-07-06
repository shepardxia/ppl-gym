"""Contract tests for the posteriordb integration (eval/posteriordb.py, stan_bundle).

Three contracts: (a) the bundle codec round-trips and the data-block parser
recovers declared names; (b) ingestion turns a gold posterior into a valid
problem + realizations; (c) the committed dataset is coherent (aligned columns,
complete plain-prose statements, documented exclusions). The Stan execution path
is slow/toolchain-bound and runs only under PPL_GYM_RUN_STAN=1.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from eval import stan_bundle
from eval.corpus import BATCH_EXECUTORS
from eval.gt_cache import EXECUTOR_VERSION
from eval.io import load_jsonl
from eval.posteriordb import _PDB_ROOT, data_block_vars

_HAVE_PDB = _PDB_ROOT.exists()
_pdb = pytest.mark.skipif(not _HAVE_PDB, reason="posteriordb clone not present")


# ── (a) bundle codec + data-block parsing ──────────────────────────────────

def test_bundle_roundtrips():
    model = "data { int N; }\nparameters { real x; }\nmodel { x ~ normal(0,1); }"
    data, params, sampling = {"N": 3, "y": [1.0, 2.0]}, ["x", "z[1]"], {"chains": 2}
    b = stan_bundle.unpack(stan_bundle.pack(model, data, params, sampling))
    assert b.data == data and b.params == params and b.sampling["chains"] == 2
    assert "parameters { real x; }" in b.model and "//@" not in b.model
    # omitted sampling → defaults
    assert stan_bundle.unpack(stan_bundle.pack("model {}", {"N": 1}, ["x"])).sampling == stan_bundle.DEFAULT_SAMPLING


@pytest.mark.parametrize("bundle, missing", [
    ('model {}\n//@ PARAMS ["x"]\n', "DATA"),
    ("model {}\n//@ DATA {}\n", "PARAMS"),
])
def test_bundle_missing_directive_raises(bundle, missing):
    with pytest.raises(ValueError, match=missing):
        stan_bundle.unpack(bundle)


def test_data_block_vars():
    model = ("data { int<lower=0> N; vector<lower=0,upper=200>[N] kid; // c\n"
             " array[N] real y; matrix[N,2] X; }\nparameters { real b; }")
    assert data_block_vars(model) == ["N", "kid", "y", "X"]
    assert data_block_vars("parameters { real x; }") == []


@pytest.mark.parametrize("lang", ["stan", "reference"])
def test_language_registered(lang):
    assert lang in BATCH_EXECUTORS and lang in EXECUTOR_VERSION


# ── (b) ingestion turns a gold posterior into valid records ────────────────

@_pdb
def test_ingestion_contract():
    from eval.posteriordb import (answer_spec, gold_posterior_names, param_names,
                                  reference_blocks, reference_chains, stan_realization)
    name = "eight_schools-eight_schools_noncentered"
    assert name in gold_posterior_names()
    # answer_spec is a record of per-parameter real marginals, keyed by the draw params
    spec = answer_spec(name)
    assert spec["kind"] == "record" and set(spec["fields"]) == set(param_names(name))
    assert all(f == {"kind": "dist", "domain": "real"} for f in spec["fields"].values())
    # the bundle keeps only the model's declared data (kidiq ships extra columns)
    kb = stan_bundle.unpack(stan_realization("kidiq-kidscore_momhs")["code"])
    assert set(kb.data) == {"N", "kid_score", "mom_hs"}
    # reference replay partitions the gold chains disjointly (no draw lost or duplicated)
    blocks = reference_blocks(name, 5)
    assert len(blocks) == 5
    assert sum(len(b["mu"]) for b in blocks) == sum(len(c["mu"]) for c in reference_chains(name))


# ── (c) the committed dataset is coherent ──────────────────────────────────

def test_committed_dataset_coherent():
    rows = load_jsonl("data/problems/posteriordb.jsonl")
    if not rows:
        pytest.skip("no posteriordb problems committed yet")
    probs = {r["problem_id"] for r in rows}
    stan = {r["problem_id"] for r in load_jsonl("data/realizations/stan.jsonl")}
    ref = {r["problem_id"] for r in load_jsonl("data/realizations/reference.jsonl")}
    assert probs <= stan and probs <= ref, "every problem needs a stan + reference realization"
    md = re.compile(r"##|\*\*|^\s*[-*]\s", re.M)
    kw = re.compile(r"positive_ordered|\bsimplex\b|vector\[|int<lower|real<lower|"
                    r"transformed parameters|parameters\s*\{|generated quantities")
    for r in rows:
        s = r["statement"]
        assert s.get("given") and s.get("model") and s.get("query"), f"incomplete: {r['problem_id']}"
        txt = "\n".join(s.values())
        assert not md.search(txt) and not kw.search(txt), f"markdown/Stan leak: {r['problem_id']}"


def test_exclusions_documented_and_absent():
    from eval.posteriordb import EXCLUDED_OUT
    if not EXCLUDED_OUT.exists():
        pytest.skip("no exclusions recorded")
    live = {r["problem_id"] for r in load_jsonl("data/problems/posteriordb.jsonl")}
    for rec in load_jsonl(EXCLUDED_OUT):
        assert rec.get("reason") and rec["problem_id"] not in live


# ── Stan execution path (opt-in) ───────────────────────────────────────────

@pytest.mark.skipif(not (_HAVE_PDB and os.environ.get("PPL_GYM_RUN_STAN") == "1"),
                    reason="set PPL_GYM_RUN_STAN=1 (and clone posteriordb) for the Stan exec test")
def test_stan_executor_reproduces_gold_mean():
    import statistics
    from eval.executor_stan import execute_stan_batch
    from eval.posteriordb import reference_blocks, stan_realization
    name = "eight_schools-eight_schools_noncentered"
    out, errs = execute_stan_batch(stan_realization(name)["code"], [42], timeout=120, workers=4)
    ref = reference_blocks(name, 1)[0]
    assert out[0] is not None and errs[0] is None
    for p in ("mu", "tau"):
        assert abs(statistics.mean(out[0][p]) - statistics.mean(ref[p])) < 0.5


# ---------------------------------------------------------------------------
# Self-generated reference overlay (eval/reference_gen.py)
# ---------------------------------------------------------------------------

def test_overlay_resolution(tmp_path, monkeypatch):
    """Overlay draws resolve through reference_chains/info for non-gold names;
    gold names never read the overlay; validated = gold + overlay."""
    import eval.reference_gen as rg
    import eval.posteriordb as pdb

    monkeypatch.setattr(rg, "OVERLAY_DIR", tmp_path)
    name = "fake_data-fake_model"
    chains = [{"mu": [0.1, 0.2]}, {"mu": [0.3, 0.4]}]
    (tmp_path / f"{name}.json").write_text(json.dumps(chains))
    (tmp_path / f"{name}.info.json").write_text(json.dumps(
        {"provenance": "self-generated", "inference": {"method_arguments": {}}}))

    assert rg.overlay_names() == [name]
    assert name in pdb.validated_posterior_names()
    pdb.reference_chains.cache_clear()
    assert pdb.reference_chains(name) == chains
    assert pdb.reference_info(name)["provenance"] == "self-generated"
    pdb.reference_chains.cache_clear()


def test_overlay_refuses_gold(monkeypatch):
    import eval.reference_gen as rg
    gold = rg.gold_posterior_names()[0]
    r = rg.generate_reference(gold)
    assert r["status"] == "error" and "gold" in r["error"]
