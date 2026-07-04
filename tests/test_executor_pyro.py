"""Contract tests for eval/executor_pyro.py.

The executor's job: run a Pyro program via its own machinery and serialize the
ANSWER into a form eval/algebra.py accepts under the problem's spec. Tested at
that boundary (execute → canonicalize) rather than per serialization detail.
Each case spawns a real subprocess (~1-2s), so cases are kept few.
"""

import pytest

from eval.algebra import EnumDist, ParamDist, canonicalize, parse_spec
from eval.executor_pyro import execute_pyro


@pytest.mark.parametrize("code, spec_d, check", [
    # discrete distribution → EnumDist over bool, P(True) ≈ 0.7
    ("ANSWER = dist.Bernoulli(0.7)", {"kind": "dist", "domain": "bool"},
     lambda c: isinstance(c, EnumDist) and dict(zip(c.support, c.probs))[True] == pytest.approx(0.7, abs=1e-4)),
    # categorical → EnumDist over a 3-way finite support
    ("ANSWER = dist.Categorical(probs=torch.tensor([0.1, 0.3, 0.6]))",
     {"kind": "dist", "domain": "int"},
     lambda c: isinstance(c, EnumDist) and len(c.support) == 3),
    # continuous distribution → ParamDist (family alias normalized by the algebra)
    ("ANSWER = dist.Normal(0., 1.)", {"kind": "dist", "domain": "real"},
     lambda c: isinstance(c, ParamDist)),
    # inference output (Importance + EmpiricalMarginal) → EnumDist
    ("""
from pyro.infer import Importance, EmpiricalMarginal
def model():
    return pyro.sample('rain', dist.Bernoulli(0.3))
ANSWER = EmpiricalMarginal(Importance(model, guide=None, num_samples=200).run())
""", {"kind": "dist", "domain": "bool"},
     lambda c: isinstance(c, EnumDist) and sum(c.probs) == pytest.approx(1.0, abs=1e-4)),
    # plain tensor → real-vector value
    ("ANSWER = torch.tensor([1.0, 2.0, 3.0])", {"kind": "value", "domain": "realvec"},
     lambda c: c.value == [1.0, 2.0, 3.0]),
])
def test_executor_output_canonicalizes(code, spec_d, check):
    r = execute_pyro(code, random_seed=42)
    assert r.success, r.stderr
    assert check(canonicalize(r.answer, parse_spec(spec_d)))


def test_seeding():
    code = "ANSWER = [pyro.sample(f'x{i}', dist.Normal(0., 1.)).item() for i in range(5)]"
    a = execute_pyro(code, random_seed=42)
    b = execute_pyro(code, random_seed=42)
    c = execute_pyro(code, random_seed=99)
    assert a.success and b.success and c.success
    assert a.answer == b.answer        # same seed → identical
    assert a.answer != c.answer        # different seed → different draws


def test_missing_answer_fails():
    r = execute_pyro("x = 1 + 1", random_seed=42)
    assert not r.success and "ANSWER" in r.error_message


# ---------------------------------------------------------------------------
# Budget policy + chunking (contract-level; no subprocess spawned)
# ---------------------------------------------------------------------------

def test_chunk_budget_policy(monkeypatch):
    """Each chunk gets per-seed budget x its seed count, capped; chunks cover
    all seeds in order and results reassemble aligned."""
    import eval.executor_pyro as ep
    from eval.config import PYRO_CHUNK_BUDGET_CAP, PYRO_SEED_BUDGET_SCALE

    calls = []

    def fake_chunk(code, seeds, timeout):
        calls.append((tuple(seeds), timeout))
        return [f"s{s}" for s in seeds]

    monkeypatch.setattr(ep, "_run_pyro_chunk", fake_chunk)

    # k=5 exact GT with ample workers: one seed per chunk, full scaled budget.
    out = ep.execute_pyro_batch("x", [1, 2, 3, 4, 5], timeout=60, workers=8)
    assert out == ["s1", "s2", "s3", "s4", "s5"]
    assert all(t == 60 * PYRO_SEED_BUDGET_SCALE for _, t in calls)
    assert [s for seeds, _ in calls for s in seeds] == [1, 2, 3, 4, 5]

    # Many-seed draws chunk: budget capped, never hours.
    calls.clear()
    ep.execute_pyro_batch("x", list(range(200)), timeout=60, workers=1)
    assert len(calls) == 1
    assert calls[0][1] == PYRO_CHUNK_BUDGET_CAP


def test_env_worker_budget(monkeypatch):
    from eval.config import DEFAULT_MC_WORKERS, total_exec_workers
    monkeypatch.delenv("PPL_GYM_EXEC_WORKERS", raising=False)
    assert total_exec_workers() == DEFAULT_MC_WORKERS
    monkeypatch.setenv("PPL_GYM_EXEC_WORKERS", "48")
    assert total_exec_workers() == 48
    monkeypatch.setenv("PPL_GYM_EXEC_WORKERS", "junk")
    assert total_exec_workers() == DEFAULT_MC_WORKERS
