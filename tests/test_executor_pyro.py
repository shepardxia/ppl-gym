"""Tests for eval/executor_pyro.py.

Each test spawns a real subprocess (~1-2s). Keep the total count small (~8-10).
"""

import pytest
from eval.executor_pyro import execute_pyro
from eval.algebra import canonicalize, parse_spec


# ── Seeding ──────────────────────────────────────────────────────────────────

def test_same_seed_deterministic():
    """Same seed -> identical output across two runs."""
    code = "ANSWER = [pyro.sample(f'x{i}', dist.Normal(0., 1.)).item() for i in range(5)]"
    r1 = execute_pyro(code, random_seed=42)
    r2 = execute_pyro(code, random_seed=42)
    assert r1.success, r1.stderr
    assert r2.success, r2.stderr
    assert r1.answer == r2.answer, (
        f"Same seed should give identical output.\n"
        f"run1={r1.answer}\nrun2={r2.answer}"
    )


def test_different_seed_different():
    """Different seeds -> different output (with overwhelming probability for 5 Normal draws)."""
    code = "ANSWER = [pyro.sample(f'x{i}', dist.Normal(0., 1.)).item() for i in range(5)]"
    r42 = execute_pyro(code, random_seed=42)
    r99 = execute_pyro(code, random_seed=99)
    assert r42.success, r42.stderr
    assert r99.success, r99.stderr
    assert r42.answer != r99.answer, (
        "Different seeds should produce different draws (astronomically unlikely to collide)."
    )


# ── Discrete distributions ────────────────────────────────────────────────────

def test_bernoulli_dist_enum():
    """dist.Bernoulli -> dist_enum with numeric support [0.0, 1.0]."""
    r = execute_pyro("ANSWER = dist.Bernoulli(0.7)", random_seed=42)
    assert r.success, r.stderr
    ans = r.answer
    assert ans["kind"] == "dist_enum"
    assert len(ans["support"]) == 2
    assert len(ans["probs"]) == 2
    assert abs(sum(ans["probs"]) - 1.0) < 1e-6, f"probs sum={sum(ans['probs'])}"
    # 0.7 mass on 1.0
    idx1 = ans["support"].index(1.0)
    assert abs(ans["probs"][idx1] - 0.7) < 1e-5


def test_categorical_dist_enum():
    """dist.Categorical -> dist_enum with integer support."""
    code = "ANSWER = dist.Categorical(probs=torch.tensor([0.1, 0.3, 0.6]))"
    r = execute_pyro(code, random_seed=42)
    assert r.success, r.stderr
    ans = r.answer
    assert ans["kind"] == "dist_enum"
    assert len(ans["support"]) == 3
    assert abs(sum(ans["probs"]) - 1.0) < 1e-5


# ── Parametric (continuous) distributions ────────────────────────────────────

def test_normal_dist_param():
    """dist.Normal -> dist_param with family 'normal' and loc/scale params (not remapped)."""
    r = execute_pyro("ANSWER = dist.Normal(0., 1.)", random_seed=42)
    assert r.success, r.stderr
    ans = r.answer
    assert ans["kind"] == "dist_param"
    assert ans["family"] == "normal"
    # executor does NOT remap; loc/scale are Pyro's native names
    assert "loc" in ans["params"]
    assert "scale" in ans["params"]
    assert abs(ans["params"]["loc"] - 0.0) < 1e-6
    assert abs(ans["params"]["scale"] - 1.0) < 1e-6


# ── Tensor serialization ──────────────────────────────────────────────────────

def test_1d_tensor_is_list():
    """1-D torch.Tensor -> plain Python list."""
    r = execute_pyro("ANSWER = torch.tensor([1.0, 2.0, 3.0])", random_seed=42)
    assert r.success, r.stderr
    assert r.answer == [1.0, 2.0, 3.0]


# ── Dict key coercion ─────────────────────────────────────────────────────────

def test_dict_bool_keys():
    """Dict with bool keys -> JSON string keys 'true'/'false'."""
    r = execute_pyro("ANSWER = {True: 0.6, False: 0.4}", random_seed=42)
    assert r.success, r.stderr
    assert set(r.answer.keys()) == {"true", "false"}
    assert abs(r.answer["true"] - 0.6) < 1e-9
    assert abs(r.answer["false"] - 0.4) < 1e-9


# ── Empirical / EmpiricalMarginal ─────────────────────────────────────────────

def test_importance_empirical_marginal():
    """Importance + EmpiricalMarginal -> dist_enum whose probs sum to ~1."""
    code = """
from pyro.infer import Importance, EmpiricalMarginal
def model():
    rain = pyro.sample('rain', dist.Bernoulli(0.3))
    return rain
posterior = Importance(model, guide=None, num_samples=200).run()
ANSWER = EmpiricalMarginal(posterior)
"""
    r = execute_pyro(code, random_seed=42)
    assert r.success, r.stderr
    ans = r.answer
    assert ans["kind"] == "dist_enum"
    assert abs(sum(ans["probs"]) - 1.0) < 1e-5, f"probs sum={sum(ans['probs'])}"
    # Should have support {0.0, 1.0}
    assert len(ans["support"]) == 2


# ── Error cases ───────────────────────────────────────────────────────────────

def test_missing_answer_fails():
    """Program without ANSWER binding -> failure with clear message."""
    r = execute_pyro("x = 1 + 1", random_seed=42)
    assert not r.success
    assert "ANSWER" in r.error_message


# ── End-to-end algebra integration ───────────────────────────────────────────

def test_bernoulli_through_algebra():
    """Bernoulli dist_enum wired through algebra.canonicalize under dist/bool spec."""
    r = execute_pyro("ANSWER = dist.Bernoulli(0.7)", random_seed=42)
    assert r.success, r.stderr
    spec = parse_spec({"kind": "dist", "domain": "bool"})
    # canonicalize should succeed (EnumDist with bool-coerced support)
    canon = canonicalize(r.answer, spec)
    from eval.algebra import EnumDist
    assert isinstance(canon, EnumDist)
    assert abs(sum(canon.probs) - 1.0) < 1e-5
