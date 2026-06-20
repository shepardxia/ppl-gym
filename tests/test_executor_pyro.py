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
