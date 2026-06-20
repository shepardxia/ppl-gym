"""Contract tests for eval/algebra.py — the answer comparator (data/SCHEMA.md).

Organized by the promises the comparator makes, exercised through its PUBLIC
surface (judge / distance / parse_spec / canonicalize) via golden case tables —
not one test per internal form. The implementation is free to change as long as
these contracts hold.

The comparator promises:
  1. judge() classifies candidate-vs-GT as pass / fail / ill_posed / malformed.
  2. Comparison is representation-invariant: the same answer in any wire form
     (enum / mapping / __kind / int-vs-float labels / parametric aliases) compares equal.
  3. The metric numbers are correct (TV, W1, abs-diff).
  4. Tolerance is measured (GT noise floor + candidate self-noise), never authored.
  5. Specs parse and round-trip; malformed specs and answers are rejected.
"""

import math
import random

import pytest

from eval.algebra import (
    AlgebraError, ParamDist,
    canonicalize, distance, judge, parse_spec, spec_to_dict, answer_to_dict,
)


def sp(d):
    return parse_spec(d)


BOOL = {"kind": "dist", "domain": "bool"}
FIN = {"kind": "dist", "domain": "finite"}
REAL = {"kind": "dist", "domain": "real"}
VREAL = {"kind": "value", "domain": "real"}
VFIN = {"kind": "value", "domain": "finite"}


def cloud(spec_d, mu, n, seed):
    rng = random.Random(seed)
    return canonicalize([rng.gauss(mu, 1) for _ in range(n)], sp(spec_d))


# ---------------------------------------------------------------------------
# 1. judge() classifies candidate vs GT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cand, gt, spec_d, status", [
    # exact distribution: matching answer passes, including the pyro 0.0/1.0 form
    ({"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]},
     {"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]}, BOOL, "pass"),
    ({"kind": "dist_enum", "support": [1.0, 0.0], "probs": [0.7, 0.3]},
     {"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]}, BOOL, "pass"),
    # wrong distribution fails
    ({"kind": "dist_enum", "support": [True, False], "probs": [0.1, 0.9]},
     {"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]}, BOOL, "fail"),
    # garbage / non-finite are malformed
    ("not_a_dist", {"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]}, BOOL, "malformed"),
    ({"kind": "dist_enum", "support": [True, False], "probs": [float("nan"), 0.3]},
     {"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]}, BOOL, "malformed"),
])
def test_judge_classifies(cand, gt, spec_d, status):
    spec = sp(spec_d)
    g = canonicalize(gt, spec)
    assert judge(cand, [g, g], spec)["status"] == status


def test_judge_ill_posed_when_gt_scatters():
    # GT runs land in wildly different places → problem can't discriminate
    gts = [cloud(REAL, 0, 300, 1), cloud(REAL, 50, 300, 2)]
    assert judge([random.Random(3).gauss(0, 1) for _ in range(300)], gts, sp(REAL))["status"] == "ill_posed"
    # a "deterministic" value that varies across GT seeds is also ill-posed
    from eval.algebra import Exact
    assert judge(1.0, [Exact(1.0), Exact(2.0)], sp(VREAL))["status"] == "ill_posed"


# ---------------------------------------------------------------------------
# 2. Representation invariance — equal answers compare equal across wire forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, spec_d", [
    # native enum == mapping form == legacy __kind form
    ({"kind": "dist_enum", "support": ["a", "b"], "probs": [0.25, 0.75]}, {"a": 0.25, "b": 0.75}, FIN),
    ({"kind": "dist_enum", "support": ["a", "b"], "probs": [0.25, 0.75]},
     {"__kind": "distribution", "support": ["a", "b"], "probs": [1.0, 3.0]}, FIN),
    # int label 1 == float label 1.0 (cross-PPL)
    ({"kind": "dist_enum", "support": [1, 2], "probs": [0.6, 0.4]},
     {"kind": "dist_enum", "support": [1.0, 2.0], "probs": [0.6, 0.4]}, FIN),
    # a matching cloud histograms to the same distribution as the enum
    (["a", "a", "b", "b"], {"kind": "dist_enum", "support": ["a", "b"], "probs": [0.5, 0.5]}, FIN),
    # parametric aliases: Gaussian repr == Normal native; gamma scale == gamma rate
    ({"__kind": "distribution_continuous", "repr": "Gaussian({ mu: 0, sigma: 1 })"},
     {"kind": "dist_param", "family": "Normal", "params": {"loc": 0, "scale": 1}}, REAL),
    ({"kind": "dist_param", "family": "gamma", "params": {"shape": 2.0, "scale": 2.0}},
     {"kind": "dist_param", "family": "gamma", "params": {"concentration": 2.0, "rate": 0.5}}, REAL),
])
def test_representation_invariant(a, b, spec_d):
    spec = sp(spec_d)
    assert distance(canonicalize(a, spec), canonicalize(b, spec), spec).value == pytest.approx(0.0)


def test_bools_never_merge_with_ints():
    """A bool answer must not be judged equal to an int 0/1 answer."""
    a = canonicalize({"kind": "dist_enum", "support": [True, False], "probs": [0.5, 0.5]}, sp(FIN))
    b = canonicalize({"kind": "dist_enum", "support": [1, 0], "probs": [0.5, 0.5]}, sp(FIN))
    assert distance(a, b, sp(FIN)).value > 0.0


# ---------------------------------------------------------------------------
# 3. Metric numbers are correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, spec_d, expected", [
    # TV on finite
    ({"kind": "dist_enum", "support": ["A", "B"], "probs": [0.5, 0.5]},
     {"kind": "dist_enum", "support": ["A"], "probs": [1.0]}, FIN, 0.5),
    # W1 between unit point masses on the real line
    ({"kind": "dist_enum", "support": [0.0], "probs": [1.0]},
     {"kind": "dist_enum", "support": [1.0], "probs": [1.0]}, REAL, 1.0),
    # abs-diff on a real value
    (1.0, 1.5, VREAL, 0.5),
    # mismatched finite values are infinitely far
    ("H", "T", VFIN, math.inf),
])
def test_distance_values(a, b, spec_d, expected):
    spec = sp(spec_d)
    assert distance(canonicalize(a, spec), canonicalize(b, spec), spec).value == pytest.approx(expected)


def test_w1_disjoint_float_supports_is_sane():
    """Two samples of the same continuous law are W1-close (the legacy TV~1 bug)."""
    assert distance(cloud(REAL, 0, 500, 10), cloud(REAL, 0, 500, 11), sp(REAL)).value < 0.2


# ---------------------------------------------------------------------------
# 4. Tolerance is measured, not authored
# ---------------------------------------------------------------------------

def test_measured_tolerance():
    # sampled GT defines a noise floor: a same-law candidate passes, a shifted one fails.
    # One RNG stream for GTs + candidates so the floor reflects genuine MC noise.
    rng = random.Random(7)
    gts = [canonicalize([rng.gauss(0, 1) for _ in range(400)], sp(REAL)) for _ in range(5)]
    assert judge([rng.gauss(0, 1) for _ in range(400)], gts, sp(REAL))["status"] == "pass"
    assert judge([rng.gauss(2, 1) for _ in range(400)], gts, sp(REAL))["status"] == "fail"
    # candidate self-noise widens tolerance: a Beta(2,5) cloud matches a ParamDist GT
    g = ParamDist("beta", (("a", 2.0), ("b", 5.0)))
    brng = random.Random(42)
    beta = [brng.betavariate(2, 5) for _ in range(4096)]
    assert judge(beta, [g, g], sp(REAL))["status"] == "pass"
    # estimated value compared within the GT spread
    est = {"kind": "value", "domain": "real", "estimated": True}
    from eval.algebra import Exact
    assert judge(10.05, [Exact(10.1), Exact(9.9), Exact(10.0)], sp(est))["status"] == "pass"


# ---------------------------------------------------------------------------
# 5. Spec schema: parse / round-trip / reject
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [
    {"kind": "record", "fields": {"rain": {"kind": "dist", "domain": "bool"},
                                  "n": {"kind": "value", "domain": "int"}}},
    {"kind": "dist", "domain": "finite", "labels": {"record": {"sneeze": "bool", "fever": "bool"}}},
    {"kind": "dist", "domain": "finite", "support": ["C", "A", "B"]},   # support order preserved
])
def test_spec_round_trips(d):
    assert spec_to_dict(parse_spec(d)) == d


@pytest.mark.parametrize("bad", [
    {"kind": "dist", "domain": "realvec"},                         # unsupported
    {"kind": "dist", "domain": "real", "estimated": True},          # estimated is value-only
    {"kind": "value", "domain": "real", "protocol": "draws"},       # protocol is dist-only
    {"kind": "dist", "domain": "real", "labels": {"record": {"x": "bool"}}},   # labels need finite
    {"kind": "dist", "domain": "bool", "support": [True, False]},   # support is finite-only
    {"kind": "dist", "domain": "finite", "support": ["H", "H"]},    # duplicate support label
])
def test_spec_rejects_malformed(bad):
    with pytest.raises(AlgebraError):
        parse_spec(bad)


# ---------------------------------------------------------------------------
# 5b. Malformed / out-of-contract answers are rejected
# ---------------------------------------------------------------------------

_LABELED = {"kind": "dist", "domain": "finite", "labels": {"record": {"sneeze": "bool", "fever": "bool"}}}
_SUPP = {"kind": "dist", "domain": "finite", "support": ["H", "T"]}

@pytest.mark.parametrize("raw, spec_d", [
    ([], REAL),                                                                     # empty cloud
    (2.5, {"kind": "value", "domain": "int"}),                                      # non-integral int
    ({"kind": "dist_enum", "support": ["H", "X"], "probs": [0.6, 0.4]}, _SUPP),      # label outside declared support
    ({"kind": "dist_enum", "support": [{"sneeze": True}], "probs": [1.0]}, _LABELED),  # missing declared label field
    ({"rain": {"kind": "dist_enum", "support": [True], "probs": [1.0]}},             # record missing a field
     {"kind": "record", "fields": {"rain": {"kind": "dist", "domain": "bool"},
                                   "n": {"kind": "value", "domain": "int"}}}),
])
def test_canonicalize_rejects_malformed(raw, spec_d):
    with pytest.raises(AlgebraError):
        canonicalize(raw, sp(spec_d))


# ---------------------------------------------------------------------------
# answer_to_dict is the inverse of canonicalize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec_d, raw", [
    (BOOL, {"kind": "dist_enum", "support": [True, False], "probs": [0.6, 0.4]}),
    (REAL, {"kind": "dist_param", "family": "gaussian", "params": {"mu": 0.0, "sigma": 1.0}}),
    (REAL, {"kind": "cloud", "samples": [0.1, 0.5, 0.9, 0.5]}),
    ({"kind": "record", "fields": {"p": {"kind": "value", "domain": "real"}}}, {"p": 1.5}),
])
def test_answer_to_dict_round_trips(spec_d, raw):
    spec = sp(spec_d)
    canon = canonicalize(raw, spec)
    assert canonicalize(answer_to_dict(canon), spec) == canon
