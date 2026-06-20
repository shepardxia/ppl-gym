"""Tests for eval/algebra.py — the answer comparator (contract in data/SCHEMA.md).

Lean and contract-focused: one function per promise the comparator makes, with
representative parametrized cases. Deliberately does NOT pin internal helpers,
exact error wording, or diagnostic-dict structure — those are implementation,
not contract, and locking them only over-constrains the code.
"""

import math
import random

import pytest

from eval.algebra import (
    AlgebraError, Cloud, EnumDist, Exact, ParamDist, Rec,
    agreement, canonicalize, distance, judge, noise_floor, parse_spec, self_noise,
    spec_to_dict, verdict, answer_to_dict,
)


def spec(d):
    return parse_spec(d)


DIST_BOOL = spec({"kind": "dist", "domain": "bool"})
DIST_FIN = spec({"kind": "dist", "domain": "finite"})
DIST_REAL = spec({"kind": "dist", "domain": "real"})
DIST_INT = spec({"kind": "dist", "domain": "int"})
VAL_REAL = spec({"kind": "value", "domain": "real"})
VAL_REAL_EST = spec({"kind": "value", "domain": "real", "estimated": True})
VAL_FIN = spec({"kind": "value", "domain": "finite"})
VAL_VEC = spec({"kind": "value", "domain": "realvec"})

_RAIN_SPEC = spec({"kind": "record", "fields": {
    "rain": {"kind": "dist", "domain": "bool"},
    "n": {"kind": "value", "domain": "int"}}})
_RAIN_GT = canonicalize(
    {"rain": {"kind": "dist_enum", "support": [True, False], "probs": [0.6, 0.4]}, "n": 3},
    _RAIN_SPEC)
DIST_LABELED = spec({"kind": "dist", "domain": "finite",
                     "labels": {"record": {"sneeze": "bool", "fever": "bool"}}})


def enum(support, probs, sp=DIST_FIN):
    return canonicalize({"kind": "dist_enum", "support": support, "probs": probs}, sp)


# ---------------------------------------------------------------------------
# Specs: parse / roundtrip / validation
# ---------------------------------------------------------------------------

def test_spec_roundtrip():
    d = {"kind": "record", "fields": {
        "rain": {"kind": "dist", "domain": "bool"},
        "n": {"kind": "value", "domain": "int"}}}
    assert spec_to_dict(spec(d)) == d


@pytest.mark.parametrize("bad", [
    {"kind": "dist", "domain": "realvec"},                       # dist/realvec unsupported
    {"kind": "dist", "domain": "floats"},                       # unknown domain
    {"kind": "record", "fields": {}},                           # empty record
    {"kind": "dist", "domain": "real", "estimated": True},       # estimated is value-only
    {"kind": "value", "domain": "real", "protocol": "draws"},    # protocol is dist-only
])
def test_spec_rejects_malformed(bad):
    with pytest.raises(AlgebraError):
        spec(bad)


# ---------------------------------------------------------------------------
# Canonicalization: dist / value / cloud / record forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    # native enum
    ({"kind": "dist_enum", "support": ["a", "b"], "probs": [0.25, 0.75]}, {"a": 0.25, "b": 0.75}),
    # live executor wire form (__kind: distribution)
    ({"__kind": "distribution", "support": ["a", "b"], "probs": [2.0, 6.0]}, {"a": 0.25, "b": 0.75}),
    # mapping form (plain {label: prob}), renormalized
    ({"a": 2.0, "b": 6.0}, {"a": 0.25, "b": 0.75}),
])
def test_canonicalize_enum_forms(raw, expected):
    c = canonicalize(raw, DIST_FIN)
    assert isinstance(c, EnumDist)
    assert dict(zip(c.support, c.probs)) == pytest.approx(expected)


def test_canonicalize_bool_coerces_pyro_floats():
    c = canonicalize({"kind": "dist_enum", "support": [0.0, 1.0], "probs": [0.3, 0.7]}, DIST_BOOL)
    assert set(c.support) == {False, True}


def test_canonicalize_dup_merge_and_none_drop_and_zero_mass():
    c = canonicalize({"kind": "dist_enum",
                      "support": [{"a": 1, "b": 2}, {"b": 2, "a": 1}, "x"],
                      "probs": [0.25, 0.25, None]}, DIST_FIN)
    assert len(c.support) == 1 and c.probs == (1.0,)
    with pytest.raises(AlgebraError):  # all-zero mass
        canonicalize({"kind": "dist_enum", "support": ["a"], "probs": [0.0]}, DIST_FIN)
    with pytest.raises(AlgebraError):  # support/probs length mismatch
        canonicalize({"kind": "dist_enum", "support": ["a"], "probs": [0.5, 0.5]}, DIST_FIN)


def test_canonicalize_cloud_and_value():
    assert canonicalize([1, 2.0, 3], DIST_REAL) == Cloud((1.0, 2.0, 3.0))
    assert canonicalize(2.0, spec({"kind": "value", "domain": "int"})) == Exact(2)
    with pytest.raises(AlgebraError):
        canonicalize(2.5, spec({"kind": "value", "domain": "int"}))   # non-integral float
    with pytest.raises(AlgebraError):
        canonicalize([], DIST_REAL)                                    # empty cloud


def test_canonicalize_record_and_missing_field():
    raw = {"rain": {"kind": "dist_enum", "support": [True], "probs": [1.0]}, "n": 3}
    c = canonicalize(raw, _RAIN_SPEC)
    assert isinstance(c, Rec) and isinstance(c.field_map()["rain"], EnumDist)
    with pytest.raises(AlgebraError):
        canonicalize({"rain": raw["rain"]}, _RAIN_SPEC)   # missing 'n'


def test_canonicalize_record_draws():
    r_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "dist", "domain": "real", "protocol": "draws"},
        "n": {"kind": "value", "domain": "int"}}})
    # draws accumulate into a cloud; a value field must agree across runs
    c = canonicalize([{"x": 1.0, "n": 5}, {"x": 1.5, "n": 5}], r_spec)
    assert c.field_map()["x"] == Cloud((1.0, 1.5)) and c.field_map()["n"] == Exact(5)
    with pytest.raises(AlgebraError):           # value field varies across runs
        canonicalize([{"x": 1.0, "n": 5}, {"x": 1.5, "n": 6}], r_spec)
    with pytest.raises(AlgebraError):           # a record expects an object, not a bare list
        canonicalize([{"rain": True}], _RAIN_SPEC)


@pytest.mark.parametrize("raw, sp", [
    ({"kind": "dist_enum", "support": [True, False], "probs": [float("nan"), 0.3]}, DIST_BOOL),
    (float("nan"), VAL_REAL),
    ([1.0, math.inf, 2.0], DIST_REAL),
])
def test_non_finite_rejected(raw, sp):
    with pytest.raises(AlgebraError):
        canonicalize(raw, sp)


# ---------------------------------------------------------------------------
# Canonicalization: parametric families (aliases, gamma)
# ---------------------------------------------------------------------------

def test_parametric_aliases_and_gamma_rate():
    # Normal/Gaussian + concentration1/0 → a/b aliasing all land on one canonical form
    beta = canonicalize({"kind": "dist_param", "family": "Beta",
                         "params": {"concentration1": 10, "concentration0": 10}}, DIST_REAL)
    assert beta == ParamDist("beta", (("a", 10.0), ("b", 10.0)))
    # Gamma scale↔rate: both reduce to the canonical rate parameterization
    g_scale = canonicalize({"kind": "dist_param", "family": "gamma",
                            "params": {"shape": 2.0, "scale": 2.0}}, DIST_REAL)
    g_rate = canonicalize({"kind": "dist_param", "family": "gamma",
                           "params": {"concentration": 2.0, "rate": 0.5}}, DIST_REAL)
    assert g_scale == g_rate == ParamDist("gamma", (("rate", 0.5), ("shape", 2.0)))
    with pytest.raises(AlgebraError):  # over-specified gamma is ambiguous
        canonicalize({"kind": "dist_param", "family": "gamma",
                      "params": {"shape": 2.0, "rate": 0.5, "scale": 3.0}}, DIST_REAL)


@pytest.mark.parametrize("dom", [DIST_BOOL, DIST_FIN])
def test_parametric_under_finite_domain_raises(dom):
    with pytest.raises(AlgebraError):
        canonicalize({"kind": "dist_param", "family": "beta", "params": {"a": 2, "b": 5}}, dom)


# ---------------------------------------------------------------------------
# Distances: tv / w1 / value / record
# ---------------------------------------------------------------------------

def test_tv_distance():
    a = enum(["A", "B"], [0.5, 0.5])
    assert distance(a, enum(["A"], [1.0]), DIST_FIN).value == pytest.approx(0.5)
    assert distance(a, a, DIST_FIN).value == 0.0
    # a cloud histograms to the same distribution as the matching enum
    assert distance(canonicalize(["A", "A", "B", "B"], DIST_FIN), a, DIST_FIN).value == pytest.approx(0.0)


def test_w1_distance():
    d = distance(enum([0.0], [1.0], DIST_REAL), enum([1.0], [1.0], DIST_REAL), DIST_REAL)
    assert d.metric == "w1" and d.value == pytest.approx(1.0)
    # disjoint float supports → small W1 (the legacy TV~1 failure), and int domain works
    rng = random.Random(0)
    a = canonicalize([rng.gauss(0, 1) for _ in range(500)], DIST_REAL)
    b = canonicalize([rng.gauss(0, 1) for _ in range(500)], DIST_REAL)
    assert distance(a, b, DIST_REAL).value < 0.2
    assert distance(enum([0, 1], [0.5, 0.5], DIST_INT), enum([0, 1], [0.5, 0.5], DIST_INT), DIST_INT).value == 0.0


def test_w1_param_vs_cloud():
    g = canonicalize({"kind": "dist_param", "family": "gaussian",
                      "params": {"mu": 0, "sigma": 1}}, DIST_REAL)
    rng = random.Random(1)
    near = canonicalize([rng.gauss(0, 1) for _ in range(2000)], DIST_REAL)
    far = canonicalize([rng.gauss(5, 1) for _ in range(2000)], DIST_REAL)
    assert distance(g, near, DIST_REAL).value < 0.15
    assert distance(g, far, DIST_REAL).value == pytest.approx(5.0, abs=0.3)


def test_value_distances():
    assert distance(Exact(1.0), Exact(1.5), VAL_REAL).value == pytest.approx(0.5)
    assert distance(Exact("H"), Exact("H"), VAL_FIN).value == 0.0
    assert distance(Exact("H"), Exact("T"), VAL_FIN).value == math.inf
    assert distance(Exact([1.0, 2.0]), Exact([1.0, 2.5]), VAL_VEC).value == pytest.approx(0.5)
    assert distance(Exact([1.0]), Exact([1.0, 2.0]), VAL_VEC).value == math.inf  # length mismatch


def test_distance_incompatible_or_missing_field_raises():
    with pytest.raises(AlgebraError):
        distance(Exact(1.0), enum(["a"], [1.0]), DIST_FIN)   # value vs dist
    xy_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "value", "domain": "real"}, "y": {"kind": "value", "domain": "real"}}})
    a = Rec(fields=(("x", Exact(1.0)), ("y", Exact(2.0))))
    b = Rec(fields=(("x", Exact(1.0)),))   # missing 'y'
    with pytest.raises(AlgebraError):
        distance(a, b, xy_spec)


def test_label_normalization_cross_ppl():
    # int 1 and float 1.0 are the same finite label; bools never merge with 0/1
    assert distance(enum([1, 2], [0.6, 0.4]), enum([1.0, 2.0], [0.6, 0.4]), DIST_FIN).value == 0.0
    assert distance(Exact(1), Exact(1.0), VAL_FIN).value == 0.0
    assert distance(enum([True, False], [0.5, 0.5]), enum([1, 0], [0.5, 0.5]), DIST_FIN).value > 0.0


# ---------------------------------------------------------------------------
# Verdicts / judge: measured tolerance, ill-posedness, record recursion
# ---------------------------------------------------------------------------

def test_verdict_exact_dist():
    gt = enum([True, False], [0.75, 0.25], DIST_BOOL)
    good = enum([1.0, 0.0], [0.75, 0.25], DIST_BOOL)   # pyro-flavored same answer
    assert verdict(good, [gt, gt], DIST_BOOL)["passed"]
    assert not verdict(enum([True, False], [0.5, 0.5], DIST_BOOL), [gt, gt], DIST_BOOL)["passed"]


def test_verdict_sampled_floor():
    rng = random.Random(7)
    gts = [canonicalize([rng.gauss(0, 1) for _ in range(400)], DIST_REAL) for _ in range(5)]
    same = canonicalize([rng.gauss(0, 1) for _ in range(400)], DIST_REAL)
    v = verdict(same, gts, DIST_REAL)
    assert v["passed"] and v["floor"] > 0
    assert not verdict(canonicalize([rng.gauss(2, 1) for _ in range(400)], DIST_REAL), gts, DIST_REAL)["passed"]


@pytest.mark.parametrize("cand, gts, sp", [
    ("sampled", None, DIST_REAL),                                   # GT runs scatter → ill-posed
    (Exact(1.0), [Exact(1.0), Exact(2.0)], VAL_REAL),              # deterministic value varies
])
def test_verdict_ill_posed(cand, gts, sp):
    if gts is None:
        rng = random.Random(3)
        gts = [canonicalize([rng.gauss(0, 1) for _ in range(300)], DIST_REAL),
               canonicalize([rng.gauss(50, 1) for _ in range(300)], DIST_REAL)]
        cand = gts[0]
    v = verdict(cand, gts, sp)
    assert v["ill_posed"] and not v["passed"]


def test_verdict_estimated_value():
    gts = [Exact(10.1), Exact(9.9), Exact(10.0)]
    assert verdict(Exact(10.05), gts, VAL_REAL_EST)["passed"]
    assert not verdict(Exact(12.0), gts, VAL_REAL_EST)["passed"]


def test_verdict_record_recurses():
    bad = canonicalize({"rain": {"kind": "dist_enum", "support": [True, False],
                                 "probs": [0.4, 0.6]}, "n": 3}, _RAIN_SPEC)
    assert verdict(_RAIN_GT, [_RAIN_GT, _RAIN_GT], _RAIN_SPEC)["passed"]
    v = verdict(bad, [_RAIN_GT, _RAIN_GT], _RAIN_SPEC)
    assert not v["passed"] and v["fields"]["n"]["passed"]


def test_verdict_k1_raises():
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    with pytest.raises(AlgebraError):
        verdict(gt, [gt], DIST_BOOL)


def test_verdict_cloud_vs_param_uses_self_noise():
    """Tolerance includes candidate self-noise: a Beta(2,5) cloud vs a ParamDist GT passes."""
    gt = ParamDist("beta", (("a", 2.0), ("b", 5.0)))
    rng = random.Random(42)
    cand = Cloud(tuple(rng.betavariate(2, 5) for _ in range(4096)))
    assert verdict(cand, [gt, gt], DIST_REAL)["passed"]


@pytest.mark.parametrize("raw, status", [
    ({"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]}, "pass"),
    ({"kind": "dist_enum", "support": [True, False], "probs": [0.1, 0.9]}, "fail"),
    ("not_a_dist", "malformed"),
    ({"kind": "dist_enum", "support": [True, False], "probs": [float("nan"), 0.3]}, "malformed"),
])
def test_judge_statuses(raw, status):
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    assert judge(raw, [gt, gt], DIST_BOOL)["status"] == status


# ---------------------------------------------------------------------------
# noise_floor / self_noise / agreement
# ---------------------------------------------------------------------------

def test_noise_floor():
    rng = random.Random(99)
    gts = [canonicalize([rng.gauss(0, 1) for _ in range(200)], DIST_REAL) for _ in range(3)]
    assert noise_floor(gts, DIST_REAL) > 0.0
    assert noise_floor([_RAIN_GT, _RAIN_GT], _RAIN_SPEC) == 0.0   # identical → 0, recurses


@pytest.mark.parametrize("cand, sp, zero", [
    (Exact(1.0), VAL_REAL, True),
    (EnumDist(("H", "T"), (0.6, 0.4)), DIST_FIN, True),
    (Cloud(tuple(random.Random(5).gauss(0, 1) for _ in range(400))), DIST_REAL, False),
])
def test_self_noise(cand, sp, zero):
    sn = self_noise(cand, sp)
    assert (sn == 0.0) if zero else (sn >= 0.0)


@pytest.mark.parametrize("mu_b, agree", [(0, True), (10, False)])
def test_agreement_clouds(mu_b, agree):
    rng = random.Random(99 if agree else 42)
    a = Cloud(tuple(rng.gauss(0, 1) for _ in range(600)))
    b = Cloud(tuple(rng.gauss(mu_b, 1) for _ in range(600)))
    assert agreement(a, b, DIST_REAL)["agree"] is agree


def test_agreement_scalar_and_record():
    assert agreement(Exact("H"), Exact("H"), VAL_FIN)["agree"] is True
    assert agreement(Exact("H"), Exact("T"), VAL_FIN)["agree"] is False
    r_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "value", "domain": "real"}, "y": {"kind": "value", "domain": "finite"}}})
    same = Rec(fields=(("x", Exact(1.0)), ("y", Exact("H"))))
    diff = Rec(fields=(("x", Exact(1.0)), ("y", Exact("T"))))
    assert agreement(same, same, r_spec)["agree"] is True
    assert agreement(same, diff, r_spec)["agree"] is False


# ---------------------------------------------------------------------------
# Structured finite labels + declared support
# ---------------------------------------------------------------------------

def test_labels_roundtrip_and_rejections():
    d = {"kind": "dist", "domain": "finite", "labels": {"record": {"sneeze": "bool", "fever": "bool"}}}
    s = parse_spec(d)
    assert s.labels == (("fever", "bool"), ("sneeze", "bool")) and spec_to_dict(s) == d
    for bad in [{"kind": "value", "domain": "finite", "labels": {"record": {"x": "bool"}}},
                {"kind": "dist", "domain": "real", "labels": {"record": {"x": "bool"}}},
                {"kind": "dist", "domain": "finite", "labels": {"record": {"x": "finite"}}}]:
        with pytest.raises(AlgebraError):
            spec(bad)


def test_labels_validation_and_coercion():
    # pyro-style 0.0/1.0 field values coerce to bool; equal to the bool-typed answer
    fl = canonicalize({"kind": "dist_enum",
                       "support": [{"sneeze": 1.0, "fever": 0.0}, {"sneeze": 0.0, "fever": 0.0}],
                       "probs": [0.44, 0.56]}, DIST_LABELED)
    bl = canonicalize({"kind": "dist_enum",
                       "support": [{"sneeze": True, "fever": False}, {"sneeze": False, "fever": False}],
                       "probs": [0.44, 0.56]}, DIST_LABELED)
    assert all(isinstance(d["sneeze"], bool) for d in fl.support)
    assert distance(fl, bl, DIST_LABELED).value == pytest.approx(0.0)
    for raw in [{"kind": "dist_enum", "support": [{"sneeze": True}], "probs": [1.0]},          # missing field
                {"kind": "dist_enum", "support": [{"sneeze": True, "fever": False, "x": 1}], "probs": [1.0]}]:  # undeclared
        with pytest.raises(AlgebraError):
            canonicalize(raw, DIST_LABELED)


def test_support_parse_roundtrip_and_rejections():
    for d in [{"kind": "dist", "domain": "finite", "support": ["C", "A", "B"]},
              {"kind": "value", "domain": "finite", "support": [1, 2, 3]}]:
        assert spec_to_dict(parse_spec(d)) == d          # order preserved, roundtrips
    for bad in [{"kind": "dist", "domain": "finite", "support": ["H", "H"]},   # duplicate
                {"kind": "dist", "domain": "bool", "support": [True, False]}]:  # support is finite-only
        with pytest.raises(AlgebraError):
            parse_spec(bad)


def test_support_membership_enforced():
    s = parse_spec({"kind": "dist", "domain": "finite", "support": ["H", "T"]})
    assert isinstance(canonicalize({"kind": "dist_enum", "support": ["H", "T"], "probs": [0.6, 0.4]}, s), EnumDist)
    for raw in [{"kind": "dist_enum", "support": ["H", "X"], "probs": [0.6, 0.4]},   # enum out-of-space
                {"H": 0.6, "X": 0.4}]:                                                # mapping out-of-space
        with pytest.raises(AlgebraError):
            canonicalize(raw, s)
    # int/float labels normalize to the same declared key (cross-PPL)
    si = parse_spec({"kind": "dist", "domain": "finite", "support": [1, 2]})
    c1 = canonicalize({"kind": "dist_enum", "support": [1, 2], "probs": [0.5, 0.5]}, si)
    c2 = canonicalize({"kind": "dist_enum", "support": [1.0, 2.0], "probs": [0.5, 0.5]}, si)
    assert distance(c1, c2, si).value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# answer_to_dict round-trip (inverse of canonicalize)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec_d, raw", [
    ({"kind": "dist", "domain": "bool"},
     {"kind": "dist_enum", "support": [True, False], "probs": [0.6, 0.4]}),
    ({"kind": "dist", "domain": "real"},
     {"kind": "dist_param", "family": "gaussian", "params": {"mu": 0.0, "sigma": 1.0}}),
    ({"kind": "dist", "domain": "real"}, {"kind": "cloud", "samples": [0.1, 0.5, 0.9, 0.5]}),
    ({"kind": "value", "domain": "real"}, 1.5),
    ({"kind": "record", "fields": {"p": {"kind": "value", "domain": "real"}}}, {"p": 1.5}),
])
def test_answer_to_dict_round_trips(spec_d, raw):
    sp = parse_spec(spec_d)
    canon = canonicalize(raw, sp)
    assert canonicalize(answer_to_dict(canon), sp) == canon
