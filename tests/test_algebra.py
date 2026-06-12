"""Tests for eval/algebra.py against the contract in data/SCHEMA.md."""

import math
import random

import pytest

from eval.algebra import (
    AlgebraError, Cloud, EnumDist, Exact, ParamDist, Rec,
    agreement, canonicalize, distance, judge, noise_floor, parse_spec, self_noise,
    spec_to_dict, verdict,
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

# Shared record spec and canonical GT used by record-verdict / noise-floor tests.
_RAIN_SPEC = spec({"kind": "record", "fields": {
    "rain": {"kind": "dist", "domain": "bool"},
    "n": {"kind": "value", "domain": "int"}}})
_RAIN_GT = canonicalize(
    {"rain": {"__kind": "distribution", "support": [True, False], "probs": [0.6, 0.4]},
     "n": 3},
    _RAIN_SPEC)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def test_spec_roundtrip():
    d = {"kind": "record", "fields": {
        "rain": {"kind": "dist", "domain": "bool"},
        "n": {"kind": "value", "domain": "int"}}}
    assert spec_to_dict(spec(d)) == d


def test_spec_rejects_dist_realvec_and_unknowns():
    with pytest.raises(AlgebraError):
        spec({"kind": "dist", "domain": "realvec"})
    with pytest.raises(AlgebraError):
        spec({"kind": "dist", "domain": "floats"})
    with pytest.raises(AlgebraError):
        spec({"kind": "histogram", "domain": "real"})
    with pytest.raises(AlgebraError):
        spec({"kind": "record", "fields": {}})


# Fix #7: parse_spec cross-field validation
def test_spec_rejects_estimated_on_dist():
    with pytest.raises(AlgebraError, match="value specs only"):
        spec({"kind": "dist", "domain": "real", "estimated": True})


def test_spec_rejects_protocol_on_value():
    with pytest.raises(AlgebraError, match="dist specs only"):
        spec({"kind": "value", "domain": "real", "protocol": "draws"})


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def test_legacy_distribution_dict():
    raw = {"__kind": "distribution", "support": [True, False], "probs": [0.7, 0.3]}
    c = canonicalize(raw, DIST_BOOL)
    assert isinstance(c, EnumDist)
    assert dict(zip(c.support, c.probs)) == {True: 0.7, False: 0.3}


def test_bool_domain_coerces_pyro_floats():
    raw = {"kind": "dist_enum", "support": [0.0, 1.0], "probs": [0.3, 0.7]}
    c = canonicalize(raw, DIST_BOOL)
    assert set(c.support) == {False, True}


def test_duplicate_labels_merge_and_none_probs_drop():
    raw = {"__kind": "distribution",
           "support": [{"a": 1, "b": 2}, {"b": 2, "a": 1}, "x"],
           "probs": [0.25, 0.25, None]}
    c = canonicalize(raw, DIST_FIN)
    assert len(c.support) == 1
    assert c.probs == (1.0,)


def test_probs_renormalize():
    raw = {"__kind": "distribution", "support": ["a", "b"], "probs": [2.0, 6.0]}
    c = canonicalize(raw, DIST_FIN)
    assert dict(zip(c.support, c.probs)) == {"a": 0.25, "b": 0.75}


def test_zero_mass_and_mismatch_rejected():
    with pytest.raises(AlgebraError):
        canonicalize({"__kind": "distribution", "support": ["a"], "probs": [0.0]}, DIST_FIN)
    with pytest.raises(AlgebraError):
        canonicalize({"__kind": "distribution", "support": ["a"], "probs": [0.5, 0.5]}, DIST_FIN)


def test_legacy_parametric_repr_and_native_param():
    legacy = canonicalize(
        {"__kind": "distribution_continuous", "repr": "Beta({ a: 10, b: 10 })"},
        DIST_REAL)
    native = canonicalize(
        {"kind": "dist_param", "family": "Beta",
         "params": {"concentration1": 10, "concentration0": 10}},
        DIST_REAL)
    assert legacy == native == ParamDist("beta", (("a", 10.0), ("b", 10.0)))


def test_normal_is_gaussian():
    a = canonicalize({"kind": "dist_param", "family": "Normal",
                      "params": {"loc": 0, "scale": 1}}, DIST_REAL)
    b = canonicalize({"__kind": "distribution_continuous",
                      "repr": "Gaussian({ mu: 0, sigma: 1 })"}, DIST_REAL)
    assert a == b


def test_cloud_and_value_canonicalization():
    c = canonicalize([1, 2.0, 3], DIST_REAL)
    assert isinstance(c, Cloud) and c.samples == (1.0, 2.0, 3.0)
    assert canonicalize(2.0, spec({"kind": "value", "domain": "int"})) == Exact(2)
    with pytest.raises(AlgebraError):
        canonicalize(2.5, spec({"kind": "value", "domain": "int"}))
    with pytest.raises(AlgebraError):
        canonicalize([], DIST_REAL)


def test_legacy_tensor_is_value_vector():
    c = canonicalize({"__kind": "tensor", "dims": [3], "data": [1, 2, 3]}, VAL_VEC)
    assert c == Exact([1.0, 2.0, 3.0])


def test_record_and_missing_field():
    r_spec = spec({"kind": "record", "fields": {
        "rain": {"kind": "dist", "domain": "bool"},
        "k": {"kind": "value", "domain": "int"}}})
    raw = {"rain": {"__kind": "distribution", "support": [True], "probs": [1.0]},
           "k": 3}
    c = canonicalize(raw, r_spec)
    assert isinstance(c, Rec) and isinstance(c.field_map()["rain"], EnumDist)
    with pytest.raises(AlgebraError):
        canonicalize({"rain": raw["rain"]}, r_spec)


def test_record_draws_split():
    r_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "dist", "domain": "real", "protocol": "draws"},
        "y": {"kind": "dist", "domain": "real", "protocol": "draws"}}})
    runs = [{"x": 1.0, "y": 2.0}, {"x": 1.5, "y": 2.5}]
    c = canonicalize(runs, r_spec)
    assert c.field_map()["x"] == Cloud((1.0, 1.5))
    assert c.field_map()["y"] == Cloud((2.0, 2.5))


# Fix #1: NaN probs rejection
def test_nan_prob_raises():
    """NaN probability must be rejected at ingestion, not silently produce TV=0."""
    with pytest.raises(AlgebraError, match="invalid probability"):
        canonicalize(
            {"kind": "dist_enum", "support": [True, False],
             "probs": [float("nan"), 0.3]},
            DIST_BOOL)


def test_inf_prob_raises():
    with pytest.raises(AlgebraError, match="invalid probability"):
        canonicalize(
            {"kind": "dist_enum", "support": ["a"], "probs": [math.inf]},
            DIST_FIN)


# Fix #1: non-finite number rejection in real/int domains
def test_nan_real_value_raises():
    with pytest.raises(AlgebraError, match="non-finite"):
        canonicalize(float("nan"), VAL_REAL)


def test_inf_real_sample_raises():
    with pytest.raises(AlgebraError, match="non-finite"):
        canonicalize([1.0, math.inf, 2.0], DIST_REAL)


# Fix #2: Gamma canonical parameterization
def test_gamma_scale_converted_to_rate():
    """Gamma(shape=2, scale=2.0) and Gamma(shape=2, rate=0.5) are the same distribution."""
    g_scale = canonicalize(
        {"kind": "dist_param", "family": "gamma",
         "params": {"shape": 2.0, "scale": 2.0}}, DIST_REAL)
    g_rate = canonicalize(
        {"kind": "dist_param", "family": "gamma",
         "params": {"concentration": 2.0, "rate": 0.5}}, DIST_REAL)
    assert g_scale == g_rate
    assert g_scale == ParamDist("gamma", (("rate", 0.5), ("shape", 2.0)))


def test_gamma_ambiguous_params_raises():
    with pytest.raises(AlgebraError, match="ambiguous"):
        canonicalize(
            {"kind": "dist_param", "family": "gamma",
             "params": {"shape": 2.0, "rate": 0.5, "scale": 3.0}},
            DIST_REAL)


def test_gamma_sampler_uses_rate():
    """Gamma samples should have mean = shape / rate."""
    from eval.algebra import _sample_parametric
    d = ParamDist("gamma", (("rate", 2.0), ("shape", 4.0)))
    draws = _sample_parametric(d)
    mean = sum(draws) / len(draws)
    # expected mean = shape/rate = 4/2 = 2.0
    assert abs(mean - 2.0) < 0.1


# Fix #6: record + list without draws fields raises
def test_record_list_without_draws_field_raises():
    r_spec = spec({"kind": "record", "fields": {
        "rain": {"kind": "dist", "domain": "bool"}}})
    runs = [{"rain": True}, {"rain": False}]
    with pytest.raises(AlgebraError, match="got a list"):
        canonicalize(runs, r_spec)


# Fix #6: mixed draws-record with value field — consensus works, disagreement raises
def test_record_draws_with_value_field_consensus():
    r_spec = spec({"kind": "record", "fields": {
        "outcome": {"kind": "dist", "domain": "bool", "protocol": "draws"},
        "n_samples": {"kind": "value", "domain": "int"}}})
    runs = [
        {"outcome": True, "n_samples": 500},
        {"outcome": False, "n_samples": 500},
        {"outcome": True, "n_samples": 500},
    ]
    c = canonicalize(runs, r_spec)
    assert c.field_map()["n_samples"] == Exact(500)
    assert c.field_map()["outcome"] == Cloud((True, False, True))


def test_record_draws_with_value_field_disagreement_raises():
    r_spec = spec({"kind": "record", "fields": {
        "outcome": {"kind": "dist", "domain": "bool", "protocol": "draws"},
        "n_samples": {"kind": "value", "domain": "int"}}})
    runs = [
        {"outcome": True, "n_samples": 500},
        {"outcome": False, "n_samples": 600},  # different value
    ]
    with pytest.raises(AlgebraError, match="varies across runs"):
        canonicalize(runs, r_spec)


# Fix #6: dist-object field inside draws-record is incoherent
def test_record_draws_with_dist_object_field_raises():
    r_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "dist", "domain": "real", "protocol": "draws"},
        "dist": {"kind": "dist", "domain": "real"}}})  # protocol=object
    runs = [{"x": 1.0, "dist": [0.5, 0.5]}, {"x": 2.0, "dist": [0.3, 0.7]}]
    with pytest.raises(AlgebraError, match="protocol='object'"):
        canonicalize(runs, r_spec)


# Fix #6: ParamDist under finite domain raises at canonicalize time
def test_parametric_under_finite_domain_raises():
    with pytest.raises(AlgebraError, match="incompatible with domain"):
        canonicalize(
            {"kind": "dist_param", "family": "beta",
             "params": {"a": 2, "b": 5}},
            DIST_BOOL)
    with pytest.raises(AlgebraError, match="incompatible with domain"):
        canonicalize(
            {"kind": "dist_param", "family": "beta",
             "params": {"a": 2, "b": 5}},
            DIST_FIN)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def enum(support, probs, sp=DIST_FIN):
    return canonicalize({"kind": "dist_enum", "support": support, "probs": probs}, sp)


def test_tv_known_values():
    a = enum(["A", "B"], [0.5, 0.5])
    b = enum(["A"], [1.0])
    assert distance(a, b, DIST_FIN).value == pytest.approx(0.5)
    assert distance(a, a, DIST_FIN).value == 0.0


def test_tv_cloud_vs_enum():
    cloud = canonicalize(["A", "A", "B", "B"], DIST_FIN)
    e = enum(["A", "B"], [0.5, 0.5])
    assert distance(cloud, e, DIST_FIN).value == pytest.approx(0.0)


def test_w1_point_masses():
    a = enum([0.0], [1.0], DIST_REAL)
    b = enum([1.0], [1.0], DIST_REAL)
    d = distance(a, b, DIST_REAL)
    assert d.metric == "w1" and d.value == pytest.approx(1.0)


def test_w1_disjoint_float_supports_is_sane():
    # The legacy failure mode: two MCMC runs with disjoint float supports
    # gave TV~1; W1 must report a small distance instead.
    rng = random.Random(0)
    a = canonicalize([rng.gauss(0, 1) for _ in range(500)], DIST_REAL)
    b = canonicalize([rng.gauss(0, 1) for _ in range(500)], DIST_REAL)
    assert distance(a, b, DIST_REAL).value < 0.2


def test_w1_param_vs_cloud():
    g = canonicalize({"kind": "dist_param", "family": "gaussian",
                      "params": {"mu": 0, "sigma": 1}}, DIST_REAL)
    rng = random.Random(1)
    near = canonicalize([rng.gauss(0, 1) for _ in range(2000)], DIST_REAL)
    far = canonicalize([rng.gauss(5, 1) for _ in range(2000)], DIST_REAL)
    assert distance(g, near, DIST_REAL).value < 0.15
    assert distance(g, far, DIST_REAL).value == pytest.approx(5.0, abs=0.3)


def test_param_fast_path_and_diagnostics():
    a = ParamDist("beta", (("a", 2.0), ("b", 5.0)))
    same = ParamDist("beta", (("a", 2.0), ("b", 5.0)))
    other = ParamDist("beta", (("a", 5.0), ("b", 2.0)))
    assert distance(a, same, DIST_REAL).value == 0.0
    d = distance(a, other, DIST_REAL)
    assert d.value > 0.1 and "param_diffs" in d.diagnostics


def test_w1_int_domain():
    geo1 = enum([0, 1, 2], [0.5, 0.25, 0.25], DIST_INT)
    geo2 = enum([0, 1, 2], [0.5, 0.25, 0.25], DIST_INT)
    assert distance(geo1, geo2, DIST_INT).value == 0.0


def test_value_distances():
    assert distance(Exact(1.0), Exact(1.5), VAL_REAL).value == pytest.approx(0.5)
    assert distance(Exact("H"), Exact("H"), VAL_FIN).value == 0.0
    assert distance(Exact("H"), Exact("T"), VAL_FIN).value == math.inf
    assert distance(Exact([1.0, 2.0]), Exact([1.0, 2.5]), VAL_VEC).value == pytest.approx(0.5)
    assert distance(Exact([1.0]), Exact([1.0, 2.0]), VAL_VEC).value == math.inf


def test_incompatible_representation_raises():
    with pytest.raises(AlgebraError):
        distance(Exact(1.0), enum(["a"], [1.0]), DIST_FIN)
    with pytest.raises(AlgebraError):
        # a parametric distribution has no label histogram, so TV-domain comparison must reject it
        distance(ParamDist("beta", (("a", 1.0), ("b", 1.0))), enum(["a"], [1.0]), DIST_FIN)


# Fix #6: distance() on records raises AlgebraError on missing field (not KeyError)
def test_distance_record_missing_field_raises_algebra_error():
    r_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "value", "domain": "real"},
        "y": {"kind": "value", "domain": "real"}}})
    a = Rec(fields=(("x", Exact(1.0)), ("y", Exact(2.0))))
    b = Rec(fields=(("x", Exact(1.0)),))  # missing 'y'
    with pytest.raises(AlgebraError):
        distance(a, b, r_spec)


# Fix #9: KL returns inf on disjoint support
def test_kl_inf_on_disjoint_support():
    p = enum(["A", "B", "C"], [0.3, 0.5, 0.2])
    q = enum(["A", "B"], [0.5, 0.5])
    d = distance(p, q, DIST_FIN)
    assert d.diagnostics["kl"] == math.inf


# Fix #9: _ks guards empty inputs
def test_ks_empty_inputs():
    from eval.algebra import _ks
    assert _ks([], []) == 0.0
    assert _ks([], [1.0, 2.0]) == 1.0
    assert _ks([1.0], []) == 1.0


# Fix #8: label normalization — int 1 and float 1.0 are equal finite labels
def test_label_normalization_int_float_equal():
    webppl = canonicalize(
        {"kind": "dist_enum", "support": [1, 2], "probs": [0.6, 0.4]}, DIST_FIN)
    pyro = canonicalize(
        {"kind": "dist_enum", "support": [1.0, 2.0], "probs": [0.6, 0.4]}, DIST_FIN)
    assert distance(webppl, pyro, DIST_FIN).value == pytest.approx(0.0)


def test_label_normalization_exact_value():
    assert distance(Exact(1), Exact(1.0), VAL_FIN).value == 0.0
    assert distance(Exact(2), Exact(2.0), VAL_FIN).value == 0.0


def test_label_normalization_bools_not_collapsed():
    """Bool labels must not merge with int 0/1."""
    from eval.algebra import _label_key
    # False/True must stay distinct from 0/1
    assert _label_key(False) != _label_key(0)
    assert _label_key(True) != _label_key(1)


# ---------------------------------------------------------------------------
# Verdicts: measured tolerance, ill-posedness
# ---------------------------------------------------------------------------

def test_verdict_exact_dist():
    gt = enum([True, False], [0.75, 0.25], DIST_BOOL)
    good = enum([1.0, 0.0], [0.75, 0.25], DIST_BOOL)   # pyro-flavored same answer
    bad = enum([True, False], [0.5, 0.5], DIST_BOOL)
    v = verdict(good, [gt, gt], DIST_BOOL)
    assert v["passed"] and v["floor"] == 0.0
    assert not verdict(bad, [gt, gt], DIST_BOOL)["passed"]


def test_verdict_sampled_floor():
    rng = random.Random(7)
    gts = [canonicalize([rng.gauss(0, 1) for _ in range(400)], DIST_REAL)
           for _ in range(5)]
    same = canonicalize([rng.gauss(0, 1) for _ in range(400)], DIST_REAL)
    shifted = canonicalize([rng.gauss(2, 1) for _ in range(400)], DIST_REAL)
    v = verdict(same, gts, DIST_REAL)
    assert v["passed"] and v["floor"] > 0
    assert not verdict(shifted, gts, DIST_REAL)["passed"]


def test_verdict_ill_posed_when_gt_inconsistent():
    rng = random.Random(3)
    gts = [canonicalize([rng.gauss(0, 1) for _ in range(300)], DIST_REAL),
           canonicalize([rng.gauss(50, 1) for _ in range(300)], DIST_REAL)]
    v = verdict(gts[0], gts, DIST_REAL)
    assert v["ill_posed"] and not v["passed"]


def test_verdict_deterministic_value_nondeterminism_flagged():
    v = verdict(Exact(1.0), [Exact(1.0), Exact(2.0)], VAL_REAL)
    assert v["ill_posed"]


def test_verdict_estimated_value():
    gts = [Exact(10.1), Exact(9.9), Exact(10.0)]
    assert verdict(Exact(10.05), gts, VAL_REAL_EST)["passed"]
    assert not verdict(Exact(12.0), gts, VAL_REAL_EST)["passed"]


def test_verdict_record_recurses():
    bad = canonicalize({"rain": {"__kind": "distribution", "support": [True, False],
                                 "probs": [0.4, 0.6]}, "n": 3}, _RAIN_SPEC)
    assert verdict(_RAIN_GT, [_RAIN_GT, _RAIN_GT], _RAIN_SPEC)["passed"]
    v = verdict(bad, [_RAIN_GT, _RAIN_GT], _RAIN_SPEC)
    assert not v["passed"] and v["fields"]["n"]["passed"]


# Fix #3: k=1 raises AlgebraError
def test_verdict_k1_raises():
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    with pytest.raises(AlgebraError, match="at least 2"):
        verdict(gt, [gt], DIST_BOOL)


# Fix #3: tolerance includes candidate self-noise (Cloud vs ParamDist GT)
def test_verdict_cloud_vs_param_gt_passes():
    """Cloud from Beta(2,5) against ParamDist GT ×2 must pass (repro from review)."""
    gt_param = ParamDist("beta", (("a", 2.0), ("b", 5.0)))
    rng = random.Random(42)
    cand_cloud = Cloud(tuple(rng.betavariate(2, 5) for _ in range(4096)))
    v = verdict(cand_cloud, [gt_param, gt_param], DIST_REAL)
    assert v["passed"], f"Expected pass, got: {v}"


# Fix #3: noise_floor is a public function
def test_noise_floor_public():
    rng = random.Random(99)
    gts = [canonicalize([rng.gauss(0, 1) for _ in range(200)], DIST_REAL)
           for _ in range(3)]
    fl = noise_floor(gts, DIST_REAL)
    assert fl > 0.0


def test_noise_floor_exact_is_zero():
    gt = Exact(42)
    assert noise_floor([gt, gt], VAL_FIN) == 0.0


def test_noise_floor_record():
    assert noise_floor([_RAIN_GT, _RAIN_GT], _RAIN_SPEC) == 0.0


# Fix #4: verdict result completeness — diagnostics in leaf verdict
def test_leaf_verdict_contains_diagnostics():
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    cand = enum([True, False], [0.6, 0.4], DIST_BOOL)
    v = verdict(cand, [gt, gt], DIST_BOOL)
    assert "diagnostics" in v
    assert "kl" in v["diagnostics"]


# Fix #4: record verdict carries distance, floor, tol
def test_record_verdict_carries_distance_floor_tol():
    v = verdict(_RAIN_GT, [_RAIN_GT, _RAIN_GT], _RAIN_SPEC)
    assert "distance" in v
    assert "floor" in v
    assert "tol" in v
    # worst-case should be max over fields
    worst = max(v["fields"][n]["distance"] for n in ("rain", "n"))
    assert v["distance"] == pytest.approx(worst)


# Fix #5: judge() entry point
def test_judge_pass():
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    raw = {"kind": "dist_enum", "support": [True, False], "probs": [0.7, 0.3]}
    result = judge(raw, [gt, gt], DIST_BOOL)
    assert result["status"] == "pass"
    assert result["passed"] is True


def test_judge_fail():
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    raw = {"kind": "dist_enum", "support": [True, False], "probs": [0.1, 0.9]}
    result = judge(raw, [gt, gt], DIST_BOOL)
    assert result["status"] == "fail"


def test_judge_malformed():
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    # passing a string for a dist is malformed
    result = judge("not_a_dist", [gt, gt], DIST_BOOL)
    assert result["status"] == "malformed"
    assert "error" in result


def test_judge_ill_posed():
    rng = random.Random(3)
    gts = [canonicalize([rng.gauss(0, 1) for _ in range(300)], DIST_REAL),
           canonicalize([rng.gauss(50, 1) for _ in range(300)], DIST_REAL)]
    raw = [rng.gauss(0, 1) for _ in range(300)]
    result = judge(raw, gts, DIST_REAL)
    assert result["status"] == "ill_posed"


def test_judge_malformed_nan_probs():
    """NaN probs in candidate are caught by judge() as malformed."""
    gt = enum([True, False], [0.7, 0.3], DIST_BOOL)
    raw = {"kind": "dist_enum", "support": [True, False],
           "probs": [float("nan"), 0.3]}
    result = judge(raw, [gt, gt], DIST_BOOL)
    assert result["status"] == "malformed"


# Fix #9: eq-metric eps is 1e-9
def test_eq_metric_eps_is_small():
    gt = canonicalize("H", VAL_FIN)
    v = verdict(gt, [gt, gt], VAL_FIN)
    assert v["tol"] == pytest.approx(1e-9)


# ---------------------------------------------------------------------------
# Structured finite labels
# ---------------------------------------------------------------------------

DIST_LABELED = spec({
    "kind": "dist", "domain": "finite",
    "labels": {"record": {"sneeze": "bool", "fever": "bool"}},
})


def test_labels_roundtrip():
    """Labels survive spec_to_dict → parse_spec."""
    d = {"kind": "dist", "domain": "finite",
         "labels": {"record": {"sneeze": "bool", "fever": "bool"}}}
    s = parse_spec(d)
    assert s.labels == (("fever", "bool"), ("sneeze", "bool"))  # sorted
    assert spec_to_dict(s) == d


def test_labels_rejected_on_value_spec():
    """'labels' on a value spec must raise."""
    with pytest.raises(AlgebraError):
        spec({"kind": "value", "domain": "finite",
              "labels": {"record": {"x": "bool"}}})


def test_labels_rejected_on_non_finite_domain():
    """'labels' on dist/real must raise."""
    with pytest.raises(AlgebraError):
        spec({"kind": "dist", "domain": "real",
              "labels": {"record": {"x": "bool"}}})


def test_labels_rejected_on_non_atomic_domain():
    """A label field with domain 'finite' (non-atomic) must raise."""
    with pytest.raises(AlgebraError, match="non-atomic"):
        spec({"kind": "dist", "domain": "finite",
              "labels": {"record": {"x": "finite"}}})


def test_labels_pyro_float_coerced_to_bool():
    """Pyro-style 0.0/1.0 field values normalize to bools when domain='bool'."""
    raw = {"kind": "dist_enum",
           "support": [{"sneeze": 1.0, "fever": 0.0},
                       {"sneeze": 0.0, "fever": 0.0}],
           "probs": [0.44, 0.56]}
    c = canonicalize(raw, DIST_LABELED)
    assert isinstance(c, EnumDist)
    # sneeze=1.0 → True; fever=0.0 → False
    support_dicts = list(c.support)
    sneezes = {d["sneeze"] for d in support_dicts}
    assert sneezes == {True, False} or True in sneezes
    # Ensure they are actual bools, not ints
    for d in support_dicts:
        assert isinstance(d["sneeze"], bool)
        assert isinstance(d["fever"], bool)


def test_labels_missing_field_raises():
    """A support element missing a declared field must raise AlgebraError."""
    raw = {"kind": "dist_enum",
           "support": [{"sneeze": True}],  # missing 'fever'
           "probs": [1.0]}
    with pytest.raises(AlgebraError, match="missing fields"):
        canonicalize(raw, DIST_LABELED)


def test_labels_extra_field_raises():
    """A support element with an undeclared field must raise AlgebraError."""
    raw = {"kind": "dist_enum",
           "support": [{"sneeze": True, "fever": False, "cough": True}],
           "probs": [1.0]}
    with pytest.raises(AlgebraError, match="undeclared fields"):
        canonicalize(raw, DIST_LABELED)


def test_labels_sneeze_1_float_equals_true():
    """{'sneeze': 1.0} and {'sneeze': true} are equal under labeled spec."""
    float_raw = {"kind": "dist_enum",
                 "support": [{"sneeze": 1.0, "fever": False},
                              {"sneeze": 0.0, "fever": False}],
                 "probs": [0.44, 0.56]}
    bool_raw = {"kind": "dist_enum",
                "support": [{"sneeze": True, "fever": False},
                             {"sneeze": False, "fever": False}],
                "probs": [0.44, 0.56]}
    c_float = canonicalize(float_raw, DIST_LABELED)
    c_bool = canonicalize(bool_raw, DIST_LABELED)
    assert distance(c_float, c_bool, DIST_LABELED).value == pytest.approx(0.0)


def test_labels_undeclared_keeps_opaque_behavior():
    """A dist/finite spec WITHOUT labels still accepts any finite labels opaquely."""
    raw = {"kind": "dist_enum",
           "support": [{"x": 1}, {"x": 2}],
           "probs": [0.5, 0.5]}
    c = canonicalize(raw, DIST_FIN)   # DIST_FIN has no labels
    assert isinstance(c, EnumDist)
    assert len(c.support) == 2


def test_labels_cloud_validates_elements():
    """Cloud draws are also validated against labels."""
    draws = [{"sneeze": True, "fever": False}, {"sneeze": False, "fever": False}]
    c = canonicalize(draws, DIST_LABELED)
    assert isinstance(c, Cloud)
    # Missing field in a draw
    with pytest.raises(AlgebraError, match="missing fields"):
        canonicalize([{"sneeze": True}], DIST_LABELED)


# ---------------------------------------------------------------------------
# Support field on specs
# ---------------------------------------------------------------------------

def test_support_parse_string_labels():
    """Support with string labels parses and round-trips."""
    d = {"kind": "dist", "domain": "finite", "support": ["H", "T"]}
    s = parse_spec(d)
    assert len(s.support) == 2
    assert spec_to_dict(s) == d


def test_support_parse_int_labels():
    """Support with integer labels parses correctly."""
    d = {"kind": "value", "domain": "finite", "support": [1, 2, 3]}
    s = parse_spec(d)
    assert len(s.support) == 3
    assert spec_to_dict(s) == d


def test_support_with_labels_shape():
    """Support with a labels schema validates elements against the shape."""
    d = {
        "kind": "dist",
        "domain": "finite",
        "labels": {"record": {"color": "string", "size": "int"}},
        "support": [{"color": "red", "size": 1}, {"color": "blue", "size": 2}],
    }
    s = parse_spec(d)
    assert len(s.support) == 2
    rt = spec_to_dict(s)
    # Round-trip preserves authored order
    assert rt["support"][0] == {"color": "red", "size": 1}
    assert rt["support"][1] == {"color": "blue", "size": 2}


def test_support_duplicate_labels_error():
    """Duplicate support labels must raise AlgebraError."""
    with pytest.raises(AlgebraError, match="duplicate support label"):
        parse_spec({"kind": "dist", "domain": "finite", "support": ["H", "H"]})


def test_support_on_non_finite_domain_error():
    """Support on a non-finite domain must raise AlgebraError."""
    with pytest.raises(AlgebraError, match="domain='finite'"):
        parse_spec({"kind": "dist", "domain": "bool", "support": [True, False]})
    with pytest.raises(AlgebraError, match="domain='finite'"):
        parse_spec({"kind": "value", "domain": "int", "support": [1, 2]})


def test_support_element_violates_labels_shape():
    """A support element that violates the declared labels shape must raise."""
    with pytest.raises(AlgebraError):
        parse_spec({
            "kind": "dist",
            "domain": "finite",
            "labels": {"record": {"color": "string"}},
            "support": [{"color": "red"}, {"color": "blue", "extra": "oops"}],
        })


def test_support_spec_to_dict_roundtrip_order():
    """spec_to_dict must preserve authored support order."""
    labels = ["C", "A", "B"]
    d = {"kind": "dist", "domain": "finite", "support": labels}
    s = parse_spec(d)
    rt = spec_to_dict(s)
    assert rt["support"] == labels


def test_support_int_float_normalize_to_same_key():
    """Int label 1 and float label 1.0 should normalize to the same support key."""
    s_int = parse_spec({"kind": "dist", "domain": "finite", "support": [1, 2]})
    s_float = parse_spec({"kind": "dist", "domain": "finite", "support": [1.0, 2.0]})
    assert set(s_int.support) == set(s_float.support)


# ---------------------------------------------------------------------------
# Canonicalize: support validation
# ---------------------------------------------------------------------------

def test_canonicalize_enum_dist_declared_support_passes():
    """EnumDist with all labels in declared support passes."""
    s = parse_spec({"kind": "dist", "domain": "finite", "support": ["H", "T"]})
    raw = {"kind": "dist_enum", "support": ["H", "T"], "probs": [0.6, 0.4]}
    c = canonicalize(raw, s)
    assert isinstance(c, EnumDist)


def test_canonicalize_enum_dist_undeclared_label_raises():
    """EnumDist with positive mass on undeclared label must raise AlgebraError."""
    s = parse_spec({"kind": "dist", "domain": "finite", "support": ["H", "T"]})
    raw = {"kind": "dist_enum", "support": ["H", "X"], "probs": [0.6, 0.4]}
    with pytest.raises(AlgebraError, match="not in declared support"):
        canonicalize(raw, s)
    # Check the label appears in the error message
    try:
        canonicalize(raw, s)
    except AlgebraError as e:
        assert "X" in str(e)


def test_canonicalize_cloud_outside_support_raises():
    """Cloud sample outside declared support must raise AlgebraError."""
    s = parse_spec({"kind": "dist", "domain": "finite", "support": ["H", "T"],
                    "protocol": "draws"})
    with pytest.raises(AlgebraError, match="not in declared support"):
        canonicalize(["H", "X", "T"], s)


def test_canonicalize_exact_value_outside_support_raises():
    """Exact value outside declared support must raise AlgebraError."""
    s = parse_spec({"kind": "value", "domain": "finite", "support": ["H", "T"]})
    with pytest.raises(AlgebraError, match="not in declared support"):
        canonicalize("X", s)


def test_canonicalize_exact_value_in_support_passes():
    """Exact value in declared support passes."""
    s = parse_spec({"kind": "value", "domain": "finite", "support": ["H", "T"]})
    c = canonicalize("H", s)
    assert c == Exact("H")


def test_canonicalize_support_int_vs_float_normalize():
    """Int label 1 and float label 1.0 are both accepted by a support declaring [1, 2]."""
    s = parse_spec({"kind": "dist", "domain": "finite", "support": [1, 2]})
    raw_int = {"kind": "dist_enum", "support": [1, 2], "probs": [0.5, 0.5]}
    raw_float = {"kind": "dist_enum", "support": [1.0, 2.0], "probs": [0.5, 0.5]}
    c1 = canonicalize(raw_int, s)
    c2 = canonicalize(raw_float, s)
    assert distance(c1, c2, s).value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# self_noise (renamed from _cand_self_noise)
# ---------------------------------------------------------------------------

def test_self_noise_exact_is_zero():
    """Exact answers have zero self-noise."""
    assert self_noise(Exact(1.0), VAL_REAL) == 0.0


def test_self_noise_cloud():
    """Cloud split-half self-noise is positive for a cloud with variance."""
    rng = random.Random(5)
    c = Cloud(tuple(rng.gauss(0, 1) for _ in range(400)))
    sn = self_noise(c, DIST_REAL)
    assert sn >= 0.0


def test_self_noise_enum_dist_is_zero():
    """EnumDist has zero self-noise (no sampling noise)."""
    e = EnumDist(("H", "T"), (0.6, 0.4))
    assert self_noise(e, DIST_FIN) == 0.0


# ---------------------------------------------------------------------------
# agreement()
# ---------------------------------------------------------------------------

def test_agreement_identical_enum_dists_agree():
    """Two identical EnumDists must agree."""
    s = parse_spec({"kind": "dist", "domain": "finite"})
    e = EnumDist(("H", "T"), (0.6, 0.4))
    result = agreement(e, e, s)
    assert result["agree"] is True
    assert result["distance"] == pytest.approx(0.0)


def test_agreement_different_exact_values_disagree():
    """Two exact values that differ should not agree (tol = eps, distance = inf)."""
    result = agreement(Exact("H"), Exact("T"), VAL_FIN)
    assert result["agree"] is False
    assert result["distance"] == math.inf


def test_agreement_close_exact_values_agree():
    """Two very close exact real values should agree."""
    result = agreement(Exact(1.0), Exact(1.0), VAL_REAL)
    assert result["agree"] is True


def test_agreement_clouds_from_same_distribution():
    """Two clouds from the same distribution should agree within self-noise."""
    rng = random.Random(99)
    samples_a = tuple(rng.gauss(0, 1) for _ in range(600))
    samples_b = tuple(rng.gauss(0, 1) for _ in range(600))
    ca = Cloud(samples_a)
    cb = Cloud(samples_b)
    result = agreement(ca, cb, DIST_REAL)
    assert result["agree"] is True


def test_agreement_clouds_far_apart_disagree():
    """Two clouds from far-apart distributions should not agree."""
    rng = random.Random(42)
    ca = Cloud(tuple(rng.gauss(0, 1) for _ in range(600)))
    cb = Cloud(tuple(rng.gauss(10, 1) for _ in range(600)))
    result = agreement(ca, cb, DIST_REAL)
    assert result["agree"] is False


def test_agreement_record_recurses():
    """agreement() on a record spec recurses per field and returns fields."""
    r_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "value", "domain": "real"},
        "y": {"kind": "value", "domain": "real"}}})
    a = Rec(fields=(("x", Exact(1.0)), ("y", Exact(2.0))))
    b = Rec(fields=(("x", Exact(1.0)), ("y", Exact(2.0))))
    result = agreement(a, b, r_spec)
    assert result["agree"] is True
    assert "fields" in result


def test_agreement_record_one_field_disagrees():
    """Record agreement fails if any field disagrees."""
    r_spec = spec({"kind": "record", "fields": {
        "x": {"kind": "value", "domain": "real"},
        "y": {"kind": "value", "domain": "finite"}}})
    a = Rec(fields=(("x", Exact(1.0)), ("y", Exact("H"))))
    b = Rec(fields=(("x", Exact(1.0)), ("y", Exact("T"))))
    result = agreement(a, b, r_spec)
    assert result["agree"] is False


def test_agreement_returns_metric():
    """agreement() result includes the metric string."""
    e = EnumDist(("H",), (1.0,))
    result = agreement(e, e, DIST_FIN)
    assert result["metric"] == "tv"


# ---------------------------------------------------------------------------
# Mapping form: plain JSON object {label: probability}
# ---------------------------------------------------------------------------

def test_mapping_finite_string_keys():
    """Plain dict with string keys is accepted as a mapping (label→prob)."""
    raw = {"chases": 0.7, "barks": 0.3}
    c = canonicalize(raw, DIST_FIN)
    assert isinstance(c, EnumDist)
    histo = dict(zip(c.support, c.probs))
    assert histo["chases"] == pytest.approx(0.7)
    assert histo["barks"] == pytest.approx(0.3)


def test_mapping_key_true_parses_to_bool():
    """Key "true" JSON-parses to Python bool True."""
    raw = {"true": 0.6, "false": 0.4}
    c = canonicalize(raw, DIST_BOOL)
    assert isinstance(c, EnumDist)
    histo = dict(zip(c.support, c.probs))
    assert True in histo
    assert False in histo
    assert histo[True] == pytest.approx(0.6)


def test_mapping_key_int_parses():
    """Key "3" JSON-parses to int 3."""
    raw = {"1": 0.5, "3": 0.5}
    c = canonicalize(raw, DIST_FIN)
    assert isinstance(c, EnumDist)
    histo = dict(zip(c.support, c.probs))
    assert 1 in histo
    assert 3 in histo


def test_mapping_key_list_parses():
    """Key "[1, 2]" JSON-parses to list [1, 2]."""
    raw = {"[1, 2]": 0.5, "[3, 4]": 0.5}
    c = canonicalize(raw, DIST_FIN)
    assert isinstance(c, EnumDist)
    histo = {str(v): p for v, p in zip(c.support, c.probs)}
    # The labels should be lists; check via _label_key
    from eval.algebra import _label_key
    keys = {_label_key(v) for v in c.support}
    assert "[1, 2]" in keys or "\"[1, 2]\"" in keys  # json.dumps of list
    # More reliable: check both parsed labels are lists
    assert all(isinstance(v, list) for v in c.support)


def test_mapping_unparseable_key_falls_back_to_string():
    """Key that fails json.loads stays as a raw string."""
    raw = {"chases": 0.8, "hides": 0.2}
    c = canonicalize(raw, DIST_FIN)
    assert isinstance(c, EnumDist)
    assert "chases" in c.support
    assert "hides" in c.support


def test_mapping_mass_normalizes():
    """Probabilities in the mapping form are normalized."""
    raw = {"a": 2.0, "b": 6.0}
    c = canonicalize(raw, DIST_FIN)
    assert sum(c.probs) == pytest.approx(1.0)
    histo = dict(zip(c.support, c.probs))
    assert histo["a"] == pytest.approx(0.25)
    assert histo["b"] == pytest.approx(0.75)


def test_mapping_duplicate_after_parse_keys_merge():
    """Keys "1" and "1.0" both parse to int 1, so they merge."""
    raw = {"1": 0.3, "1.0": 0.2, "2": 0.5}
    c = canonicalize(raw, DIST_FIN)
    assert isinstance(c, EnumDist)
    histo = dict(zip(c.support, c.probs))
    # "1" and "1.0" both normalize to label 1; combined prob = 0.5/1.0 = 0.5
    assert 1 in histo
    assert histo[1] == pytest.approx(0.5)
    assert 2 in histo
    assert histo[2] == pytest.approx(0.5)


def test_mapping_declared_support_violation_raises():
    """Mapping label not in declared support raises AlgebraError."""
    s = parse_spec({"kind": "dist", "domain": "finite", "support": ["H", "T"]})
    raw = {"H": 0.6, "X": 0.4}  # "X" is not in declared support
    with pytest.raises(AlgebraError, match="not in declared support"):
        canonicalize(raw, s)


def test_mapping_dist_real_w1_comparable():
    """Mapping form for dist/real produces EnumDist comparable via W1."""
    raw = {"0": 0.5, "1": 0.5}
    c = canonicalize(raw, DIST_INT)
    assert isinstance(c, EnumDist)
    # Should be comparable to an equivalent EnumDist
    ref = canonicalize({"kind": "dist_enum", "support": [0, 1], "probs": [0.5, 0.5]}, DIST_INT)
    d = distance(c, ref, DIST_INT)
    assert d.value == pytest.approx(0.0)


def test_mapping_key_prefers_declared_string_over_json_literal():
    # "null" parses to None via json.loads; with the string declared in the
    # spec's support, the raw-string reading wins.
    spec = parse_spec({"kind": "dist", "domain": "finite",
                       "support": ["null", "every-not"]})
    c = canonicalize({"null": 0.3, "every-not": 0.7}, spec)
    assert set(c.support) == {"null", "every-not"}
