"""Answer algebra: the single place answer semantics live.

Implements `data/SCHEMA.md`. An answer is a mathematical object —
Value(domain) | Dist(domain) | Record{...} — independent of how an executed
program represents it (exact value, enumerated distribution, parametric
distribution, sample cloud). Comparison is defined on the object, between
any pair of representations. Tolerance is measured (GT-vs-GT noise floor),
not authored.

The legacy comparators (eval/metrics.py, eval/spec_metrics.py) were ripped
in P2; this module is the only comparison surface (see data/REDESIGN.md).
"""

from __future__ import annotations

import functools
import json
import math
import random
import re
import statistics
from dataclasses import dataclass, field


class AlgebraError(Exception):
    """Canonicalization or comparison failure (structural, not numeric)."""


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

DOMAINS = ("bool", "finite", "int", "real", "realvec")

# Atomic domains valid as label-field domains (SCHEMA.md §Structured finite labels).
_LABEL_FIELD_DOMAINS = ("bool", "int", "real", "string")


@dataclass(frozen=True)
class Spec:
    kind: str                                   # value | dist | record
    domain: str | None = None                   # value/dist only
    fields: tuple[tuple[str, "Spec"], ...] = ()  # record only
    protocol: str = "object"                    # dist only: object | draws
    estimated: bool = False                     # value only
    labels: tuple[tuple[str, str], ...] = ()    # sorted (field, atomic_domain) pairs; empty = undeclared
    support: tuple = ()                         # tuple of label-key strings; empty = undeclared (finite domain only)

    def field_map(self) -> dict[str, "Spec"]:
        return dict(self.fields)


def _parse_support(support_raw, domain: str, labels: tuple[tuple[str, str], ...]) -> tuple:
    """Parse and validate a 'support' list from a spec dict.

    Returns a tuple of label-key strings (canonical, ordered as authored).
    Raises AlgebraError on:
      - non-finite domain
      - empty list
      - duplicate keys
      - element violating the declared labels shape (when labels are declared)
    """
    if domain != "finite":
        raise AlgebraError("'support' applies only to specs with domain='finite'")
    if not isinstance(support_raw, list) or not support_raw:
        raise AlgebraError("'support' must be a non-empty list")
    seen_keys: set[str] = set()
    result: list[str] = []
    for elem in support_raw:
        if labels:
            elem = _norm_labeled_element(elem, labels)
        else:
            elem = _normalize_label(elem)
        k = _label_key(elem)
        if k in seen_keys:
            raise AlgebraError(f"duplicate support label: {json.loads(k)!r}")
        seen_keys.add(k)
        result.append(k)
    return tuple(result)


def parse_spec(d: dict) -> Spec:
    kind = d.get("kind")
    if kind == "record":
        fields = d.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise AlgebraError("record spec needs non-empty 'fields'")
        return Spec(kind="record",
                    fields=tuple((n, parse_spec(s)) for n, s in fields.items()))
    if kind not in ("value", "dist"):
        raise AlgebraError(f"unknown spec kind: {kind!r}")
    domain = d.get("domain")
    if domain not in DOMAINS:
        raise AlgebraError(f"unknown domain: {domain!r}")
    if kind == "dist":
        if domain == "realvec":
            raise AlgebraError("dist over realvec is not supported; respec the problem")
        # cross-field guards: estimated is value-only, protocol is dist-only
        if d.get("estimated"):
            raise AlgebraError("'estimated' applies to value specs only")
        protocol = d.get("protocol", "object")
        if protocol not in ("object", "draws"):
            raise AlgebraError(f"unknown protocol: {protocol!r}")
        # labels — only valid on dist/finite
        labels_raw = d.get("labels")
        labels: tuple[tuple[str, str], ...] = ()
        if labels_raw is not None:
            if domain != "finite":
                raise AlgebraError("'labels' applies only to dist specs with domain='finite'")
            if not isinstance(labels_raw, dict) or "record" not in labels_raw:
                raise AlgebraError("'labels' must be {'record': {name: domain, ...}}")
            record = labels_raw["record"]
            if not isinstance(record, dict) or not record:
                raise AlgebraError("'labels.record' must be a non-empty object")
            parsed: list[tuple[str, str]] = []
            for name, fdomain in record.items():
                if fdomain not in _LABEL_FIELD_DOMAINS:
                    raise AlgebraError(
                        f"label field {name!r} has non-atomic domain {fdomain!r}; "
                        f"must be one of {_LABEL_FIELD_DOMAINS}"
                    )
                parsed.append((name, fdomain))
            labels = tuple(sorted(parsed))
        # support — only valid on finite domain
        support_raw = d.get("support")
        support: tuple = ()
        if support_raw is not None:
            support = _parse_support(support_raw, domain, labels)
        return Spec(kind="dist", domain=domain, protocol=protocol, labels=labels,
                    support=support)
    # value
    if d.get("protocol", "object") != "object":
        raise AlgebraError("'protocol' applies to dist specs only")
    if d.get("labels") is not None:
        raise AlgebraError("'labels' applies only to dist specs with domain='finite'")
    # support — only valid on finite domain for value specs too
    support_raw = d.get("support")
    support: tuple = ()
    if support_raw is not None:
        support = _parse_support(support_raw, domain, ())
    return Spec(kind="value", domain=domain,
                estimated=bool(d.get("estimated", False)),
                support=support)


def spec_to_dict(s: Spec) -> dict:
    if s.kind == "record":
        return {"kind": "record",
                "fields": {n: spec_to_dict(f) for n, f in s.fields}}
    out: dict = {"kind": s.kind, "domain": s.domain}
    if s.kind == "dist" and s.protocol != "object":
        out["protocol"] = s.protocol
    if s.kind == "value" and s.estimated:
        out["estimated"] = True
    if s.kind == "dist" and s.labels:
        out["labels"] = {"record": dict(s.labels)}
    if s.support:
        out["support"] = [json.loads(k) for k in s.support]
    return out


# ---------------------------------------------------------------------------
# Canonical representations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Exact:
    value: object


@dataclass(frozen=True)
class EnumDist:
    support: tuple        # domain-normalized values
    probs: tuple          # normalized to sum 1, aligned with support


@dataclass(frozen=True)
class ParamDist:
    family: str           # canonical lowercase family name
    params: tuple         # sorted (name, float) pairs, canonical names

    def param_map(self) -> dict[str, float]:
        return dict(self.params)


@dataclass(frozen=True)
class Cloud:
    samples: tuple        # domain-normalized draws


@dataclass(frozen=True)
class Rec:
    fields: tuple         # ((name, canon), ...)

    def field_map(self) -> dict:
        return dict(self.fields)


# ---------------------------------------------------------------------------
# Domain normalization
# ---------------------------------------------------------------------------

def _norm_atom(v, domain: str):
    if domain == "bool":
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(v)
        raise AlgebraError(f"not coercible to bool: {v!r}")
    if domain == "int":
        if isinstance(v, bool):
            raise AlgebraError(f"bool where int expected: {v!r}")
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        raise AlgebraError(f"not coercible to int: {v!r}")
    if domain == "real":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AlgebraError(f"not coercible to real: {v!r}")
        if not math.isfinite(float(v)):
            raise AlgebraError(f"non-finite real value: {v!r}")
        return float(v)
    if domain == "realvec":
        if not isinstance(v, list):
            raise AlgebraError(f"not a vector: {v!r}")
        return [_norm_atom(x, "real") for x in v]
    # finite: opaque labels, passed through untouched
    return v


def _label_key(v) -> str:
    """Canonical hashable key for an opaque label.

    Whole-number floats are normalised to int before serialisation so that
    WebPPL's int label 1 and Pyro's float label 1.0 produce the same key.
    Booleans are exempt (False/True must not collapse to 0/1).
    """
    return json.dumps(_normalize_label(v), sort_keys=True)


def _normalize_label(v):
    """Recursively coerce whole-number floats to int for finite-domain keys."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_normalize_label(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize_label(val) for k, val in v.items()}
    return v


def _norm_labeled_element(v, labels: tuple[tuple[str, str], ...]):
    """Validate and normalize a finite-domain support element against a labels schema.

    `labels` is a sorted tuple of (field_name, atomic_domain) pairs.
    Returns a normalized dict with exactly the declared fields.
    Raises AlgebraError on missing field, extra field, or domain coercion failure.
    """
    if not isinstance(v, dict):
        raise AlgebraError(
            f"labeled spec expects each support element to be an object, got {type(v).__name__}"
        )
    declared = dict(labels)
    declared_names = set(declared)
    got_names = set(v)
    missing = declared_names - got_names
    extra = got_names - declared_names
    if missing:
        raise AlgebraError(f"labeled support element missing fields: {sorted(missing)}")
    if extra:
        raise AlgebraError(f"labeled support element has undeclared fields: {sorted(extra)}")
    out: dict = {}
    for name, fdomain in labels:
        raw_val = v[name]
        if fdomain == "string":
            if not isinstance(raw_val, str):
                raise AlgebraError(
                    f"label field {name!r} expected string, got {type(raw_val).__name__}: {raw_val!r}"
                )
            out[name] = raw_val
        else:
            out[name] = _norm_atom(raw_val, fdomain)
    return out


# ---------------------------------------------------------------------------
# Parametric families: canonical names, aliases, sampling
# ---------------------------------------------------------------------------

_FAMILY_NAME_ALIASES = {
    "normal": "gaussian",
}

_PARAM_ALIASES = {
    "beta": {"a": "a", "b": "b", "alpha": "a", "beta": "b",
             "concentration1": "a", "concentration0": "b"},
    "gaussian": {"mu": "mu", "sigma": "sigma", "loc": "mu", "scale": "sigma",
                 "mean": "mu", "std": "sigma"},
    # canonical gamma parameterisation is shape + rate; scale is converted at
    # alias time so cross-PPL Gamma comparisons produce identical tuples.
    "gamma": {"shape": "shape", "rate": "rate", "concentration": "shape"},
    "uniform": {"a": "a", "b": "b", "low": "a", "high": "b"},
    "exponential": {"a": "rate", "rate": "rate"},
}

_PARAM_SAMPLE_N = 16384


def _gamma_transform(p: dict) -> dict:
    # Normalise scale -> rate so cross-PPL identical distributions
    # produce the same canonical tuple and hit the fast path.
    if "scale" in p and "rate" in p:
        raise AlgebraError("gamma: ambiguous params: both scale and rate")
    if "scale" in p:
        p = {**p, "rate": 1.0 / p["scale"]}
        del p["scale"]
    return p


# Applied after alias mapping; family -> callable(params_dict) -> params_dict.
_PARAM_TRANSFORMS: dict[str, object] = {
    "gamma": _gamma_transform,
}


def _canonical_family(name: str) -> str:
    low = name.strip().lower()
    return _FAMILY_NAME_ALIASES.get(low, low)


def _canonical_params(family: str, params: dict) -> tuple:
    aliases = _PARAM_ALIASES.get(family, {})
    out = {}
    for k, v in params.items():
        try:
            out[aliases.get(k, k)] = float(v)
        except (TypeError, ValueError):
            raise AlgebraError(f"non-numeric param {k}={v!r} for family {family}")
    transform = _PARAM_TRANSFORMS.get(family)
    if transform is not None:
        out = transform(out)
    return tuple(sorted(out.items()))


# Dispatch table: family -> callable(params_dict, rng, n) -> list[float].
# Canonical form is always rate for gamma; scale was converted in _canonical_params.
_FAMILY_SAMPLERS: dict[str, object] = {
    "beta":        lambda p, rng, n: [rng.betavariate(p["a"], p["b"]) for _ in range(n)],
    "gaussian":    lambda p, rng, n: [rng.gauss(p["mu"], p["sigma"]) for _ in range(n)],
    "gamma":       lambda p, rng, n: [rng.gammavariate(p["shape"], 1.0 / p["rate"]) for _ in range(n)],
    "uniform":     lambda p, rng, n: [rng.uniform(p["a"], p["b"]) for _ in range(n)],
    "exponential": lambda p, rng, n: [rng.expovariate(p["rate"]) for _ in range(n)],
}


@functools.lru_cache(maxsize=256)
def _sample_parametric(d: ParamDist, n: int = _PARAM_SAMPLE_N) -> tuple | None:
    """Seeded draws from a known family; None if the family is unknown.

    Common-random-numbers: when two GT runs are both ParamDist and use the
    same seed the draws are identical, giving W1=0. This is deliberate — for
    parametric-vs-parametric comparison it is variance reduction, not a bug.
    The floor is still set correctly by the fast path (identical objects → 0).

    Memoized: the draws are a pure function of (family, params, n), and a
    verdict call samples the same ParamDist k+ times.
    """
    sampler = _FAMILY_SAMPLERS.get(d.family)
    if sampler is None:
        return None
    p = d.param_map()
    rng = random.Random(f"algebra:{d.family}:{sorted(p.items())}")
    try:
        return tuple(sampler(p, rng, n))
    except KeyError as e:
        raise AlgebraError(f"family {d.family} missing param {e}")


# Serializer repr strings for continuous distributions: 'Gaussian({ mu: 10,
# sigma: 1 })' / 'Beta(a=2, b=5)'. WebPPL emits this form by design (continuous
# dists can't toJSON); this parser is its canonical reader. Only the Pyro
# serializer emits native dist_param.
_LEGACY_REPR_RE = re.compile(r"^(\w+)\s*\(\s*\{?\s*(.*?)\s*\}?\s*\)\s*$")


def _parse_legacy_repr(repr_s: str) -> ParamDist:
    m = _LEGACY_REPR_RE.match(repr_s.strip())
    if not m:
        raise AlgebraError(f"unparseable parametric repr: {repr_s!r}")
    family = _canonical_family(m.group(1))
    params: dict = {}
    for part in m.group(2).split(","):
        part = part.strip()
        if not part:
            continue
        kv = re.split(r"[:=]", part, maxsplit=1)
        if len(kv) != 2:
            raise AlgebraError(f"unparseable param {part!r} in {repr_s!r}")
        params[kv[0].strip().strip("\"'")] = kv[1].strip()
    return ParamDist(family=family, params=_canonical_params(family, params))


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def _kind_tag(d: dict):
    """Single source of truth for the wire-kind discriminant."""
    return d.get("kind") or d.get("__kind")


def _is_dist_dict(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    if _kind_tag(raw) in ("dist_enum", "distribution"):
        return True
    # permissive: anything carrying parallel probs/support lists
    return isinstance(raw.get("probs"), list) and isinstance(raw.get("support"), list)


def _is_param_dict(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    return _kind_tag(raw) in ("dist_param", "distribution_continuous")


def _enum_from_dict(
    raw: dict, domain: str,
    labels: tuple[tuple[str, str], ...] = (),
    declared_support: tuple = (),
) -> EnumDist:
    raw_support = raw.get("support") or []
    probs = raw.get("probs") or []
    if len(raw_support) != len(probs):
        raise AlgebraError("support/probs length mismatch")
    support_set: set[str] = set(declared_support) if declared_support else set()
    merged: dict[str, list] = {}
    for v, p in zip(raw_support, probs):
        if p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            # a malformed candidate (e.g. a probability that is a dict/list) must
            # be rejected as malformed, never crash the whole scoring run.
            raise AlgebraError(f"non-numeric probability {p!r}")
        # reject non-finite probabilities: NaN/Inf corrupt normalisation
        if not math.isfinite(p) or p < 0:
            raise AlgebraError(f"invalid probability {p!r}")
        if p == 0:
            continue
        if labels:
            v = _norm_labeled_element(v, labels)
        else:
            v = _norm_atom(v, domain)
        k = _label_key(v)
        if support_set and k not in support_set:
            raise AlgebraError(f"label {json.loads(k)!r} not in declared support")
        if k in merged:
            merged[k][1] += p
        else:
            merged[k] = [v, p]
    total = sum(p for _, p in merged.values())
    if total <= 0:
        raise AlgebraError("empty or zero-mass distribution")
    items = sorted(merged.values(), key=lambda vp: _label_key(vp[0]))
    return EnumDist(support=tuple(v for v, _ in items),
                    probs=tuple(p / total for _, p in items))


def _has_draws_field(spec: Spec) -> bool:
    """True if any leaf field under a record spec has protocol='draws'."""
    if spec.kind == "record":
        return any(_has_draws_field(f) for _, f in spec.fields)
    return spec.kind == "dist" and spec.protocol == "draws"


def canonicalize(raw, spec: Spec):
    """Program output (native or legacy wire JSON) -> canonical representation.

    For dist specs with protocol="draws", `raw` is the collected list of
    draws (the harness gathers N seeded runs; a program that samples
    internally yields the same shape in one run).
    """
    if spec.kind == "record":
        fmap = spec.field_map()
        if isinstance(raw, dict) and _kind_tag(raw) == "record" and "fields" in raw:
            raw = raw["fields"]
        if isinstance(raw, list):
            # draws protocol over a record: only valid when at least one leaf
            # field carries protocol="draws"; otherwise a list is structurally wrong.
            if not _has_draws_field(spec):
                raise AlgebraError(
                    "record expected a single object, got a list")
            if not all(isinstance(r, dict) for r in raw):
                raise AlgebraError("record draws must be a list of objects")
            # split per-field; value fields require consensus across runs
            result_fields = []
            for n, f in fmap.items():
                vals = [r.get(n) for r in raw]
                if f.kind == "value":
                    # all runs must agree; take consensus. Compare by equality,
                    # not set() — a realvec value field is an Exact over a list,
                    # which is unhashable and would raise TypeError on set().
                    canons = [canonicalize(v, f) for v in vals]
                    if any(c != canons[0] for c in canons[1:]):
                        raise AlgebraError(
                            f"value field {n!r} varies across runs")
                    result_fields.append((n, canons[0]))
                elif f.kind == "dist" and f.protocol == "object":
                    # dist-object field inside a draws-record is incoherent:
                    # the harness is collecting scalars, not distribution objects
                    raise AlgebraError(
                        f"dist field {n!r} has protocol='object' inside a draws-record")
                else:
                    result_fields.append((n, canonicalize(vals, f)))
            return Rec(fields=tuple(result_fields))
        if not isinstance(raw, dict):
            raise AlgebraError(f"record answer must be an object, got {type(raw).__name__}")
        missing = [n for n in fmap if n not in raw]
        if missing:
            raise AlgebraError(f"record missing fields: {missing}")
        return Rec(fields=tuple(
            (n, canonicalize(raw[n], f)) for n, f in fmap.items()))

    if spec.kind == "dist":
        if _is_param_dict(raw):
            # parametric continuous distributions are incoherent over finite label domains
            if spec.domain in ("bool", "finite"):
                raise AlgebraError(
                    f"parametric distribution is incompatible with domain {spec.domain!r}")
            if "family" in raw:
                family = _canonical_family(str(raw["family"]))
                return ParamDist(family=family,
                                 params=_canonical_params(family, raw.get("params") or {}))
            return _parse_legacy_repr(raw.get("repr", ""))
        if _is_dist_dict(raw):
            return _enum_from_dict(raw, spec.domain, spec.labels, spec.support)
        if (isinstance(raw, dict)
                and not _is_param_dict(raw)
                and not _is_dist_dict(raw)
                and _kind_tag(raw) is None):
            # Mapping form: plain JSON object {label: probability}.
            # This is the language-neutral "mapping" representation — natural in Python
            # (Pyro) and constructible in any language. Keys are JSON-parsed back to
            # labels; values must be numeric probabilities.
            declared = set(spec.support)
            support_list = []
            probs_list = []
            for k, v in raw.items():
                try:
                    label = json.loads(k)
                except (json.JSONDecodeError, ValueError):
                    label = k
                # Disambiguate keys that parse as JSON literals ("null", "true",
                # "1") when the problem's declared support says the label is the
                # raw STRING: prefer the declared reading.
                if (declared and label != k
                        and _label_key(label) not in declared
                        and _label_key(k) in declared):
                    label = k
                support_list.append(label)
                probs_list.append(v)
            return _enum_from_dict(
                {"support": support_list, "probs": probs_list},
                spec.domain, spec.labels, spec.support,
            )
        if isinstance(raw, dict) and _kind_tag(raw) == "cloud":
            raw = raw.get("samples")
        if isinstance(raw, list):
            if not raw:
                raise AlgebraError("empty sample cloud")
            support_set: set[str] = set(spec.support) if spec.support else set()
            if spec.labels:
                normed = tuple(_norm_labeled_element(v, spec.labels) for v in raw)
            else:
                normed = tuple(_norm_atom(v, spec.domain) for v in raw)
            if support_set:
                for v in normed:
                    k = _label_key(v)
                    if k not in support_set:
                        raise AlgebraError(f"label {json.loads(k)!r} not in declared support")
            return Cloud(samples=normed)
        raise AlgebraError(
            f"cannot read a dist({spec.domain}) from {type(raw).__name__}")

    # value
    if isinstance(raw, dict) and _kind_tag(raw) == "tensor":
        raw = raw.get("data")
    if isinstance(raw, dict) and _kind_tag(raw) == "exact":
        raw = raw.get("value")
    v = _norm_atom(raw, spec.domain)
    if spec.support:
        k = _label_key(v)
        if k not in set(spec.support):
            raise AlgebraError(f"label {json.loads(k)!r} not in declared support")
    return Exact(value=v)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

@dataclass
class Distance:
    value: float
    metric: str
    diagnostics: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)   # records only


def _hist(c) -> dict[str, float]:
    """Label histogram {label_key: prob} for finite/bool TV."""
    if isinstance(c, EnumDist):
        return {_label_key(v): p for v, p in zip(c.support, c.probs)}
    if isinstance(c, Cloud):
        n = len(c.samples)
        out: dict[str, float] = {}
        for s in c.samples:
            k = _label_key(s)
            out[k] = out.get(k, 0.0) + 1.0 / n
        return out
    raise AlgebraError(f"{type(c).__name__} is not a distribution over labels")


def _numeric_pairs(c) -> list[tuple[float, float]]:
    """Weighted points [(x, w)] for W1 on int/real."""
    if isinstance(c, EnumDist):
        try:
            return [(float(v), p) for v, p in zip(c.support, c.probs)]
        except (TypeError, ValueError):
            raise AlgebraError("non-numeric support in a numeric-domain distribution")
    if isinstance(c, Cloud):
        w = 1.0 / len(c.samples)
        return [(float(s), w) for s in c.samples]
    if isinstance(c, ParamDist):
        draws = _sample_parametric(c)
        if draws is None:
            raise AlgebraError(f"cannot sample unknown family {c.family!r}")
        w = 1.0 / len(draws)
        return [(x, w) for x in draws]
    raise AlgebraError(f"{type(c).__name__} is not a numeric distribution")


def _tv(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return min(1.0, max(0.0, 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)))


def _kl(p: dict, q: dict) -> float:
    """KL(p||q); returns math.inf when q has zero mass anywhere p has positive mass."""
    for k, pv in p.items():
        if pv > 0 and q.get(k, 0.0) == 0.0:
            return math.inf
    return sum(pv * math.log(pv / q[k])
               for k, pv in p.items() if pv > 0)


def _w1(p: list[tuple[float, float]], q: list[tuple[float, float]]) -> float:
    """W1 between weighted point sets via single-sweep CDF integration."""
    p = sorted(p)
    q = sorted(q)
    pts = sorted({x for x, _ in p} | {x for x, _ in q})
    if len(pts) < 2:
        return 0.0
    ip = iq = 0
    cp = cq = 0.0
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        while ip < len(p) and p[ip][0] <= a:
            cp += p[ip][1]
            ip += 1
        while iq < len(q) and q[iq][0] <= a:
            cq += q[iq][1]
            iq += 1
        total += abs(cp - cq) * (b - a)
    return total


def _ks(xs: list[float], ys: list[float]) -> float:
    # guard empty inputs
    if not xs and not ys:
        return 0.0
    if not xs or not ys:
        return 1.0
    xs, ys = sorted(xs), sorted(ys)
    i = j = 0
    d = 0.0
    for p in sorted(set(xs) | set(ys)):
        while i < len(xs) and xs[i] <= p:
            i += 1
        while j < len(ys) and ys[j] <= p:
            j += 1
        d = max(d, abs(i / len(xs) - j / len(ys)))
    return d


def _metric(spec: Spec) -> str:
    """Primary metric string determined solely by spec — no GT needed."""
    if spec.kind == "record":
        return "record"
    if spec.kind == "dist":
        return "tv" if spec.domain in ("bool", "finite") else "w1"
    # value
    return "absdiff" if spec.domain in ("int", "real", "realvec") else "eq"


def distance(a, b, spec: Spec) -> Distance:
    """Distance between two canonical answers under a spec.

    Defined on the object, between any pair of representations.
    """
    m = _metric(spec)
    if m == "record":
        fmap = spec.field_map()
        if not isinstance(a, Rec) or not isinstance(b, Rec):
            raise AlgebraError("record spec requires record answers")
        am, bm = a.field_map(), b.field_map()
        try:
            fields = {n: distance(am[n], bm[n], f) for n, f in fmap.items()}
        except KeyError as e:
            raise AlgebraError(f"record missing field {e}")
        worst = max((d.value for d in fields.values()), default=0.0)
        return Distance(value=worst, metric="record", fields=fields)

    if m == "tv":
        pa, pb = _hist(a), _hist(b)
        return Distance(value=_tv(pa, pb), metric="tv",
                        diagnostics={"kl": _kl(pa, pb)})

    if m == "w1":
        diagnostics: dict = {}
        if isinstance(a, ParamDist) and isinstance(b, ParamDist):
            diagnostics["families"] = (a.family, b.family)
            if a == b:
                return Distance(value=0.0, metric="w1", diagnostics=diagnostics)
            if a.family == b.family:
                pa, pb = a.param_map(), b.param_map()
                diagnostics["param_diffs"] = {
                    k: {"a": pa.get(k), "b": pb.get(k)}
                    for k in set(pa) | set(pb)}
        na, nb = _numeric_pairs(a), _numeric_pairs(b)
        if isinstance(a, Cloud) and isinstance(b, Cloud):
            diagnostics["ks"] = _ks([x for x, _ in na], [x for x, _ in nb])
        if isinstance(a, EnumDist) and isinstance(b, EnumDist):
            diagnostics["tv"] = _tv(_hist(a), _hist(b))
        return Distance(value=_w1(na, nb), metric="w1", diagnostics=diagnostics)

    # absdiff or eq — value spec
    if not isinstance(a, Exact) or not isinstance(b, Exact):
        raise AlgebraError("value spec requires exact answers")
    va, vb = a.value, b.value
    if m == "absdiff":
        if spec.domain == "realvec":
            if len(va) != len(vb):
                return Distance(value=math.inf, metric="absdiff",
                                diagnostics={"length": (len(va), len(vb))})
            worst = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
            return Distance(value=worst, metric="absdiff")
        return Distance(value=abs(float(va) - float(vb)), metric="absdiff")
    # eq: bool / finite — equality on canonical labels (eps = 1e-9 per schema)
    eq = _label_key(va) == _label_key(vb)
    return Distance(value=0.0 if eq else math.inf, metric="eq")


# ---------------------------------------------------------------------------
# Noise floor and verdicts
# ---------------------------------------------------------------------------

# Discriminability caps: a noise floor beyond these means the problem cannot
# tell answers apart and is flagged ill-posed (see SCHEMA.md).
_TV_FLOOR_CAP = 0.3
_W1_FLOOR_CAP_FRAC = 0.5    # of the pooled GT spread
_VALUE_FLOOR_CAP_FRAC = 0.5  # of the GT magnitude


def noise_floor(answers: list, spec: Spec) -> float:
    """Max pairwise distance among `answers` under `spec`.

    For records, returns the worst-case (max) over fields, recursing.
    This is the GT noise floor when answers are GT runs; the gate uses
    the same function on solver runs to detect solver scatter.
    """
    if spec.kind == "record":
        fmap = spec.field_map()
        worst = 0.0
        for n, f in fmap.items():
            sub = [a.field_map()[n] for a in answers]
            worst = max(worst, noise_floor(sub, f))
        return worst
    return max(
        (distance(answers[i], answers[j], spec).value
         for i in range(len(answers))
         for j in range(i + 1, len(answers))),
        default=0.0,
    )


def self_noise(cand, spec: Spec) -> float:
    """Candidate split-half self-noise: distance between first and second half.

    Measures the candidate's own sampling noise. Only meaningful for Cloud
    (split samples) and Rec (recurse per field). All other representations
    yield 0.0 because they carry no sampling noise.
    """
    if isinstance(cand, Cloud):
        n = len(cand.samples)
        if n < 2:
            return 0.0
        half = n // 2
        a = Cloud(cand.samples[:half])
        b = Cloud(cand.samples[half:])
        return distance(a, b, spec).value
    if isinstance(cand, Rec):
        fmap = spec.field_map()
        worst = 0.0
        for name, f in fmap.items():
            sub = cand.field_map().get(name)
            if sub is not None:
                worst = max(worst, self_noise(sub, f))
        return worst
    return 0.0


def _magnitude(answers: list) -> float:
    """Max |x| over the Exact answers in a pool — the scale for absdiff epsilons."""
    return max(
        (abs(float(x)) for a in answers if isinstance(a, Exact)
         for x in (a.value if isinstance(a.value, list) else [a.value])),
        default=1.0,
    )


def _eps_for(metric: str, answers: list, spec: Spec) -> float:
    """Compute the epsilon floor for a given metric and answer pool."""
    if metric in ("tv", "eq"):
        return 1e-9
    if metric == "w1":
        return 1e-9 * max(1.0, _pooled_spread(answers, spec))
    # absdiff
    return 1e-9 * max(1.0, _magnitude(answers))


def _pooled_spread(gts: list, spec: Spec) -> float:
    """Scale of the GT answers, for W1/value epsilons and ill-posedness caps.

    Only numeric domains have a scale; finite/bool answers return 0.
    """
    if spec.domain not in ("int", "real", "realvec"):
        return 0.0
    xs: list[float] = []
    for g in gts:
        if isinstance(g, Exact):
            v = g.value
            xs.extend(float(x) for x in (v if isinstance(v, list) else [v]))
        else:
            xs.extend(x for x, _ in _numeric_pairs(g))
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def _leaf_verdict(cand, gts: list, spec: Spec, margin: float) -> dict:
    gt_floor = noise_floor(gts, spec)
    cand_floor = self_noise(cand, spec)
    # compute distances once; take diagnostics from first GT run (convention)
    dists = [distance(cand, g, spec) for g in gts]
    d = statistics.median([di.value for di in dists])
    diag = dists[0].diagnostics

    metric = _metric(spec)
    eps = _eps_for(metric, gts, spec)
    if metric == "tv":
        ill = gt_floor > _TV_FLOOR_CAP
    elif metric == "w1":
        spread = _pooled_spread(gts, spec)
        ill = spread > 0 and gt_floor > _W1_FLOOR_CAP_FRAC * spread
    elif metric == "absdiff":
        ill = (gt_floor > _VALUE_FLOOR_CAP_FRAC * max(_magnitude(gts), eps)
               if spec.estimated else gt_floor > eps)
    else:  # eq
        ill = gt_floor > eps  # exact values disagreeing across seeds = nondeterminism

    tol = max(margin * max(gt_floor, cand_floor), eps)
    return {
        "passed": bool(d <= tol and not ill),
        "distance": d, "floor": gt_floor, "tol": tol,
        "metric": metric, "ill_posed": bool(ill),
        "diagnostics": diag,
    }


def agreement(a, b, spec: Spec, margin: float = 2.0) -> dict:
    """Do two canonical answers agree relative to their own measured noise?

    tol = max(margin * max(self_noise(a), self_noise(b)), eps)
    Records recurse per field: agree iff every field agrees; distance/tol are worst-case.
    Returns {"agree": bool, "distance": float, "tol": float, "metric": str}.
    """
    if spec.kind == "record":
        fmap = spec.field_map()
        if not isinstance(a, Rec) or not isinstance(b, Rec):
            raise AlgebraError("record spec requires record answers")
        am, bm = a.field_map(), b.field_map()
        fields = {}
        for n, f in fmap.items():
            try:
                fields[n] = agreement(am[n], bm[n], f, margin)
            except KeyError as e:
                raise AlgebraError(f"record missing field {e}")
        worst_dist = max(v["distance"] for v in fields.values())
        worst_tol = max(v["tol"] for v in fields.values())
        worst_metric = max(fields.values(), key=lambda v: v["distance"])["metric"]
        return {
            "agree": all(v["agree"] for v in fields.values()),
            "distance": worst_dist,
            "tol": worst_tol,
            "metric": worst_metric,
            "fields": fields,
        }
    metric = _metric(spec)
    pool = [a, b]
    eps = _eps_for(metric, pool, spec)
    sn_a = self_noise(a, spec)
    sn_b = self_noise(b, spec)
    tol = max(margin * max(sn_a, sn_b), eps)
    d = distance(a, b, spec)
    return {
        "agree": bool(d.value <= tol),
        "distance": d.value,
        "tol": tol,
        "metric": metric,
    }


def verdict(cand, gts: list, spec: Spec, margin: float = 2.0) -> dict:
    """Judge a candidate against k GT runs under the spec.

    `cand` and every element of `gts` are canonical (from `canonicalize`).
    The GT runs both set the noise floor (tolerance) and check the problem's
    own consistency: an out-of-cap floor flags the problem ill-posed.

    Requires len(gts) >= 2; k=1 silently zeroes the floor and produces
    false negatives for any stochastic answer.
    """
    if len(gts) < 2:
        raise AlgebraError("verdict requires at least 2 GT runs")
    if spec.kind == "record":
        fmap = spec.field_map()
        fields = {}
        for n, f in fmap.items():
            try:
                sub_gts = [g.field_map()[n] for g in gts]
            except KeyError:
                raise AlgebraError(f"record GT missing field {n!r}")
            try:
                sub_cand = cand.field_map()[n]
            except KeyError:
                raise AlgebraError(f"record candidate missing field {n!r}")
            fields[n] = verdict(sub_cand, sub_gts, f, margin)
        # aggregate worst-case scalars for uniform consumer access
        worst_dist = max(v["distance"] for v in fields.values())
        worst_floor = max(v["floor"] for v in fields.values())
        worst_tol = max(v["tol"] for v in fields.values())
        return {
            "passed": all(v["passed"] for v in fields.values()),
            "ill_posed": any(v["ill_posed"] for v in fields.values()),
            "metric": "record",
            "distance": worst_dist,
            "floor": worst_floor,
            "tol": worst_tol,
            "fields": fields,
        }
    return _leaf_verdict(cand, gts, spec, margin)


def answer_to_dict(canon, *, max_samples: int | None = None) -> dict:
    """Canonical representation → native wire dict (inverse of canonicalize).

    Round-trips: canonicalize(answer_to_dict(c), spec) reproduces c (Cloud
    modulo max_samples truncation, which is presentation policy for exports).
    """
    if isinstance(canon, EnumDist):
        return {"kind": "dist_enum", "support": list(canon.support),
                "probs": list(canon.probs)}
    if isinstance(canon, ParamDist):
        return {"kind": "dist_param", "family": canon.family,
                "params": dict(canon.params)}
    if isinstance(canon, Cloud):
        samples = list(canon.samples)
        if max_samples is not None:
            samples = samples[:max_samples]
        return {"kind": "cloud", "samples": samples}
    if isinstance(canon, Rec):
        return {"kind": "record",
                "fields": {n: answer_to_dict(v, max_samples=max_samples)
                           for n, v in canon.fields}}
    if isinstance(canon, Exact):
        return {"kind": "exact", "value": canon.value}
    raise AlgebraError(f"not a canonical answer: {type(canon).__name__}")


def status_of(v: dict) -> str:
    """Verdict dict → status string. ill_posed wins over pass/fail."""
    if v.get("ill_posed"):
        return "ill_posed"
    return "pass" if v.get("passed") else "fail"


def judge(cand_raw, gts: list, spec: Spec, margin: float = 2.0) -> dict:
    """Canonicalize `cand_raw` and judge it against pre-canonical GT runs.

    Single entry point for harness/web consumers. Returns a dict with:
      {"status": "malformed", "error": <str>}          — canonicalization failed
      {"status": "ill_posed"|"pass"|"fail", **verdict}  — ill_posed wins

    `gts` are already canonical (from `canonicalize`).
    """
    try:
        cand = canonicalize(cand_raw, spec)
    except AlgebraError as e:
        return {"status": "malformed", "error": str(e)}
    v = verdict(cand, gts, spec, margin)
    return {"status": status_of(v), **v}
