"""Spec-aware comparator (phase 2 of the output-taxonomy refactor).

Dispatches on `output_spec` (from `scripts/classify_atom_specs.py`) rather
than the legacy 4-shape `answer_shape` enum. Lives parallel to
`eval/metrics.py` so it can be validated against the legacy comparator on
existing scored data before any harness rewiring.

Spec fields read here:
  role        : deterministic | summary | distribution | samples
              | trajectory | record
  domain      : discrete_finite | discrete_large | continuous_1d
              | continuous_nd | structured
  support     : enumerated | empirical | parametric | implicit
  dtype       : bool | int | float | string | vector | structured | mixed
  equiv       : {method: ..., threshold/rtol/atol: ...}
  fields      : {name: <spec>, ...}    (records only)
  family,repr : for parametric distributions

Returns a comparison tree with the same outer shape as `compare_by_shape`:
  {shape: ..., ok?, error?, tv?, kl?, exact_match?, approx_match?, ...}

Note: this is a working comparator that uses Wasserstein, KS, and
parametric-match metrics not present in legacy metrics.py. The trade-off
vs full reimplementation of legacy: shared helpers (`empirical_tv`,
`_normalize_dist`, etc.) are reused from `eval.metrics` to avoid
divergence — the new logic only adds what's missing.
"""

from __future__ import annotations

import re
from typing import Any

from eval.metrics import (
    _normalize_dist,
    _kl,
    _tv,
    _distribution_to_samples,
    empirical_tv,
    _looks_like_distribution,
    value_match,
)


def _normalize_bool_int(x: Any) -> Any:
    """Coerce 0.0/1.0 from Pyro into True/False to match WebPPL bools."""
    if isinstance(x, float) and x in (0.0, 1.0):
        return bool(x)
    return x


def _normalize_support(support: list, dtype: str) -> list:
    if dtype == 'bool':
        return [_normalize_bool_int(s) for s in support]
    return support


_FAMILY_PARAM_ALIASES = {
    # Map cross-PPL parameter names back to a canonical family form.
    'Beta':    {'a': 'a', 'b': 'b', 'concentration1': 'a', 'concentration0': 'b'},
    'Normal':  {'mu': 'mu', 'sigma': 'sigma', 'loc': 'mu', 'scale': 'sigma'},
    'Gamma':   {'shape': 'shape', 'rate': 'rate', 'scale': 'scale',
                'concentration': 'shape'},
    'Uniform': {'a': 'a', 'b': 'b', 'low': 'a', 'high': 'b'},
}


_REPR_RE = re.compile(r'^(\w+)\s*\(\s*\{?\s*(.*?)\s*\}?\s*\)\s*$')


def _parse_parametric_repr(repr_s: str) -> tuple[str, dict[str, float]]:
    """Parse 'Beta({ a: 10, b: 10 })' or 'Beta(a=10, b=10)' into (family, params)."""
    m = _REPR_RE.match(repr_s.strip())
    if not m:
        return ('unknown', {})
    family = m.group(1)
    body = m.group(2)
    params: dict[str, float] = {}
    # tolerate both `a: 10` and `a=10` separators
    for part in re.split(r',', body):
        part = part.strip()
        if not part:
            continue
        kv = re.split(r'[:=]', part, maxsplit=1)
        if len(kv) != 2:
            continue
        k, v = kv[0].strip().strip('"').strip("'"), kv[1].strip()
        try:
            params[k] = float(v)
        except ValueError:
            continue
    return (family, params)


def _canonicalize_params(family: str, params: dict[str, float]) -> dict[str, float]:
    aliases = _FAMILY_PARAM_ALIASES.get(family, {})
    return {aliases.get(k, k): v for k, v in params.items()}


def cmp_parametric(gen: Any, gt: Any, equiv: dict) -> dict:
    """Compare two parametric continuous distributions by canonical params."""
    if not (isinstance(gen, dict) and gen.get('__kind') == 'distribution_continuous'):
        return {'shape': 'parametric', 'ok': False, 'error': 'gen not parametric continuous'}
    if not (isinstance(gt, dict) and gt.get('__kind') == 'distribution_continuous'):
        return {'shape': 'parametric', 'ok': False, 'error': 'gt not parametric continuous'}
    fg, pg = _parse_parametric_repr(gen.get('repr', ''))
    ft, pt = _parse_parametric_repr(gt.get('repr', ''))
    if fg != ft:
        return {'shape': 'parametric', 'family_match': False,
                'gen_family': fg, 'gt_family': ft}
    cg = _canonicalize_params(fg, pg)
    ct = _canonicalize_params(ft, pt)
    rtol = equiv.get('rtol', 0.01)
    keys = set(cg) | set(ct)
    diffs = {}
    matched = True
    for k in keys:
        a, b = cg.get(k), ct.get(k)
        if a is None or b is None:
            matched = False
            diffs[k] = {'gen': a, 'gt': b}
            continue
        ref = max(abs(b), 1e-12)
        rel = abs(a - b) / ref
        diffs[k] = {'gen': a, 'gt': b, 'rel': rel}
        if rel > rtol:
            matched = False
    return {'shape': 'parametric', 'family': fg, 'family_match': True,
            'params_match': matched, 'param_diffs': diffs}


def _wasserstein1_1d(p_support: list, p_probs: list,
                     q_support: list, q_probs: list) -> float:
    """Discrete W_1 via CDF integration for 1d numeric supports.

    Single-sweep O(n log n) implementation: sort once, then advance
    pointers monotonically along the merged support to maintain running
    CDFs. Naive nested-CDF was O(n²) and blew up at support_size~24k.
    """
    p_items = sorted((float(v), w) for v, w in zip(p_support, p_probs))
    q_items = sorted((float(v), w) for v, w in zip(q_support, q_probs))
    all_pts = sorted({v for v, _ in p_items} | {v for v, _ in q_items})
    if len(all_pts) < 2:
        return 0.0
    i_p = i_q = 0
    cum_p = cum_q = 0.0
    total = 0.0
    for a, b in zip(all_pts, all_pts[1:]):
        while i_p < len(p_items) and p_items[i_p][0] <= a:
            cum_p += p_items[i_p][1]
            i_p += 1
        while i_q < len(q_items) and q_items[i_q][0] <= a:
            cum_q += q_items[i_q][1]
            i_q += 1
        total += abs(cum_p - cum_q) * (b - a)
    return total


def _wasserstein1_distribution(gen_dict: dict, gt_dict: dict) -> float | None:
    gs, gp = gen_dict.get('support') or [], gen_dict.get('probs') or []
    ts, tp = gt_dict.get('support') or [], gt_dict.get('probs') or []
    if not gs or not ts or len(gs) != len(gp) or len(ts) != len(tp):
        return None
    try:
        gs_f = [float(v) for v in gs]
        ts_f = [float(v) for v in ts]
    except (TypeError, ValueError):
        return None
    return _wasserstein1_1d(gs_f, gp, ts_f, tp)


def _ks_1d(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        return 1.0
    xs, ys = sorted(xs), sorted(ys)
    nx, ny = len(xs), len(ys)
    i = j = 0
    cx = cy = 0.0
    d = 0.0
    pts = sorted(set(xs) | set(ys))
    for p in pts:
        while i < nx and xs[i] <= p:
            i += 1
        while j < ny and ys[j] <= p:
            j += 1
        cx, cy = i / nx, j / ny
        d = max(d, abs(cx - cy))
    return d


def _cmp_distribution_enumerated(gen, gt, spec) -> dict:
    if not (_looks_like_distribution(gen) and _looks_like_distribution(gt)):
        return {'shape': 'distribution', 'ok': False, 'error': 'not a distribution'}
    # Bool/int normalization on support
    dtype = spec.get('dtype')
    if dtype == 'bool':
        gen = {**gen, 'support': _normalize_support(gen.get('support') or [], 'bool')}
        gt = {**gt, 'support': _normalize_support(gt.get('support') or [], 'bool')}
    p = _normalize_dist(gen)
    q = _normalize_dist(gt)
    if p is None or q is None:
        return {'shape': 'distribution', 'ok': False, 'error': 'empty distribution'}
    return {'shape': 'distribution', 'kl': _kl(p, q), 'tv': _tv(p, q)}


def _cmp_distribution_empirical_continuous(gen, gt, spec) -> dict:
    if not (_looks_like_distribution(gen) and _looks_like_distribution(gt)):
        return {'shape': 'distribution', 'ok': False, 'error': 'not a distribution'}
    w = _wasserstein1_distribution(gen, gt)
    p = _normalize_dist(gen)
    q = _normalize_dist(gt)
    tv = _tv(p, q) if (p and q) else None
    return {'shape': 'distribution', 'w1': w, 'tv': tv,
            'metric': 'wasserstein1'}


def _cmp_distribution_empirical_structured(gen, gt, spec) -> dict:
    # Best we can do without a structure metric: report TV (will be near 1
    # if supports don't overlap) and mark for future Wasserstein-on-structures.
    if not (_looks_like_distribution(gen) and _looks_like_distribution(gt)):
        return {'shape': 'distribution', 'ok': False, 'error': 'not a distribution'}
    p = _normalize_dist(gen)
    q = _normalize_dist(gt)
    if p is None or q is None:
        return {'shape': 'distribution', 'ok': False, 'error': 'empty distribution'}
    overlap = len(set(p) & set(q)) / max(1, len(set(p) | set(q)))
    return {'shape': 'distribution', 'tv': _tv(p, q), 'kl': _kl(p, q),
            'support_jaccard': overlap,
            'metric': 'tv+jaccard',
            'warning': 'large empirical support; Wasserstein-on-structures not implemented'}


def _cmp_samples_iid(gen, gt, spec) -> dict:
    if _looks_like_distribution(gen):
        gen = _distribution_to_samples(gen) or gen
    if _looks_like_distribution(gt):
        gt = _distribution_to_samples(gt) or gt
    if not isinstance(gen, list) or not isinstance(gt, list):
        return {'shape': 'samples', 'ok': False, 'error': 'samples must be lists'}
    dtype = spec.get('dtype')
    if dtype == 'bool':
        gen = [_normalize_bool_int(x) for x in gen]
        gt = [_normalize_bool_int(x) for x in gt]
    if dtype == 'float':
        try:
            gen_f = [float(x) for x in gen]
            gt_f = [float(x) for x in gt]
            return {'shape': 'samples', 'n_gen': len(gen), 'n_gt': len(gt),
                    'tv': empirical_tv(gen, gt), 'ks': _ks_1d(gen_f, gt_f),
                    'metric': 'tv+ks'}
        except (TypeError, ValueError):
            pass
    return {'shape': 'samples', 'n_gen': len(gen), 'n_gt': len(gt),
            'tv': empirical_tv(gen, gt)}


def _cmp_trajectory(gen, gt, spec) -> dict:
    if not isinstance(gen, list) or not isinstance(gt, list):
        return {'shape': 'trajectory', 'ok': False, 'error': 'trajectory must be lists'}
    max_steps = max(
        max((len(t) for t in gen if isinstance(t, list)), default=0),
        max((len(t) for t in gt if isinstance(t, list)), default=0),
    )
    per_step_tv = []
    for k in range(min(max_steps, 20)):  # cap at 20 steps for cost
        gen_at_k = [t[k] for t in gen if isinstance(t, list) and k < len(t)]
        gt_at_k = [t[k] for t in gt if isinstance(t, list) and k < len(t)]
        if not gen_at_k or not gt_at_k:
            continue
        per_step_tv.append(empirical_tv(gen_at_k, gt_at_k))
    # Also compare length distribution
    gen_lens = [len(t) if isinstance(t, list) else 0 for t in gen]
    gt_lens = [len(t) if isinstance(t, list) else 0 for t in gt]
    length_tv = empirical_tv(gen_lens, gt_lens)
    mean_step_tv = sum(per_step_tv) / len(per_step_tv) if per_step_tv else None
    return {'shape': 'trajectory', 'n_gen': len(gen), 'n_gt': len(gt),
            'length_tv': length_tv, 'mean_step_tv': mean_step_tv,
            'per_step_tv': per_step_tv,
            'metric': 'stepwise_marginal_tv'}


def _cmp_value(gen, gt, spec) -> dict:
    dtype = spec.get('dtype')
    equiv = spec.get('equiv') or {}
    method = equiv.get('method', 'exact')

    if dtype == 'bool':
        gen = _normalize_bool_int(gen)
        gt = _normalize_bool_int(gt)

    if method == 'exact':
        return {'shape': 'value', 'exact_match': gen == gt, 'approx_match': gen == gt}
    if method == 'tolerance':
        rtol = equiv.get('rtol', 0.05)
        return {'shape': 'value', **value_match(gen, gt, rtol=rtol)}
    return {'shape': 'value', 'ok': False, 'error': f'unknown equiv method: {method}'}


def compare_by_spec(gen: Any, gt: Any, spec: dict) -> dict:
    """Compare two answers under an output_spec. Mirrors compare_by_shape."""
    role = spec.get('role')

    if role == 'record':
        fields = spec.get('fields') or {}
        if not isinstance(gen, dict) or not isinstance(gt, dict):
            return {'shape': 'record', 'ok': False, 'error': 'non-record answer'}
        return {'shape': 'record', 'fields': {
            fname: compare_by_spec(gen.get(fname), gt.get(fname), fspec)
            for fname, fspec in fields.items()
        }}

    if role == 'distribution':
        support = spec.get('support')
        domain = spec.get('domain')
        if support == 'parametric':
            return cmp_parametric(gen, gt, spec.get('equiv') or {})
        if support == 'empirical' and domain == 'continuous_1d':
            return _cmp_distribution_empirical_continuous(gen, gt, spec)
        if support == 'empirical' and domain in ('discrete_large', 'structured'):
            return _cmp_distribution_empirical_structured(gen, gt, spec)
        # enumerated (discrete_finite or discrete_large)
        return _cmp_distribution_enumerated(gen, gt, spec)

    if role == 'samples':
        return _cmp_samples_iid(gen, gt, spec)

    if role == 'trajectory':
        return _cmp_trajectory(gen, gt, spec)

    if role in ('deterministic', 'summary'):
        return _cmp_value(gen, gt, spec)

    return {'shape': str(role), 'ok': False, 'error': f'unknown role: {role}'}


def collect_metrics_spec(comparison: dict) -> dict:
    out: dict = {}
    def walk(node, prefix: str):
        if not isinstance(node, dict):
            return
        shape = node.get('shape')
        if shape == 'record':
            for fname, fnode in (node.get('fields') or {}).items():
                walk(fnode, f'{prefix}{fname}.')
            return
        if shape == 'distribution':
            if node.get('kl') is not None:
                out[prefix + 'kl'] = node['kl']
            if node.get('tv') is not None:
                out[prefix + 'tv'] = node['tv']
            if node.get('w1') is not None:
                out[prefix + 'w1'] = node['w1']
        elif shape == 'samples':
            if node.get('tv') is not None:
                out[prefix + 'tv'] = node['tv']
            if node.get('ks') is not None:
                out[prefix + 'ks'] = node['ks']
        elif shape == 'value':
            out[prefix + 'exact'] = 1.0 if node.get('exact_match') else 0.0
            out[prefix + 'approx'] = 1.0 if node.get('approx_match') else 0.0
        elif shape == 'parametric':
            out[prefix + 'family_match'] = 1.0 if node.get('family_match') else 0.0
            out[prefix + 'params_match'] = 1.0 if node.get('params_match') else 0.0
        elif shape == 'trajectory':
            if node.get('length_tv') is not None:
                out[prefix + 'length_tv'] = node['length_tv']
            if node.get('mean_step_tv') is not None:
                out[prefix + 'mean_step_tv'] = node['mean_step_tv']
    walk(comparison, '')
    return out
