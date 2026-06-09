"""Phase 1: classify each atom under the proposed output_spec schema.

Reads all atoms in the four collections, applies deterministic heuristics
to produce an output_spec per atom, writes data/atom_specs.jsonl, and
emits data/_unclassifiable.md listing atoms flagged for human review.

This is calibration scaffolding, not a production tool. The schema:

  role        = deterministic | summary | distribution | sample | samples
              | trajectory | record | unknown
  domain      = discrete_finite | discrete_large | continuous_1d
              | continuous_nd | structured | mixed
  container   = scalar | vector | matrix | object | list
  support     = enumerated | empirical | parametric | implicit  (distributions)
  dtype       = bool | int | float | string | vector | structured | mixed
  equiv       = {method: ..., threshold/rtol/atol: ...}
  fields      = {field_name: <spec>, ...} (records only)
  review_note = (optional) human-readable flag for borderline cases

Heuristics are deliberately conservative: low-confidence cases get a
`review_note` so they show up in _unclassifiable.md and don't pretend the
classifier was certain.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.io import load_jsonl

COLLECTIONS = [
    'data/atomized_v2.jsonl',
    'data/curated_v3/dippl.jsonl',
    'data/curated_v3/forestdb.jsonl',
    'data/pyro_v3/probmods.jsonl',
]

ENUMERATED_MAX = 50          # support <= this is "small enumerated"
LARGE_ENUMERATED_MAX = 500   # support <= this is "large enumerated"


# ---------------------------------------------------------------------------
# Distribution classification
# ---------------------------------------------------------------------------

def _support_dtype(support: list) -> str:
    if not support:
        return 'empty'
    # bool first (bool is subclass of int)
    if all(isinstance(s, bool) for s in support):
        return 'bool'
    if all(isinstance(s, int) and not isinstance(s, bool) for s in support):
        return 'int'
    if all(isinstance(s, str) for s in support):
        return 'string'
    if all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in support):
        return 'float'
    if all(isinstance(s, list) for s in support):
        return 'vector'
    if all(isinstance(s, dict) for s in support):
        return 'structured'
    return 'mixed'


def classify_distribution(out) -> dict | None:
    """Inspect a distribution-shaped dict and produce a spec.

    Handles both `__kind:distribution` (enumerated support+probs) and
    `__kind:distribution_continuous` (parametric, e.g. `Beta({a:10,b:10})`).
    """
    if not isinstance(out, dict):
        return None

    if out.get('__kind') == 'distribution_continuous':
        repr_s = out.get('repr', '')
        family = repr_s.split('(')[0].strip() if '(' in repr_s else 'unknown'
        return {
            'role': 'distribution',
            'domain': 'continuous_1d',
            'support': 'parametric',
            'dtype': 'float',
            'family': family,
            'repr': repr_s,
            'equiv': {'method': 'parametric_match', 'rtol': 0.01},
            'review_note': 'parametric continuous distribution — comparator needs family-aware param match; cross-PPL param-name remap required',
        }

    if not (isinstance(out.get('support'), list) and isinstance(out.get('probs'), list)):
        return None

    support = out['support']
    n = len(support)
    dtype = _support_dtype(support)

    if dtype == 'float' and n > ENUMERATED_MAX:
        return {
            'role': 'distribution',
            'domain': 'continuous_1d',
            'support': 'empirical',
            'dtype': 'float',
            'support_size': n,
            'equiv': {'method': 'ks_marginal', 'threshold': 0.15},
            'review_note': 'large-support float-valued distribution — likely MCMC posterior; ks_marginal is a placeholder metric',
        }

    if n <= ENUMERATED_MAX:
        support_kind = 'enumerated'
        domain = 'discrete_finite' if dtype not in ('vector', 'structured') else 'structured'
        equiv = {'method': 'tv', 'threshold': 0.05}
    elif n <= LARGE_ENUMERATED_MAX:
        support_kind = 'enumerated'
        domain = 'discrete_large' if dtype not in ('vector', 'structured') else 'structured'
        equiv = {'method': 'tv', 'threshold': 0.1, 'alignment': 'permissive'}
    else:
        support_kind = 'empirical'
        domain = 'discrete_large' if dtype not in ('vector', 'structured') else 'structured'
        equiv = {'method': 'wasserstein', 'threshold': 0.15}

    spec = {
        'role': 'distribution',
        'domain': domain,
        'support': support_kind,
        'dtype': dtype,
        'support_size': n,
        'equiv': equiv,
    }
    if support_kind == 'empirical':
        spec['review_note'] = 'empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder'
    elif support_kind == 'enumerated' and n > ENUMERATED_MAX:
        spec['review_note'] = f'large enumerated support ({n}) — TV with permissive alignment; verify metric choice'
    return spec


# ---------------------------------------------------------------------------
# Value classification
# ---------------------------------------------------------------------------

def classify_value(out) -> dict:
    """Inspect a value-shape output; map to deterministic/summary/unknown."""
    # bool first since bool is subclass of int
    if isinstance(out, bool):
        return {'role': 'deterministic', 'container': 'scalar', 'dtype': 'bool',
                'equiv': {'method': 'exact'}}
    if isinstance(out, int):
        return {'role': 'deterministic', 'container': 'scalar', 'dtype': 'int',
                'equiv': {'method': 'exact'}}
    if isinstance(out, float):
        return {
            'role': 'summary',
            'container': 'scalar', 'dtype': 'float',
            'equiv': {'method': 'tolerance', 'rtol': 0.05},
            'review_note': 'scalar float — could be a deterministic closed-form value or a posterior summary; default tolerance assumes summary',
        }
    if isinstance(out, str):
        return {'role': 'deterministic', 'container': 'scalar', 'dtype': 'string',
                'equiv': {'method': 'exact'}}
    if isinstance(out, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in out):
            return {
                'role': 'deterministic',
                'container': 'vector', 'dtype': 'float', 'length': len(out),
                'equiv': {'method': 'tolerance', 'rtol': 0.05},
                'review_note': 'numeric vector classified as deterministic — verify GT code is not stochastic',
            }
        return {'role': 'unknown',
                'review_note': f'value-shape non-numeric list of length {len(out)}'}
    if isinstance(out, dict):
        # case A: dict is a single distribution
        if 'probs' in out and 'support' in out:
            spec = classify_distribution(out) or {}
            spec['review_note'] = 'declared value-shape but output is a __kind:distribution — likely curation-time mislabel; should be answer_shape="distribution"'
            return spec
        if out.get('__kind') == 'distribution_continuous':
            spec = classify_distribution(out) or {}
            spec['review_note'] = 'declared value-shape but output is parametric continuous — likely curation-time mislabel; should be answer_shape="distribution"'
            return spec
        # case B: dict is a bundle of heterogeneous fields → auto-record
        fields = {fname: classify_atom_value(fval) for fname, fval in out.items()}
        return {
            'role': 'record',
            'fields': fields,
            'review_note': 'declared value-shape but output is a dict bundle — likely curation-time mislabel; should be answer_shape={"record": {...}}',
        }
    return {'role': 'unknown',
            'review_note': f'unrecognized value-shape output type: {type(out).__name__}'}


def classify_atom_value(out) -> dict:
    """Classify a value at unknown declared shape — used for auto-record fields.
    Picks distribution if the value looks distributional, else falls through to
    the scalar/vector value rules.
    """
    if isinstance(out, dict):
        if 'probs' in out and 'support' in out:
            return classify_distribution(out) or {'role': 'unknown'}
        if out.get('__kind') == 'distribution_continuous':
            return classify_distribution(out) or {'role': 'unknown'}
    return classify_value(out)


# ---------------------------------------------------------------------------
# Samples classification
# ---------------------------------------------------------------------------

def classify_samples(out) -> dict:
    """Inspect a samples-shape output.

    Cached samples GT is a list of N per-seed runs. Per-run output type
    drives the spec; trajectory vs samples ambiguity is flagged for review.
    """
    if not isinstance(out, list) or not out:
        return {'role': 'unknown', 'review_note': 'samples-shape but output is not a non-empty list'}

    per_run = out[0]
    n_runs = len(out)

    if isinstance(per_run, bool):
        return {'role': 'samples', 'domain': 'discrete_finite', 'container': 'scalar',
                'dtype': 'bool', 'n_runs': n_runs,
                'equiv': {'method': 'tv', 'threshold': 0.15}}
    if isinstance(per_run, int) and not isinstance(per_run, bool):
        return {'role': 'samples', 'domain': 'discrete_finite', 'container': 'scalar',
                'dtype': 'int', 'n_runs': n_runs,
                'equiv': {'method': 'tv', 'threshold': 0.15}}
    if isinstance(per_run, float):
        return {'role': 'samples', 'domain': 'continuous_1d', 'container': 'scalar',
                'dtype': 'float', 'n_runs': n_runs,
                'equiv': {'method': 'ks', 'threshold': 0.15},
                'review_note': 'per-run float samples — KS is a placeholder; verify metric for continuous-1d'}
    if isinstance(per_run, str):
        return {'role': 'samples', 'domain': 'discrete_finite', 'container': 'scalar',
                'dtype': 'string', 'n_runs': n_runs,
                'equiv': {'method': 'tv', 'threshold': 0.15}}

    if isinstance(per_run, list):
        lens = {len(r) if isinstance(r, list) else None for r in out}
        if len(lens) == 1 and None not in lens:
            return {
                'role': 'samples',
                'domain': 'structured', 'container': 'vector',
                'n_runs': n_runs, 'vector_length': lens.pop(),
                'equiv': {'method': 'tv', 'threshold': 0.15},
                'review_note': 'per-run output is fixed-length list — ambiguous between vector-valued sample and repeated trajectory',
            }
        return {
            'role': 'trajectory',
            'domain': 'structured', 'container': 'list',
            'n_runs': n_runs,
            'equiv': {'method': 'ks_marginal_stepwise', 'threshold': 0.15},
            'review_note': 'per-run outputs are variable-length lists — likely trajectory; metric is placeholder',
        }

    if isinstance(per_run, dict):
        return {'role': 'samples', 'domain': 'structured', 'container': 'object',
                'n_runs': n_runs,
                'equiv': {'method': 'tv', 'threshold': 0.15},
                'review_note': 'per-run output is dict — TV on JSON-encoded key works only for exact match'}

    return {'role': 'unknown', 'review_note': f'samples-shape with unrecognized per-run type {type(per_run).__name__}'}


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def classify_atom(answer_shape, groundtruth_output) -> dict:
    if isinstance(answer_shape, dict) and 'record' in answer_shape:
        fields = {}
        for fname, fshape in answer_shape['record'].items():
            field_out = groundtruth_output.get(fname) if isinstance(groundtruth_output, dict) else None
            fields[fname] = classify_atom(fshape, field_out)
        return {'role': 'record', 'fields': fields}

    if answer_shape == 'distribution':
        spec = classify_distribution(groundtruth_output)
        if spec is None:
            return {'role': 'unknown',
                    'review_note': 'declared distribution but output is not a {probs,support} dict'}
        return spec
    if answer_shape == 'value':
        return classify_value(groundtruth_output)
    if answer_shape == 'samples':
        return classify_samples(groundtruth_output)
    return {'role': 'unknown', 'review_note': f'unhandled answer_shape: {answer_shape}'}


def walk_review(atom_id: str, collection: str, spec: dict, items: list) -> None:
    """Collect review_note from a spec tree (records walked recursively)."""
    if spec.get('role') == 'record':
        for fname, fspec in spec.get('fields', {}).items():
            walk_review(f'{atom_id}.{fname}', collection, fspec, items)
        return
    if 'review_note' in spec or spec.get('role') == 'unknown':
        items.append({
            'atom_id': atom_id,
            'collection': collection,
            'role': spec.get('role'),
            'old_shape': spec.get('_old_shape', '?'),
            'note': spec.get('review_note', '(role=unknown, no note)'),
            'support_size': spec.get('support_size'),
            'dtype': spec.get('dtype'),
        })


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    specs_out = []
    review_items = []
    role_counter = Counter()

    for rel in COLLECTIONS:
        for atom in load_jsonl(repo / rel):
            spec = classify_atom(atom.get('answer_shape'), atom.get('groundtruth_output'))
            record = {
                'atom_id': atom['id'],
                'collection': rel,
                'old_shape': atom.get('answer_shape'),
                'spec': spec,
            }
            specs_out.append(record)
            walk_review(atom['id'], rel, spec, review_items)
            if spec.get('role') == 'record':
                for fspec in spec['fields'].values():
                    role_counter[f"record.{fspec.get('role')}"] += 1
            else:
                role_counter[spec.get('role')] += 1

    out_specs = repo / 'data' / 'atom_specs.jsonl'
    with out_specs.open('w') as f:
        for s in specs_out:
            f.write(json.dumps(s) + '\n')

    out_review = repo / 'data' / '_unclassifiable.md'
    with out_review.open('w') as f:
        f.write('# Atom Spec Calibration — Review List\n\n')
        f.write(f'Generated from `scripts/classify_atom_specs.py`.\n\n')
        f.write(f'**Total atoms across collections**: {len(specs_out)}.\n')
        f.write(f'**Items flagged for human review**: {len(review_items)}.\n\n')
        f.write('Each item is either:\n')
        f.write('- `role=unknown` — the script could not assign a role.\n')
        f.write('- has a `review_note` — assigned but with a flagged concern (metric placeholder, '
                'curation-time misclassification, ambiguous trajectory vs samples, etc.).\n\n')

        f.write('## Role distribution (after classification)\n\n')
        f.write('| Role | Count |\n|---|---|\n')
        for role, n in sorted(role_counter.items(), key=lambda x: -x[1]):
            f.write(f'| `{role}` | {n} |\n')
        f.write('\n')

        # Group review items by note category for digestibility
        by_note: dict[str, list] = {}
        for item in review_items:
            key = item['note'].split(' — ')[0].split(' -- ')[0]
            by_note.setdefault(key, []).append(item)

        f.write('## Review items, grouped by issue\n\n')
        for note_key in sorted(by_note.keys(), key=lambda k: -len(by_note[k])):
            items = by_note[note_key]
            f.write(f'### {note_key}  ({len(items)})\n\n')
            for item in items:
                meta = []
                if item.get('support_size') is not None:
                    meta.append(f'support={item["support_size"]}')
                if item.get('dtype') is not None:
                    meta.append(f'dtype={item["dtype"]}')
                if item.get('role') is not None:
                    meta.append(f'role={item["role"]}')
                meta_s = (' (' + ', '.join(meta) + ')') if meta else ''
                f.write(f'- `{item["atom_id"]}`{meta_s}  — {item["note"]}\n')
            f.write('\n')

    print(f'Wrote {len(specs_out)} specs to {out_specs.relative_to(repo)}')
    print(f'Wrote {len(review_items)} review items to {out_review.relative_to(repo)}')
    print()
    print('Role distribution:')
    for k, v in sorted(role_counter.items(), key=lambda x: -x[1]):
        print(f'  {v:4d}  {k}')


if __name__ == '__main__':
    main()
