"""Side-by-side validation: spec_metrics vs legacy metrics on scored data.

For a given scored.jsonl, recompute comparison via:
  (1) legacy `eval.metrics.compare_by_shape(gen, gt, atom.answer_shape)`
  (2) new    `eval.spec_metrics.compare_by_spec(gen, gt, spec_for_atom)`

Report per-atom metric diffs (legacy TV vs new TV, KS-only-in-new, etc.)
plus aggregate stats.

This script is calibration scaffolding. It does not modify the harness,
the scored files, or the atom files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from eval.io import load_jsonl
from eval.metrics import compare_by_shape, collect_metrics
from eval.spec_metrics import compare_by_spec, collect_metrics_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scored', required=True, help='Path to scored.jsonl')
    parser.add_argument('--atoms', required=True, help='Path to the matching atoms jsonl')
    parser.add_argument('--specs', default='data/atom_specs.jsonl', help='atom_specs.jsonl')
    parser.add_argument('--limit', type=int, default=0, help='only process first N atoms (0=all)')
    args = parser.parse_args()

    atoms = {a['id']: a for a in load_jsonl(args.atoms)}
    specs = {s['atom_id']: s['spec'] for s in load_jsonl(args.specs)}

    print(f'Loaded {len(atoms)} atoms, {len(specs)} specs.')

    legacy_metrics = defaultdict(list)
    new_metrics = defaultdict(list)
    legacy_keys = Counter()
    new_keys = Counter()
    diffs_per_atom = []
    n_processed = 0
    n_no_gen = 0
    n_no_spec = 0

    for row in load_jsonl(args.scored):
        if not row.get('id'):
            continue
        aid = row['id']
        atom = atoms.get(aid)
        spec = specs.get(aid)
        if not atom:
            continue
        if not spec:
            n_no_spec += 1
            continue
        eval_block = row.get('evaluation') or {}
        gen_block = eval_block.get('gen') or {}
        gen = gen_block.get('answer')
        gt = atom.get('groundtruth_output')
        if gen is None or not gen_block.get('executed', True):
            n_no_gen += 1
            continue
        try:
            legacy = compare_by_shape(gen, gt, atom['answer_shape'])
            new = compare_by_spec(gen, gt, spec)
        except Exception as e:
            print(f'  ERROR on {aid}: {type(e).__name__}: {e}')
            continue

        lm = collect_metrics(legacy)
        nm = collect_metrics_spec(new)
        for k, v in lm.items():
            legacy_metrics[k.split('.')[-1]].append(v)
            legacy_keys[k] += 1
        for k, v in nm.items():
            new_metrics[k.split('.')[-1]].append(v)
            new_keys[k] += 1

        # per-atom diff on shared TV signal
        ltv = lm.get('tv', lm.get('rain.tv', None))
        ntv = nm.get('tv', nm.get('rain.tv', None))
        if ltv is not None and ntv is not None:
            diff = abs(ltv - ntv)
            if diff > 1e-6:
                diffs_per_atom.append({
                    'atom_id': aid, 'old_tv': ltv, 'new_tv': ntv, 'diff': diff,
                    'spec_role': spec.get('role'),
                    'spec_support': spec.get('support'),
                })

        n_processed += 1
        if args.limit and n_processed >= args.limit:
            break

    print(f'\nProcessed {n_processed} scored atoms.')
    if n_no_spec:
        print(f'  ({n_no_spec} had no spec)')
    if n_no_gen:
        print(f'  ({n_no_gen} had no generated_output)')

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    print('\nMetric aggregates (mean across atoms):')
    all_keys = sorted(set(legacy_metrics) | set(new_metrics))
    print(f"  {'metric':14s}  {'legacy':>12s}  {'new':>12s}  {'n_legacy':>10s}  {'n_new':>10s}")
    for k in all_keys:
        lv = _mean(legacy_metrics.get(k, []))
        nv = _mean(new_metrics.get(k, []))
        print(f"  {k:14s}  {str(round(lv,4)) if lv is not None else '-':>12s}  "
              f"{str(round(nv,4)) if nv is not None else '-':>12s}  "
              f"{len(legacy_metrics.get(k, [])):>10d}  {len(new_metrics.get(k, [])):>10d}")

    print('\nTop 10 per-atom TV diffs (where legacy and new disagree by >1e-6):')
    for d in sorted(diffs_per_atom, key=lambda x: -x['diff'])[:10]:
        print(f"  {d['atom_id']:60s}  old_tv={d['old_tv']:.4f}  new_tv={d['new_tv']:.4f}  diff={d['diff']:.4f}  "
              f"({d['spec_role']}/{d['spec_support']})")

    if not diffs_per_atom:
        print('  (none — TV metrics agree on all comparable atoms)')


if __name__ == '__main__':
    main()
