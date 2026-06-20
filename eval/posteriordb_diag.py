"""Per-field crosscheck diagnostics for posteriordb problems.

`gate crosscheck` reports one status per problem; when a problem is ill_posed or
fails, this pins *which parameter* is responsible and *which side* carries the
noise — our sampled `stan` column (fixable by heavier / reference-matched
sampling) or the gold `reference` posterior itself (genuinely non-discriminable
→ respec or exclude). Reuses the gate's cached GT collection, so it is cheap
once a crosscheck has run.
"""

from __future__ import annotations

import argparse
import statistics

from eval.algebra import (_W1_FLOOR_CAP_FRAC, _pooled_spread, distance,
                          noise_floor, parse_spec)
from eval.corpus import load_problems, load_realizations
from eval.config import DEFAULT_SEED, DEFAULT_TIMEOUT


def _real_by_id(language: str) -> dict:
    return {r["problem_id"]: r for r in load_realizations(language)}


def diagnose(pid: str, *, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Per-field floor/spread/cross-distance breakdown for one problem."""
    from eval.gate import collect_gt_answers  # lazy: gate -> corpus -> posteriordb

    prob = next(p for p in load_problems({pid}) if p["problem_id"] == pid)
    spec = parse_spec(prob["answer_spec"])
    stan_real = _real_by_id("stan")[pid]
    ref_real = _real_by_id("reference")[pid]

    stan_gts, _ = collect_gt_answers(stan_real["code"], spec, language="stan",
                                     base_seed=DEFAULT_SEED, timeout=timeout)
    ref_gts, _ = collect_gt_answers(ref_real["code"], spec, language="reference",
                                    base_seed=DEFAULT_SEED, timeout=timeout)

    fmap = spec.field_map()
    fields = []
    for name, fspec in fmap.items():
        sfield = [a.field_map()[name] for a in stan_gts]
        rfield = [a.field_map()[name] for a in ref_gts]
        stan_floor = noise_floor(sfield, fspec)
        ref_floor = noise_floor(rfield, fspec)
        spread = _pooled_spread(sfield + rfield, fspec)
        cap = _W1_FLOOR_CAP_FRAC * spread
        # representative cross-column distance (median run vs median run)
        cross = distance(rfield[len(rfield) // 2], sfield[len(sfield) // 2], fspec).value
        fields.append({
            "field": name,
            "stan_floor": round(stan_floor, 4),
            "ref_floor": round(ref_floor, 4),
            "spread": round(spread, 4),
            "cap": round(cap, 4),
            "cross": round(cross, 4),
            "stan_ill": spread > 0 and stan_floor > cap,
            "ref_ill": spread > 0 and ref_floor > cap,
        })
    return {"problem_id": pid, "fields": fields}


def _fmt(d: dict) -> str:
    lines = [d["problem_id"]]
    lines.append(f"  {'field':<12} {'stan_fl':>8} {'ref_fl':>8} {'spread':>8} "
                 f"{'cap':>8} {'cross':>8}  flag")
    for f in d["fields"]:
        flag = ""
        if f["stan_ill"]:
            flag += " STAN-ILL"
        if f["ref_ill"]:
            flag += " REF-ILL"
        lines.append(f"  {f['field']:<12} {f['stan_floor']:>8} {f['ref_floor']:>8} "
                     f"{f['spread']:>8} {f['cap']:>8} {f['cross']:>8} {flag}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="posteriordb per-field crosscheck diagnostics")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()
    for pid in args.ids:
        print(_fmt(diagnose(pid, timeout=args.timeout)))
        print()


if __name__ == "__main__":
    main()
