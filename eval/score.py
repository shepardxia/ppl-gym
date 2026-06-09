"""Stage 2: generation JSONL -> scored JSONL. Free to re-run."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from eval.config import DEFAULT_MC_WORKERS, DEFAULT_N_MC, DEFAULT_SEED, DEFAULT_TIMEOUT, EvalConfig
from eval.harness import evaluate_atom
from eval.io import iter_scored, load_atom_specs, load_jsonl
from eval.metrics import aggregate_metrics


def run_scoring(
    dataset_path: Path,
    generations_path: Path,
    output_path: Path,
    *,
    cfg: EvalConfig,
    workers: int = 4,
    specs_path: Path | None = None,
    use_specs: bool = True,
):
    atoms_by_id = {a["id"]: a for a in load_jsonl(dataset_path)}
    gens = [r for r in load_jsonl(generations_path) if not r.get("summary")]
    specs_by_id = load_atom_specs(specs_path) if use_specs else {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = Lock()
    completed = [0]
    t_start = time.time()

    def _score_one(rec):
        atom = atoms_by_id.get(rec["id"])
        if atom is None:
            return {**rec, "evaluation": {"error": "atom not found in dataset"}}
        t0 = time.time()
        spec = specs_by_id.get(rec["id"]) if use_specs else None
        evaluation = evaluate_atom(atom, rec["generation"]["code"], cfg=cfg, spec=spec)
        return {**rec, "evaluation": evaluation, "score_runtime_sec": round(time.time() - t0, 2)}

    all_results = []
    with open(output_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_score_one, r) for r in gens]
            for fut in futures:
                rec = fut.result()
                with write_lock:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    completed[0] += 1
                    ev = rec.get("evaluation", {})
                    print(
                        f"[{completed[0]}/{len(gens)}] {rec['id']:55s} "
                        f"executed={ev.get('gen', {}).get('executed')}, "
                        f"metrics={ev.get('metrics', {})}",
                        flush=True,
                    )
                    all_results.append(rec)

        cross = _aggregate(all_results)
        summary = {
            "summary": True,
            "n_atoms": len(all_results),
            "total_runtime_sec": round(time.time() - t_start, 2),
            "cross": cross,
        }
        out_f.write(json.dumps(summary) + "\n")

    return summary


def _aggregate(scored_records: list[dict]) -> dict:
    n = len(scored_records)
    n_executed = 0
    parse_failures = 0
    metrics_per = []
    for r in scored_records:
        gen = r.get("evaluation", {}).get("gen", {})
        if gen.get("executed"):
            n_executed += 1
        warns = r.get("generation", {}).get("parse_warnings") or []
        if any("no fenced" in w or "API error" in w for w in warns):
            parse_failures += 1
        metrics_per.append(r.get("evaluation", {}).get("metrics") or {})

    return {
        "n_atoms": n,
        "n_executed": n_executed,
        "exec_rate": (n_executed / n) if n else 0.0,
        "n_parse_failures": parse_failures,
        **aggregate_metrics(metrics_per),
    }


def main():
    p = argparse.ArgumentParser(description="Stage 2: score generations.")
    p.add_argument("--dataset", default="data/atomized_v2.jsonl")
    p.add_argument("--generations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-mc", type=int, default=DEFAULT_N_MC)
    p.add_argument("--mc-workers", type=int, default=DEFAULT_MC_WORKERS)
    p.add_argument("--workers", type=int, default=4,
                   help="Atom-level concurrent scoring jobs")
    p.add_argument("--specs", default=None,
                   help="Path to atom_specs.jsonl (default: data/atom_specs.jsonl if present)")
    p.add_argument("--no-specs", action="store_true",
                   help="Force legacy comparator (compare_by_shape)")
    args = p.parse_args()

    # Cap mc_workers so total in-flight WebPPL processes <= ~mc_workers cap.
    # With workers=4 atoms × mc_workers=8 we'd peak at 32 node procs; clamp.
    effective_mc_workers = max(1, args.mc_workers // max(1, args.workers))
    cfg = EvalConfig(
        timeout=args.timeout, seed=args.seed,
        n_mc=args.n_mc, mc_workers=effective_mc_workers,
    )

    summary = run_scoring(
        dataset_path=Path(args.dataset),
        generations_path=Path(args.generations),
        output_path=Path(args.output),
        cfg=cfg,
        workers=args.workers,
        specs_path=Path(args.specs) if args.specs else None,
        use_specs=not args.no_specs,
    )

    cross = summary["cross"]
    print()
    print("=" * 60)
    print("SCORING DONE")
    print("=" * 60)
    print(f"  Atoms:           {cross['n_atoms']}")
    print(f"  Executed:        {cross['n_executed']} ({cross['exec_rate']:.1%})")
    print(f"  Parse failures:  {cross['n_parse_failures']}")
    # Legacy aggregate keys
    if cross["mean_kl"] is not None:
        print(f"  Mean KL:         {cross['mean_kl']:.4f} (n={cross['n_kl_metrics']})")
    if cross["mean_tv"] is not None:
        print(f"  Mean TV:         {cross['mean_tv']:.4f} (n={cross['n_tv_metrics']})")
    if cross["mean_value_exact"] is not None:
        print(f"  Mean value-exact: {cross['mean_value_exact']:.2%} (n={cross['n_exact_metrics']})")
    # Spec-aware aggregates (only print when present)
    for label, key, fmt in (
        ("Mean W1",            "mean_w1",            "{:.4f}"),
        ("Mean KS",            "mean_ks",            "{:.4f}"),
        ("Mean length-TV",     "mean_length_tv",     "{:.4f}"),
        ("Mean step-TV",       "mean_mean_step_tv",  "{:.4f}"),
        ("Mean family-match",  "mean_family_match",  "{:.2%}"),
        ("Mean params-match",  "mean_params_match",  "{:.2%}"),
    ):
        v = cross.get(key)
        if v is None:
            continue
        n_key = key.replace("mean_", "n_") + "_metrics"
        print(f"  {label:18s} {fmt.format(v)} (n={cross.get(n_key, 0)})")
    print(f"  Wall clock:      {summary['total_runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
