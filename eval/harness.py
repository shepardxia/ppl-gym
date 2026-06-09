"""Atom-based eval harness.

Runs (or reads cached) the groundtruth answer, runs the generated answer,
dispatches comparison via metrics.compare_by_shape.

For top-level `samples`-shape atoms, the answer is a list of N seeded
runs of the program. The cache (built by
`scripts/cache_groundtruth_outputs.py`) stores those N gt samples
directly, so the harness only does N reruns of the *generated* code.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.config import (
    DEFAULT_MC_WORKERS, DEFAULT_N_MC, DEFAULT_SEED, DEFAULT_TIMEOUT, EvalConfig,
)
from eval.executor import execute_webppl
from eval.executor_pyro import execute_pyro


def _execute_for(atom: dict):
    """Return the executor function appropriate to the atom's language.

    Defaults to WebPPL for atoms without a `language` field (preserves the
    v2 dataset's implicit contract).
    """
    if atom.get("language") == "pyro":
        return execute_pyro
    return execute_webppl
from eval.io import load_jsonl
from eval.metrics import (
    SHAPE_SAMPLES,
    aggregate_metrics,
    code_exact_match,
    code_jaccard,
    collect_metrics,
    compare_by_shape,
)
from eval.spec_metrics import compare_by_spec, collect_metrics_spec


def _is_top_samples(shape) -> bool:
    return shape == SHAPE_SAMPLES


def _is_aggregate_samples(answer) -> bool:
    """An aggregated-samples answer is a list of scalars / shallow records.

    Programs that already produce a list of samples internally (via
    `repeat(N, fn)`, `_.map`, etc.) shouldn't be re-run N times by the
    harness — the list itself IS the sample collection.
    """
    if not isinstance(answer, list):
        return False
    return len(answer) >= 5 and all(
        not isinstance(x, list) or len(x) <= 8
        for x in answer[:10]
    )


def _run_mc(code, n, timeout, base_seed=DEFAULT_SEED, workers=DEFAULT_MC_WORKERS,
            executor=execute_webppl):
    """Run `code` with seeds base_seed..base_seed+n-1 in parallel.

    Returns (answers, first_error). Each entry of `answers` is the run's
    parsed answer or None on failure, in seed order. `first_error` is the
    error message from the lowest-seed failing run, or None.
    """
    answers: list = [None] * n
    errors: list = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(executor, code, timeout=timeout, random_seed=base_seed + i): i
            for i in range(n)
        }
        for fut in futures:
            i = futures[fut]
            r = fut.result()
            if r.success:
                answers[i] = r.answer
            else:
                errors[i] = r.error_message
    first_error = next((e for e in errors if e), None)
    return answers, first_error


def _ensure_groundtruth_output(atom, *, cfg: EvalConfig):
    """Return the gt's serialized answer (or list of N for samples shape).

    Always uses the cached `groundtruth_output` if present. For samples
    atoms the cache must have been built with `cache_groundtruth_outputs.py
    --n-mc <N> --seed <S>` to contain the list of N samples.
    """
    cached = atom.get("groundtruth_output")
    if cached is not None:
        return cached, {"source": "cached"}
    exec_fn = _execute_for(atom)
    if _is_top_samples(atom["answer_shape"]):
        answers, _ = _run_mc(atom["groundtruth_code"], cfg.n_mc, cfg.timeout,
                             base_seed=cfg.seed, workers=cfg.mc_workers,
                             executor=exec_fn)
        non_null = [a for a in answers if a is not None]
        return non_null, {"source": "executed", "n_mc": cfg.n_mc}
    res = exec_fn(atom["groundtruth_code"], timeout=cfg.timeout, random_seed=cfg.seed)
    if not res.success:
        return None, {"source": "executed", "error": res.error_message}
    return res.answer, {"source": "executed"}


def _run_gen(atom, generated_code, *, cfg: EvalConfig, gt_answer=None):
    exec_fn = _execute_for(atom)
    if _is_top_samples(atom["answer_shape"]):
        # If the GT's cached answer is already a list (aggregated samples),
        # the gen should also run once — otherwise we'd compare list-of-N-lists
        # vs flat-list-of-scalars and trivially get TV=1. The comparator will
        # coerce a Distribution-shaped gen result to samples if needed.
        if _is_aggregate_samples(gt_answer):
            res = exec_fn(generated_code, timeout=cfg.timeout, random_seed=cfg.seed)
            return {
                "executed": res.success,
                "answer": res.answer if res.success else None,
                "error": None if res.success else res.error_message,
            }
        answers, first_error = _run_mc(generated_code, cfg.n_mc, cfg.timeout,
                                       base_seed=cfg.seed, workers=cfg.mc_workers,
                                       executor=exec_fn)
        non_null = [a for a in answers if a is not None]
        return {
            "executed": bool(non_null),
            "answer": non_null,
            "n_ok": len(non_null), "n_total": cfg.n_mc,
            "error": None if non_null else (first_error or "all reruns failed"),
        }
    res = exec_fn(generated_code, timeout=cfg.timeout, random_seed=cfg.seed)
    return {
        "executed": res.success,
        "answer": res.answer if res.success else None,
        "error": None if res.success else res.error_message,
    }


def evaluate_atom(
    atom: dict,
    generated_code: str,
    *,
    cfg: EvalConfig | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    seed: int = DEFAULT_SEED,
    n_mc: int = DEFAULT_N_MC,
    spec: dict | None = None,
) -> dict:
    """Score one generated program against one atom.

    When `spec` is provided, dispatches via `compare_by_spec` (spec-aware
    metrics: Wasserstein, KS, parametric param-match, stepwise trajectory).
    Falls back to `compare_by_shape` (legacy 4-shape enum) when omitted.
    """
    if cfg is None:
        cfg = EvalConfig(timeout=timeout, seed=seed, n_mc=n_mc)
    gt_answer, gt_meta = _ensure_groundtruth_output(atom, cfg=cfg)
    gen = _run_gen(atom, generated_code, cfg=cfg, gt_answer=gt_answer)

    out = {
        "id": atom["id"],
        "answer_shape": atom["answer_shape"],
        "gt": gt_meta,
        "gen": {k: v for k, v in gen.items()},
        "string": {
            "exact": code_exact_match(generated_code, atom["groundtruth_code"]),
            "jaccard": code_jaccard(generated_code, atom["groundtruth_code"]),
        },
    }
    if spec is not None:
        out["spec"] = spec

    if not gen["executed"] or gt_answer is None:
        out["comparison"] = {"shape": str(atom["answer_shape"]), "ok": False,
                             "reason": "execution failure"}
        out["metrics"] = {}
        return out

    if spec is not None:
        cmp = compare_by_spec(gen["answer"], gt_answer, spec)
        out["comparison"] = cmp
        out["metrics"] = collect_metrics_spec(cmp)
    else:
        cmp = compare_by_shape(gen["answer"], gt_answer, atom["answer_shape"])
        out["comparison"] = cmp
        out["metrics"] = collect_metrics(cmp)
    return out


def evaluate_atoms_groundtruth_self(atoms, *, cfg: EvalConfig, verbose=False):
    results = []
    for i, atom in enumerate(atoms):
        t0 = time.time()
        r = evaluate_atom(atom, atom["groundtruth_code"], cfg=cfg)
        r["runtime_sec"] = round(time.time() - t0, 2)
        results.append(r)
        if verbose:
            shape = atom["answer_shape"]
            tag = "OK" if r["gen"]["executed"] else "FAIL"
            shape_str = shape if isinstance(shape, str) else "record"
            print(f"[{i+1}/{len(atoms)}] {atom['id']:50s} {tag:5s} "
                  f"{r['runtime_sec']:5.1f}s shape={shape_str} metrics={r['metrics']}")
    return results


def aggregate(results):
    n = len(results)
    n_gen_ok = sum(1 for r in results if r["gen"]["executed"])
    return {
        "n_atoms": n,
        "n_gen_executed": n_gen_ok,
        "exec_rate": (n_gen_ok / n) if n else 0.0,
        **aggregate_metrics([r.get("metrics") for r in results]),
    }


def main():
    p = argparse.ArgumentParser(description="Atom eval harness (default: gt-vs-self).")
    p.add_argument("--dataset", default="data/atomized_v2.jsonl")
    p.add_argument("--max-atoms", type=int, default=None)
    p.add_argument("--ids", nargs="+", default=None)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-mc", type=int, default=DEFAULT_N_MC)
    p.add_argument("--mc-workers", type=int, default=DEFAULT_MC_WORKERS)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    atoms = load_jsonl(args.dataset)
    if args.ids:
        atoms = [a for a in atoms if a["id"] in set(args.ids)]
    if args.max_atoms is not None:
        atoms = atoms[: args.max_atoms]

    cfg = EvalConfig(timeout=args.timeout, seed=args.seed,
                     n_mc=args.n_mc, mc_workers=args.mc_workers)

    print(f"Loaded {len(atoms)} atoms from {args.dataset}")
    print(f"Sanity check: gt vs self ({cfg})\n")

    results = evaluate_atoms_groundtruth_self(atoms, cfg=cfg, verbose=args.verbose)
    agg = aggregate(results)

    print()
    print("=" * 60)
    print("AGGREGATE (gt vs self — should be ~perfect)")
    print("=" * 60)
    print(f"  Atoms:           {agg['n_atoms']}")
    print(f"  Executed:        {agg['n_gen_executed']} ({agg['exec_rate']:.1%})")
    if agg["mean_kl"] is not None:
        print(f"  Mean KL:         {agg['mean_kl']:.6f} (n={agg['n_kl_metrics']})")
    if agg["mean_tv"] is not None:
        print(f"  Mean TV:         {agg['mean_tv']:.6f} (n={agg['n_tv_metrics']})")
    if agg["mean_value_exact"] is not None:
        print(f"  Mean exact-rate: {agg['mean_value_exact']:.2%} (n={agg['n_exact_metrics']})")


if __name__ == "__main__":
    main()
