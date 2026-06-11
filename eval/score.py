"""Stage 2: generations JSONL → scored JSONL (problem-centric).

Each input row must have at minimum {"problem_id": ..., "code": ...}.
All other fields are preserved in the output row.

Output row schema:
  {<original fields>, "status", "distance", "tol", "floor", "metric",
   "ill_posed", "error" (when present), "code_jaccard", "runtime_sec"}

Final summary line:
  {"summary": true, "n": <int>, "pass": <int>, "fail": <int>,
   "ill_posed": <int>, "malformed": <int>, "exec_error": <int>,
   "pass_rate": <float>}

CLI:
  PYTHONPATH=. .venv/bin/python -m eval.score \\
      --generations <path> \\
      --output <path> \\
      [--language webppl] \\
      [--ids ID ...] \\
      [--timeout 60] \\
      [--seed 42] \\
      [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Lock

from eval.algebra import AlgebraError, parse_spec, status_of, verdict
from eval.config import DEFAULT_MC_WORKERS, DEFAULT_N_MC, DEFAULT_SEED, DEFAULT_TIMEOUT
from eval.corpus import load_corpus
from eval.gate import (
    collect_gt_answers,
    code_jaccard,
    execute_candidate_answer,
)
from eval.io import load_jsonl, write_jsonl


def _score_one(
    row: dict,
    problem: dict,
    realization: dict,
    *,
    seed: int,
    n_draws: int,
    timeout: int,
    workers: int,
    gt_cache: dict,
    gt_cache_lock: Lock,
) -> dict:
    """Score one generation row. Returns the row dict augmented with result fields."""
    t0 = time.time()
    pid = row["problem_id"]
    code = row.get("code", "")

    def _error_row(status: str, msg: str) -> dict:
        return {
            **row,
            "status": status,
            "distance": None,
            "tol": None,
            "floor": None,
            "metric": None,
            "ill_posed": False,
            "error": msg,
            "code_jaccard": code_jaccard(code, realization.get("code", "")),
            "runtime_sec": round(time.time() - t0, 3),
        }

    # Parse spec
    try:
        spec = parse_spec(problem["answer_spec"])
    except (AlgebraError, KeyError, TypeError) as exc:
        return _error_row("malformed", f"bad spec: {exc}")

    # Collect GT canonicals once per problem_id: the first thread to arrive
    # installs an Event and computes; later threads wait on it instead of
    # launching a duplicate collection.
    with gt_cache_lock:
        entry = gt_cache.get(pid)
        if entry is None:
            entry = gt_cache[pid] = {"event": Event(), "gts": None, "error": None}
            is_owner = True
        else:
            is_owner = False

    if is_owner:
        try:
            entry["gts"], _ = collect_gt_answers(
                realization["code"],
                spec,
                base_seed=seed,
                n_draws=n_draws,
                timeout=timeout,
                workers=workers,
            )
        except (RuntimeError, AlgebraError) as exc:
            entry["error"] = f"GT collection failed: {exc}"
        finally:
            entry["event"].set()
    else:
        entry["event"].wait()

    if entry["error"] is not None:
        return _error_row("exec_error", entry["error"])
    gts = entry["gts"]

    # Execute candidate code
    if not code or not code.strip():
        return _error_row("exec_error", "empty code")

    try:
        canon = execute_candidate_answer(
            code,
            spec,
            base_seed=seed,
            n_draws=n_draws,
            timeout=timeout,
            workers=workers,
        )
    except AlgebraError as exc:
        return _error_row("malformed", str(exc))
    except RuntimeError as exc:
        return _error_row("exec_error", str(exc))

    # Judge
    try:
        v = verdict(canon, gts, spec)
    except AlgebraError as exc:
        return _error_row("malformed", f"verdict failed: {exc}")

    status = status_of(v)
    jac = code_jaccard(code, realization.get("code", ""))

    return {
        **row,
        "status": status,
        "distance": round(v.get("distance", 0.0), 6),
        "tol": v.get("tol"),
        "floor": v.get("floor"),
        "metric": v.get("metric"),
        "ill_posed": bool(v.get("ill_posed")),
        "code_jaccard": round(jac, 6),
        "runtime_sec": round(time.time() - t0, 3),
    }


def run_scoring(
    generations_path: Path,
    output_path: Path,
    *,
    language: str = "webppl",
    problem_ids: set[str] | None = None,
    seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = 4,
) -> dict:
    """Score a generations JSONL; write scored JSONL + summary row.

    Returns the summary dict.
    """
    rows = [r for r in load_jsonl(generations_path) if not r.get("summary")]

    # Optionally filter by problem_id
    if problem_ids:
        rows = [r for r in rows if r.get("problem_id") in problem_ids]

    # Load corpus for all problem_ids present in rows
    row_pids = {r["problem_id"] for r in rows if "problem_id" in r}
    problems, realizations = load_corpus(row_pids, language=language)
    prob_by_id = {p["problem_id"]: p for p in problems}
    real_by_id = {r["problem_id"]: r for r in realizations}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-process GT cache: avoids re-running GT collection when multiple
    # generation slots share the same problem_id.
    gt_cache: dict = {}
    gt_cache_lock = Lock()

    # Clamp per-problem workers so total WebPPL processes stay bounded:
    # workers (problems in flight) × mc_workers = max WebPPL processes.
    mc_workers = max(1, DEFAULT_MC_WORKERS // max(1, workers))

    scored_rows: list[dict] = []
    write_lock = Lock()
    completed = [0]

    def _process(row: dict) -> dict:
        pid = row.get("problem_id")
        problem = prob_by_id.get(pid) if pid else None
        realization = real_by_id.get(pid) if pid else None

        if problem is None or realization is None:
            return {
                **row,
                "status": "exec_error",
                "distance": None,
                "tol": None,
                "floor": None,
                "metric": None,
                "ill_posed": False,
                "error": f"problem/realization not found for problem_id={pid!r}",
                "code_jaccard": 0.0,
                "runtime_sec": 0.0,
            }

        return _score_one(
            row, problem, realization,
            seed=seed,
            n_draws=n_draws,
            timeout=timeout,
            workers=mc_workers,
            gt_cache=gt_cache,
            gt_cache_lock=gt_cache_lock,
        )

    with open(output_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, row): row for row in rows}
            for fut in as_completed(futures):
                scored = fut.result()
                with write_lock:
                    out_f.write(json.dumps(scored) + "\n")
                    out_f.flush()
                    completed[0] += 1
                    status = scored.get("status", "?")
                    pid = scored.get("problem_id", "?")
                    print(
                        f"[{completed[0]}/{len(rows)}] {pid}  status={status}"
                        + (f"  dist={scored['distance']}" if scored.get("distance") is not None else ""),
                        flush=True,
                    )
                    scored_rows.append(scored)

        # Summary
        counts: dict[str, int] = {}
        for r in scored_rows:
            s = r.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        n = len(scored_rows)
        n_pass = counts.get("pass", 0)
        summary = {
            "summary": True,
            "n": n,
            "pass": n_pass,
            "fail": counts.get("fail", 0),
            "ill_posed": counts.get("ill_posed", 0),
            "malformed": counts.get("malformed", 0),
            "exec_error": counts.get("exec_error", 0),
            "pass_rate": round(n_pass / n, 4) if n else 0.0,
        }
        out_f.write(json.dumps(summary) + "\n")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Score a generations JSONL (problem-centric).")
    p.add_argument("--generations", required=True, help="Path to generations JSONL.")
    p.add_argument("--output", required=True, help="Path for scored output JSONL.")
    p.add_argument(
        "--language", default="webppl",
        help="PPL language; selects realization file (default: webppl).",
    )
    p.add_argument(
        "--ids", nargs="+", default=None, metavar="ID",
        help="Restrict to these problem IDs.",
    )
    p.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-execution timeout in seconds (default {DEFAULT_TIMEOUT}).",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base random seed (default {DEFAULT_SEED}).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Problem-level concurrency (default 4).",
    )
    args = p.parse_args()

    id_set = set(args.ids) if args.ids else None

    summary = run_scoring(
        generations_path=Path(args.generations),
        output_path=Path(args.output),
        language=args.language,
        problem_ids=id_set,
        seed=args.seed,
        timeout=args.timeout,
        workers=args.workers,
    )

    print()
    print("=" * 60)
    print("SCORING DONE")
    print("=" * 60)
    print(f"  n:           {summary['n']}")
    print(f"  pass:        {summary['pass']}  ({summary['pass_rate']:.1%})")
    print(f"  fail:        {summary['fail']}")
    print(f"  ill_posed:   {summary['ill_posed']}")
    print(f"  malformed:   {summary['malformed']}")
    print(f"  exec_error:  {summary['exec_error']}")


if __name__ == "__main__":
    main()
