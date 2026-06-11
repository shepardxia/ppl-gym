"""Gate: Phase-A GT noise-floor measurement + Phase-B solver re-derivation.

Phase A (default / ``phaseA`` subcommand):
  For each problem with a WebPPL realization, runs k independent GT answer
  collections under the problem's answer_spec, computes the GT noise floor
  via algebra.noise_floor, and flags ill-posed problems.

Phase B (``solve`` / ``judge`` subcommands):
  ``solve``  — render each problem's prompt and submit an Anthropic batch
               with two solver requests per problem.  ``--dry-run`` writes
               the manifest and prints sample prompts without submitting.
  ``judge``  — poll/fetch the solver batch, execute the extracted code,
               classify each problem (accept / gt_suspect / underdetermined /
               solver_failure), and write a solver report JSONL.

Contract: data/SCHEMA.md §Gate.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic

from eval.algebra import (
    AlgebraError,
    Spec,
    _has_draws_field,
    agreement,
    canonicalize,
    noise_floor,
    parse_spec,
    verdict,
)
from eval.config import DEFAULT_MC_WORKERS, DEFAULT_N_MC, DEFAULT_SEED, DEFAULT_TIMEOUT
from eval.corpus import executor_for, load_corpus, load_problems
from eval.executor import execute_webppl
from eval.generate_batch import (
    build_requests,
    cid_to_problem_slot,
    collect_results,
    problem_id_to_cid,
    submit_batch,
    wait_for_batch,
)
from eval.io import load_jsonl, write_jsonl
from eval.prompt import system_prompt
from eval.render import render_problem


# ---------------------------------------------------------------------------
# Code similarity (eval/metrics.py was ripped in P2; logic lives here only).
# ---------------------------------------------------------------------------

def _normalize_code(code: str) -> str:
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"\s+", " ", code).strip()
    code = re.sub(r";\s*$", "", code)
    return code


def code_jaccard(generated: str, ground_truth: str) -> float:
    g = set(_normalize_code(generated).split())
    t = set(_normalize_code(ground_truth).split())
    if not g and not t:
        return 1.0
    if not g or not t:
        return 0.0
    return len(g & t) / len(g | t)

# ---------------------------------------------------------------------------
# Report merge helper (shared by phaseA and cmd_judge)
# ---------------------------------------------------------------------------

def _merge_report(report_path: Path, new_rows: list[dict]) -> int:
    """Load existing report at `report_path`, update rows for judged ids, write sorted.

    Returns total row count after merge.
    """
    merged: dict[str, dict] = {}
    if report_path.exists():
        for row in load_jsonl(report_path):
            merged[row["problem_id"]] = row
    for r in new_rows:
        merged[r["problem_id"]] = r
    sorted_rows = sorted(merged.values(), key=lambda r: r["problem_id"])
    write_jsonl(report_path, sorted_rows)
    return len(sorted_rows)


# Defaults match SCHEMA.md: k=5 for non-draws, k=3 for draws blocks.
_DEFAULT_K_EXACT = 5
_DEFAULT_K_DRAWS = 3

# Data paths relative to repo root.
_DEFAULT_REPORT = Path("data/problems/_gate_report.jsonl")


# ---------------------------------------------------------------------------
# GT answer collection (reusable by Phase B)
# ---------------------------------------------------------------------------

def _run_single(code: str, seed: int, timeout: int, executor) -> object:
    """Execute code once; return the raw answer or raise RuntimeError on failure."""
    r = executor(code, timeout=timeout, random_seed=seed)
    if not r.success:
        raise RuntimeError(r.error_message or "execution failed")
    return r.answer


def _run_block(
    code: str,
    base_seed: int,
    n: int,
    timeout: int,
    workers: int,
    executor,
) -> list:
    """Run code at seeds base_seed..base_seed+n-1 in parallel; collect raw answers.

    Returns the list of raw answers (None entries removed).  Raises RuntimeError
    if every run fails (the caller can treat partial success as an error or not,
    depending on tolerance requirements).
    """
    answers: list = [None] * n
    errors: list = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(executor, code, timeout=timeout, random_seed=base_seed + i): i
            for i in range(n)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
                if r.success:
                    answers[i] = r.answer
                else:
                    errors[i] = r.error_message
            except Exception as exc:
                errors[i] = str(exc)
    good = [a for a in answers if a is not None]
    if not good:
        first_err = next((e for e in errors if e), "all runs failed")
        raise RuntimeError(first_err)
    return good


def collect_gt_answers(
    code: str,
    spec: Spec,
    *,
    base_seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    k_exact: int = _DEFAULT_K_EXACT,
    k_draws: int = _DEFAULT_K_DRAWS,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_MC_WORKERS,
    executor=execute_webppl,
) -> tuple[list, int]:
    """Collect k independent canonical GT answers for (code, spec).

    Returns (canonical_answers, n_runs_total).

    Draws protocol (spec has draws anywhere):
      One GT answer = the collected list of n_draws seeded executions.
      Seed block i uses base_seed + i*n_draws.  k = k_draws (default 3).
      Total runs = k_draws * n_draws.

    Non-draws (exact/enum/parametric):
      One GT answer = one execution.
      Seeds = base_seed, base_seed+1, ..., base_seed+k_exact-1.
      Total runs = k_exact (default 5).

    Each collected answer is canonicalized under the spec.  AlgebraError or
    RuntimeError propagates to the caller; gate_problem() catches both.
    """
    uses_draws = _has_draws_field(spec)

    if uses_draws:
        k = k_draws
        canonical: list = []
        n_runs = 0
        for block in range(k):
            block_base = base_seed + block * n_draws
            raw_list = _run_block(
                code, block_base, n_draws, timeout, workers, executor
            )
            n_runs += len(raw_list)
            canonical.append(canonicalize(raw_list, spec))
        return canonical, n_runs
    else:
        k = k_exact
        canonical = []
        # Single-run seeds are executed individually (each is a separate task
        # in the pool) so the thread pool stays fully busy.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(
                    _run_single, code, base_seed + i, timeout, executor
                ): i
                for i in range(k)
            }
            results_raw: list = [None] * k
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results_raw[idx] = fut.result()  # propagates RuntimeError
        for raw in results_raw:
            canonical.append(canonicalize(raw, spec))
        return canonical, k


# ---------------------------------------------------------------------------
# Per-problem gate
# ---------------------------------------------------------------------------

def gate_problem(
    problem: dict,
    realization: dict,
    *,
    base_seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    k_exact: int = _DEFAULT_K_EXACT,
    k_draws: int = _DEFAULT_K_DRAWS,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_MC_WORKERS,
    executor=execute_webppl,
) -> dict:
    """Run the Phase-A gate for one (problem, realization) pair.

    Returns a report dict:
      {problem_id, status, k, n_runs, floor, metric, error?, runtime_sec}

    status is one of:
      "ok"        — GT floor within discriminability caps
      "ill_posed" — GT floor exceeds cap; problem cannot discriminate
      "error"     — execution or algebra failure; goes to triage
    """
    problem_id = problem["problem_id"]
    t0 = time.time()

    try:
        spec = parse_spec(problem["answer_spec"])
    except (AlgebraError, KeyError, TypeError) as exc:
        return {
            "problem_id": problem_id,
            "status": "error",
            "k": 0, "n_runs": 0,
            "floor": None, "metric": None,
            "error": f"bad spec: {exc}",
            "runtime_sec": round(time.time() - t0, 3),
        }

    try:
        gts, n_runs = collect_gt_answers(
            realization["code"],
            spec,
            base_seed=base_seed,
            n_draws=n_draws,
            k_exact=k_exact,
            k_draws=k_draws,
            timeout=timeout,
            workers=workers,
            executor=executor,
        )
    except (RuntimeError, AlgebraError) as exc:
        return {
            "problem_id": problem_id,
            "status": "error",
            "k": 0, "n_runs": 0,
            "floor": None, "metric": None,
            "error": str(exc),
            "runtime_sec": round(time.time() - t0, 3),
        }

    k = len(gts)

    try:
        floor = noise_floor(gts, spec)
        # verdict(gts[0], gts, spec) uses gts[0] as candidate against all GTs
        # to check the problem's own consistency — same call the Phase-B
        # solver harness will make for each solver answer.
        v = verdict(gts[0], gts, spec)
        ill = bool(v.get("ill_posed"))
        metric = v.get("metric")
    except AlgebraError as exc:
        return {
            "problem_id": problem_id,
            "status": "error",
            "k": k, "n_runs": n_runs,
            "floor": None, "metric": None,
            "error": f"algebra: {exc}",
            "runtime_sec": round(time.time() - t0, 3),
        }

    return {
        "problem_id": problem_id,
        "status": "ill_posed" if ill else "ok",
        "k": k,
        "n_runs": n_runs,
        "floor": round(floor, 6),
        "metric": metric,
        "runtime_sec": round(time.time() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Summary table (Phase A)
# ---------------------------------------------------------------------------

def _print_summary(reports: list[dict]) -> None:
    counts: dict[str, int] = {"ok": 0, "ill_posed": 0, "error": 0}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("\n" + "=" * 64)
    print("GATE SUMMARY")
    print("=" * 64)
    total = len(reports)
    for status, n in sorted(counts.items()):
        print(f"  {status:<12s} {n:>4d} / {total}")

    # Worst 10 floors (non-error, sorted descending)
    ranked = sorted(
        [r for r in reports if r["status"] != "error" and r["floor"] is not None],
        key=lambda r: r["floor"],
        reverse=True,
    )[:10]
    if ranked:
        print()
        print("  Worst 10 GT floors:")
        print(f"  {'problem_id':<50s}  {'floor':>8s}  {'metric':<8s}  status")
        print("  " + "-" * 80)
        for r in ranked:
            print(f"  {r['problem_id']:<50s}  {r['floor']:>8.4f}  {r['metric'] or '?':<8s}  {r['status']}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Phase-B: solve (batch submission)
# ---------------------------------------------------------------------------

_SOLVE_MODEL = "claude-sonnet-4-6"
_SOLVE_MAX_TOKENS = 4096
_SOLVE_TEMPERATURE = 1.0
_SOLVE_MANIFEST = Path("data/problems/_gate_solve_batch.json")


def cmd_solve(args) -> None:
    """Phase-B solve: render prompts and submit batch (or dry-run)."""
    id_set = set(args.ids) if args.ids else None
    problems = load_problems(id_set)

    if not problems:
        print("No matching problems found.")
        return

    print(f"[solve] {len(problems)} problem(s) → {len(problems) * 2} requests")

    requests = build_requests(problems, language=args.language, model=args.model)

    # Build manifest (always, whether dry-run or not)
    manifest = {
        "batch_id": None,
        "model": args.model,
        "language": args.language,
        "temperature": _SOLVE_TEMPERATURE,
        "max_tokens": _SOLVE_MAX_TOKENS,
        "n_problems": len(problems),
        "n_requests": len(requests),
        "problem_ids": [p["problem_id"] for p in problems],
        "requests": [
            {
                "custom_id": r["custom_id"],
                "problem_id": cid_to_problem_slot(r["custom_id"])[0],
                "slot": cid_to_problem_slot(r["custom_id"])[1],
                "rendered_prompt": r["params"]["messages"][0]["content"],
            }
            for r in requests
        ],
    }

    manifest_path = Path(args.manifest) if args.manifest else _SOLVE_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_path = manifest_path.with_suffix(".dry.json")
        manifest["batch_id"] = "DRY_RUN"
        dry_path.write_text(json.dumps(manifest, indent=2))
        print(f"[dry-run] Manifest written to {dry_path} (live manifest not touched)")
        print(f"[dry-run] Printing 3 sample rendered prompts:\n")

        # Print up to 3 prompts (one per problem, slot 0 only)
        shown = 0
        for entry in manifest["requests"]:
            if entry["slot"] != 0:
                continue
            if shown >= 3:
                break
            print("=" * 70)
            print(f"problem_id: {entry['problem_id']}")
            print("=" * 70)
            print(entry["rendered_prompt"])
            print()
            shown += 1
        return

    # Live submission
    client = Anthropic()
    print(f"[solve] submitting {len(requests)} requests to Anthropic batch API...")
    batch_id = submit_batch(client, requests)
    manifest["batch_id"] = batch_id
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[solve] batch_id={batch_id}")
    print(f"[solve] manifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# Phase-B: judge (execute solver code, classify problems)
# ---------------------------------------------------------------------------

_SOLVER_REPORT = Path("data/problems/_gate_solver_report.jsonl")
_MEMORIZATION_JACCARD_THRESHOLD = 0.6


def execute_candidate_answer(
    code: str,
    spec: Spec,
    *,
    base_seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_MC_WORKERS,
    executor=execute_webppl,
) -> object:
    """Execute candidate code and return a canonical answer.

    The k=1 case of collect_gt_answers: draws-spec problems collect n_draws
    seeded runs into one canonical answer; others run once at base_seed.
    """
    answers, _ = collect_gt_answers(
        code, spec,
        base_seed=base_seed, n_draws=n_draws,
        k_exact=1, k_draws=1,
        timeout=timeout, workers=workers,
        executor=executor,
    )
    return answers[0]


def _judge_problem_b(
    problem: dict,
    realization: dict | None,
    solver_codes: list[str | None],  # [code_s0, code_s1]
    solver_errors: list[str | None],  # [err_s0, err_s1]
    *,
    base_seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    k_exact: int = _DEFAULT_K_EXACT,
    k_draws: int = _DEFAULT_K_DRAWS,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_MC_WORKERS,
    stamp: dict | None = None,
    executor=execute_webppl,
) -> dict:
    """Judge one problem: collect GTs, execute solvers, classify.

    Returns the solver report record for this problem.
    `stamp` is merged into every returned row (gate_model, timeout, n_solvers).
    """
    _stamp = stamp or {}
    problem_id = problem["problem_id"]

    try:
        spec = parse_spec(problem["answer_spec"])
    except (AlgebraError, KeyError, TypeError) as exc:
        return {
            "problem_id": problem_id,
            "status": "solver_failure",
            "n_pass": 0,
            "distances": [None, None],
            "solver_agree_distance": None,
            "solver_agree_tol": None,
            "gt_floor": None,
            "memorization_suspect": False,
            "errors": [f"bad spec: {exc}"],
            **_stamp,
        }

    # Collect GT answers (recomputed every call — accept the cost per spec)
    if realization is None:
        return {
            "problem_id": problem_id,
            "status": "solver_failure",
            "n_pass": 0,
            "distances": [None, None],
            "solver_agree_distance": None,
            "solver_agree_tol": None,
            "gt_floor": None,
            "memorization_suspect": False,
            "errors": ["no realization available for GT computation"],
            **_stamp,
        }

    try:
        gts, _ = collect_gt_answers(
            realization["code"],
            spec,
            base_seed=base_seed,
            n_draws=n_draws,
            k_exact=k_exact,
            k_draws=k_draws,
            timeout=timeout,
            workers=workers,
            executor=executor,
        )
        gt_floor = noise_floor(gts, spec)
    except (RuntimeError, AlgebraError) as exc:
        return {
            "problem_id": problem_id,
            "status": "solver_failure",
            "n_pass": 0,
            "distances": [None, None],
            "solver_agree_distance": None,
            "solver_agree_tol": None,
            "gt_floor": None,
            "memorization_suspect": False,
            "errors": [f"GT collection failed: {exc}"],
            **_stamp,
        }

    # Execute solver codes
    solver_canons: list = [None, None]
    solver_exec_errors: list[str | None] = list(solver_errors)  # copy API errors
    errors_out: list[str] = []

    for i, code in enumerate(solver_codes):
        if code is None or not code.strip():
            solver_exec_errors[i] = solver_exec_errors[i] or "no code"
            continue
        try:
            solver_canons[i] = execute_candidate_answer(
                code, spec,
                base_seed=base_seed,
                n_draws=n_draws,
                timeout=timeout,
                workers=workers,
                executor=executor,
            )
        except (RuntimeError, AlgebraError) as exc:
            solver_exec_errors[i] = str(exc)

    # Both structurally failed → solver_failure
    both_failed = all(c is None for c in solver_canons)
    if both_failed:
        for err in solver_exec_errors:
            if err:
                errors_out.append(err)
        return {
            "problem_id": problem_id,
            "status": "solver_failure",
            "n_pass": 0,
            "distances": [None, None],
            "solver_agree_distance": None,
            "solver_agree_tol": None,
            "gt_floor": round(gt_floor, 6),
            "memorization_suspect": False,
            "errors": errors_out or ["both solvers failed structurally"],
            **_stamp,
        }

    # Compute distances and verdicts for each solver
    distances: list = [None, None]
    passed: list = [False, False]

    for i, canon in enumerate(solver_canons):
        if canon is None:
            if solver_exec_errors[i]:
                errors_out.append(f"s{i}: {solver_exec_errors[i]}")
            continue
        try:
            v = verdict(canon, gts, spec)
            passed[i] = bool(v.get("passed"))
            distances[i] = round(v.get("distance", 0.0), 6)
        except AlgebraError as exc:
            solver_exec_errors[i] = f"verdict failed: {exc}"
            errors_out.append(f"s{i}: {solver_exec_errors[i]}")

    n_pass = sum(passed)

    # Solver-agreement: use algebra.agreement() when both canonicals exist.
    solver_agree_dist: float | None = None
    solver_agree_tol: float | None = None
    both_canon = solver_canons[0] is not None and solver_canons[1] is not None
    ag: dict | None = None
    if both_canon:
        try:
            ag = agreement(solver_canons[0], solver_canons[1], spec)
            solver_agree_dist = round(ag["distance"], 6)
            solver_agree_tol = ag["tol"]
        except AlgebraError:
            pass

    # Classify
    if n_pass >= 1:
        status = "accept"
    elif both_canon and ag is not None:
        status = "gt_suspect" if ag["agree"] else "underdetermined"
    else:
        status = "underdetermined"

    # Memorization check: jaccard > 0.6 for any passing solver vs GT code
    gt_code = realization.get("code", "")
    memorization_suspect = False
    if n_pass >= 1 and gt_code:
        for i, (code, did_pass) in enumerate(zip(solver_codes, passed)):
            if did_pass and code:
                jac = code_jaccard(code, gt_code)
                if jac > _MEMORIZATION_JACCARD_THRESHOLD:
                    memorization_suspect = True
                    break

    return {
        "problem_id": problem_id,
        "status": status,
        "n_pass": n_pass,
        "distances": distances,
        "solver_agree_distance": solver_agree_dist,
        "solver_agree_tol": solver_agree_tol,
        "gt_floor": round(gt_floor, 6),
        "memorization_suspect": memorization_suspect,
        "errors": errors_out,
        **_stamp,
    }


def cmd_judge(args) -> None:
    """Phase-B judge: poll batch, execute solver code, write solver report."""
    # Determine batch_id and manifest
    manifest_path = Path(args.manifest) if args.manifest else _SOLVE_MANIFEST
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    batch_id = args.batch_id or manifest.get("batch_id")
    if not batch_id or batch_id == "DRY_RUN":
        print("Error: no batch_id available (pass --batch-id or run solve first).")
        return

    client = Anthropic()

    # Poll if still running
    print(f"[judge] retrieving batch {batch_id}...")
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"[judge] batch status={batch.processing_status}, waiting...")
        batch = wait_for_batch(client, batch_id)

    # Collect raw results: {custom_id: {ok, code, ...}}
    print("[judge] streaming results...")
    raw_results = collect_results(client, batch_id)

    # Index solver codes by problem_id
    solver_codes_by_pid: dict[str, list[str | None]] = {}
    solver_errors_by_pid: dict[str, list[str | None]] = {}
    for cid, res in raw_results.items():
        try:
            pid, slot = cid_to_problem_slot(cid)
        except ValueError:
            continue
        if pid not in solver_codes_by_pid:
            solver_codes_by_pid[pid] = [None, None]
            solver_errors_by_pid[pid] = [None, None]
        if slot in (0, 1):
            if res["ok"]:
                solver_codes_by_pid[pid][slot] = res["code"]
            else:
                errs = res.get("warnings") or []
                solver_errors_by_pid[pid][slot] = "; ".join(errs) if errs else "batch error"

    # Load problems and realizations for GT re-computation
    language = manifest.get("language", "webppl")
    executor = executor_for(language)
    id_set = set(solver_codes_by_pid.keys()) if solver_codes_by_pid else None
    problems = load_problems(id_set)
    _, realizations_list = load_corpus(id_set, language=language)
    real_by_id = {r["problem_id"]: r for r in realizations_list if "code" in r}

    # Build protocol stamp from manifest metadata
    stamp = {
        "gate_model": manifest.get("model"),
        "language": language,
        "timeout": args.timeout,
        "n_solvers": 2,
    }

    print(f"[judge] judging {len(problems)} problem(s)...")
    parallel = min(4, max(1, len(problems)))
    per_problem_workers = max(1, args.workers // parallel)

    def _judge_one(prob: dict) -> dict:
        pid = prob["problem_id"]
        return _judge_problem_b(
            prob, real_by_id.get(pid),
            solver_codes_by_pid.get(pid, [None, None]),
            solver_errors_by_pid.get(pid, [None, None]),
            base_seed=args.seed,
            n_draws=args.n_draws,
            k_exact=args.k_exact,
            k_draws=args.k_draws,
            timeout=args.timeout,
            workers=per_problem_workers,
            stamp=stamp,
            executor=executor,
        )

    reports = _map_problems(
        problems, _judge_one, parallel=parallel,
        line=lambda r: f"{r['problem_id']} ... {r['status']} "
                       f"(n_pass={r['n_pass']}, floor={r['gt_floor']})",
    )

    report_path = Path(args.report) if args.report else _SOLVER_REPORT
    total = _merge_report(report_path, reports)
    print(f"\n[judge] report written to {report_path} "
          f"({len(reports)} judged, {total} total rows)")
    _print_solver_summary(reports)


def _map_problems(items: list, fn, *, line, parallel: int) -> list[dict]:
    """Run fn(item) over items with `parallel` problems in flight.

    Prints `line(result)` in completion order; returns results in input order.
    Problem-level parallelism multiplies each problem's internal WebPPL
    workers — callers split their worker budget accordingly.
    """
    results: list = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            print(f"  [{done}/{len(items)}] {line(results[i])}", flush=True)
    return results


def _print_solver_summary(reports: list[dict]) -> None:
    status_counts: dict[str, int] = {}
    non_accept: list[dict] = []
    for r in reports:
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        if st != "accept":
            non_accept.append(r)

    print("\n" + "=" * 70)
    print("SOLVER GATE SUMMARY")
    print("=" * 70)
    total = len(reports)
    for status, n in sorted(status_counts.items()):
        print(f"  {status:<20s} {n:>4d} / {total}")

    if non_accept:
        print()
        print("  Non-accept problems:")
        print(f"  {'problem_id':<50s}  {'status':<20s}  reason")
        print("  " + "-" * 100)
        for r in non_accept:
            reason = (r["errors"] or [""])[ 0][:60] if r["errors"] else ""
            dist_info = ""
            if r["distances"] and any(d is not None for d in r["distances"]):
                ds = [str(d) if d is not None else "?" for d in r["distances"]]
                dist_info = f"d=[{','.join(ds)}]"
            note = f"{dist_info} {reason}".strip()
            print(f"  {r['problem_id']:<50s}  {r['status']:<20s}  {note}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Cross-language consistency check
# ---------------------------------------------------------------------------

_CROSSCHECK_REPORT = Path("data/problems/_gate_crosscheck_report.jsonl")


def crosscheck_problem(
    problem: dict,
    target_real: dict,
    reference_real: dict,
    *,
    target_executor,
    reference_executor,
    base_seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    k_exact: int = _DEFAULT_K_EXACT,
    k_draws: int = _DEFAULT_K_DRAWS,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_MC_WORKERS,
) -> dict:
    """Judge the reference-language GT against k target-language GT runs.

    The target's own multi-seed noise floor sets the tolerance, so a sampled
    target realization (e.g. Pyro importance sampling) is comparable against
    an exact reference (e.g. WebPPL enumeration). Statuses mirror judge():
    pass / fail / ill_posed (target floor exceeds caps) / error.
    """
    pid = problem["problem_id"]
    t0 = time.time()
    try:
        spec = parse_spec(problem["answer_spec"])
    except (AlgebraError, KeyError, TypeError) as exc:
        return {"problem_id": pid, "status": "error", "error": f"bad spec: {exc}"}

    try:
        target_gts, _ = collect_gt_answers(
            target_real["code"], spec,
            base_seed=base_seed, n_draws=n_draws,
            k_exact=k_exact, k_draws=k_draws,
            timeout=timeout, workers=workers, executor=target_executor,
        )
    except (RuntimeError, AlgebraError) as exc:
        return {"problem_id": pid, "status": "error",
                "error": f"target GT failed: {exc}",
                "runtime_sec": round(time.time() - t0, 3)}

    try:
        ref_canon = execute_candidate_answer(
            reference_real["code"], spec,
            base_seed=base_seed, n_draws=n_draws,
            timeout=timeout, workers=workers, executor=reference_executor,
        )
    except (RuntimeError, AlgebraError) as exc:
        return {"problem_id": pid, "status": "error",
                "error": f"reference GT failed: {exc}",
                "runtime_sec": round(time.time() - t0, 3)}

    try:
        v = verdict(ref_canon, target_gts, spec)
    except AlgebraError as exc:
        return {"problem_id": pid, "status": "error",
                "error": f"verdict failed: {exc}",
                "runtime_sec": round(time.time() - t0, 3)}

    from eval.algebra import status_of
    return {
        "problem_id": pid,
        "status": status_of(v),
        "distance": round(v["distance"], 6),
        "tol": v["tol"],
        "target_floor": round(v["floor"], 6),
        "metric": v["metric"],
        "runtime_sec": round(time.time() - t0, 3),
    }


def cmd_crosscheck(args) -> None:
    """Cross-language gate: target-language realizations vs reference GT."""
    target_executor = executor_for(args.language)
    reference_executor = executor_for(args.reference)

    id_set = set(args.ids) if args.ids else None
    problems, target_reals = load_corpus(id_set, language=args.language)
    _, ref_list = load_corpus(
        {p["problem_id"] for p in problems}, language=args.reference)
    ref_by_id = {r["problem_id"]: r for r in ref_list}

    pairs = [(p, tr) for p, tr in zip(problems, target_reals)
             if p["problem_id"] in ref_by_id]
    if not pairs:
        print("No problems with realizations in both languages.")
        return

    print(f"[crosscheck] {len(pairs)} problem(s): "
          f"{args.language} (target) vs {args.reference} (reference)")
    parallel = min(4, max(1, len(pairs)))
    per_problem_workers = max(1, args.workers // parallel)

    def _check_one(pair: tuple) -> dict:
        prob, target_real = pair
        row = crosscheck_problem(
            prob, target_real, ref_by_id[prob["problem_id"]],
            target_executor=target_executor,
            reference_executor=reference_executor,
            base_seed=args.seed, n_draws=args.n_draws,
            k_exact=args.k_exact, k_draws=args.k_draws,
            timeout=args.timeout, workers=per_problem_workers,
        )
        return {**row, "language": args.language, "reference": args.reference}

    def _line(r: dict) -> str:
        d = f"d={r['distance']} tol={round(r['tol'], 6)}" if "distance" in r else \
            f"ERR: {r.get('error', '')[:70]}"
        return f"{r['problem_id']} ... {r['status']} {d}"

    reports = _map_problems(pairs, _check_one, parallel=parallel, line=_line)

    report_path = Path(args.report) if args.report else _CROSSCHECK_REPORT
    total = _merge_report(report_path, reports)
    print(f"\n[crosscheck] report written to {report_path} "
          f"({len(reports)} checked, {total} total rows)")
    counts: dict[str, int] = {}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for s, n in sorted(counts.items()):
        print(f"  {s:<12s} {n:>4d} / {len(reports)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    top = argparse.ArgumentParser(
        description="Gate: Phase-A GT noise-floor + Phase-B solver re-derivation."
    )
    subs = top.add_subparsers(dest="subcommand")

    # ------------------------------------------------------------------
    # Phase-A (default behaviour — invoked when no subcommand is given)
    # ------------------------------------------------------------------
    def _add_phase_a_args(p):
        p.add_argument(
            "--ids", nargs="+", default=None, metavar="ID",
            help="Restrict to specific problem IDs.",
        )
        p.add_argument(
            "--language", default="webppl",
            help="Realization language (selects realization file + executor).",
        )
        p.add_argument(
            "--workers", type=int, default=DEFAULT_MC_WORKERS,
            help=f"Thread-pool workers for WebPPL execution (default {DEFAULT_MC_WORKERS}).",
        )
        p.add_argument(
            "--n-draws", type=int, default=DEFAULT_N_MC,
            help=f"Draws per GT block for draws-protocol specs (default {DEFAULT_N_MC}).",
        )
        p.add_argument(
            "--k-exact", type=int, default=_DEFAULT_K_EXACT,
            help=f"GT runs for non-draws specs (default {_DEFAULT_K_EXACT}).",
        )
        p.add_argument(
            "--k-draws", type=int, default=_DEFAULT_K_DRAWS,
            help=f"GT blocks for draws-protocol specs (default {_DEFAULT_K_DRAWS}).",
        )
        p.add_argument(
            "--seed", type=int, default=DEFAULT_SEED,
            help=f"Base random seed (default {DEFAULT_SEED}).",
        )
        p.add_argument(
            "--timeout", type=int, default=DEFAULT_TIMEOUT,
            help=f"Per-execution timeout in seconds (default {DEFAULT_TIMEOUT}).",
        )
        p.add_argument(
            "--report", default=str(_DEFAULT_REPORT), metavar="PATH",
            help=f"Output JSONL path (default {_DEFAULT_REPORT}).",
        )

    phase_a = subs.add_parser(
        "phaseA",
        help="Phase-A gate: measure GT noise floor for each (problem, realization).",
    )
    _add_phase_a_args(phase_a)

    # ------------------------------------------------------------------
    # Cross-language check
    # ------------------------------------------------------------------
    cross_p = subs.add_parser(
        "crosscheck",
        help="Judge a reference language's GT against the target language's "
             "multi-seed GT runs (target floor sets tolerance).",
    )
    _add_phase_a_args(cross_p)
    cross_p.add_argument(
        "--reference", default="webppl",
        help="Reference language whose GT is judged (default webppl).",
    )
    cross_p.set_defaults(report=str(_CROSSCHECK_REPORT))

    # ------------------------------------------------------------------
    # Phase-B: solve
    # ------------------------------------------------------------------
    solve_p = subs.add_parser(
        "solve",
        help="Phase-B: render prompts and submit solver batch (or --dry-run).",
    )
    solve_p.add_argument(
        "--ids", nargs="+", default=None, metavar="ID",
        help="Restrict to specific problem IDs.",
    )
    solve_p.add_argument(
        "--dry-run", action="store_true",
        help="Write manifest with rendered prompts; print 3 samples; do not submit.",
    )
    solve_p.add_argument(
        "--model", default=_SOLVE_MODEL,
        help=f"Solver model (default {_SOLVE_MODEL}).",
    )
    solve_p.add_argument(
        "--language", default="webppl",
        help="Target language for prompts + judging (recorded in the manifest).",
    )
    solve_p.add_argument(
        "--manifest", default=None, metavar="PATH",
        help=f"Path for batch manifest JSON (default {_SOLVE_MANIFEST}).",
    )

    # ------------------------------------------------------------------
    # Phase-B: judge
    # ------------------------------------------------------------------
    judge_p = subs.add_parser(
        "judge",
        help="Phase-B: poll batch, execute solvers, write solver report.",
    )
    judge_p.add_argument(
        "--batch-id", default=None,
        help="Batch ID to retrieve (default: read from manifest).",
    )
    judge_p.add_argument(
        "--manifest", default=None, metavar="PATH",
        help=f"Path to manifest JSON (default {_SOLVE_MANIFEST}).",
    )
    judge_p.add_argument(
        "--report", default=None, metavar="PATH",
        help=f"Output JSONL path (default {_SOLVER_REPORT}).",
    )
    judge_p.add_argument(
        "--workers", type=int, default=DEFAULT_MC_WORKERS,
    )
    judge_p.add_argument(
        "--n-draws", type=int, default=DEFAULT_N_MC,
    )
    judge_p.add_argument(
        "--k-exact", type=int, default=_DEFAULT_K_EXACT,
    )
    judge_p.add_argument(
        "--k-draws", type=int, default=_DEFAULT_K_DRAWS,
    )
    judge_p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
    )
    judge_p.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
    )

    args = top.parse_args()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if args.subcommand == "crosscheck":
        cmd_crosscheck(args)
        return

    if args.subcommand == "solve":
        cmd_solve(args)
        return

    if args.subcommand == "judge":
        cmd_judge(args)
        return

    # No subcommand or "phaseA" → run Phase A (preserves existing default behaviour)
    if args.subcommand is None:
        # Re-parse with Phase-A defaults applied to top-level args
        # by adding Phase-A args directly to the top-level parser as well
        # so bare invocation `python -m eval.gate` still works.
        # We need a second parse with just Phase-A args.
        pa = argparse.ArgumentParser(
            description="Phase-A gate: measure GT noise floor for each (problem, realization)."
        )
        _add_phase_a_args(pa)
        args = pa.parse_args()

    id_set = set(args.ids) if args.ids else None
    problems, realizations = load_corpus(id_set, language=args.language)
    executor = executor_for(args.language)

    if not problems:
        print("No matching problems found.")
        return

    print(
        f"Running Phase-A gate on {len(problems)} problem(s)  "
        f"[workers={args.workers}, n_draws={args.n_draws}, "
        f"k_exact={args.k_exact}, k_draws={args.k_draws}]"
    )

    parallel = min(4, max(1, len(problems)))
    per_problem_workers = max(1, args.workers // parallel)

    def _gate_one(pair: tuple) -> dict:
        prob, real = pair
        return gate_problem(
            prob, real,
            base_seed=args.seed,
            n_draws=args.n_draws,
            k_exact=args.k_exact,
            k_draws=args.k_draws,
            timeout=args.timeout,
            workers=per_problem_workers,
            executor=executor,
        )

    def _pa_line(r: dict) -> str:
        floor = f"floor={r['floor']}" if r["floor"] is not None else ""
        err = f" ERR: {r.get('error', '')[:80]}" if r["status"] == "error" else ""
        return f"{r['problem_id']} ... {r['status']} {floor}{err}  ({r['runtime_sec']}s)"

    reports = _map_problems(
        list(zip(problems, realizations)), _gate_one,
        parallel=parallel, line=_pa_line,
    )

    report_path = Path(args.report)
    total = _merge_report(report_path, reports)
    print(f"\nReport written to {report_path} ({len(reports)} judged, {total} total rows)")
    _print_summary(reports)


if __name__ == "__main__":
    main()
