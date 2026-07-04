"""Recover the real cause behind exec_error rows in a benchmark matrix.

The scoring harness collapses an executor failure to ``None`` and then to the
generic ``RuntimeError("execution failed")`` (eval/harness.py), discarding the
``ExecutionResult.error_message`` / ``.stderr`` the executor actually captured.
For triage we re-execute the *generic* exec_error candidates through the executor
directly and surface the real error, so a downstream classifier can split the
exec_error bucket into harness-artifact vs genuine-model-failure.

Mechanical categories (no re-execution needed):
  timeout     — error already says "timeout"
  compile     — Stan candidate failed to compile (already surfaced)
  gt_side     — GT collection failed (our ground truth could not run -> harness)
  empty_code  — model emitted nothing
  generic     — bare "execution failed"; re-executed to recover real_error

CLI:
  PYTHONPATH=. .venv/bin/python -m eval.triage_exec_errors \\
      --dirs 'runs/matrix/*' \\
      --out runs/matrix/triage/exec_errors.jsonl \\
      [--per-cell 30] [--timeout 20] [--workers 12]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.config import DEFAULT_SEED


def _mech_cat(err: str) -> str:
    e = (err or "").lower()
    if "timeout" in e:
        return "timeout"
    if "gt collection failed" in e:
        return "gt_side"
    if "empty code" in e:
        return "empty_code"
    if "compile" in e:
        return "compile"
    if "not found" in e:
        return "corpus_miss"
    if e == "execution failed" or re.match(r"^\d+/\d+ seeded runs failed$", e):
        # Legacy bare form + the current n/k form: both mean the candidate died
        # without a captured reason — re-execute to recover it.
        return "generic"
    return "other"


def _reexec(language: str, code: str, gt_bundle: str | None, timeout: int) -> tuple[str, str]:
    """Re-run one candidate; return (real_error, stderr_head)."""
    try:
        if language == "pyro":
            from eval.executor_pyro import execute_pyro
            r = execute_pyro(code, timeout=timeout, random_seed=DEFAULT_SEED)
        elif language == "webppl":
            from eval.executor import execute_webppl
            r = execute_webppl(code, timeout=timeout, random_seed=DEFAULT_SEED)
        elif language == "stan":
            from eval.stan_bundle import DEFAULT_SAMPLING, repack_model, unpack
            from eval.executor_stan import _compiled_model, _cwd_guard
            packed = repack_model(gt_bundle, code, sampling=DEFAULT_SAMPLING) if gt_bundle else code
            # cmdstanpy compile/sample leaks process CWD; guard the boundary so a
            # later relative-path file op (e.g. the triage write) is not redirected.
            with _cwd_guard():
                try:
                    b = unpack(packed)
                    model = _compiled_model(b.model)
                except Exception as e:  # compile error -> already a real reason
                    return (f"stan compile/unpack: {str(e).splitlines()[0][:200]}", str(e)[:500])
                # _one_fit swallows its exception; reproduce it here to surface the cause.
                try:
                    fit = model.sample(
                        data=b.data, seed=DEFAULT_SEED,
                        chains=b.sampling.get("chains", 4),
                        parallel_chains=b.sampling.get("chains", 4),
                        iter_warmup=b.sampling.get("iter_warmup", 1000),
                        iter_sampling=b.sampling.get("iter_sampling", 1000),
                        show_progress=False, show_console=False, timeout=timeout,
                    )
                    df = fit.draws_pd()
                    missing = [p for p in b.params if p not in df.columns]
                    if missing:
                        return (f"stan: param(s) not exposed: {missing}", "")
                    return ("(re-run succeeded — nondeterministic / flaky)", "")
                except Exception as e:
                    return (f"stan fit: {str(e).splitlines()[0][:200]}", str(e)[:500])
        else:
            return (f"unknown language {language}", "")
        if r.success:
            return ("(re-run succeeded — nondeterministic / flaky)", "")
        return (r.error_message or "(no error_message)", (r.stderr or "")[:500])
    except Exception as e:  # noqa: BLE001 — triage must never crash on one row
        return (f"reexec raised: {type(e).__name__}: {str(e)[:200]}", "")


def run(dirs_glob: str, out_path: Path, *, per_cell: int, timeout: int, workers: int,
        reexec_langs: set[str]) -> None:
    # Anchor the output path before any Stan execution: cmdstanpy leaks process
    # CWD, so a relative write after a Stan re-run lands in the wrong directory.
    out_path = out_path.resolve()
    scored_files = sorted(glob.glob(f"{dirs_glob}/scored.jsonl"))
    if not scored_files:
        raise SystemExit(f"no scored.jsonl under {dirs_glob}")

    # Stan GT bundles (needed to repack a bare candidate model).
    stan_gt: dict[str, str] = {}
    if "stan" in reexec_langs and any("__stan" in f for f in scored_files):
        from eval.corpus import load_realizations
        stan_gt = {r["problem_id"]: r.get("code", "") for r in load_realizations("stan")}

    jobs: list[dict] = []          # generic rows needing re-execution
    rows: list[dict] = []          # all exec_error rows (mechanical cats filled)
    for f in scored_files:
        cell = Path(f).parent.name
        model, _, language = cell.partition("__")
        generic_seen = 0
        for line in open(f):
            r = json.loads(line)
            if r.get("status") != "exec_error":
                continue
            cat = _mech_cat(r.get("error"))
            base = {
                "model": model, "language": language, "cell": cell,
                "problem_id": r.get("problem_id"), "slot": r.get("slot"),
                "orig_error": r.get("error"), "mech_cat": cat,
                "code_len": len(r.get("code", "")),
            }
            if cat == "generic" and language in reexec_langs:
                generic_seen += 1
                if generic_seen <= per_cell:
                    jobs.append({**base, "_code": r.get("code", ""),
                                 "_gt": stan_gt.get(r.get("problem_id"), "")})
                else:
                    base["sampled"] = False
                    rows.append(base)
            else:
                # not re-executed (non-generic, or a language excluded from
                # re-exec); keep the candidate code so a classifier can read it.
                if cat == "generic":
                    base["code"] = r.get("code", "")[:6000]
                rows.append(base)

    print(f"[triage] {len(rows)} non-generic exec_error rows; "
          f"re-executing {len(jobs)} sampled generic rows "
          f"(per_cell={per_cell})...", flush=True)

    def work(j: dict) -> dict:
        real, stderr = _reexec(j["language"], j["_code"], j.get("_gt"), timeout)
        out = {k: v for k, v in j.items() if not k.startswith("_")}
        out.update(sampled=True, real_error=real, stderr_head=stderr,
                   code=j["_code"][:6000])
        return out

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(work, jobs):
            rows.append(res)
            done += 1
            if done % 25 == 0:
                print(f"  [{done}/{len(jobs)}] re-executed", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[triage] wrote {len(rows)} rows -> {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Recover real cause of exec_error rows.")
    p.add_argument("--dirs", default="runs/matrix/*/*",
                   help="glob matching cell dirs (each holds scored.jsonl).")
    p.add_argument("--out", default="runs/matrix/triage/exec_errors.jsonl")
    p.add_argument("--per-cell", type=int, default=30,
                   help="max generic rows to re-execute per cell (sample cap).")
    p.add_argument("--timeout", type=int, default=20,
                   help="per-candidate re-exec timeout (generic = fast crashes).")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--reexec-langs", default="webppl,pyro",
                   help="languages to re-execute (default skips stan: cmdstan "
                        "compile/NUTS is heavy and stan's generic bucket is tiny).")
    a = p.parse_args()
    run(a.dirs, Path(a.out), per_cell=a.per_cell, timeout=a.timeout, workers=a.workers,
        reexec_langs=set(x for x in a.reexec_langs.split(",") if x))


if __name__ == "__main__":
    main()
