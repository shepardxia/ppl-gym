"""Backfill the real cause behind LEGACY exec_error rows in a benchmark matrix.

Fresh runs now carry the real reason + ``error_tag`` at scoring time (the
executors surface each failed seed's ``error_message`` and eval.error_tags
classifies it). This tool exists for matrices scored BEFORE that change, whose
exec_error rows still hold the collapsed placeholder ("execution failed" /
"n/k seeded runs failed") with the real reason lost. It re-executes those
candidates to recover the cause, then classifies with the SAME eval.error_tags
``classify`` — so a re-executed legacy row and a fresh scored row land in the
same bucket and carry the same ``error`` + ``error_tag`` fields.

Rows that already carry a real reason (non-collapsed) are classified in place,
no re-execution.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.config import DEFAULT_SEED
from eval.error_tags import classify, is_collapsed
from eval.io import load_jsonl, write_jsonl


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
        reexec_langs: set[str], apply: bool = False) -> None:
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

    jobs: list[dict] = []          # collapsed rows needing re-execution
    rows: list[dict] = []          # all exec_error rows (error_tag filled)
    for f in scored_files:
        cell = Path(f).parent.name
        model, _, language = cell.partition("__")
        collapsed_seen = 0
        for line in open(f):
            r = json.loads(line)
            if r.get("status") != "exec_error":
                continue
            orig = r.get("error")
            base = {
                "model": model, "language": language, "cell": cell,
                "problem_id": r.get("problem_id"), "slot": r.get("slot"),
                "orig_error": orig, "error": orig,
                "error_tag": classify(orig, language),
                "code_len": len(r.get("code", "")),
            }
            if is_collapsed(orig) and language in reexec_langs:
                collapsed_seen += 1
                if collapsed_seen <= per_cell:
                    jobs.append({**base, "_code": r.get("code", ""),
                                 "_gt": stan_gt.get(r.get("problem_id"), "")})
                else:
                    base["sampled"] = False
                    rows.append(base)
            else:
                # not re-executed (already has a real reason, or a language
                # excluded from re-exec); keep candidate code if still collapsed.
                if is_collapsed(orig):
                    base["code"] = r.get("code", "")[:6000]
                rows.append(base)

    print(f"[triage] {len(rows)} rows tagged in place; "
          f"re-executing {len(jobs)} sampled collapsed rows "
          f"(per_cell={per_cell})...", flush=True)

    def work(j: dict) -> dict:
        real, stderr = _reexec(j["language"], j["_code"], j.get("_gt"), timeout)
        out = {k: v for k, v in j.items() if not k.startswith("_")}
        # Recovered reason replaces the collapsed placeholder; classify with the
        # SAME function score.py uses -> identical `error` + `error_tag` schema.
        out.update(sampled=True, stderr_head=stderr,
                   error=real, error_tag=classify(real, j["language"]),
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

    if apply:
        apply_to_scored(scored_files, rows)


def apply_to_scored(scored_files: list[str], rows: list[dict]) -> None:
    """Write recovered ``error`` + ``error_tag`` back onto the matrix's exec_error
    rows in place — so legacy scored data matches what fresh runs now produce and
    export_rollouts (HF + web) carries the real failure class. Keyed by
    (cell, problem_id, slot); non-exec_error rows and summaries are untouched.
    """
    recovered = {(r["cell"], r.get("problem_id"), r.get("slot")):
                 (r.get("error"), r.get("error_tag")) for r in rows}
    n_rows = n_files = 0
    for f in scored_files:
        cell = Path(f).parent.name
        srows = load_jsonl(f)
        changed = False
        for sr in srows:
            if sr.get("summary") or sr.get("status") != "exec_error":
                continue
            hit = recovered.get((cell, sr.get("problem_id"), sr.get("slot")))
            if hit is not None:
                sr["error"], sr["error_tag"] = hit
                changed = True
                n_rows += 1
        if changed:
            write_jsonl(f, srows)
            n_files += 1
    print(f"[triage] applied error_tag to {n_rows} exec_error rows "
          f"across {n_files} files", flush=True)


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
    p.add_argument("--apply", action="store_true",
                   help="write recovered error + error_tag back onto the matrix's "
                        "scored.jsonl exec_error rows (legacy backfill).")
    a = p.parse_args()
    run(a.dirs, Path(a.out), per_cell=a.per_cell, timeout=a.timeout, workers=a.workers,
        reexec_langs=set(x for x in a.reexec_langs.split(",") if x), apply=a.apply)


if __name__ == "__main__":
    main()
