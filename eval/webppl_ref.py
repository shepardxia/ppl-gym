"""Generate multi-seed WebPPL reference answers for the Gen crosscheck.

Runs on the LAPTOP (needs node + the probmods2 webppl bundle; the box has no
webppl). For each problem it executes the WebPPL realization over k seeds and
dumps the raw (untrimmed) wire answers to JSON keyed by problem_id.

Why multi-seed: a single stored ``_gt_answers`` answer — cloud or dist_enum
histogram — cannot reveal an estimator's run-to-run noise (``self_noise`` is 0
for a histogram, tiny for one long cloud). A tolerance floor built from it is
too tight for a sharply-peaked MCMC posterior, and false-fails a correct Gen
realization whenever Gen mixes differently than WebPPL. ``eval/gen_validate.py``
loads these refs and judges the Gen answer against the k WebPPL runs, so the
floor is WebPPL's OWN cross-run noise — the same path ``score.py`` uses with a
multi-seed GT.

CLI:
  PYTHONPATH=. .venv/bin/python -m eval.webppl_ref \\
      --ids <pid> ... | --missing-gen   --out data/webppl_ref.json \\
      [--seeds 42 43 44 45 46] [--timeout 300] [--workers 5]

``--missing-gen`` selects every WebPPL-realized problem that has no Gen
realization yet (the crosscheck backlog).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.algebra import parse_spec
from eval.config import DEFAULT_N_MC
from eval.corpus import batch_executor_for, load_problems, load_realizations
from eval.harness import _DEFAULT_K_DRAWS, _has_draws_field


def missing_gen_ids() -> list[str]:
    gen = {r["problem_id"] for r in load_realizations("gen")}
    wppl = [r["problem_id"] for r in load_realizations("webppl")]
    return [p for p in wppl if p not in gen]


def generate(ids: list[str], *, seeds: list[int], timeout: int, workers: int, out: Path) -> None:
    code_by_id = {r["problem_id"]: r.get("code", "") for r in load_realizations("webppl")}
    spec_by_id = {r["problem_id"]: parse_spec(r["answer_spec"]) for r in load_problems()}
    run = batch_executor_for("webppl")

    # Merge into any existing ref file so partial re-runs accumulate.
    ref: dict = {}
    if out.exists():
        ref = json.loads(out.read_text())

    for i, pid in enumerate(ids, 1):
        code = code_by_id.get(pid)
        if not code:
            print(f"[{i}/{len(ids)}] {pid}  SKIP (no webppl code)", flush=True)
            continue
        spec = spec_by_id.get(pid)
        # A draws-protocol answer is built by pooling n_draws single draws; the
        # GT is k_draws such blocks (matches eval.harness.collect_gt_answers).
        draws = spec is not None and _has_draws_field(spec)
        if draws:
            n_draws = DEFAULT_N_MC
            run_seeds = list(range(seeds[0], seeds[0] + _DEFAULT_K_DRAWS * n_draws))
        else:
            run_seeds = seeds
        try:
            answers, errors = run(code, run_seeds, timeout, workers)
        except Exception as e:  # whole-run failure
            print(f"[{i}/{len(ids)}] {pid}  RUN-FAIL {str(e)[:120]}", flush=True)
            continue
        ok = [a for a in answers if a is not None]
        entry = {"answers": answers}
        if draws:
            entry.update(draws=True, n_draws=n_draws, k_draws=_DEFAULT_K_DRAWS)
        else:
            entry["seeds"] = run_seeds
        ref[pid] = entry
        out.write_text(json.dumps(ref))  # checkpoint after each (heavy runs)
        print(f"[{i}/{len(ids)}] {pid}  {len(ok)}/{len(run_seeds)} runs ok"
              + ("  [draws-pooled]" if draws else "")
              + (f"  ERR {next((e for e in errors if e), '')[:100]}" if len(ok) < len(run_seeds) else ""),
              flush=True)
    print(f"wrote {len(ref)} refs -> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed WebPPL reference answers for Gen crosscheck.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", nargs="+", metavar="ID")
    g.add_argument("--missing-gen", action="store_true", help="all WebPPL problems lacking a Gen realization.")
    p.add_argument("--out", default="data/webppl_ref.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    ids = missing_gen_ids() if a.missing_gen else a.ids
    print(f"generating WebPPL refs for {len(ids)} problems, seeds={a.seeds}", flush=True)
    generate(ids, seeds=a.seeds, timeout=a.timeout, workers=a.workers, out=Path(a.out))


if __name__ == "__main__":
    main()
