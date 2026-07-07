"""Validate Gen (Gen.jl) realizations against a multi-seed WebPPL reference.

Box-only: needs Julia + Gen.jl (set PPL_GYM_JULIA). This is the reusable
authoring gate for the Gen realization column — it replaces the per-batch
scratch scripts.

The gate is a cross-*language* check (Gen realization vs WebPPL realization),
so the tolerance floor must reflect the estimator's real run-to-run noise. A
single stored ``_gt_answers`` answer cannot reveal that noise (``self_noise`` is
0 for a dist_enum histogram, tiny for one long cloud), which false-fails a
correct Gen posterior whenever Gen mixes differently than WebPPL. So the GT is a
**multi-seed WebPPL reference** (``eval.webppl_ref``, generated on the laptop):
for each problem we canonicalize its k WebPPL runs into the GT set, run the Gen
program over ``seeds``, and judge each Gen run as a candidate against that GT
(``eval.algebra.judge`` — the exact path ``score.py`` uses; the floor is
WebPPL's own cross-run noise). The realization passes iff every Gen run passes.

Two executor regimes fall out for free:

  - **exact** (no ``PPLGYM_SAMPLE`` marker): ``enumerative_inference`` is
    deterministic, so the executor replicates one answer across seeds; the GT
    (deterministic WebPPL enumerate) floor is ~eps → a near-exact match required.
  - **sampling** (``PPLGYM_SAMPLE`` marker): mh/importance/forward runs per seed.
    Author it with the SAME sample counts the statement/WebPPL GT pins so the two
    estimators carry comparable noise.

CLI:
  PPL_GYM_JULIA=/workspace/julia-1.10.5/bin/julia \\
  PYTHONPATH=. /venv/main/bin/python -m eval.gen_validate \\
      --batch batch.json [--ref data/webppl_ref.json] [--seeds 42 43 ...] [--merge]

``batch.json`` = a JSON list of ``{"problem_id", "code"}`` records, or a
``{problem_id: code}`` object. ``--merge`` upserts the passers into
``data/realizations/gen.jsonl`` (merge-by-problem_id, never clobbers other rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.algebra import AlgebraError, canonicalize, judge, parse_spec
from eval.corpus import batch_executor_for, load_problems
from eval.io import merge_jsonl

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _gt_canons(ref_entry: dict, spec) -> list:
    """Canonicalize a WebPPL ref entry into the GT answer set.

    Non-draws: one GT answer per WebPPL run. Draws-protocol: pool the flat run
    list into ``k_draws`` blocks of ``n_draws`` single draws, one GT answer per
    block — matching ``eval.harness.collect_gt_answers``.
    """
    raw = ref_entry["answers"]
    if ref_entry.get("draws"):
        n = ref_entry["n_draws"]
        gts = []
        for b in range(ref_entry["k_draws"]):
            chunk = [a for a in raw[b * n:(b + 1) * n] if a is not None]
            if chunk:
                gts.append(canonicalize(chunk, spec))
        return gts
    return [canonicalize(a, spec) for a in raw if a is not None]


def validate_one(
    pid: str, code: str, spec, ref_entry: dict, *, language: str, seeds: list[int], timeout: int, workers: int
) -> dict:
    """Judge a target-language realization against the multi-seed WebPPL reference.

    ``ref_entry`` = the WebPPL ref for this problem (from ``eval.webppl_ref``).
    Canonicalize it into the GT set, run the ``language`` program over ``seeds``,
    then judge each run as a candidate against that GT (``eval.algebra.judge`` —
    the exact path ``score.py`` uses; the tolerance floor is WebPPL's real
    cross-run noise). The realization passes iff every run passes. Language-
    agnostic: works for gen (box, Julia) and pyro (box, torch) alike.
    """
    try:
        gts = _gt_canons(ref_entry, spec)
    except AlgebraError as e:
        return {"problem_id": pid, "status": "gt_error", "error": f"webppl ref: {str(e)[:280]}"}
    if len(gts) < 2:
        return {"problem_id": pid, "status": "gt_error", "error": "webppl ref has <2 usable runs"}

    try:
        answers, errors = batch_executor_for(language)(code, seeds, timeout, workers)
    except Exception as e:  # whole-run failure surfaces the real reason
        return {"problem_id": pid, "status": "exec_error", "error": str(e)[:300]}
    gen_ok = [a for a in answers if a is not None]
    if not gen_ok:
        reason = next((e for e in errors if e), f"{language} produced no runs")
        return {"problem_id": pid, "status": "exec_error", "error": str(reason)[:300]}

    verdicts = [judge(a, gts, spec) for a in gen_ok]  # each Gen run vs the WebPPL GT set
    n_pass = sum(1 for v in verdicts if v.get("status") == "pass")
    worst = max(verdicts, key=lambda v: (v.get("distance") or 0.0))
    status = "pass" if n_pass == len(verdicts) else worst.get("status", "fail")
    return {
        "problem_id": pid,
        "status": status,
        "distance": worst.get("distance"),
        "tol": worst.get("tol"),
        "floor": worst.get("floor"),
        "pass_frac": f"{n_pass}/{len(verdicts)}",
        "n_gt": len(gts),
        "sampling": "PPLGYM_SAMPLE" in code,
    }


def load_batch(path: Path) -> list[dict]:
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict):
        return [{"problem_id": k, "code": v} for k, v in obj.items()]
    return obj


def run(batch, *, language, ref_path, seeds, timeout, workers, merge):
    real_path = Path(f"data/realizations/{language}.jsonl")
    problems = {p["problem_id"]: p for p in load_problems()}
    ref = json.loads(ref_path.read_text())  # {pid: {seeds, answers}} from eval.webppl_ref

    results, passers = [], []
    for rec in batch:
        pid, code = rec["problem_id"], rec["code"]
        if pid not in problems:
            results.append({"problem_id": pid, "status": "corpus_miss", "error": "no problem"})
        elif pid not in ref:
            results.append({"problem_id": pid, "status": "gt_error", "error": f"no WebPPL ref (run eval.webppl_ref --ids {pid})"})
        else:
            spec = parse_spec(problems[pid]["answer_spec"])
            r = validate_one(pid, code, spec, ref[pid], language=language, seeds=seeds, timeout=timeout, workers=workers)
            results.append(r)
            if r["status"] == "pass":
                passers.append({"problem_id": pid, "language": language, "code": code.strip(), "available": True})

        r = results[-1]
        extra = (f"  d={r.get('distance')} tol={r.get('tol')}  ({r.get('pass_frac')})"
                 if r.get("distance") is not None else f"  {r.get('error', '')}")
        print(f"[{r['status']:>10}] {pid}{extra}", flush=True)

    n_pass = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{n_pass}/{len(batch)} pass", flush=True)

    if merge and passers:
        n = merge_jsonl(real_path, passers)
        print(f"merged {len(passers)} passers -> {real_path} ({n} total rows)", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a target-language realization column vs the multi-seed WebPPL reference.")
    p.add_argument("--batch", required=True, help="JSON: list of {problem_id,code} or {pid:code}.")
    p.add_argument("--language", default="gen", help="target realization language (gen/pyro/...); selects executor + merge file.")
    p.add_argument("--ref", default="data/webppl_ref.json",
                   help="multi-seed WebPPL reference (from eval.webppl_ref).")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--timeout", type=int, default=300, help="per-run timeout seconds (default 300).")
    p.add_argument("--workers", type=int, default=len(DEFAULT_SEEDS))
    p.add_argument("--merge", action="store_true", help="upsert passers into data/realizations/<language>.jsonl.")
    a = p.parse_args()
    run(load_batch(Path(a.batch)), language=a.language, ref_path=Path(a.ref), seeds=a.seeds,
        timeout=a.timeout, workers=a.workers, merge=a.merge)


if __name__ == "__main__":
    main()
