"""Multi-model benchmark runner.

Orchestrates the existing pipeline (generate_batch -> score) across a grid of
(model x language) combos and aggregates the per-combo summaries into one
comparison table. Reuses build_requests / submit_batch / collect_results /
write_generation_rows and score.run_scoring -- no duplicated generation or
scoring logic.

Layout under --out:
  <out>/<model-slug>__<language>/generations.jsonl
  <out>/<model-slug>__<language>/scored.jsonl
  <out>/comparison.jsonl   (one summary row per combo)
  <out>/comparison.md      (the rendered table)

WebPPL and Pyro share the 115 textbook problems; Stan is the separate
posteriordb 45. Scorable problems are selected per language via load_corpus,
so the grid handles the corpus split automatically (a problem is only prompted
when a GT realization exists to score it against).

Flow: all (model x language) batches are submitted up front (they run
concurrently on Anthropic's side), then each is polled, collected, and scored.

CLI:
  PYTHONPATH=. .venv/bin/python -m eval.benchmark run \\
      --models sonnet,haiku --languages webppl,pyro,stan \\
      --n-samples 3 --out runs/<name> [--limit N] [--ids ID ...]
  PYTHONPATH=. .venv/bin/python -m eval.benchmark report --out runs/<name>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import Anthropic

from eval.config import DEFAULT_N_MC, DEFAULT_SEED, DEFAULT_TIMEOUT
from eval.corpus import load_corpus
from eval.generate_batch import (
    build_requests,
    collect_results,
    submit_batch,
    wait_for_batch,
    write_generation_rows,
)
from eval.score import run_scoring

# Short aliases -> exact model ids (full ids pass through unchanged).
MODEL_ALIASES = {
    "opus": "claude-opus-4-8",
    "opus-4-8": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "sonnet-4-6": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "haiku-4-5": "claude-haiku-4-5-20251001",
}

# Per-combo summary fields lifted from score.run_scoring's summary row.
SUMMARY_COLS = ["n", "pass", "fail", "ill_posed", "malformed", "exec_error", "pass_rate"]


def resolve_model(m: str) -> str:
    return MODEL_ALIASES.get(m, m)


def model_slug(m: str) -> str:
    """Filesystem-safe short label for a model id."""
    return resolve_model(m).replace("claude-", "").replace("/", "_")


@dataclass
class Combo:
    model: str            # resolved model id
    language: str
    problems: list = field(default_factory=list)  # scorable set, carried submit->write
    batch_id: str = ""
    needs_collect: bool = True  # False when generations.jsonl is already on disk


def _scorable_problems(language: str, ids, limit) -> list[dict]:
    """Problems that have a GT realization for `language` (optionally first N)."""
    problems, _ = load_corpus(ids, language=language)
    return problems[:limit] if limit else problems


# ---------------------------------------------------------------------------
# Resume state: a batch manifest + cached-combo detection
# ---------------------------------------------------------------------------

def _manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / "batches.jsonl"


def _load_manifest(out_dir: Path) -> dict:
    """slug -> batch_id for batches already submitted (so re-runs never re-spend)."""
    p = _manifest_path(out_dir)
    m: dict = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                m[d["slug"]] = d["batch_id"]
    return m


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    with open(_manifest_path(out_dir), "w") as f:
        for slug, bid in manifest.items():
            f.write(json.dumps({"slug": slug, "batch_id": bid}) + "\n")


def _scored_summary(scored_path: Path) -> dict | None:
    """The summary row of a scored.jsonl, or None if absent/incomplete."""
    if not Path(scored_path).exists():
        return None
    summary = None
    for line in Path(scored_path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("summary"):
                summary = d
    return summary


# ---------------------------------------------------------------------------
# Submit / collect / score
# ---------------------------------------------------------------------------

def collect_and_score(client, combo: Combo, out_dir: Path, *, n_solvers, seed,
                      n_draws, timeout, workers, poll_interval, poll_timeout) -> dict:
    slug = f"{model_slug(combo.model)}__{combo.language}"
    cdir = out_dir / slug
    cdir.mkdir(parents=True, exist_ok=True)
    gens = cdir / "generations.jsonl"
    scored = cdir / "scored.jsonl"

    if combo.needs_collect:
        print(f"[collect] {slug}: polling {combo.batch_id} ...", flush=True)
        wait_for_batch(client, combo.batch_id, poll_interval=poll_interval, timeout=poll_timeout)
        results = collect_results(client, combo.batch_id)
        write_generation_rows(combo.problems, results, gens,
                              model=combo.model, language=combo.language, n_solvers=n_solvers)

    print(f"[score]   {slug}: executing + judging {len(combo.problems)} problems ...", flush=True)
    summary = run_scoring(gens, scored, language=combo.language,
                          seed=seed, n_draws=n_draws, timeout=timeout, workers=workers)
    return {"model": combo.model, "language": combo.language,
            "batch_id": combo.batch_id, **{k: summary.get(k) for k in SUMMARY_COLS}}


def run_benchmark(models, languages, *, out_dir, ids=None, limit=None, n_solvers=3,
                  with_primer=True, verbose_primer=False,
                  seed=DEFAULT_SEED, n_draws=DEFAULT_N_MC, timeout=DEFAULT_TIMEOUT,
                  score_workers=4, poll_interval=30, poll_timeout=3600) -> list[dict]:
    """Submit + score a (model x language) grid. Resumable: a combo already
    scored is reused; an already-submitted batch (in batches.jsonl) or
    already-collected generations.jsonl is never re-submitted."""
    # Absolute path: the Stan executor (cmdstanpy) can leave the process CWD
    # changed, so any relative file op here could land in the wrong directory.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Anthropic()
    manifest = _load_manifest(out_dir)

    rows: list[dict] = []
    pending: list[Combo] = []
    for model in models:
        rid = resolve_model(model)
        for language in languages:
            slug = f"{model_slug(rid)}__{language}"
            cdir = out_dir / slug
            cached = _scored_summary(cdir / "scored.jsonl")
            if cached is not None:
                print(f"[cached] {slug}: already scored", flush=True)
                rows.append({"model": rid, "language": language,
                             **{k: cached.get(k) for k in SUMMARY_COLS}})
                continue
            problems = _scorable_problems(language, ids, limit)
            if not problems:
                print(f"[skip]   {slug}: no scorable problems", flush=True)
                continue
            gens_exist = (cdir / "generations.jsonl").exists()
            batch_id = manifest.get(slug, "")
            if not gens_exist and not batch_id:
                requests = build_requests(problems, language=language, model=rid,
                                          n_solvers=n_solvers, with_primer=with_primer,
                                          verbose_primer=verbose_primer)
                batch_id = submit_batch(client, requests)
                manifest[slug] = batch_id
                _save_manifest(out_dir, manifest)
                print(f"[submit] {slug}: {len(problems)} x {n_solvers} = "
                      f"{len(requests)} reqs -> {batch_id}", flush=True)
            elif gens_exist:
                print(f"[rescore]{slug}: generations on disk, re-scoring", flush=True)
            else:
                print(f"[reuse]  {slug}: batch {batch_id}", flush=True)
            pending.append(Combo(rid, language, problems, batch_id,
                                 needs_collect=not gens_exist))

    for combo in pending:
        rows.append(collect_and_score(
            client, combo, out_dir, n_solvers=n_solvers, seed=seed, n_draws=n_draws,
            timeout=timeout, workers=score_workers,
            poll_interval=poll_interval, poll_timeout=poll_timeout))
        _write_comparison(out_dir, rows)  # checkpoint after every combo
    return rows


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def render_comparison(rows: list[dict]) -> str:
    header = ["model", "language", *SUMMARY_COLS]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for r in sorted(rows, key=lambda r: (r["language"], model_slug(r["model"]))):
        cells = [model_slug(r["model"]), r["language"]]
        for c in SUMMARY_COLS:
            v = r.get(c)
            if c == "pass_rate" and v is not None:
                cells.append(f"{v:.3f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_comparison(out_dir: Path, rows: list[dict]) -> None:
    out_dir = Path(out_dir)
    with open(out_dir / "comparison.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md = render_comparison(rows)
    (out_dir / "comparison.md").write_text(md + "\n")
    print("\n" + md + "\n", flush=True)


def dump_failures(out_dir) -> None:
    """Per combo, write failures.jsonl: every non-pass scored row joined with the
    rendered prompt, the GT realization code, and the spec — for investigation.

    The scored row already carries the solver `code`, status, error, distance and
    metric; this adds what the solver actually saw (the prompt) and the GT it was
    judged against, so a reviewer can tell prompt issues from model failures.
    """
    from eval.corpus import load_corpus
    from eval.render import render_problem

    out_dir = Path(out_dir).resolve()
    for cdir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        scored = cdir / "scored.jsonl"
        if not scored.exists():
            continue
        _, _, language = cdir.name.partition("__")
        rows = [json.loads(l) for l in scored.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if not r.get("summary") and r.get("status") != "pass"]
        pids = {r["problem_id"] for r in rows}
        problems, reals = load_corpus(pids, language=language) if pids else ([], [])
        prob_by = {p["problem_id"]: p for p in problems}
        real_by = {r["problem_id"]: r for r in reals}
        with open(cdir / "failures.jsonl", "w") as f:
            for r in rows:
                pid = r["problem_id"]
                prob = prob_by.get(pid)
                f.write(json.dumps({
                    "problem_id": pid, "slot": r.get("slot"),
                    "status": r.get("status"), "error": r.get("error"),
                    "distance": r.get("distance"), "tol": r.get("tol"),
                    "metric": r.get("metric"),
                    "spec": prob.get("answer_spec") if prob else None,
                    "prompt": render_problem(prob, language=language) if prob else None,
                    "solver_code": r.get("code", ""),
                    "gt_code": real_by.get(pid, {}).get("code", ""),
                }) + "\n")
        print(f"{cdir.name}: {len(rows)} failures")


def report(out_dir) -> list[dict]:
    """Re-aggregate the comparison from each combo's scored.jsonl summary on disk."""
    out_dir = Path(out_dir).resolve()
    rows: list[dict] = []
    for cdir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        scored = cdir / "scored.jsonl"
        if not scored.exists():
            continue
        model, _, language = cdir.name.partition("__")
        summary = None
        for line in scored.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("summary"):
                summary = d
        if summary:
            rows.append({"model": model, "language": language,
                         **{k: summary.get(k) for k in SUMMARY_COLS}})
    _write_comparison(out_dir, rows)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-model benchmark runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="Submit + score a model x language grid.")
    rp.add_argument("--models", required=True,
                    help="Comma-separated model ids/aliases (e.g. sonnet,haiku).")
    rp.add_argument("--languages", required=True,
                    help="Comma-separated languages (webppl,pyro,stan).")
    rp.add_argument("--out", required=True, help="Output run directory.")
    rp.add_argument("--ids", nargs="*", default=None, help="Restrict to specific problem IDs.")
    rp.add_argument("--limit", type=int, default=None,
                    help="Pilot: first N scorable problems per language.")
    rp.add_argument("--n-samples", type=int, default=3, help="Solver attempts per problem.")
    rp.add_argument("--no-primer", action="store_true", help="No-primer arm (system prompt only).")
    rp.add_argument("--verbose-primer", action="store_true",
                    help="Use the heavier hand-holding primer instead of the lean one.")
    rp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    rp.add_argument("--n-draws", type=int, default=DEFAULT_N_MC)
    rp.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    rp.add_argument("--workers", type=int, default=4, help="Score-time problem-parallelism.")
    rp.add_argument("--poll-interval", type=int, default=30)
    rp.add_argument("--poll-timeout", type=int, default=3600)

    rep = sub.add_parser("report", help="Re-aggregate comparison from a run directory.")
    rep.add_argument("--out", required=True)

    fp = sub.add_parser("failures", help="Dump per-combo failures.jsonl (with prompt + GT) for investigation.")
    fp.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "failures":
        dump_failures(args.out)
    elif args.cmd == "run":
        run_benchmark(
            [m.strip() for m in args.models.split(",") if m.strip()],
            [l.strip() for l in args.languages.split(",") if l.strip()],
            out_dir=args.out, ids=args.ids, limit=args.limit, n_solvers=args.n_samples,
            with_primer=not args.no_primer, verbose_primer=args.verbose_primer,
            seed=args.seed, n_draws=args.n_draws, timeout=args.timeout,
            score_workers=args.workers, poll_interval=args.poll_interval,
            poll_timeout=args.poll_timeout,
        )
    else:
        report(args.out)


if __name__ == "__main__":
    main()
