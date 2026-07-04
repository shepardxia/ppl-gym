"""Export the benchmark matrix as a self-contained HuggingFace dataset.

Three configs, joinable on problem_id:
  rollouts   — one row per generation: model, language, problem_id, slot, the
               generated code, and the scored verdict (status/distance/tol/...).
  problems   — the language-neutral problem statements + answer_spec + corpus.
  gt_answers — canonical ground-truth answers per (problem, language).

A row's problem may be UNSCORABLE if our ground truth could not be computed
(e.g. a pyro GT that times out at the collection budget); such rows carry
gt_unscorable=true so a consumer can drop them before reading pass rates.

CLI:
  PYTHONPATH=. .venv/bin/python -m eval.export_rollouts \\
      --runs runs/matrix --out <dir> [--repo shepardxia/ppl-gym-rollouts]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Columns carried from a scored row into the rollouts config (in order).
_ROLLOUT_COLS = [
    "model", "language", "problem_id", "slot", "code", "status",
    "distance", "tol", "floor", "metric", "ill_posed", "code_jaccard",
    "runtime_sec", "stop_reason", "output_tokens",
]


def _gt_broken_by_lang(scored_files: list[str]) -> dict[str, set[str]]:
    """problem_ids whose ground truth failed to collect, per language."""
    out: dict[str, set[str]] = {}
    for f in scored_files:
        lang = Path(f).parent.name.split("__")[1]
        s = out.setdefault(lang, set())
        for line in open(f):
            r = json.loads(line)
            if r.get("summary"):
                continue
            if (r.get("status") == "exec_error"
                    and "gt collection failed" in (r.get("error") or "").lower()):
                s.add(r["problem_id"])
    return out


def build_rollouts(runs_dir: Path) -> list[dict]:
    scored_files = sorted(glob.glob(str(runs_dir / "*/*/scored.jsonl")))
    if not scored_files:
        raise SystemExit(f"no scored.jsonl under {runs_dir}")
    gt_broken = _gt_broken_by_lang(scored_files)
    rows: list[dict] = []
    for f in scored_files:
        lang = Path(f).parent.name.split("__")[1]
        for line in open(f):
            r = json.loads(line)
            if r.get("summary") or not r.get("problem_id"):
                continue
            row = {c: r.get(c) for c in _ROLLOUT_COLS}
            row["gt_unscorable"] = r["problem_id"] in gt_broken.get(lang, set())
            rows.append(row)
    return rows


# Status rank for picking a model's representative slot: a passing attempt beats
# a wrong-but-running one beats a malformed/crashing one.
_STATUS_RANK = {"pass": 0, "fail": 1, "ill_posed": 2, "malformed": 3, "exec_error": 4}


# The browser surfaces rollouts in each corpus's solver language only; pyro is a
# cross-realization column, never a solver target. Keeping web to these langs
# halves the committed build-input file.
_WEB_LANGS = {"webppl", "stan"}


def build_web_rollouts(runs_dir: Path, keep_langs: set[str] = _WEB_LANGS) -> list[dict]:
    """One representative rollout per (problem, model, language), for the web build.

    Picks the best slot (passing first, then lowest distance, then lowest slot)
    so the browser shows each model's strongest attempt with its verdict. Written
    to a committed file under data/ because runs/ is gitignored (the Cloudflare
    build only sees the repo tree).
    """
    scored_files = sorted(glob.glob(str(runs_dir / "*/*/scored.jsonl")))
    best: dict[tuple, dict] = {}
    for f in scored_files:
        lang = Path(f).parent.name.split("__")[1]
        if lang not in keep_langs:
            continue
        model = Path(f).parents[1].name  # short run-dir name (sonnet, gpt-oss-120b)
        for line in open(f):
            r = json.loads(line)
            if r.get("summary") or not r.get("problem_id") or not r.get("code"):
                continue
            key = (r["problem_id"], model, lang)
            cand = {
                "problem_id": r["problem_id"], "model": model, "language": lang,
                "code": r["code"], "status": r.get("status"),
                "distance": r.get("distance"), "error": r.get("error"),
                "slot": r.get("slot", 0),
            }
            cur = best.get(key)
            if cur is None or _rollout_rank(cand) < _rollout_rank(cur):
                best[key] = cand
    return sorted(best.values(), key=lambda r: (r["problem_id"], r["language"], r["model"]))


def _rollout_rank(r: dict) -> tuple:
    return (_STATUS_RANK.get(r.get("status"), 9),
            r["distance"] if r.get("distance") is not None else float("inf"),
            r.get("slot", 0))


def _round_sig(x, sig: int = 6):
    if not isinstance(x, float) or x == 0 or x != x or x in (float("inf"), float("-inf")):
        return x
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def trim_answer(a: dict, max_support: int = 256, max_samples: int = 256,
                max_fields: int = 40) -> dict:
    """Shrink a wire answer for web embedding: round floats to 6 sig figs, cap
    enumerated support / sample clouds to the highest-mass entries, and cap a
    record's marginals to the first ``max_fields`` parameters. Display-only —
    the full answer stays in scored data and the HF dataset. A 500-point support
    overlay is unreadable anyway; a 1000-parameter record (lsat) overlay even
    less so — the cap keeps the committed web file from ballooning."""
    k = a.get("kind")
    if k == "dist_enum":
        pairs = sorted(zip(a["support"], a["probs"]), key=lambda t: -t[1])[:max_support]
        return {"kind": "dist_enum", "support": [p[0] for p in pairs],
                "probs": [_round_sig(p[1]) for p in pairs]}
    if k == "cloud":
        s = a["samples"][:max_samples]
        return {"kind": "cloud", "samples": [_round_sig(x) for x in s]}
    if k == "dist_param":
        return {"kind": "dist_param", "family": a["family"],
                "params": {kk: _round_sig(vv) for kk, vv in a["params"].items()}}
    if k == "record":
        items = list(a["fields"].items())[:max_fields]
        return {"kind": "record",
                "fields": {n: trim_answer(v, max_support, max_samples, max_fields)
                           for n, v in items}}
    if k == "exact":
        return {"kind": "exact", "value": _round_sig(a["value"])}
    return a


def trim_web_answers(web_path: Path) -> None:
    """Apply trim_answer to every captured answer in the web rollouts file."""
    rows = [json.loads(l) for l in open(web_path)]
    n = 0
    for r in rows:
        if r.get("answer"):
            r["answer"] = trim_answer(r["answer"]); n += 1
    _write_jsonl(rows, web_path)
    print(f"[trim] trimmed {n} answers -> {web_path} ({web_path.stat().st_size // 1024} KB)")


def augment_rollout_answers(web_path: Path, langs: set[str], *,
                            timeout: int = 60, workers: int = 8) -> None:
    """Re-execute non-error rollouts in `langs` to capture each model's computed
    answer (the distribution/value), stored as the `answer` wire dict so the web
    overlay can chart a model's posterior against the GT. Skips error/malformed
    rollouts (no valid answer). Edits the file in place.

    webppl is cheap; stan recompiles a unique model per candidate (slow) — pass it
    explicitly only when you can afford the cmdstan cost.
    """
    from concurrent.futures import ThreadPoolExecutor
    from eval.algebra import answer_to_dict, parse_spec
    from eval.corpus import load_problems
    from eval.harness import execute_candidate_answer

    rows = [json.loads(l) for l in open(web_path)]
    specs: dict[str, object] = {}
    for p in load_problems(None):
        try:
            specs[p["problem_id"]] = parse_spec(p["answer_spec"])
        except Exception:  # noqa: BLE001
            pass
    stan_gt: dict[str, str] = {}
    if "stan" in langs:
        from eval.corpus import load_realizations
        stan_gt = {r["problem_id"]: r.get("code", "") for r in load_realizations("stan")}

    targets = [r for r in rows if r["language"] in langs and r.get("code")
               and r.get("status") in ("pass", "fail", "ill_posed")]

    def work(r: dict):
        spec = specs.get(r["problem_id"])
        if spec is None:
            return None
        try:
            canon = execute_candidate_answer(
                r["code"], spec, language=r["language"], timeout=timeout,
                gt_bundle=stan_gt.get(r["problem_id"]) if r["language"] == "stan" else None)
            return answer_to_dict(canon, max_samples=400)
        except Exception:  # noqa: BLE001 — a failed re-exec just leaves no overlay
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(work, targets))
    got = 0
    for r, ans in zip(targets, results):
        if ans is not None:
            r["answer"] = trim_answer(ans)
            got += 1
    _write_jsonl(rows, web_path)
    print(f"[answers] re-executed {len(targets)} {sorted(langs)} rollouts; "
          f"captured {got} answers -> {web_path}")


def build_problems() -> list[dict]:
    out: list[dict] = []
    for f in sorted(glob.glob(str(_REPO / "data/problems/*.jsonl"))):
        corpus = Path(f).stem
        if corpus.startswith("_"):  # _retired, _gate_*, _gt_answers — not corpora
            continue
        for line in open(f):
            p = json.loads(line)
            if not p.get("problem_id"):
                continue
            out.append({
                "problem_id": p["problem_id"],
                "corpus": corpus,
                "status_review": (p.get("status") or {}).get("review"),
                "given": (p.get("statement") or {}).get("given"),
                "model": (p.get("statement") or {}).get("model"),
                "query": (p.get("statement") or {}).get("query"),
                "answer_spec": json.dumps(p.get("answer_spec")),
                "provenance": json.dumps(p.get("provenance")),
            })
    return out


def build_gt_answers() -> list[dict]:
    path = _REPO / "data/problems/_gt_answers.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in open(path):
        r = json.loads(line)
        out.append({
            "problem_id": r.get("problem_id"),
            "language": r.get("language"),
            "answer": json.dumps(r.get("answer")) if r.get("answer") is not None else None,
            "error": r.get("error"),
        })
    return out


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


_CARD = """---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - probabilistic-programming
  - code-generation
  - llm-evaluation
  - webppl
  - pyro
  - stan
configs:
  - config_name: rollouts
    data_files: rollouts.jsonl
  - config_name: problems
    data_files: problems.jsonl
  - config_name: gt_answers
    data_files: gt_answers.jsonl
---

# ppl-gym rollouts

Model rollouts from **ppl-gym**, a benchmark of probabilistic-programming
*problems*: language-neutral problem statements with per-language ground-truth
realizations. Each rollout is one LLM-generated solution (program) for a
(model, language, problem) triple, executed and scored against ground truth
under a single answer comparator.

- Code & methodology: https://github.com/shepardxia/ppl-gym
- Browser: https://pplgym.kingdomofends.org

## Configs

- **rollouts** ({n_rollouts} rows) — one generated solution per row:
  `model`, `language` (webppl / pyro / stan), `problem_id`, `slot` (sample
  index), `code` (the generated program), and the scored verdict: `status`
  (pass / fail / ill_posed / malformed / exec_error), `distance`, `tol`,
  `floor`, `metric`, `code_jaccard` (vs the reference realization),
  `runtime_sec`. `gt_unscorable=true` marks rows whose ground truth could not
  be computed (drop these before reading pass rates — see below).
- **problems** ({n_problems} rows) — the problem statements: `given` / `model`
  / `query`, `answer_spec`, `corpus`, `provenance`. Join on `problem_id`.
- **gt_answers** ({n_gt} rows) — canonical ground-truth answers per
  (`problem_id`, `language`).

## Scoring note (gt_unscorable)

`status` is a per-generation verdict. `exec_error` means the candidate failed to
execute OR the ground truth itself could not be computed. A subset of problems
(notably heavier pyro MCMC models) have a ground truth that exceeds the
collection time budget; those are **unscorable for every model** and are flagged
`gt_unscorable=true`. Pass rates should be read over the scorable subset
(`gt_unscorable=false`).

## Models

claude-sonnet-4-6, claude-haiku-4-5, gpt-oss-20b, gpt-oss-120b,
Llama-3.3-70B-Instruct, Qwen3-235B-A22B-Instruct, Qwen3.5-9B.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the matrix as a HF dataset dir.")
    ap.add_argument("--runs", default="runs/matrix", help="matrix run dir.")
    ap.add_argument("--out", help="output HF dataset directory.")
    ap.add_argument("--web-out", default=None,
                    help="also write the curated web build-input rollouts JSONL here "
                         "(best slot per problem×model×language).")
    ap.add_argument("--repo", default="Sheppp/ppl-gym-rollouts")
    ap.add_argument("--answers-langs", default=None,
                    help="comma langs to re-execute for the answer overlay (e.g. webppl). "
                         "stan is slow (per-candidate cmdstan compile).")
    ap.add_argument("--answers-only", action="store_true",
                    help="skip rebuilding web rollouts; only augment answers in --web-out.")
    ap.add_argument("--trim-only", action="store_true",
                    help="just shrink existing answers in --web-out for embedding (no exec).")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    runs = (_REPO / a.runs) if not Path(a.runs).is_absolute() else Path(a.runs)

    if a.web_out:
        wp = Path(a.web_out).resolve()
        if a.trim_only:
            trim_web_answers(wp)
            return
        if not a.answers_only:
            web = build_web_rollouts(runs)
            wp.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(web, wp)
            print(f"[export] web rollouts={len(web)} -> {wp}")
        if a.answers_langs:
            augment_rollout_answers(wp, set(a.answers_langs.split(",")),
                                    timeout=a.timeout, workers=a.workers)
    if not a.out:
        return

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rollouts = build_rollouts(runs)
    problems = build_problems()
    gt = build_gt_answers()

    _write_jsonl(rollouts, out / "rollouts.jsonl")
    _write_jsonl(problems, out / "problems.jsonl")
    _write_jsonl(gt, out / "gt_answers.jsonl")
    (out / "README.md").write_text(
        _CARD.format(n_rollouts=len(rollouts), n_problems=len(problems), n_gt=len(gt)))

    n_unscorable = sum(1 for r in rollouts if r["gt_unscorable"])
    print(f"[export] rollouts={len(rollouts)} ({n_unscorable} gt_unscorable) "
          f"problems={len(problems)} gt_answers={len(gt)}")
    print(f"[export] wrote dataset dir -> {out}")
    print(f"[export] upload with: huggingface-cli upload {a.repo} {out} --repo-type=dataset")


if __name__ == "__main__":
    main()
