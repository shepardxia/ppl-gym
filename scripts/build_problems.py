"""Merge authored problem emissions into the canonical corpus files.

Used when growing the corpus (new source material authored per
data/problems/_AUTHORING_BRIEF.md). Emission rows carry
{problem_id, action: keep|respec|retire, provenance, statement, answer_spec,
status, realization}; retires go to _retired.jsonl with their full record.

Merging is by problem_id into the EXISTING corpus/realization files — existing
rows for other problems are preserved, and re-merging the same emissions is
idempotent. (The P1 bootstrap half of this script — spec drafting from legacy
atom outputs — was removed after P1 closed; see data/REDESIGN.md.)

Usage:
  PYTHONPATH=. .venv/bin/python -m scripts.build_problems <emissions.jsonl> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from eval.io import load_jsonl, write_jsonl

# Maps problem_id prefix -> corpus file stem; unknown prefix is a hard error.
CORPUS_PREFIXES = {
    "dippl":     "dippl",
    "forestdb":  "forestdb",
    "probmods2": "probmods2",
}


def _merge_rows(path: Path, new_rows: list[dict]) -> int:
    rows = {r["problem_id"]: r for r in load_jsonl(path)} if path.exists() else {}
    for r in new_rows:
        rows[r["problem_id"]] = r
    write_jsonl(path, sorted(rows.values(), key=lambda r: r["problem_id"]))
    return len(rows)


def merge_emissions(paths: list[Path]) -> None:
    emissions = [row for p in paths for row in load_jsonl(p)]
    by_corpus: dict[str, list] = {}
    realizations, retired = [], []
    for e in emissions:
        pid = e["problem_id"]
        if e["action"] == "retire":
            retired.append(e)
            continue
        prefix = pid.split("-")[0]
        if prefix not in CORPUS_PREFIXES:
            raise ValueError(f"unknown problem_id prefix: {prefix!r} (id={pid!r})")
        problem = {"problem_id": pid, "provenance": e.get("provenance", {}),
                   "statement": e["statement"], "answer_spec": e["answer_spec"],
                   "status": e["status"]}
        by_corpus.setdefault(CORPUS_PREFIXES[prefix], []).append(problem)
        realizations.append({"problem_id": pid, **e["realization"], "gate": {}})

    for corpus, rows in by_corpus.items():
        out = Path(f"data/problems/{corpus}.jsonl")
        total = _merge_rows(out, rows)
        print(f"{out}: +{len(rows)} merged, {total} total")
    if realizations:
        real_path = Path("data/realizations/webppl.jsonl")
        total = _merge_rows(real_path, realizations)
        print(f"{real_path}: +{len(realizations)} merged, {total} total")
    if retired:
        retired_path = Path("data/problems/_retired.jsonl")
        total = _merge_rows(retired_path, retired)
        print(f"{retired_path}: +{len(retired)} merged, {total} total")


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    if not paths:
        print(__doc__)
        sys.exit(1)
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"missing emission files: {missing}")
    merge_emissions(paths)


if __name__ == "__main__":
    main()
