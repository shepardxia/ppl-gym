"""Single source of truth for loading the problem-centric dataset.

Provides:
  load_problems   — problems from data/problems/{probmods2,dippl,forestdb}.jsonl
  load_realizations — realizations from data/realizations/<language>.jsonl
  load_corpus     — joined (problems, realizations) intersection
"""

from __future__ import annotations

from pathlib import Path

from eval.io import load_jsonl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROBLEMS_DIR = Path("data/problems")
_PROBLEM_FILES = ["probmods2.jsonl", "dippl.jsonl", "forestdb.jsonl"]


def _realizations_path(language: str) -> Path:
    return Path("data/realizations") / f"{language}.jsonl"


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_problems(
    problem_ids: set[str] | None = None,
    *,
    include_retired: bool = False,
) -> list[dict]:
    """Load all non-retired problems from the three canonical corpus files.

    Parameters
    ----------
    problem_ids:
        When provided, restrict results to these IDs (applied after the
        retired filter).
    include_retired:
        When True, also include problems whose review status is "retired".
    """
    problems: list[dict] = []
    for fname in _PROBLEM_FILES:
        p = _PROBLEMS_DIR / fname
        if p.exists():
            problems.extend(load_jsonl(p))

    if not include_retired:
        problems = [
            prob for prob in problems
            if prob.get("status", {}).get("review") != "retired"
        ]

    if problem_ids:
        problems = [p for p in problems if p["problem_id"] in problem_ids]

    return problems


def load_realizations(language: str = "webppl") -> list[dict]:
    """Load all realizations for the given language."""
    path = _realizations_path(language)
    if not path.exists():
        return []
    return load_jsonl(path)


def load_corpus(
    problem_ids: set[str] | None = None,
    language: str = "webppl",
) -> tuple[list[dict], list[dict]]:
    """Load problems joined with realizations on problem_id.

    Returns (problems, realizations) filtered to the intersection of problems
    that have a realization for the given language, and optionally to the
    requested problem_ids subset.

    Both lists are 1-1 aligned: problems[i] corresponds to realizations[i].
    """
    problems = load_problems(include_retired=False)
    realizations = load_realizations(language)

    real_by_id = {r["problem_id"]: r for r in realizations if "code" in r}

    joined: list[tuple[dict, dict]] = [
        (prob, real_by_id[prob["problem_id"]])
        for prob in problems
        if prob["problem_id"] in real_by_id
    ]

    if problem_ids:
        joined = [(p, r) for p, r in joined if p["problem_id"] in problem_ids]

    return [p for p, _ in joined], [r for _, r in joined]
