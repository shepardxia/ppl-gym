"""Single source of truth for loading the problem-centric dataset.

Provides:
  load_problems   — problems from data/problems/{probmods2,dippl,forestdb,posteriordb}.jsonl
  load_realizations — realizations from data/realizations/<language>.jsonl
  load_corpus     — joined (problems, realizations) intersection
"""

from __future__ import annotations

from pathlib import Path

from eval.executor import execute_webppl_batch
from eval.executor_gen import execute_gen_batch
from eval.executor_pyro import execute_pyro_batch
from eval.executor_reference import execute_reference_batch
from eval.executor_stan import execute_stan_batch
from eval.io import load_jsonl

# Language bindings: the batched executor (run a set of seeds, answers out) per
# realization language — the primitive used by GT collection. (For ad-hoc n=1 runs
# import execute_webppl / execute_pyro directly from their executor modules.)
#   stan      — compile + NUTS-sample a self-contained Stan bundle (posteriordb)
#   reference — replay posteriordb's stored gold draws as the GT column
#   gen       — Gen.jl (Julia) subprocess, exact/enumerative discrete inference
BATCH_EXECUTORS = {
    "webppl": execute_webppl_batch,
    "pyro": execute_pyro_batch,
    "stan": execute_stan_batch,
    "reference": execute_reference_batch,
    "gen": execute_gen_batch,
}


def batch_executor_for(language: str):
    """Return execute_batch(code, seeds, timeout, workers) -> (answers, errors).

    Contract (uniform across languages):
      - Returns ``(answers, errors)``, both aligned with ``seeds``: ``answers[i]``
        is the parsed answer or ``None`` for a failed seed; ``errors[i]`` is that
        seed's real failure reason (``None`` on success), so callers surface the
        actual cause instead of a generic count.
      - A whole-run failure (nothing executed: compile error, subprocess death,
        every unit failed) raises RuntimeError carrying the REAL reason —
        never a generic message.
      - ``timeout`` is the per-run budget for one seed's execution; executors
        apply the budget policy in eval.config on top (Pyro seed scale + chunk
        cap, Stan data/regime scaling).
      - ``workers`` bounds the executor's concurrent OS processes for this
        call: WebPPL = process per seed (workers threads spawning them),
        Pyro = chunk subprocesses, Stan = concurrent fits (workers // chains).
    """
    try:
        return BATCH_EXECUTORS[language]
    except KeyError:
        raise ValueError(
            f"no batch executor for language {language!r} "
            f"(known: {sorted(BATCH_EXECUTORS)})")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repo-anchored absolutes: cmdstanpy (the Stan executor) can leave the process
# CWD changed, so relative data paths would read the wrong directory in any
# later op of the same process (e.g. a pyro combo scored after a stan one).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBLEMS_DIR = _REPO_ROOT / "data/problems"
_PROBLEM_FILES = ["probmods2.jsonl", "dippl.jsonl", "forestdb.jsonl",
                  "posteriordb.jsonl"]


def _realizations_path(language: str) -> Path:
    return _REPO_ROOT / "data/realizations" / f"{language}.jsonl"


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_problems(
    problem_ids: set[str] | None = None,
    *,
    include_retired: bool = False,
) -> list[dict]:
    """Load all non-retired problems from the four canonical corpus files.

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


def is_available(rec: dict) -> bool:
    """Return True iff a realization record is available for execution.

    A record is available when it carries a ``code`` key AND does not
    explicitly declare ``available: false``.  The ``available`` field
    defaults to True when absent (existing records without the field are
    available).
    """
    return rec.get("available", True) and "code" in rec


def load_realizations(language: str = "webppl") -> list[dict]:
    """Load all realizations for the given language."""
    path = _realizations_path(language)
    if not path.exists():
        return []
    return load_jsonl(path)


def load_unavailable(language: str = "webppl") -> list[dict]:
    """Return realization records for which ``is_available`` is False.

    Each returned record has at minimum ``problem_id``, ``language``, and
    ``reason``; it has no ``code`` key.
    """
    return [r for r in load_realizations(language) if not is_available(r)]


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

    real_by_id = {r["problem_id"]: r for r in realizations if is_available(r)}

    joined: list[tuple[dict, dict]] = [
        (prob, real_by_id[prob["problem_id"]])
        for prob in problems
        if prob["problem_id"] in real_by_id
    ]

    if problem_ids:
        joined = [(p, r) for p, r in joined if p["problem_id"] in problem_ids]

    return [p for p, _ in joined], [r for _, r in joined]
