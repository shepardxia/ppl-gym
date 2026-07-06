"""Persistent content-addressed cache for GT executions.

A GT run is deterministic in (language, code, seeds) — seeded execution always
yields the same answers. So the raw serialized executor outputs are cached on
disk, keyed by a hash of those inputs (plus an executor-version tag that busts
the cache when the serializer/executor changes). Spec-independent: the same
cached run serves any spec interpretation, canonicalized on use.

Used transparently by eval.gate.collect_gt_answers, so every consumer
(crosscheck / phaseA / judge / answers / score) and every future language
benefits. Self-invalidating: changed code → different key → recompute.

Disable with env PPL_GYM_NO_CACHE=1.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.corpus import batch_executor_for

# Bump a language's tag when its executor/serializer output format changes,
# so stale cached runs are never read.
EXECUTOR_VERSION = {
    "webppl": "wp1",
    "pyro": "py3",        # py3: float64 default + import-free preamble
    "stan": "st1",        # cmdstanpy NUTS, record{param:[draws]}
    "reference": "ref1",  # replayed posteriordb gold draws
    "gen": "gen1",        # Gen.jl subprocess, exact/enumerative, mapping-dict answer
}

# Absolute, repo-anchored: a concurrent Stan compile (cmdstanpy) transiently
# chdir's the process, so a relative cache path could resolve against the wrong
# CWD in a worker thread writing the cache. Absolute is CWD-independent.
_CACHE_DIR = Path(__file__).resolve().parents[1] / "data/.gt_cache"


def _disabled() -> bool:
    return os.environ.get("PPL_GYM_NO_CACHE") == "1"


def _key(language: str, code: str, seeds: list[int]) -> str:
    payload = json.dumps(
        {
            "lang": language,
            "ver": EXECUTOR_VERSION.get(language, "?"),
            "code": code,
            "seeds": list(seeds),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cached_run(
    language: str,
    code: str,
    seeds,
    *,
    timeout: int,
    workers: int,
    use_cache: bool = True,
) -> tuple[list, list]:
    """Return ``(answers, errors)`` aligned with ``seeds``.

    ``answers[i]`` is the serialized answer or ``None`` for a failed seed;
    ``errors[i]`` is that seed's real failure reason (``None`` on success), so
    callers can raise the actual cause. On a cache hit, loads answers from disk
    (a cached run has no failures → all-None errors). On a miss, runs the
    language's batch executor and writes the ANSWERS ONLY if every seed
    succeeded (a partial / failed run is never cached, so a transient failure
    can be retried; the on-disk cache format is unchanged — answers, no errors).
    """
    seeds = list(seeds)
    if not seeds:
        return [], []

    active = use_cache and not _disabled()
    path = _CACHE_DIR / f"{_key(language, code, seeds)}.json" if active else None
    if path is not None and path.exists():
        try:
            answers = json.loads(path.read_text())
            return answers, [None] * len(answers)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry → recompute

    answers, errors = batch_executor_for(language)(code, seeds, timeout, workers)

    if path is not None and all(a is not None for a in answers):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(answers))
        tmp.replace(path)  # atomic publish
    return answers, errors
