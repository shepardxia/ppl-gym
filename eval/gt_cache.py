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
) -> list:
    """Return raw serialized answers aligned with ``seeds``.

    On a cache hit, loads from disk. On a miss, runs the language's batch
    executor and writes the result ONLY if every seed succeeded (a partial /
    failed run is never cached, so a transient failure can be retried).
    """
    seeds = list(seeds)
    if not seeds:
        return []

    active = use_cache and not _disabled()
    path = _CACHE_DIR / f"{_key(language, code, seeds)}.json" if active else None
    if path is not None and path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry → recompute

    raw = batch_executor_for(language)(code, seeds, timeout, workers)

    if path is not None and all(a is not None for a in raw):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw))
        tmp.replace(path)  # atomic publish
    return raw
