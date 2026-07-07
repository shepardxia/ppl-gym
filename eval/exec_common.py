"""Shared executor primitives: the ExecutionResult struct and lenient JSON
parsing of subprocess stdout. Single source for both the WebPPL and Pyro
executors (the result shape and the warning-tolerant answer parse).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    success: bool
    answer: object = None
    raw_stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    code: str = ""


def loads_lenient(stdout: str):
    """Parse ``stdout`` as JSON, retrying on the last non-empty line.

    WebPPL/Pyro may interleave warnings before the answer; the serializer always
    emits the answer as the final line. Raises json.JSONDecodeError if neither
    the whole stdout nor the last non-empty line parses.
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        last = next((ln for ln in reversed(stdout.split("\n")) if ln.strip()), "")
        return json.loads(last)


def strip_ansi(text: str) -> str:
    """Drop ANSI color escapes from subprocess output (the first step every
    executor's ``_extract_error`` takes before scanning for the cause line)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text or "")


def run_per_seed(run_one, seeds, *, workers: int, default_error: str = "execution failed"):
    """Fan out ``run_one(seed) -> ExecutionResult`` across a thread pool; return
    ``(answers, errors)`` aligned with ``seeds`` (answer or None; error or None).

    The shared process-per-seed batch shape for executors that spawn one
    subprocess per seed (WebPPL, and Gen's sampling mode). Out-of-order thread
    completion is handled by the index map, so results stay seed-aligned.
    """
    from concurrent.futures import ThreadPoolExecutor

    seeds = list(seeds)
    answers: list = [None] * len(seeds)
    errors: list = [None] * len(seeds)
    if not seeds:
        return answers, errors
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(seeds)))) as pool:
        futs = {pool.submit(run_one, s): i for i, s in enumerate(seeds)}
        for fut, i in futs.items():
            r = fut.result()
            if r.success:
                answers[i] = r.answer
            else:
                errors[i] = r.error_message or default_error
    return answers, errors
