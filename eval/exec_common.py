"""Shared executor primitives: the ExecutionResult struct and lenient JSON
parsing of subprocess stdout. Single source for both the WebPPL and Pyro
executors (the result shape and the warning-tolerant answer parse).
"""

from __future__ import annotations

import json
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
