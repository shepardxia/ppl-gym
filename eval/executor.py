"""WebPPL executor.

Contract: the user program binds `var ANSWER = <expression>;` as its last
statement. The harness wraps the program with a serializer header and
appends `JSON.stringify(__serialize(ANSWER))`. No JS-text parsing.

Serialization uses WebPPL's built-in `serializeDist` for Marginal/Categorical
distributions (those have a JSON-friendly `{probs, support}` form).
Continuous distributions (Beta, Gaussian, ...) fall back to a string
representation. Records and arrays recurse. Primitives pass through.

Output schema (always JSON-parseable):
  - discrete distribution → {"__kind": "distribution", "probs": [...], "support": [...]}
  - continuous distribution → {"__kind": "distribution_continuous", "repr": "..."}
  - function → {"__kind": "function"}
  - tensor → {"__kind": "tensor", "dims": [...], "data": [...]}
  - everything else → emitted as-is (numbers, strings, booleans, arrays, plain objects)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from eval.exec_common import ExecutionResult, loads_lenient, run_per_seed, strip_ansi


# Header injected before user code. Defines `__serialize(x)`. The trailer
# is appended after user code: it stringifies `ANSWER` (a top-level var
# the user is expected to bind).
SERIALIZER_HEADER = r"""
var __serialize = function(x) {
  if (x === null || x === undefined) return x;
  if (typeof x === 'function') return {"__kind": "function"};
  if (typeof x !== 'object') return x;
  // Marginal / Categorical from Infer: WebPPL has a built-in JSON form
  // ({probs, support}). Use it for the canonical cross-PPL schema.
  if (typeof x.getDist === 'function') {
    return _.assign({"__kind": "distribution"}, JSON.parse(serializeDist(x)));
  }
  // Continuous distributions (Beta, Gaussian, ...) don't implement toJSON.
  if (typeof x.score === 'function' && typeof x.sample === 'function') {
    return {"__kind": "distribution_continuous", "repr": ('' + x)};
  }
  // Tensors expose .dims / .data
  if (x.dims !== undefined && x.length !== undefined && x.data !== undefined) {
    return {"__kind": "tensor", "dims": x.dims, "data": T.toScalars(x)};
  }
  if (Array.isArray(x)) {
    return map(__serialize, x);
  }
  var keys = Object.keys(x);
  var pairs = map(function(k) { return [k, __serialize(x[k])]; }, keys);
  return _.fromPairs(pairs);
};
"""

SERIALIZER_FOOTER = "JSON.stringify(__serialize(ANSWER))"


# Probmods2's WebPPL install with pre-installed packages.
_PROBMODS_DIR = Path(__file__).parent.parent / "data" / "sources" / "probmods2"
_PROBMODS_MODULES = _PROBMODS_DIR / "node_modules"
_WEBPPL_BIN = _PROBMODS_MODULES / "webppl" / "webppl"

# Shim packages always loaded via --require.
_DEPS_DIR = Path(__file__).parent / "deps"
_REQUIRE_PACKAGES = [
    _PROBMODS_MODULES / "webppl-agents",
    _PROBMODS_MODULES / "webppl-dp",
    _PROBMODS_MODULES / "webppl-timeit",
    _DEPS_DIR / "probmods-deps",
    _DEPS_DIR / "probmods-towdata",
    _DEPS_DIR / "probmods-physics",
    _DEPS_DIR / "probmods-draw",
    _DEPS_DIR / "probmods-viz-stub",
]

# Preload script overriding Math.random BEFORE webppl modules capture a ref.
_MATH_RANDOM_PRELOAD = _DEPS_DIR / "probmods-seeded-random" / "preload.js"


def _wrap_program(code: str) -> str:
    """Prepend the serializer header and append the ANSWER stringifier.

    Assumes the user's code binds `var ANSWER = ...;` as a top-level
    statement.
    """
    return SERIALIZER_HEADER + "\n" + code.rstrip() + "\n" + SERIALIZER_FOOTER


def execute_webppl(code: str, timeout: int = 30, random_seed: int | None = None) -> ExecutionResult:
    full_code = _wrap_program(code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wppl", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        cmd = ["node", "-r", str(_MATH_RANDOM_PRELOAD), str(_WEBPPL_BIN), tmp_path]
        if random_seed is not None:
            cmd.extend(["--random-seed", str(random_seed)])
        for pkg in _REQUIRE_PACKAGES:
            cmd.extend(["--require", str(pkg)])

        env = {
            **os.environ,
            "WEBPPL_MATH_RANDOM_SEED": str(random_seed if random_seed is not None else 42),
        }

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error_message=f"Timeout after {timeout}s",
                code=code,
            )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            return ExecutionResult(
                success=False,
                raw_stdout=stdout,
                stderr=stderr,
                error_message=_extract_error(stderr or stdout),
                code=code,
            )

        if not stdout:
            return ExecutionResult(
                success=False,
                raw_stdout=stdout,
                stderr=stderr,
                error_message=(
                    "program exited 0 but produced no output "
                    "(likely silent failure or undefined ANSWER)"
                ),
                code=code,
            )

        # WebPPL may interleave warnings on stdout; loads_lenient falls back to
        # the last non-empty line (the serializer footer always emits it last).
        try:
            answer = loads_lenient(stdout)
        except json.JSONDecodeError as e:
            return ExecutionResult(
                success=False,
                raw_stdout=stdout,
                stderr=stderr,
                error_message=f"output not valid JSON: {e}",
                code=code,
            )

        return ExecutionResult(
            success=True,
            answer=answer,
            raw_stdout=stdout,
            stderr=stderr,
            code=code,
        )
    finally:
        os.unlink(tmp_path)


def execute_webppl_batch(code: str, seeds, timeout: int = 30, workers: int = 8):
    """Run ``code`` once per seed. Returns ``(answers, errors)`` aligned with ``seeds``.

    ``answers[i]`` is the parsed answer or ``None`` for a failed seed; ``errors[i]``
    is that seed's real failure reason (``None`` on success), so the harness can
    raise the actual cause instead of a generic count.

    WebPPL seeds its RNG from a process-start env override, so it cannot reseed
    in-process — this is per-seed spawning under the batch interface (same seeds,
    byte-identical to calling execute_webppl per seed), parallelized across
    ``workers`` threads. The amortization win for fixed reference GTs comes from
    caching (eval/gt_cache), not from batching webppl.
    """
    return run_per_seed(
        lambda s: execute_webppl(code, timeout=timeout, random_seed=s),
        seeds, workers=workers)


def _extract_error(text: str) -> str:
    text = strip_ansi(text)
    for line in text.split("\n"):
        line = line.strip()
        if line and any(
            line.startswith(prefix)
            for prefix in ("ReferenceError:", "TypeError:", "Error:", "SyntaxError:", "RangeError:")
        ):
            return line
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("at ") and not line.startswith("---"):
            return line[:200]
    return text[:200] if text else "Unknown error"


if __name__ == "__main__":
    # Smoke test: a program that binds ANSWER
    r = execute_webppl(
        "var model = function() { return flip() ? \"H\" : \"T\" };\n"
        "var ANSWER = Infer({method:'enumerate'}, model);",
        random_seed=42,
    )
    print(f"success={r.success}")
    print(f"answer={json.dumps(r.answer, indent=2)}")
