"""Pyro executor for ppl-gym.

Binding contract: the user program must assign a top-level ``ANSWER`` variable.
The injected serializer JSON-stringifies that binding using the native wire forms
defined in data/SCHEMA.md §Representations.

Seeding: ``execute_pyro`` sets the environment variable ``PPL_GYM_PYRO_SEED``
before spawning the subprocess. The injected header reads it and calls
``pyro.set_rng_seed(seed)`` (which seeds torch, Python random, and numpy
together), so the same stochastic program returns identical output for the same
seed and different output for different seeds.

Native output forms (no legacy __kind wrappers):
  - None / bool / int / float / str -> JSON scalar as-is.
  - torch.Tensor: 0-D -> scalar; 1-D -> list; >=2-D -> {"kind":"tensor","dims":[...],"data":flat list}.
  - dict -> plain JSON object (values recursive; non-str keys JSON-stringified
    after converting tuples to lists, so bool keys become "true"/"false").
  - list / tuple -> JSON array, recursive.
  - pyro.distributions.Empirical (and EmpiricalMarginal) -> aggregate by distinct
    sample values with normalized weights -> {"kind":"dist_enum","support":[...],"probs":[...]}.
  - Distribution with enumerate_support() -> {"kind":"dist_enum","support":[...],"probs":[...]}.
  - other Distribution (continuous/parametric) -> {"kind":"dist_param","family":...,"params":{...}}.
    Parameter names are NOT remapped here; eval/algebra.py owns the alias table.
  - callable -> error to stderr + exit 2.
  - anything else -> error to stderr + exit 2 (no silent repr fallback).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Path to the project venv's python (so subprocess sees pyro/torch).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"


# Use ''' delimiter so """ inside user-facing docstrings don't close this string.
SERIALIZER_HEADER = r'''
import json
import math
import os
import sys
import pyro
import pyro.distributions as dist
import torch
from collections import defaultdict

# ── Seeding ──────────────────────────────────────────────────────────────────
# pyro.set_rng_seed seeds torch, Python random, and numpy in one call.
pyro.set_rng_seed(int(os.environ.get("PPL_GYM_PYRO_SEED", "42")))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_distribution(x):
    return isinstance(x, (pyro.distributions.Distribution,
                           torch.distributions.Distribution))


def _is_empirical(x):
    # True for Empirical and its subclass EmpiricalMarginal.
    return isinstance(x, pyro.distributions.Empirical)


def _tensor_to_py(t):
    # Coerce a 0-D or 1-D tensor to a Python scalar or list.
    if not isinstance(t, torch.Tensor):
        return t
    if t.dim() == 0:
        return t.item()
    return t.tolist()


def _serialize_key(k):
    # Serialize a dict key to a JSON string.
    # str  -> pass through.
    # bool -> json.dumps so True->"true", False->"false".
    # int/float -> json.dumps.
    # tuple -> serialize elements then json.dumps of the resulting list.
    # anything else -> json.dumps(_serialize(k)).
    if isinstance(k, str):
        return k
    if isinstance(k, bool):
        return json.dumps(k)
    if isinstance(k, (int, float)):
        return json.dumps(k)
    if isinstance(k, tuple):
        return json.dumps([_serialize(x) for x in k])
    return json.dumps(_serialize(k))


def _serialize_empirical(x):
    # Aggregate Empirical / EmpiricalMarginal -> dist_enum.
    # Weights are obtained via softmax over log_weights (handles both
    # uniform log_weights from unweighted samplers and IS log_weights).
    weights = torch.softmax(x._log_weights, dim=0)
    agg = defaultdict(float)
    raw_vals = {}
    for sample, w in zip(x._samples, weights):
        if sample.dim() == 0:
            key = sample.item()
        else:
            key = tuple(sample.tolist())
        str_key = json.dumps(key)
        agg[str_key] += w.item()
        if str_key not in raw_vals:
            raw_vals[str_key] = _tensor_to_py(sample)
    total = sum(agg.values())
    # Sort by JSON string key for canonical ordering.
    pairs = sorted(agg.items(), key=lambda kv: kv[0])
    support = [raw_vals[k] for k, _ in pairs]
    probs = [v / total for _, v in pairs]
    return {"kind": "dist_enum", "support": support, "probs": probs}


def _serialize_enum_dist(x):
    # Distribution with finite enumerate_support -> dist_enum.
    sup = x.enumerate_support()
    log_p = x.log_prob(sup)
    probs_t = log_p.exp().detach()
    # Flatten any batch dims by averaging.
    if probs_t.dim() > 1:
        probs_t = probs_t.view(probs_t.shape[0], -1).mean(dim=1)
    total = probs_t.sum().item()
    if total <= 0:
        sys.stderr.write("ERROR: distribution has zero total probability\n")
        sys.exit(2)
    sup_py = [_tensor_to_py(s) for s in sup]
    probs_py = [p / total for p in probs_t.tolist()]
    # Canonical sort: numeric values first (by value), strings second.
    pairs = list(zip(sup_py, probs_py))
    try:
        pairs.sort(key=lambda kv: (0, kv[0]) if isinstance(kv[0], (int, float))
                                  else (1, str(kv[0])))
    except TypeError:
        pass
    sup_sorted, probs_sorted = zip(*pairs) if pairs else ([], [])
    return {"kind": "dist_enum", "support": list(sup_sorted), "probs": list(probs_sorted)}


def _serialize_param_dist(x):
    # Continuous/parametric distribution -> dist_param.
    # Parameter names are NOT remapped here; eval/algebra.py owns the alias
    # table (e.g. concentration1/0 -> a/b, loc/scale -> mu/sigma).
    family = type(x).__name__.lower()
    params = {}
    if hasattr(x, "arg_constraints"):
        for k in x.arg_constraints:
            try:
                v = getattr(x, k, None)
            except Exception:
                continue
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item() if v.numel() == 1 else v.tolist()
            params[k] = v
    return {"kind": "dist_param", "family": family, "params": params}


def _serialize(x):
    # Recursively serialize x to a JSON-native value.
    if x is None:
        return None
    # bool must come before int (bool is a subclass of int).
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, str)):
        return x
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            return x.item()
        if x.dim() == 1:
            return x.tolist()
        # >=2-D tensor
        return {"kind": "tensor", "dims": list(x.shape),
                "data": x.flatten().tolist()}
    if _is_empirical(x):
        return _serialize_empirical(x)
    if _is_distribution(x):
        if x.has_enumerate_support:
            try:
                return _serialize_enum_dist(x)
            except (NotImplementedError, AttributeError):
                pass
        return _serialize_param_dist(x)
    if callable(x) and not isinstance(x, type):
        sys.stderr.write("ERROR: ANSWER is a function\n")
        sys.exit(2)
    if isinstance(x, dict):
        return {_serialize_key(k): _serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_serialize(v) for v in x]
    # No silent repr fallback.
    sys.stderr.write(f"ERROR: cannot serialize ANSWER of type {type(x).__name__}\n")
    sys.exit(2)


def __emit_answer():
    try:
        ans = ANSWER  # noqa: F821 -- bound by the user program
    except NameError:
        sys.stderr.write("ERROR: program did not define top-level ANSWER\n")
        sys.exit(2)
    sys.stdout.write(json.dumps(_serialize(ans)))
'''

SERIALIZER_FOOTER = "__emit_answer()\n"


@dataclass
class ExecutionResult:
    success: bool
    answer: object = None
    raw_stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    code: str = ""


def _wrap_program(code: str) -> str:
    """Prepend serializer header, append answer-emit footer."""
    return SERIALIZER_HEADER + "\n" + code.rstrip() + "\n" + SERIALIZER_FOOTER


def execute_pyro(code: str, timeout: int = 30, random_seed: int | None = None) -> ExecutionResult:
    """Execute ``code`` as a Pyro program in a subprocess and return the result.

    ``random_seed`` is passed to the subprocess via ``PPL_GYM_PYRO_SEED``; the
    injected header calls ``pyro.set_rng_seed`` so the same seed gives identical
    output across calls.  Defaults to 42 when *None*.
    """
    full_code = _wrap_program(code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        env = {**os.environ}
        seed = random_seed if random_seed is not None else 42
        env["PPL_GYM_PYRO_SEED"] = str(seed)

        cmd = [str(_VENV_PY), tmp_path]
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
                error_message="program exited 0 but produced no output",
                code=code,
            )

        try:
            answer = json.loads(stdout)
        except json.JSONDecodeError:
            last_line = next((ln for ln in reversed(stdout.split("\n")) if ln.strip()), "")
            try:
                answer = json.loads(last_line)
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


# ---------------------------------------------------------------------------
# Batched execution: run many seeds in ONE subprocess (one torch import).
# ---------------------------------------------------------------------------

# Driver appended after SERIALIZER_HEADER. Runs the user program once per seed,
# reseeding + clearing the param store + using a FRESH namespace each time, so an
# in-process iteration reproduces a fresh-process-per-seed run exactly. Emits a
# JSON list of serialized answers (sentinel dict for a per-seed failure).
_BATCH_DRIVER = '''
__base_globals = dict(globals())
__results = []
for __seed in __SEEDS:
    try:
        pyro.set_rng_seed(int(__seed))
        pyro.clear_param_store()
        __ns = dict(__base_globals)
        exec(__USER_SRC, __ns)
        if "ANSWER" not in __ns:
            __results.append({"__pplgym_error__": "program did not define ANSWER"})
        else:
            __results.append(_serialize(__ns["ANSWER"]))
    except Exception as __e:
        __results.append({"__pplgym_error__": repr(__e)})
sys.stdout.write(json.dumps(__results))
'''


def execute_pyro_batch(code: str, seeds, timeout: int = 60, workers: int = 1) -> list:
    """Run ``code`` once per seed in a single subprocess (one torch import).

    Returns a list aligned with ``seeds``: each entry is the parsed answer for
    that seed, or ``None`` if that seed failed.  ``workers`` is accepted for
    interface symmetry with the WebPPL batch executor and ignored (one process).
    In-process reseed (set_rng_seed + clear_param_store + fresh namespace per
    iteration) reproduces fresh-process-per-seed output exactly.
    """
    seeds = list(seeds)
    if not seeds:
        return []
    program = (
        SERIALIZER_HEADER + "\n"
        + "__USER_SRC = " + repr(code) + "\n"
        + "__SEEDS = " + repr(seeds) + "\n"
        + _BATCH_DRIVER
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        tmp_path = f.name
    try:
        try:
            proc = subprocess.run(
                [str(_VENV_PY), tmp_path],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            return [None] * len(seeds)
        if proc.returncode != 0:
            return [None] * len(seeds)
        stdout = proc.stdout.strip()
        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError:
            last = next((ln for ln in reversed(stdout.split("\n")) if ln.strip()), "")
            try:
                raw = json.loads(last)
            except json.JSONDecodeError:
                return [None] * len(seeds)
        if not isinstance(raw, list) or len(raw) != len(seeds):
            return [None] * len(seeds)
        return [
            None if (isinstance(a, dict) and "__pplgym_error__" in a) else a
            for a in raw
        ]
    finally:
        os.unlink(tmp_path)


def _extract_error(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    # Python tracebacks: last "...Error: ..." line is the cause.
    for line in reversed(text.split("\n")):
        line = line.strip()
        if re.match(r"^[A-Z][a-zA-Z]+(?:Error|Exception):", line):
            return line
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("File ") and not line.startswith("at "):
            return line[:200]
    return text[:200] if text else "Unknown error"


if __name__ == "__main__":
    import json as _json
    # Smoke test: discrete distribution -> dist_enum.
    r = execute_pyro("ANSWER = dist.Bernoulli(0.7)", random_seed=42)
    print(f"success={r.success}")
    print(f"answer={_json.dumps(r.answer, indent=2)}")
    if not r.success:
        print(f"stderr={r.stderr}")
