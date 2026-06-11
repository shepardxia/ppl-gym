"""Pyro executor.

Contract: the user program ends by assigning to top-level `ANSWER`. The
caller wraps the program with a serializer that JSON-stringifies the
binding using the same cross-PPL schema as `executor.py` (WebPPL), so
metrics/web app code is unchanged.

Output schema (must match `executor.py`):
  - discrete distribution → {"__kind": "distribution", "probs": [...], "support": [...]}
  - continuous distribution → {"__kind": "distribution_continuous", "repr": "..."}
  - tensor → {"__kind": "tensor", "dims": [...], "data": [...]}
  - function → {"__kind": "function"}
  - everything else → JSON-native (numbers, strings, bools, lists, dicts)

`ANSWER` may be:
  - a `pyro.distributions.Distribution` (use `enumerate_support` for discrete)
  - a `torch.distributions.Distribution` (same)
  - a Python primitive / list / dict / tuple
  - a `torch.Tensor` (scalar or vector)
  - a list of samples (returned as-is; the caller re-aggregates — see eval/gate.collect_gt_answers)
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


SERIALIZER_HEADER = r"""
import json
import math
import sys
import pyro
import pyro.distributions as dist
import torch

def __serialize(x):
    # Order matters: check torch.Tensor before iterable/dict.
    if x is None:
        return None
    if isinstance(x, bool):  # bool is subclass of int — handle first
        return x
    if isinstance(x, (int, float, str)):
        return x
    if isinstance(x, torch.Tensor):
        if x.numel() == 1 and x.dim() == 0:
            return x.item()
        return {"__kind": "tensor", "dims": list(x.shape), "data": x.flatten().tolist()}
    if callable(x) and not isinstance(x, type) and not _is_distribution(x):
        return {"__kind": "function"}
    if _is_distribution(x):
        # Discrete with finite support → emit {probs, support}.
        try:
            sup = x.enumerate_support()
        except (NotImplementedError, AttributeError):
            return {"__kind": "distribution_continuous", "repr": _continuous_repr(x)}
        # log_prob can return a tensor of shape [support_size] or [support_size, ...] (batched).
        log_p = x.log_prob(sup)
        # Reduce any batch dims by summing (no batch dim expected in our usage).
        probs = log_p.exp().detach()
        if probs.dim() > 1:
            probs = probs.view(probs.shape[0], -1).mean(dim=1)
        # Coerce support to plain Python lists / scalars.
        sup_py = [__tensor_to_py(s) for s in sup]
        # Reorder so support is canonical (numeric/string asc).
        pairs = list(zip(sup_py, probs.tolist()))
        try:
            pairs.sort(key=lambda kv: (0, kv[0]) if isinstance(kv[0], (int, float)) else (1, str(kv[0])))
        except TypeError:
            pass
        sup_sorted, probs_sorted = zip(*pairs) if pairs else ([], [])
        return {"__kind": "distribution", "probs": list(probs_sorted), "support": list(sup_sorted)}
    if isinstance(x, dict):
        return {str(k): __serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [__serialize(v) for v in x]
    return repr(x)


_PYRO_TO_WEBPPL_PARAM = {
    # Beta: Pyro uses concentration1/concentration0; WebPPL uses a/b
    ("Beta", "concentration1"): "a",
    ("Beta", "concentration0"): "b",
    # Normal: Pyro loc/scale; WebPPL mu/sigma (called Gaussian there)
    ("Normal", "loc"): "mu",
    ("Normal", "scale"): "sigma",
    ("Gaussian", "loc"): "mu",
    ("Gaussian", "scale"): "sigma",
    # Gamma: Pyro concentration/rate; WebPPL shape/scale (note webppl uses inverse 1/rate)
    ("Gamma", "concentration"): "shape",
    ("Gamma", "rate"): "rate",
}
# Distributions whose Pyro class name should be remapped to webppl convention.
_DIST_NAME_REMAP = {"Normal": "Gaussian"}


def _continuous_repr(d):
    # Match WebPPL's `Name({k: v, k: v})` repr so cross-PPL comparison
    # on a continuous distribution's params succeeds.
    name = type(d).__name__
    name = _DIST_NAME_REMAP.get(name, name)
    params = {}
    if hasattr(d, "arg_constraints"):
        for k in d.arg_constraints:
            try:
                v = getattr(d, k, None)
            except Exception:
                continue
            if v is None: continue
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.tolist()
            params[_PYRO_TO_WEBPPL_PARAM.get((name, k), k)] = v
    if not params:
        return name + "()"
    def _fmt(v):
        # WebPPL's repr writes integer-valued floats without the decimal
        # (`10` not `10.0`). Mirror that so cross-PPL string comparison
        # succeeds.
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    inner = ", ".join(f"{k}: {_fmt(v)}" for k, v in params.items())
    return f"{name}({{ {inner} }})"


def __tensor_to_py(t):
    if isinstance(t, torch.Tensor):
        if t.numel() == 1:
            v = t.item()
            # Boolean-flavored scalar tensors come from Bernoulli; preserve int/bool.
            if t.dtype == torch.bool:
                return bool(v)
            return v
        return t.tolist()
    return t


def _is_distribution(x):
    return isinstance(x, (pyro.distributions.Distribution, torch.distributions.Distribution))


def __emit_answer():
    try:
        ans = ANSWER  # noqa: F821 - bound by the user program
    except NameError:
        sys.stderr.write("ERROR: program did not define top-level ANSWER\n")
        sys.exit(2)
    sys.stdout.write(json.dumps(__serialize(ans)))
"""

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
    full_code = _wrap_program(code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        # Seed Pyro/torch deterministically via env so the user code doesn't
        # have to remember to do it.
        env = {**os.environ}
        seed = random_seed if random_seed is not None else 42
        env["PPL_GYM_PYRO_SEED"] = str(seed)
        # Prepend a tiny seeder before the user code (via a wrapper file
        # would lose line numbers in errors; cleaner: inject into the header).
        # The seeder is part of SERIALIZER_HEADER instead. Add it now.
        # (Edit: already part of header in spirit; here we just set the env.)

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
    # Smoke test: discrete distribution → {probs, support}.
    r = execute_pyro(
        "ANSWER = dist.Bernoulli(0.7)",
        random_seed=42,
    )
    print(f"success={r.success}")
    print(f"answer={json.dumps(r.answer, indent=2)}")
    if not r.success:
        print(f"stderr={r.stderr}")
