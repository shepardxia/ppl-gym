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
import tempfile
from pathlib import Path

from eval.exec_common import ExecutionResult, loads_lenient, strip_ansi


# Path to the project venv's python (so subprocess sees pyro/torch).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"


def _subprocess_env(**extra: str) -> dict:
    """Env for pyro subprocesses: BLAS/OMP capped at 4 threads.

    Torch defaults to one thread per core; our models' tensors are tiny (below
    the parallelization grain), so the extra threads only spin-wait — measured
    ~9x degradation with several concurrent subprocesses on a 128-core box, and
    15-25% slower even solo. Outputs verified bit-identical 64-vs-4 threads
    across heavy NUTS + exact problems (12/12), so cached GT runs stay valid.
    """
    env = {**os.environ,
           "OMP_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4"}
    env.update(extra)
    return env


# Use ''' delimiter so """ inside user-facing docstrings don't close this string.
SERIALIZER_HEADER = r'''
# ── Preamble ─────────────────────────────────────────────────────────────────
# The standard Pyro toolkit, provided to every realization so its code is
# import-free (mirrors WebPPL, whose deps are --require'd). Realizations use
# these names directly; the pyro primer documents the subset solvers rely on.
# Bump EXECUTOR_VERSION["pyro"] in eval/gt_cache.py when this changes.
import json
import math
import os
import sys
import random
import itertools
import functools
from collections import defaultdict, Counter
import torch
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.poutine

# ── Precision ────────────────────────────────────────────────────────────────
# Ground truth runs in float64. Torch defaults to float32 (~7 significant
# digits), which silently corrupts exact enumeration over models with extreme
# probabilities (e.g. a 6e-5 vs 1e-9 competition in a conditioned posterior).
# float64 is the right precision/speed trade for GT and removes that footgun
# from every realization at once.
torch.set_default_dtype(torch.float64)

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
        seed = random_seed if random_seed is not None else 42
        env = _subprocess_env(PPL_GYM_PYRO_SEED=str(seed))

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


def execute_pyro_batch(code: str, seeds, timeout: int = 60, workers: int = 1):
    """Run ``code`` once per seed, seeds split into <=``workers`` chunk subprocesses.

    Returns ``(answers, errors)`` aligned with ``seeds``: ``answers[i]`` is the
    parsed answer or ``None`` for a failed seed; ``errors[i]`` is that seed's
    real reason (``None`` on success). ``timeout`` is the per-run budget; the
    policy in eval.config turns it into a per-chunk bound (seed scale x chunk
    seed count, capped) — wall-clock = slowest chunk.  Per-seed reseed inside a
    chunk (set_rng_seed + clear_param_store + fresh namespace) reproduces
    fresh-process-per-seed output exactly, so chunk boundaries never affect
    results and cached runs stay valid.
    """
    from eval.config import PYRO_CHUNK_BUDGET_CAP, PYRO_SEED_BUDGET_SCALE

    seeds = list(seeds)
    if not seeds:
        return [], []
    n_chunks = min(max(1, workers), len(seeds))
    size = (len(seeds) + n_chunks - 1) // n_chunks
    chunks = [seeds[i:i + size] for i in range(0, len(seeds), size)]

    def budget(chunk: list) -> int:
        return min(timeout * PYRO_SEED_BUDGET_SCALE * len(chunk), PYRO_CHUNK_BUDGET_CAP)

    if len(chunks) == 1:
        return _run_pyro_chunk(code, chunks[0], budget(chunks[0]))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(_run_pyro_chunk, code, c, budget(c)) for c in chunks]
        chunk_out, first_err = [], None
        for fut in futures:
            try:
                chunk_out.append(fut.result())
            except RuntimeError as e:
                first_err = first_err or e
                chunk_out.append(None)
    if first_err is not None:
        raise first_err
    answers = [a for ans, _ in chunk_out for a in ans]
    errors = [e for _, errs in chunk_out for e in errs]
    return answers, errors


def _run_pyro_chunk(code: str, seeds: list, timeout: int) -> list:
    """One subprocess running ``code`` for each seed in ``seeds`` (one torch import)."""
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
                env=_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            # Whole-chunk failures (the subprocess died) propagate their reason as
            # a RuntimeError -> caught downstream as exec_error WITH the cause, not
            # the generic "execution failed". Per-seed model errors below stay None.
            raise RuntimeError(f"timeout after {timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(_extract_error(proc.stderr or proc.stdout))
        stdout = proc.stdout.strip()
        try:
            raw = loads_lenient(stdout)
        except json.JSONDecodeError:
            raise RuntimeError("non-JSON output from pyro program")
        if not isinstance(raw, list) or len(raw) != len(seeds):
            raise RuntimeError("pyro batch returned the wrong number of results")
        answers: list = []
        errors: list = []
        for a in raw:
            if isinstance(a, dict) and "__pplgym_error__" in a:
                answers.append(None)
                errors.append(str(a["__pplgym_error__"]) or "execution failed")
            else:
                answers.append(a)
                errors.append(None)
        return answers, errors
    finally:
        os.unlink(tmp_path)


def _extract_error(text: str) -> str:
    text = strip_ansi(text)
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
