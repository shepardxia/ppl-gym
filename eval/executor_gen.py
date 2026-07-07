"""Gen (Gen.jl) executor — a Julia subprocess runs a Gen program, emits a JSON answer.

Scoped to **exact discrete inference** (`enumerative_inference` → exact posterior),
the family where Gen is a clean fit (no NUTS; continuous hierarchical is out of
scope — see data/DATASET_GENERATION.md / the Gen feasibility note). Because exact
inference is deterministic given the code, a batch runs the program **once** and
replicates the answer across the requested seeds — which also sidesteps Julia's
per-`@gen`-function JIT cost (paid once, not per seed; ~9s cold).

Realization contract: the program binds a top-level `ANSWER`, exactly like the
WebPPL/Pyro executors. `ANSWER` reduces to one of:
  - a `Dict` (label => probability)  → the language-neutral mapping form that
    `eval/algebra.py:canonicalize` accepts directly for a finite `dist` spec;
  - a `Bool` / number / string       → a value;
  - a `Vector` of the above          → a list.
Anything else errors (no silent repr fallback — mirrors the Pyro executor rule).

Bump EXECUTOR_VERSION["gen"] in eval/gt_cache.py when the serializer/driver changes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from eval.exec_common import ExecutionResult, loads_lenient


def _julia_bin() -> str:
    """Julia executable. Set PPL_GYM_JULIA on a box where julia is not on PATH."""
    return os.environ.get("PPL_GYM_JULIA", "julia")


# Injected before user code: JSON, a recursive serializer that turns a Gen ANSWER
# into the wire shapes eval/algebra.py accepts, and __pplgym_enum_dist — the
# helper that reduces an enumerative_inference result into a distribution.
#
# ANSWER contract (bind a top-level ANSWER, like the WebPPL/Pyro executors):
#   dist   → __pplgym_enum_dist(enumerative_inference(...))  (a {__kind:distribution,
#            support,probs} Dict; support elements may be bool/number/string/record-Dict)
#   record → a Dict(field_name => dist_or_value)
#   value  → a Bool/number/string/vector
# The serializer errors on any type it does not recognize (no silent repr fallback).
SERIALIZER_HEADER = r"""
using Gen, JSON, Random

# Soft factor / unnormalized potential. Gen's @gen DSL has no `factor`, but a
# custom Distribution whose logpdf returns its argument, OBSERVED at a dummy
# value, adds that argument to the trace log-weight — Gen's own extension
# mechanism, the faithful translation of WebPPL's factor()/Pyro's pyro.factor.
# Usage in a realization:  {:pot} ~ __pplgym_factor(w)  + choicemap((:pot, 0.0)).
struct __PplgymFactor <: Gen.Distribution{Float64} end
const __pplgym_factor = __PplgymFactor()
(::__PplgymFactor)(w::Real) = 0.0
Gen.random(::__PplgymFactor, w::Real) = 0.0
Gen.logpdf(::__PplgymFactor, x::Real, w::Real) = Float64(w)
Gen.logpdf_grad(::__PplgymFactor, x::Real, w::Real) = (nothing, nothing, nothing)
Gen.has_output_grad(::__PplgymFactor) = false
Gen.has_argument_grads(::__PplgymFactor) = (false,)
Gen.is_discrete(::__PplgymFactor) = true

function __pplgym_serialize(x)
    if isa(x, AbstractDict)
        return Dict(string(k) => __pplgym_serialize(v) for (k, v) in x)
    elseif isa(x, Bool)
        return x
    elseif isa(x, Integer) || isa(x, AbstractFloat)
        return x
    elseif isa(x, AbstractString) || isa(x, Symbol)
        return string(x)
    elseif isa(x, NamedTuple)
        return Dict(string(k) => __pplgym_serialize(v) for (k, v) in pairs(x))
    elseif isa(x, AbstractVector) || isa(x, Tuple)
        return [__pplgym_serialize(e) for e in x]
    else
        error("Gen ANSWER has unserializable type $(typeof(x)); reduce it to a distribution (__pplgym_enum_dist), a Dict(field=>...), or a number/bool/string/vector")
    end
end

# Reduce an enumerative_inference result -> a distribution over the model's return
# value: {__kind:distribution, support:[values], probs:[weights]}, aggregating
# duplicate return values. `res` is the (traces, log_norm_weights, lml) tuple.
function __pplgym_enum_dist(res)
    traces, logw = res[1], res[2]
    w = exp.(logw)
    vals = Any[]
    probs = Float64[]
    for (t, wi) in zip(traces, w)
        v = get_retval(t)
        idx = findfirst(u -> isequal(u, v), vals)
        if idx === nothing
            push!(vals, v); push!(probs, wi)
        else
            probs[idx] += wi
        end
    end
    s = sum(probs)
    if s > 0; probs = probs ./ s; end
    return Dict("__kind" => "distribution", "support" => vals, "probs" => probs)
end
"""


def _program(code: str, seed: int) -> str:
    # Header first (brings `using Random` into scope), then seed, then user code.
    return (SERIALIZER_HEADER + "\n"
            + "Random.seed!(" + str(int(seed)) + ")\n"
            + code + "\n"
            + "println(JSON.json(__pplgym_serialize(ANSWER)))\n")


def _extract_error(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text or "")
    # Julia prints `ERROR: <message>` before the stacktrace; that line is the cause.
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("ERROR:"):
            return s[len("ERROR:"):].strip()[:200] or "Gen error"
    for line in text.split("\n"):
        s = line.strip()
        if s and not s.startswith("Stacktrace") and not s.startswith("["):
            return s[:200]
    return (text[:200] or "Unknown error").strip()


def execute_gen(code: str, timeout: int = 60, random_seed: int = 0) -> ExecutionResult:
    """Run one Gen program (one Julia subprocess); return its serialized answer."""
    program = _program(code, random_seed)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jl", delete=False) as f:
        f.write(program)
        tmp_path = f.name
    try:
        try:
            proc = subprocess.run(
                [_julia_bin(), tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, error_message=f"Timeout after {timeout}s")
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error_message=f"julia not found ({_julia_bin()!r}); set PPL_GYM_JULIA")
        if proc.returncode != 0:
            return ExecutionResult(
                success=False,
                error_message=_extract_error(proc.stderr or proc.stdout),
                stderr=proc.stderr)
        stdout = (proc.stdout or "").strip()
        if not stdout:
            return ExecutionResult(
                success=False,
                error_message="program produced no output (did it bind ANSWER?)",
                stderr=proc.stderr)
        try:
            answer = loads_lenient(stdout)
        except json.JSONDecodeError as e:
            return ExecutionResult(success=False,
                                   error_message=f"output not valid JSON: {e}",
                                   stderr=stdout)
        return ExecutionResult(success=True, answer=answer, raw_stdout=proc.stdout)
    finally:
        os.unlink(tmp_path)


def execute_gen_batch(code: str, seeds, timeout: int = 60, workers: int = 1):
    """Run ``code`` and return ``(answers, errors)`` aligned with ``seeds``.

    Gen realizations are exact (enumerative) → deterministic given the code, so a
    batch runs the program ONCE and replicates the answer across seeds (also
    amortizing Julia JIT). A run failure is a whole-run failure → raises the real
    reason (batch-executor contract), never a per-seed None.
    """
    seeds = list(seeds)
    if not seeds:
        return [], []
    r = execute_gen(code, timeout=timeout, random_seed=seeds[0])
    if not r.success:
        raise RuntimeError(r.error_message or "gen execution failed")
    return [r.answer] * len(seeds), [None] * len(seeds)


if __name__ == "__main__":
    # Smoke test: the fair/biased coin (probmods2-conditioning/ex1.c).
    demo = r"""
@gen function coin_model()
    is_fair = ({:coin} ~ bernoulli(0.5))
    p = is_fair ? 0.5 : 0.9
    {:f1} ~ bernoulli(p); {:f2} ~ bernoulli(p); {:f3} ~ bernoulli(p)
    return is_fair ? "fair" : "biased"
end
obs = choicemap((:f1, true), (:f2, true), (:f3, true))
traces, logw, _ = enumerative_inference(coin_model, (), obs, choice_vol_grid((:coin, [false, true])))
w = exp.(logw)
ANSWER = Dict{String,Float64}()
for (t, wi) in zip(traces, w); ANSWER[get_retval(t)] = get(ANSWER, get_retval(t), 0.0) + wi; end
"""
    res = execute_gen(demo)
    print(f"success={res.success}")
    print(f"answer={res.answer}" if res.success else f"error={res.error_message}")
