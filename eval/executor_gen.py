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
import subprocess
import tempfile
from pathlib import Path

from eval.config import GEN_SEED_BUDGET_SCALE
from eval.error_tags import join_reasons
from eval.exec_common import ExecutionResult, loads_lenient, run_per_seed, strip_ansi


def _julia_bin() -> str:
    """Julia executable. Set PPL_GYM_JULIA on a box where julia is not on PATH."""
    return os.environ.get("PPL_GYM_JULIA", "julia")


# Injected before user code: JSON, a recursive serializer that turns a Gen ANSWER
# into the wire shapes eval/algebra.py accepts, and enum_dist — the
# helper that reduces an enumerative_inference result into a distribution.
#
# ANSWER contract (bind a top-level ANSWER, like the WebPPL/Pyro executors):
#   dist   → enum_dist(enumerative_inference(...))  (a {__kind:distribution,
#            support,probs} Dict; support elements may be bool/number/string/record-Dict)
#   record → a Dict(field_name => dist_or_value)
#   value  → a Bool/number/string/vector
# The serializer errors on any type it does not recognize (no silent repr fallback).
SERIALIZER_HEADER = r"""
using Gen, JSON, Random

# Soft factor / unnormalized potential. Gen's @gen DSL has no `factor`, so we
# define one the way Gen's docs prescribe for a from-scratch distribution
# (gen.dev how_to/custom_distributions): its logpdf returns its argument, so
# OBSERVING it at a dummy value adds that argument to the trace log-weight — the
# faithful translation of WebPPL's factor()/Pyro's pyro.factor, read by every
# inference algorithm (enumeration, mh, importance).
# Usage in a realization:  {:pot} ~ factor(w)  + choicemap((:pot, 0.0)).
struct Factor <: Gen.Distribution{Float64} end
const factor = Factor()
(::Factor)(w::Real) = 0.0
Gen.random(::Factor, w::Real) = 0.0
Gen.logpdf(::Factor, x::Real, w::Real) = Float64(w)
Gen.logpdf_grad(::Factor, x::Real, w::Real) = (nothing, nothing, nothing)
Gen.has_output_grad(::Factor) = false
Gen.has_argument_grads(::Factor) = (false,)
Gen.is_discrete(::Factor) = true

function _serialize_answer(x)
    if isa(x, AbstractDict)
        return Dict(string(k) => _serialize_answer(v) for (k, v) in x)
    elseif isa(x, Bool)
        return x
    elseif isa(x, Integer) || isa(x, AbstractFloat)
        return x
    elseif isa(x, AbstractString) || isa(x, Symbol)
        return string(x)
    elseif isa(x, NamedTuple)
        return Dict(string(k) => _serialize_answer(v) for (k, v) in pairs(x))
    elseif isa(x, AbstractVector) || isa(x, Tuple)
        return [_serialize_answer(e) for e in x]
    else
        error("Gen ANSWER has unserializable type $(typeof(x)); reduce it to a distribution (enum_dist), a Dict(field=>...), or a number/bool/string/vector")
    end
end

# Reduce an enumerative_inference result -> a distribution over the model's return
# value: {__kind:distribution, support:[values], probs:[weights]}, aggregating
# duplicate return values. `res` is the (traces, log_norm_weights, lml) tuple.
function enum_dist(res)
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
            + "println(JSON.json(_serialize_answer(ANSWER)))\n")


def _extract_error(text: str) -> str:
    text = strip_ansi(text)
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


# A realization declares itself STOCHASTIC (sampling inference: mh / importance /
# hmc over continuous latents, or forward sampling) with a plain `# @sampling`
# annotation anywhere in its code. Then each seed is an independent reseeded run —
# the batch must NOT replicate one run (that would report zero self-noise and a
# bogus tolerance). Exact/enumerative realizations omit it and get the cheap
# run-once-replicate path. (Explicit annotation, not inferred: forward-sampling
# realizations carry no mh/importance token and some sampling ones enumerate at a
# nested level, so no syntactic rule separates the two reliably.)
SAMPLE_MARKER = "@sampling"


def execute_gen_batch(code: str, seeds, timeout: int = 60, workers: int = 1):
    """Run ``code`` and return ``(answers, errors)`` aligned with ``seeds``.

    Two modes:
      - **exact** (default): enumerative_inference → deterministic given the code,
        so run ONCE and replicate across seeds (amortizes Julia JIT). A run failure
        is a whole-run failure → raises the real reason.
      - **sampling** (code contains the ``@sampling`` marker): mh/importance/hmc
        over continuous latents is stochastic, so each seed is an independent
        reseeded run (parallel across ``workers``); ``answers[i]`` is that seed's
        cloud or ``None``, ``errors[i]`` its reason. All-failed → raises.
    """
    seeds = list(seeds)
    if not seeds:
        return [], []
    # Budget policy (eval.config): the per-run budget is scaled up so a faithful
    # heavy-MCMC cloud built in one run is not killed; a cap, not a wait, so fast
    # exact runs are unaffected. Symmetric across GT and candidate (fairness).
    timeout = timeout * GEN_SEED_BUDGET_SCALE
    if SAMPLE_MARKER in code:
        answers, errors = run_per_seed(
            lambda s: execute_gen(code, timeout=timeout, random_seed=s),
            seeds, workers=workers, default_error="gen execution failed")
        if all(a is None for a in answers):
            raise RuntimeError(join_reasons(errors))
        return answers, errors
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
