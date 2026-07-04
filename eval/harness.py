"""Shared GT/execution harness primitives.

The neutral home for the answer-collection and code-similarity logic that both
the authoring-time gate (eval.gate) and the production scorer (eval.score)
depend on — so the scorer does not import the campaign CLI. (eval/metrics.py was
ripped in P2; code_jaccard lives here now.)
"""

from __future__ import annotations

import re

from eval.algebra import Spec, _has_draws_field, canonicalize
from eval.config import DEFAULT_N_MC, DEFAULT_SEED, DEFAULT_TIMEOUT, total_exec_workers
from eval.gt_cache import cached_run

# Defaults match SCHEMA.md: k=5 for non-draws, k=3 for draws blocks.
_DEFAULT_K_EXACT = 5
_DEFAULT_K_DRAWS = 3


# ---------------------------------------------------------------------------
# Code similarity (memorization heuristic)
# ---------------------------------------------------------------------------

def _normalize_code(code: str) -> str:
    # Strip comments for all realization languages: // and /* */ (WebPPL/JS),
    # # to end-of-line (Pyro/Python, Stan). code_jaccard runs on every language.
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"\s+", " ", code).strip()
    code = re.sub(r";\s*$", "", code)
    return code


def code_jaccard(generated: str, ground_truth: str) -> float:
    g = set(_normalize_code(generated).split())
    t = set(_normalize_code(ground_truth).split())
    if not g and not t:
        return 1.0
    if not g or not t:
        return 0.0
    return len(g & t) / len(g | t)


# ---------------------------------------------------------------------------
# GT answer collection
# ---------------------------------------------------------------------------

def collect_gt_answers(
    code: str,
    spec: Spec,
    *,
    language: str = "webppl",
    base_seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    k_exact: int = _DEFAULT_K_EXACT,
    k_draws: int = _DEFAULT_K_DRAWS,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int | None = None,
    use_cache: bool = True,
) -> tuple[list, int]:
    """Collect k independent canonical GT answers for (code, spec).

    Single path: every needed seed runs through the language's batched executor,
    cached by content hash (eval.gt_cache). Returns (canonical_answers, n_runs).

    Draws protocol (spec has draws anywhere):
      One GT answer = a block of n_draws seeded draws; k_draws blocks over
      contiguous seeds base_seed .. base_seed + k_draws*n_draws - 1. A block
      with at least one good draw is kept; an all-failed block raises.
    Non-draws (exact/enum/parametric):
      One GT answer = one run; seeds base_seed .. base_seed + k_exact - 1.
      Any failed run raises RuntimeError (a healthy GT succeeds on every seed).

    AlgebraError (canonicalization) and RuntimeError propagate; gate_problem()
    catches both.
    """
    if workers is None:
        workers = total_exec_workers()
    if _has_draws_field(spec):
        total = k_draws * n_draws
        seeds = [base_seed + i for i in range(total)]
        raw = cached_run(language, code, seeds, timeout=timeout,
                         workers=workers, use_cache=use_cache)
        canonical: list = []
        n_runs = 0
        for block in range(k_draws):
            chunk = [a for a in raw[block * n_draws:(block + 1) * n_draws]
                     if a is not None]
            if not chunk:
                raise RuntimeError("all runs failed")
            n_runs += len(chunk)
            canonical.append(canonicalize(chunk, spec))
        return canonical, n_runs

    seeds = [base_seed + i for i in range(k_exact)]
    # `timeout` = per-run budget; each executor applies the budget policy
    # documented in eval.config (Pyro seed scaling + chunk cap, Stan data/regime
    # scaling, WebPPL per-process as-is).
    raw = cached_run(language, code, seeds, timeout=timeout,
                     workers=workers, use_cache=use_cache)
    n_bad = sum(a is None for a in raw)
    if n_bad:
        raise RuntimeError(f"{n_bad}/{len(raw)} seeded runs failed")
    return [canonicalize(a, spec) for a in raw], k_exact


def execute_candidate_answer(
    code: str,
    spec: Spec,
    *,
    base_seed: int = DEFAULT_SEED,
    n_draws: int = DEFAULT_N_MC,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int | None = None,
    language: str = "webppl",
    gt_bundle: str | None = None,
) -> object:
    """Execute candidate code and return a canonical answer.

    The k=1 case of collect_gt_answers: draws-spec problems collect n_draws
    seeded runs into one canonical answer; others run once at base_seed.
    Candidate (solver) code is one-off, so its runs are not cached.

    Stan: a solver writes a bare model (no data values); pass `gt_bundle` (the
    GT realization bundle) so the candidate model is repacked around the same
    data / params / sampling the GT used. Ignored for other languages, and for
    Stan GT bundles that are already self-contained (pass gt_bundle=None there).
    """
    if language == "stan" and gt_bundle is not None:
        from eval.stan_bundle import DEFAULT_SAMPLING, repack_model
        # Candidate fits use the standard sampler config, never a GT's heavy
        # gold-reproduction regime — cheaper and keeps a slow solver model
        # bounded by the per-fit timeout. Measured tolerance handles the
        # smaller sample size.
        code = repack_model(gt_bundle, code, sampling=DEFAULT_SAMPLING)
    answers, _ = collect_gt_answers(
        code, spec,
        language=language,
        base_seed=base_seed, n_draws=n_draws,
        k_exact=1, k_draws=1,
        timeout=timeout, workers=workers,
        use_cache=False,
    )
    return answers[0]
