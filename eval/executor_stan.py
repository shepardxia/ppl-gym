"""Stan binding: compile a self-contained Stan bundle and sample its posterior.

The realization ``code`` is a bundle (eval.stan_bundle): a Stan model plus
//@ DATA / PARAMS / SAMPLING directives. One run = one seeded NUTS fit; the
answer is a record {param: [draws]} over the queried parameters (the cloud
representation of each marginal). This is the executable column verified, by
`gate crosscheck`, against posteriordb's gold reference draws.

Compiled executables are cached under data/.stan_cache/<model-hash>/ (never
/tmp) so repeated runs of the same model skip the ~3s compile.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from eval.stan_bundle import unpack

# cmdstanpy is chatty (per-chain INFO); quiet it so executor output stays clean.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# Absolute, repo-anchored: cmdstanpy's compile transiently chdir's the process,
# so a relative compile path can break when concurrent fits are compiling.
_COMPILE_DIR = Path(__file__).resolve().parents[1] / "data/.stan_cache"


@contextlib.contextmanager
def _cwd_guard():
    """Restore the process CWD on exit.

    cmdstanpy's compile/sample can leave the process working directory changed
    (notably when a fit errors — e.g. an ODE CVode failure — and a chdir is not
    unwound). Because the gate runs problems in threads of one process, a leaked
    CWD breaks every later relative-path file op (the report write, the next
    reference read). Guard the boundary so a Stan batch never leaks CWD.
    """
    saved = os.getcwd()
    try:
        yield
    finally:
        if os.getcwd() != saved:
            os.chdir(saved)


@lru_cache(maxsize=None)
def _compiled_model(model_code: str):
    """Compile (or reuse) a CmdStanModel for this model source.

    Keyed by the model hash on disk; cmdstanpy skips recompilation when the exe
    is newer than the .stan source. The in-process lru_cache also memoizes the
    CmdStanModel so repeat runs of the same model skip the file read + reconstruct.
    """
    from cmdstanpy import CmdStanModel

    h = hashlib.sha256(model_code.encode()).hexdigest()[:16]
    d = _COMPILE_DIR / h
    d.mkdir(parents=True, exist_ok=True)
    src = d / "model.stan"
    if not src.exists() or src.read_text() != model_code:
        src.write_text(model_code)
    return CmdStanModel(stan_file=str(src))


def _one_fit(model, data: dict, params: list[str], sampling: dict,
             seed: int, timeout: int):
    """Return {param: [draws]} for a single seeded fit, or None on failure."""
    try:
        fit = model.sample(
            data=data,
            seed=seed,
            chains=sampling.get("chains", 4),
            parallel_chains=sampling.get("chains", 4),
            iter_warmup=sampling.get("iter_warmup", 1000),
            iter_sampling=sampling.get("iter_sampling", 1000),
            adapt_delta=sampling.get("adapt_delta", 0.8),
            max_treedepth=sampling.get("max_treedepth", 10),
            show_progress=False,
            show_console=False,
            timeout=timeout or None,
        )
        df = fit.draws_pd()
        out: dict[str, list] = {}
        for p in params:
            if p not in df.columns:
                return None  # model does not expose a queried parameter
            out[p] = df[p].astype(float).tolist()
        return out
    except Exception:
        return None


def execute_stan_batch(code: str, seeds, timeout: int, workers: int) -> list:
    """Run a Stan bundle across seeds; one record-of-clouds per seed.

    Returns a list aligned with ``seeds``; a failed fit is ``None`` (the GT
    cache never stores a batch containing a failure).
    """
    seeds = list(seeds)
    if not seeds:
        return []
    with _cwd_guard():
        b = unpack(code)
        model = _compiled_model(b.model)  # compile once, reuse across seeds

        # Seeds are independent fits. cmdstanpy already parallelizes chains within
        # a fit; run a few fits concurrently but keep total processes bounded.
        chains = b.sampling.get("chains", 4)
        fit_workers = max(1, min(len(seeds), max(1, workers // chains)))

        def run(seed: int):
            return _one_fit(model, b.data, b.params, b.sampling, seed, timeout)

        if fit_workers == 1:
            return [run(s) for s in seeds]
        with ThreadPoolExecutor(max_workers=fit_workers) as ex:
            return list(ex.map(run, seeds))
