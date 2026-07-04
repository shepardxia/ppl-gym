"""Shared eval configuration constants.

Execution budget policy (single source of truth; executors implement it):

- ``DEFAULT_TIMEOUT`` is the per-run budget for one program execution: one
  WebPPL process, one Stan fit (before executor_stan's data/regime scaling),
  one Pyro seed (before the Pyro scale below).
- Pyro runs get ``PYRO_SEED_BUDGET_SCALE`` x the per-run budget, GT and
  candidate alike. Measured: faithful heavy-MCMC programs (hierarchical /
  observing-sequences / occams ex2.3) need 84-300s per seed, so a flat 60s
  systematically kills correct-but-slow programs; symmetric scaling keeps the
  fairness invariant "a candidate is never given less budget than the GT that
  judges it was validated under."
- A Pyro chunk subprocess covering several seeds gets the per-seed budget x
  its seed count, capped at ``PYRO_CHUNK_BUDGET_CAP`` so a hung many-seed
  draws chunk cannot hold a worker for hours.
"""

from __future__ import annotations

import os

DEFAULT_TIMEOUT = 60
DEFAULT_SEED = 42
DEFAULT_N_MC = 200
DEFAULT_MC_WORKERS = 8

PYRO_SEED_BUDGET_SCALE = 10
PYRO_CHUNK_BUDGET_CAP = 3600


def total_exec_workers() -> int:
    """Machine-wide executor-process budget for a scoring/gate run.

    Callers split it between problem-level parallelism and per-problem
    ``workers`` (see batch_executor_for's contract). The default is sized for
    a laptop; on a big box set ``PPL_GYM_EXEC_WORKERS`` (e.g. 48 on 128
    cores) instead of editing worker flags in every entry point.
    """
    env = os.environ.get("PPL_GYM_EXEC_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return DEFAULT_MC_WORKERS
