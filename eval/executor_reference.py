"""Reference-draws pseudo-executor: replay posteriordb's gold posterior draws.

The "reference" language has no code to run — its realization ``code`` is a
posteriordb posterior name, and a run replays the stored reference draws (10
NUTS chains, R-hat ~ 1). To fit the uniform batch-executor interface and the
gate's k-seed noise-floor machinery, the chains are partitioned into one block
per seed; each block is returned as a record {param: [draws]} (the cloud form).

Deterministic in (code, seeds): same name + same seed count -> same blocks, so
gt_cache caches it like any other executor.
"""

from __future__ import annotations

from eval.posteriordb import reference_blocks


def execute_reference_batch(code: str, seeds, timeout: int, workers: int):
    """Replay gold draws as one record-of-clouds per seed.

    Returns ``(answers, errors)`` aligned with ``seeds`` (batch executor
    contract). ``code`` is the posteriordb posterior name. ``seeds`` only fixes
    how many disjoint chain-blocks to carve (the draws themselves are stored,
    not sampled), so each seed maps to one block. Stored draws never fail a
    single block — a shortfall raises (whole-run failure), so ``errors`` is
    all-None on success.
    """
    seeds = list(seeds)
    if not seeds:
        return [], []
    name = code.strip()
    blocks = reference_blocks(name, len(seeds))
    if len(blocks) < len(seeds):
        raise RuntimeError(
            f"reference '{name}' has {len(blocks)} chain-blocks but "
            f"{len(seeds)} seeds requested; reduce k.")
    return blocks[: len(seeds)], [None] * len(seeds)
