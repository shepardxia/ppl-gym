#!/bin/bash
# Opus 5 benchmark: 4 language arms x 1 sample, submit -> collect -> score.
# Resumable: a combo with generations on disk is re-scored, never re-generated.
set -uo pipefail
cd "$HOME/ppl-gym"
source "$SCRATCH/anthropic_key.sh"
source "$SCRATCH/pplgym_env.sh"

# Machine-wide executor-process budget; the scoring side splits it problem-level
# x per-problem. Kept at the allocation's core count, not above it.
export PPL_GYM_EXEC_WORKERS=${PPL_GYM_EXEC_WORKERS:-6}

echo "host=$(hostname) cores=$(nproc) workers=$PPL_GYM_EXEC_WORKERS"
echo "julia=$PPL_GYM_JULIA"
echo "cmdstan=$CMDSTAN"
date -Is

# Poll ceiling tracks the Batch API's own 24h maximum, not the 1h default: an
# Opus 5 batch can sit queued for hours, and a re-run resumes the same batch ids
# from runs/opus5/batches.jsonl rather than re-submitting.
PYTHONPATH=. "$PPLGYM_VENV/bin/python" -m eval.benchmark run \
  --models opus \
  --languages webppl,pyro,gen,stan \
  --n-samples 1 \
  --out runs/opus5 \
  --combo-workers 4 \
  --poll-timeout 86400

echo "RUN_EXIT=$?"
date -Is
