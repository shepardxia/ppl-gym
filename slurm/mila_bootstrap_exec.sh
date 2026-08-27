#!/bin/bash
# Build the four execution toolchains ppl-gym scoring needs, into $SCRATCH.
# Idempotent: each stage skips if its artifact is already present.
#
# Produces $SCRATCH/pplgym_env.sh, which every scoring job must source — the
# executors locate Julia and the pyro interpreter through the env vars it sets.
set -uo pipefail

VENV=$SCRATCH/venvs/pplgym
NODE_DIR=$SCRATCH/node
JULIA_DIR=$SCRATCH/julia
SRC_DIR=$HOME/ppl-gym/data/sources
export UV_CACHE_DIR=$SCRATCH/uv-cache
export JULIA_DEPOT_PATH=$SCRATCH/julia-depot
export CMDSTAN=$SCRATCH/cmdstan
export PATH="$HOME/.local/bin:$PATH"

NODE_VER=20.18.1
JULIA_VER=1.10.5

step() { echo; echo "=== $* ==="; date -Is; }
fail=0
try() { "$@" || { echo "!! STAGE FAILED: $*"; fail=1; }; }

step "1/5 python deps (torch CPU + pyro + cmdstanpy + hub)"
if "$VENV/bin/python" -c "import pyro, cmdstanpy" 2>/dev/null; then
  echo "skip: pyro + cmdstanpy present"
else
  try uv pip install --python "$VENV/bin/python" \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch numpy scipy pyro-ppl cmdstanpy huggingface_hub
fi
"$VENV/bin/python" - <<'PY'
for m in ("torch", "pyro", "cmdstanpy", "numpy", "scipy"):
    try:
        mod = __import__(m)
        print(f"  {m:10s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:10s} MISSING ({e})")
PY

step "2/5 node $NODE_VER"
if [ -x "$NODE_DIR/bin/node" ]; then
  echo "skip: $($NODE_DIR/bin/node -v)"
else
  mkdir -p "$NODE_DIR"
  try bash -c "curl -fsSL https://nodejs.org/dist/v$NODE_VER/node-v$NODE_VER-linux-x64.tar.xz \
      | tar -xJ -C '$NODE_DIR' --strip-components=1"
fi
export PATH="$NODE_DIR/bin:$PATH"
node -v || fail=1

step "3/5 probmods2 sources + bundled webppl"
# node_modules must be copied from a known-good install, NOT resolved by a fresh
# `npm install`: probmods2 pins webppl and its plugins as unlocked github: refs,
# so a present-day resolve pulls ESM-only transitive deps that the CommonJS
# require() chain cannot load (ERR_REQUIRE_ESM via @csstools/css-calc). The
# WebPPL GT was validated against the older tree, so that tree is the artifact.
mkdir -p "$SRC_DIR"
if [ ! -d "$SRC_DIR/probmods2/.git" ]; then
  try git clone --depth 1 https://github.com/probmods/probmods2.git "$SRC_DIR/probmods2"
fi
if [ -x "$SRC_DIR/probmods2/node_modules/webppl/webppl" ]; then
  echo "skip: webppl present (expected: rsynced from a known-good tree)"
else
  echo "!! node_modules absent — rsync it from a working checkout:"
  echo "     rsync -az --delete <local>/data/sources/probmods2/node_modules/ \\"
  echo "       mila:~/ppl-gym/data/sources/probmods2/node_modules/"
  fail=1
fi
ls -l "$SRC_DIR/probmods2/node_modules/webppl/webppl" 2>/dev/null \
  || { echo "!! webppl binary absent"; fail=1; }

step "4/5 julia $JULIA_VER + Gen.jl"
if [ -x "$JULIA_DIR/bin/julia" ]; then
  echo "skip: $($JULIA_DIR/bin/julia --version)"
else
  mkdir -p "$JULIA_DIR"
  MAJ=${JULIA_VER%.*}
  try bash -c "curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/$MAJ/julia-$JULIA_VER-linux-x86_64.tar.gz \
      | tar -xz -C '$JULIA_DIR' --strip-components=1"
fi
"$JULIA_DIR/bin/julia" --version || fail=1
# JSON is not optional: the gen executor's injected preamble serializes ANSWER
# through it, so a Gen-only depot fails every problem with a GT-side LoadError.
try "$JULIA_DIR/bin/julia" -e 'using Pkg; Pkg.add(["Gen", "JSON"]); using Gen, JSON; println("Gen.jl + JSON.jl ok")'

step "5/5 cmdstan"
if [ -d "$CMDSTAN" ] && ls "$CMDSTAN"/bin/stanc >/dev/null 2>&1; then
  echo "skip: cmdstan present at $CMDSTAN"
else
  try "$VENV/bin/python" -m cmdstanpy.install_cmdstan --dir "$SCRATCH" --cores 4
fi

step "writing $SCRATCH/pplgym_env.sh"
CMDSTAN_REAL=$(ls -d "$SCRATCH"/cmdstan-* 2>/dev/null | sort -V | tail -1)
cat > "$SCRATCH/pplgym_env.sh" <<EOF
# Source before any ppl-gym scoring run on Mila.
export PATH="$NODE_DIR/bin:$JULIA_DIR/bin:\$HOME/.local/bin:\$PATH"
export JULIA_DEPOT_PATH=$SCRATCH/julia-depot
export PPL_GYM_JULIA=$JULIA_DIR/bin/julia
export PPL_GYM_PYRO_PYTHON=$VENV/bin/python
export CMDSTAN=${CMDSTAN_REAL:-$CMDSTAN}
export PPLGYM_VENV=$VENV
EOF
cat "$SCRATCH/pplgym_env.sh"

step "DONE"
[ "$fail" -eq 0 ] && echo "BOOTSTRAP_EXEC_OK" || echo "BOOTSTRAP_EXEC_PARTIAL (see STAGE FAILED above)"
