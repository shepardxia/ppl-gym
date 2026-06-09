# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`ppl-gym` (the working name; the GitHub repo is `shepardxia/ppl-gym`) is a benchmark dataset of WebPPL probabilistic-programming exercises packaged as **atoms**: self-contained `(prompt, groundtruth_code, groundtruth_output)` triples used to evaluate LLMs on probabilistic programming. The README points at a broader CHI-PPL multi-language scope (memo, pluck), but **all active work is WebPPL-only**; the other language directories are stubs.

There are two halves:

1. A **Python pipeline** (`eval/`, `scripts/`) for curating atoms, running LLMs, executing WebPPL, and scoring distribution-comparison metrics.
2. A **web app** (`web/`) — Astro + Cloudflare Worker — that browses atoms with their LM eval results and collects per-atom feedback into D1.

## Atom contract (the central convention)

Every atom is one JSONL line with this shape:

```json
{ "id", "source", "task_type", "eval_mode", "answer_shape",
  "prompt", "groundtruth_code", "groundtruth_output" }
```

- `groundtruth_code` ends with `var ANSWER = <expression>;`. `eval/executor.py` wraps user code with a serializer that JSON-stringifies whatever is bound to `ANSWER`. **No regex parses the code; the contract is the binding.** When porting source to a new atom, the assembly script appends this binding via `wrap_target`.
- `answer_shape` is one of `value` | `samples` | `distribution` | `{record: {...}}`. The shape drives metric dispatch in `eval/metrics.compare_by_shape`.
- For `samples`-shape atoms, `groundtruth_output` is a *list of N seeded runs* (cached by `scripts/cache_groundtruth_outputs.py`); the harness re-runs the *generated* code N times, not the GT.
- The canonical hand-curated dataset is `data/atomized_v2.jsonl` (76 atoms). New curation rounds (e.g. `data/curated_v3/dippl.jsonl`, 17 atoms) follow the same shape; see `data/REVIEW_PROCESS.md` for the agent-driven curation/review workflow and anti-patterns (full-coverage review, no sampling, pinned `sonnet`).

## Python toolchain

- Always use **`uv`** for package management: `uv pip install ...`, `uv sync`. The repo has no `pyproject.toml`; deps are pinned in `.venv/`.
- Always run scripts via **`.venv/bin/python`** — bare `python` may pick up conda. Tests / modules: `PYTHONPATH=. .venv/bin/python -m eval.<module>`.
- WebPPL itself is a system binary (`webppl` on `$PATH`, currently from miniconda). The executor shells out to it.

## Eval pipeline (run order)

```
data/atomized_v2.jsonl
        │
        │  eval/generate_batch.py  (Anthropic batch API, 50% off, async)
        ▼
data/eval_runs/<config>/generations.jsonl
        │
        │  eval/score.py  (eval/harness, eval/metrics, eval/executor)
        ▼
data/eval_runs/<config>/scored.jsonl
        │
        ├─→  scripts/render_atoms_html.py   (legacy single-page HTML)
        └─→  web/                           (Astro site reads scored.jsonl directly)
```

Common commands (run from repo root):

```bash
# Submit a generation batch (async)
PYTHONPATH=. .venv/bin/python -m eval.generate_batch \
  --dataset data/atomized_v2.jsonl --model claude-sonnet-4-6 \
  --output data/eval_runs/<run-id>/generations.jsonl

# Score
PYTHONPATH=. .venv/bin/python -m eval.score \
  --dataset data/atomized_v2.jsonl \
  --generations data/eval_runs/<run-id>/generations.jsonl \
  --output data/eval_runs/<run-id>/scored.jsonl

# Re-render legacy HTML
PYTHONPATH=. .venv/bin/python -m scripts.render_atoms_html
```

`eval.config` defaults: `seed=42`, `n_mc=200`, `mc_workers=8`, `timeout=60`. The harness parallelizes per-seed runs; tune `mc_workers` if WebPPL spawns hit OS limits.

## WebPPL execution (non-obvious bits)

`eval/executor.py` injects a JSON serializer header before user code and appends `JSON.stringify(__serialize(ANSWER))`. Distributions become `{"__kind":"distribution", probs, support}`; tensors and continuous distributions get their own `__kind` tags.

WebPPL packages in `eval/deps/` are loaded via `--require` for every run:

- `probmods-deps`, `probmods-draw`, `probmods-physics`, `probmods-towdata`, `probmods-seeded-random`, `probmods-viz-stub`.
- `probmods-viz-stub/header.js` is the shim that makes `viz(...)`, `viz.<method>(...)`, `drawLines`, `print`, etc. into headless no-ops. **Bare-identifier calls (`viz(x)`) get CPS-transformed by WebPPL — they need `function(s,k,a,...args)` returning `k(s,...)`. Member calls (`viz.bar(x)`) stay plain JS.** Mix this up and the program halts silently with no error.
- WebPPL forbids field assignment on top-level vars (`viz.table = ...` errors). All shims must be exposed via package headers, not in-program.

## Atom curation

- **`scripts/assemble_curated.py`** is the current pipeline. Agents emit a JSONL of `{id, source, source_block_indices, prompt, wrap_target, notes}`; the script concats listed code blocks from source markdown, appends `var ANSWER = (<wrap_target>);`, executes via `execute_webppl(seed=42)`, and emits a fully-formed atom on success or a broken record on failure. Agents own *judgment* (chunking, prompt wording, helper inlining); the pipeline owns mechanics.

  ```bash
  PYTHONPATH=. .venv/bin/python -m scripts.assemble_curated \
    --emissions data/curated_v3/_<corpus>_emissions.jsonl \
    --output    data/curated_v3/<corpus>.jsonl \
    --broken    data/curated_v3/_<corpus>_broken.jsonl
  ```
- **`scripts/extract_atoms.py`** is the legacy 1-block-to-1-atom extractor. Its `wrap_with_answer`, `split_blocks`, `classify_answer` are still imported by `assemble_curated.py`; the rest is dead code from earlier rounds.
- Source corpora live at `data/sources/{dippl,forestdb.org,problang,probmods2,webppl}/` (treated as deps; gitignored).

## Web app (`web/`)

Astro 5 + `@astrojs/cloudflare` adapter. Static prerendered pages for the atom browser (`/c/<slug>/`); a single SSR endpoint `POST /api/feedback` writes to D1.

Build / deploy / dev:

```bash
cd web
npm install
npm run build                   # Astro -> dist/ + dist/_worker.js
npm run preview                 # wrangler dev (port 8787, local D1)
npm run deploy                  # build + wrangler deploy
npm run db:migrate:local        # apply migrations to local D1
npm run db:migrate:remote       # apply migrations to remote D1 (production!)
```

- Build runs with **CWD = `web/`**. `src/lib/atoms.ts` resolves `process.cwd() + '..'` to find the dataset; do not change to `import.meta.url`-based resolution (vite bundles the file into `dist/_worker.js/chunks/` and the relative path breaks).
- The site reads `data/atomized_v2.jsonl`, `data/curated_v3/*.jsonl`, and `data/eval_runs/*/scored.jsonl` at build time. Adding a new collection = drop a JSONL into `data/`, append one entry to `COLLECTIONS` in `src/lib/atoms.ts`, push.
- Bucket naming (`'TV=0'`, `'val+'`, `'shape!'`, etc.) and tone mapping are shared logic with `scripts/render_atoms_html.py`; `web/src/lib/buckets.ts` is the TypeScript port. Keep the labels in sync if you change one.
- **Prompt text** (system base + WebPPL primer) lives canonically at `data/prompts/{system_base,webppl_primer}.txt`. `eval/prompt.py` reads them at import via `Path.read_text()`; `web/src/lib/prompts.ts` uses Vite `?raw` imports so the text is inlined at build time (no Node fs at runtime). Edit the .txt files only — never inline the text in either reader.
- `dist/.assetsignore` (sourced from `public/.assetsignore`) excludes `_worker.js` and `_routes.json` from the static-asset upload — without it, `wrangler deploy` refuses to upload because it would expose server code.
- D1 binding: `env.DB` (`ppl-gym-feedback`, id in `wrangler.toml`). Schema in `migrations/0001_init.sql`. R2 binding is commented out pending `wrangler r2 bucket create ppl-gym-backups`; backups will live in a separate `ppl-gym-backup` Worker, not this one.
- **Local D1 state gotcha**: `wrangler dev --persist-to <path>` and `wrangler d1 migrations apply --local` must use the SAME persist path or they read different SQLite files (silent "no such table" at POST time). Default is `.wrangler/state/v3`; pass `--persist-to` to both or neither.
- Live URL: `https://pplgym.kingdomofends.org` (custom domain attached to the `ppl-gym` Worker).

## Cost discipline & process gotchas

- **LLM eval re-gens are expensive.** Don't re-run the full 76-atom batch unless changes target every atom. `scripts/rebuild_prompts.py` rebuilds prompts only for atoms whose source emissions changed; re-gen against that subset.
- **Subagents inherit the parent's model unless pinned.** For routine review/audit, always pass `model="sonnet"` and instructions saying "all N atoms, no sampling." See `data/REVIEW_PROCESS.md`.
- **Don't silently drop data.** Atoms that fail assembly/exec land in `_*_broken.jsonl` with the agent's notes + executor stderr — they're for triage, not for ignoring.
- **Don't push directly to `main` without authorization.** Earlier blanket "commit and push" approvals don't carry across to subsequent changes; ask each time.
