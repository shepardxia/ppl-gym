# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`ppl-gym` (the working name; the GitHub repo is `shepardxia/ppl-gym`) is a benchmark dataset of probabilistic-programming **problems**: language-neutral statements with per-language ground-truth realizations, used to evaluate LLMs on probabilistic programming. The canonical corpus is 115 problems (`data/problems/{probmods2,dippl,forestdb}.jsonl`) with two verified realization columns: WebPPL (`data/realizations/webppl.jsonl`) and Pyro (`data/realizations/pyro.jsonl`) — rebuilt to **idiomatic** Pyro: 111/115 realizations, each crosscheck-verified against WebPPL GT *and* proper-Pyro-usage audited; the remaining 4 (the `inference-algorithms` hard-condition method demos: ex1.1/1.2/1.3 + ex2.4) are marked **Pyro-unavailable** with documented reasons (per-language availability; see `data/REALIZATIONS.md`). Planned: Stan (P5), memo/pluck (with the language creators).

Two halves:

1. A **Python pipeline** (`eval/`, `scripts/`) for authoring/gating problems, generating LLM solutions, executing WebPPL, and judging answers under one comparator.
2. A **web app** (`web/`) — Astro + Cloudflare Worker — where collaborators review problems in a two-pane browser (`/problems/<corpus>/`) and leave feedback into D1.

## The contract (read first)

- **`data/SCHEMA.md` is the contract** (problem/realization records, answer algebra, measured tolerance, gate). `data/REDESIGN.md` is the rationale + phase history. `data/REALIZATIONS.md` is the working knowledge for the realization layer — how to translate, add a language, and the lessons already paid for (keep it current). P0–P4 done (P2 harness collapse; P4 Pyro column, since rebuilt to idiomatic Pyro — 111/115 crosscheck-verified + usage-audited, 4 Pyro-unavailable); **P5 (Stan) is next**.
- A problem = `{problem_id, provenance, statement{given,model,query}, answer_spec, status}`. The statement must pin the answer, never the program (determination criterion). Prompts are **rendered**: statement + spec-derived harness-contract paragraph via `eval/render.py:render_problem` — wire formats never appear in prompts.
- `eval/algebra.py` is the only comparator: answer = Value/Dist/Record over domains bool/finite/int/real/realvec; representations (exact/enumerated/parametric/cloud) orthogonal; tolerance measured (GT noise floor + candidate split-half self-noise), never authored. Entry points: `judge()` (candidate vs GTs), `agreement()` (candidate vs candidate). Finite specs may declare `support` — the label *space* (incl. zero-prob labels; never the realized support, that leaks the answer); the renderer enumerates it and the canonicalizer rejects out-of-space mass as malformed.
- Gate (`eval/gate.py`): `phaseA` (multi-seed GT floors) / `solve` (render + submit solver batch; `--model` to escalate; `--dry-run` writes a `.dry.json` sidecar) / `judge` (execute solver code, classify accept/gt_suspect/underdetermined/solver_failure, stamp gate_model/timeout/n_solvers). Both report writers merge by problem_id — partial re-runs never clobber other rows. Campaign result (report v2, re-gated under the collapsed pipeline): 115/115 solver-verified, uniformly stamped, 1 opus-gated row — history in `data/problems/_gate_triage.md`. Retired problems: `data/problems/_retired.jsonl`; authoring rules incl. hard bans: `data/problems/_AUTHORING_BRIEF.md`.
- `data/pyro_v3/` is the dead pre-P4 Pyro attempt (plain-Python GTs, wire-format prompts) — archival only; the live column is `data/realizations/pyro.jsonl`. A realization must express its model via the language's own machinery (e.g. `pyro.sample`/`pyro.infer`) — audit idiomaticity with judgment, not a regex (a mechanical rule rewards the `pyro.factor`-veneer form of handrolling, which is how half the P4 column stayed handrolled); agents WILL hand back plain-Python lookalikes whose numbers pass. See `data/REALIZATIONS.md`.
- Legacy atom JSONLs (`data/atomized_v2.jsonl`, `data/curated_v3/*`, `data/eval_runs/*`) are archives; the eval pipeline no longer reads them. The web app's legacy `/c/` browser and the legacy curation scripts still do.

## Python toolchain

- Always use **`uv`** for package management: `uv pip install ...`, `uv sync`. The repo has no `pyproject.toml`; deps are pinned in `.venv/`.
- Always run scripts via **`.venv/bin/python`** — bare `python` may pick up conda. Tests / modules: `PYTHONPATH=. .venv/bin/python -m eval.<module>`.
- Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` (166, fast; covers algebra, render, executor_pyro, gate report mechanics, score smoke, status-vocabulary drift).
- WebPPL itself is a system binary (`webppl` on `$PATH`, currently from miniconda). The executor shells out to it.

## Eval pipeline (run order)

```
data/problems/*.jsonl + data/realizations/<language>.jsonl     (eval/corpus.py loads)
        │
        │  eval/generate_batch.py   (render prompts → Anthropic batch API, 50% off)
        ▼
generations.jsonl   {problem_id, slot, code, model}
        │
        │  eval/score.py            (GT collection + execution + algebra.verdict)
        ▼
scored.jsonl        {…, status: pass|fail|ill_posed|malformed|exec_error, distance, tol, floor}
```

```bash
# Generate solutions (submit + poll; --no-poll / --collect BATCH_ID to split)
PYTHONPATH=. .venv/bin/python -m eval.generate_batch \
  --model claude-sonnet-4-6 --output <run>/generations.jsonl [--ids ID ...]

# Score
PYTHONPATH=. .venv/bin/python -m eval.score \
  --generations <run>/generations.jsonl --output <run>/scored.jsonl

# Gate campaign (authoring-time verification)
PYTHONPATH=. .venv/bin/python -m eval.gate phaseA [--ids ...]
PYTHONPATH=. .venv/bin/python -m eval.gate solve --ids ... --model claude-opus-4-8 --manifest <path>
PYTHONPATH=. .venv/bin/python -m eval.gate judge --manifest <path> [--report <path>]

# Cross-language consistency (target column vs reference GT, symmetric tolerances)
PYTHONPATH=. .venv/bin/python -m eval.gate crosscheck --language pyro [--ids ...]

# Canonical GT answers (feeds the web overlay charts) — rerun after realization changes
PYTHONPATH=. .venv/bin/python -m eval.gate answers --language webppl
PYTHONPATH=. .venv/bin/python -m eval.gate answers --language pyro
```

`eval.config` defaults: `seed=42`, `n_mc=200`, `mc_workers=8`, `timeout=60`. Workers multiply across levels (problem-level × per-seed); both score and gate clamp so WebPPL process count stays bounded.

## WebPPL execution (non-obvious bits)

`eval/executor.py` injects a JSON serializer header before user code and appends `JSON.stringify(__serialize(ANSWER))`. Distributions become `{"__kind":"distribution", probs, support}`; tensors and continuous distributions get their own `__kind` tags (legacy wire forms; `eval/algebra.py` canonicalizes them).

WebPPL packages in `eval/deps/` are loaded via `--require` for every run:

- `probmods-deps`, `probmods-draw`, `probmods-physics`, `probmods-towdata`, `probmods-seeded-random`, `probmods-viz-stub`.
- `probmods-viz-stub/header.js` is the shim that makes `viz(...)`, `viz.<method>(...)`, `drawLines`, `print`, etc. into headless no-ops. **Bare-identifier calls (`viz(x)`) get CPS-transformed by WebPPL — they need `function(s,k,a,...args)` returning `k(s,...)`. Member calls (`viz.bar(x)`) stay plain JS.** Mix this up and the program halts silently with no error.
- WebPPL forbids field assignment on top-level vars (`viz.table = ...` errors). All shims must be exposed via package headers, not in-program.

## Legacy curation (kept for sourcing new corpora)

`scripts/assemble_curated.py` + `scripts/extract_atoms.py` are the atom-era curation pipeline (emission JSONL → assembled, gated atoms). They still run but target the legacy atom format; new-corpus sourcing should re-derive problems per `data/problems/_AUTHORING_BRIEF.md` instead. Source corpora live at `data/sources/{dippl,forestdb.org,problang,probmods2,webppl}/` (treated as deps; gitignored).

## Web app (`web/`)

Astro 5 + `@astrojs/cloudflare` adapter, all pages prerendered. Routes: `/` landing (framing sentences, IR figure, corpora, updates rendered from `web/src/docs/updates.md`); `/problems/<corpus>/` the problem browser — a two-pane explorer driven unchanged by `public/browse.js` via its markup contract (`.atom-row`, `data-aid`, `#ppl-meta`, `[data-feedback]`); `/p/<slug>` redirects into the browser; `/c/<collection>` the legacy atom browser; `/methodology`. One SSR endpoint `POST /api/feedback` writes to D1.

**Design philosophy (hard-won; do not regress):**
- The browser is the product — an evidence instrument in the original academic design system (parchment tokens, mono labels, bucket glyphs, `browse.js` interactions). Show data; never replace the designed browser with plain tables.
- The landing is the name, the settled framing sentences, the one IR figure, corpora rows, updates, GitHub link. No brochure copy, no slogan headlines, no CTA buttons, no card directories.
- The site carries the ideas and the data; the repo carries the machinery. No pages that translate the pipeline back into English (a Reviewing and a Results page died for this).
- Figures must carry meaning spatially (the IR region holds the stored GT; per-language answers land on the same bars). Boxes-with-arrows pipeline charts are banned.
- Prose: plain declarative sentences; no "X, not Y" / "X — never Y" constructions. The landing framing sentences are settled wording — change only with the user.
- Site changes are design decisions: discuss before shipping; every push to main deploys.

**Pushes to `main` auto-deploy** via Cloudflare's Git integration (no GitHub Actions in the repo — it's configured in the Cloudflare dashboard). Treat every push as a production deploy; D1 migrations must be applied remotely (`npm run db:migrate:remote`) before or with any push whose worker code needs the new schema.

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

- Build runs with **CWD = `web/`**. `src/lib/problems.ts` resolves `process.cwd() + '/..'` to find the dataset; do not change to `import.meta.url`-based resolution (vite bundles the file into `dist/_worker.js/chunks/` and the relative path breaks).
- Build-time inputs: `data/problems/*.jsonl`, `data/realizations/{webppl,pyro}.jsonl`, the gate reports, `data/problems/_gt_answers.jsonl` (overlay-chart data; regenerate with `eval.gate answers`), and the legacy atom JSONLs (for `/c/`). Status tones live in `src/lib/tones.ts`.
- The name-entry modal markup is inlined in each browser page (`#name-modal` ids consumed by `public/browse.js`).
- Feedback API seam: clients post an `atom_id` field (the original widget contract); the API writes it to the D1 `problem_id` column (renamed in migration 0002). Same id values either way.
- `dist/.assetsignore` (sourced from `public/.assetsignore`) excludes `_worker.js` and `_routes.json` from the static-asset upload — without it, `wrangler deploy` refuses to upload because it would expose server code.
- D1 binding: `env.DB` (`ppl-gym-feedback`, id in `wrangler.toml`). Schema in `migrations/0001_init.sql`. R2 binding is commented out pending `wrangler r2 bucket create ppl-gym-backups`; backups will live in a separate `ppl-gym-backup` Worker, not this one.
- **Local D1 state gotcha**: `wrangler dev --persist-to <path>` and `wrangler d1 migrations apply --local` must use the SAME persist path or they read different SQLite files (silent "no such table" at POST time). Default is `.wrangler/state/v3`; pass `--persist-to` to both or neither.
- Live URL: `https://pplgym.kingdomofends.org` (custom domain attached to the `ppl-gym` Worker).

## Cost discipline & process gotchas

- **LLM re-gens are expensive.** Scope generation batches to the problems a change actually targets (`--ids`); full-corpus re-gens need explicit justification.
- **Subagents inherit the parent's model unless pinned.** For routine review/audit, always pass `model="sonnet"` and instructions saying "all N problems, no sampling." See `data/REVIEW_PROCESS.md`.
- **Don't silently drop data.** Failures land in `_*_broken.jsonl` / triage reports with evidence — they're for investigation, not for ignoring.
- **Don't push directly to `main` without authorization.** Earlier blanket "commit and push" approvals don't carry across to subsequent changes; ask each time.
- **GT edits are provenance-locked.** A realization matching its textbook source is authoritative; statements are the rewritable layer. Overriding requires evidence the source is internally inconsistent (see occams ex1.2/ex1.3 in `_gate_triage.md`) — document the deviation in the realization code itself.
