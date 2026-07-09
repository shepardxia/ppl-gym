# Dataset generation — formal procedure

How a problem and its per-language realizations enter this benchmark, end to end.
This is the **procedure of record**: any new corpus follows one of the routes
below and passes the same gate. Companion docs: `data/SCHEMA.md` (the record
contract + answer algebra + tolerance), `data/REALIZATIONS.md` (per-language
translation knowledge), `data/problems/_AUTHORING_BRIEF.md` (statement authoring
rules + hard bans). This file is the *pipeline*; those are the *contracts*.

## Pipeline at a glance

Every entry travels the same five stages. Only the **realization** stage branches
— by *where the ground-truth answer comes from* (§1) — and the gate (§4) is
identical across all branches. Nothing is published (§7) until it passes.

```
   SOURCE           STATEMENT           REALIZATION                GATE                    PUBLISH            BENCHMARK
  ────────         ───────────        ───────────────           ──────                   ─────────          ───────────
  vendored     ─▶  language-      ─▶   A. translate the    ─▶   phaseA    GT floor   ─▶   answers  overlay  ─▶  measure the
  corpus, or       neutral             verified reference       crosscheck  agree        export   → HF          models over
  posteriordb      record that         column                   solve/judge (route A)    web_rollouts          the corpus
  (mine or         pins the        ─▶  B. source-native,        ───────────────────      (committed to        (a separate,
  ingest)          ANSWER, never       gold reference draws     any failure is triaged    the repo)             deliberate step:
                   the program     ─▶  C. source-native,        to its cause (§5) or                            new problems
                                       self-generated,          the item retired with                          move the
                                       validated draws          a logged reason (§6)                            denominator)
    §1              §0, §2              §1 (routes A/B/C)        §4  +  §5 (triage)        §7                    §7 (note)
```

Read left to right. A **source** yields a **statement** whose queried quantity is
uniquely determined (the *determination criterion*, §0). That statement is
**realized** in each language by exactly one of three routes (§1), which differ
only in the provenance of the reference answer. Every realization is **gated**
against a *measured* tolerance (§4); a failure is diagnosed to its cause and fixed,
or the item is **retired** with evidence (§5–§6) — never silently dropped. Passing
entries are **published** to the web overlay and the Hugging Face dataset in one
refresh sweep (§7). **Benchmarking** models over the corpus is deliberately a
separate step, because adding problems changes the denominator every score is
reported against. The gate is a *control*, not a pre-decided policy: it surfaces
each defect and the author curates or prunes on that evidence, per problem.

## 0. What a dataset entry is

One **problem** = a language-neutral record
`{problem_id, provenance, statement{given, model, query}, answer_spec, status}`.
The statement must pin the **answer**, never the program (the *determination
criterion*): given the statement, the queried posterior/quantity is uniquely
determined, and any correct program in any language must reproduce it within the
measured tolerance. One **realization** = that problem expressed in one language's
own machinery (`data/realizations/<lang>.jsonl`), executed to produce a
canonical answer that `eval/algebra.py` compares.

A problem is only *in* the dataset for a language once that language's
realization has passed the gate (§4). Everything below is how we get there.

## 1. Three generation routes

A corpus enters by exactly one of these. The route determines where the
ground-truth answer comes from; the gate (§4) is identical across all three.

### Route A — translate the reference column (textbook corpora)
Used for: probmods2 (70), dippl (16), forestdb (29) — WebPPL-native sources.

1. **Source** a model from the vendored corpus (`data/sources/<corpus>/`).
2. **Author the problem**: extract a language-neutral statement per
   `_AUTHORING_BRIEF.md`. The WebPPL realization that matches the textbook source
   is **authoritative by provenance** (GT edits are provenance-locked, per
   CLAUDE.md) — it defines the answer.
3. **WebPPL is the ground truth.** Its executed answer, over multiple seeds, is
   the reference the answer_spec is measured against (phaseA noise floor, §4).
4. **Translate** to other languages (e.g. Pyro) from the statement — never by
   transliterating the WebPPL code. The translation must use the target
   language's idioms (`pyro.sample`/`pyro.infer`), audited by judgment, not a
   regex (`data/REALIZATIONS.md`). Cross-check the translation against the WebPPL
   GT (§4 crosscheck).
5. A language that cannot express the problem is marked **available:false** with
   a documented reason (e.g. the 4 Pyro-unavailable hard-condition demos).

### Route B — source-native with gold draws (posteriordb original)
Used for: posteriordb 45 (the subset shipping gold reference draws).

posteriordb inverts Route A: it brings its **own** problems (a Stan model + data)
**and** its own ground truth (gold MCMC reference draws, ~10 chains × long NUTS,
convergence-vetted by posteriordb authors). So:

1. **Ingest** the Stan model + data + gold draws (`eval/posteriordb.py`).
2. The **gold draws are the GT answer** (the `reference` column — replayed, never
   re-sampled: `eval/executor_reference.py`).
3. The **Stan program is validated against the gold draws** by crosscheck (§4):
   Stan NUTS must reproduce the gold posterior within symmetric tolerance.
4. Author the statement from the model + data (`authoring_material(name)`).

The Stan program here is a *realization to be validated*, not the source of
truth — the gold draws are. (arma11 was pruned as non-discriminable; §6.)

### Route C — source-native with self-generated (validated) draws
Used for: posteriordb remainder (the ~101 posteriors with a model + data but
**no** gold draws). This is the route that scales posteriordb beyond its gold
subset, and the worked example for the rest of this doc.

The move: **generate our own reference draws** under convergence gates strong
enough to call them reference-grade, converting a "gold-draws corpus" into a
"validated-draws corpus." Provenance is stated honestly — "self-generated
(eval/reference_gen.py); NOT posteriordb gold" — never laundered as gold.

1. **Inventory & select** candidates (`notes/posteriordb_expansion_inventory.json`).
   Exclude by rule *before* spending compute:
   - **Monsters** (huge N or exotic structure: covid19 imperial, mnist RBM) —
     out of scope, not benchmark-shaped.
   - **Duplicates** — one problem per *mathematical posterior*. Centered and
     non-centered parameterizations of the same model = the same posterior =
     one entry (keep one). Different priors = different posterior = distinct
     (e.g. `seeds_model` vague priors vs `seeds_stanified` narrow priors: both
     kept). Dedup is an authoring judgment on the models, not a name match.
2. **Generate reference draws** (`eval/reference_gen.py`, BOX only — heavy):
   long NUTS, **10 chains × 10 000 warmup × 1 000 kept after thinning 10**,
   adapt_delta 0.9, per model. Written as an **overlay** at
   `data/reference_draws/<name>.json` (+ `.info.json` provenance/diagnostics);
   the vendored posteriordb tree is never written. `eval/posteriordb.py` resolves
   gold first, overlay second (`validated_posterior_names()`).
3. **Convergence gate** (inside reference_gen; reference-grade or rejected):
   **R-hat ≤ 1.01** and **ESS_bulk ≥ 2000** on every queried parameter, or the
   candidate is REJECTED. This is what earns the "validated" claim. Bad geometry
   self-rejects here: label-switching mixtures (R-hat ≫ 1), non-identified /
   heavy-tailed models (timeout or low ESS). A timeout is an implicit reject.
4. **Ingest** passers (`eval.posteriordb build --ids <names>`): merges problem +
   Stan realization + reference realization records; existing statements/status
   preserved on rebuild.
5. **Curate the queried parameter set** (§3) — required before finalizing.
6. **Author the statement** from `authoring_material(name)` (§2).
7. **Gate**: crosscheck Stan-vs-reference + coherence + answers (§4).

**The translation loop (Route A, and adding any new language).** Translating a
column is not a single pass; it is a staged loop that centralizes the heavy work
and keeps the parallel work cheap (full detail in `data/REALIZATIONS.md` §4):
**(i)** *calibrate the hard idioms* against the WebPPL GT before fanning out —
never fan out on an unproven idiom, or you get a uniform column of
plausible-but-wrong code; **(ii)** *pure rewrite in parallel* — agents translate
every problem, execute nothing, self-verify nothing; **(iii)** *centralized
crosscheck → repair*, looped until numerically converged — the one heavy,
batched, cached step; **(iv)** *idiomatic-usage audit → repair* — numbers passing
is not the bar, a reviewer with the target API flags hand-rolled inference until
zero remain; **(v)** *decide availability from evidence* — only after real
attempts, never by a pre-emptive statement scan. The orchestrator holds the
contract and verifies harshly; subagents implement (self-written code biases
self-judgment). All work is staged in the repo, never `/tmp`.

## 2. Statement authoring

The statement is the language-neutral layer (rewritable; the realization is the
provenance-locked layer). Author from the model's actual priors + likelihood,
in the house style (see any ingested posteriordb statement, e.g.
`eight_schools_noncentered`):

- **given**: the data setup + **every prior stated explicitly** in prose
  (`Normal(mean 0, sd 5)`, `half-Cauchy(location 0, scale 5)`, `improper uniform
  (flat)` when the model declares none). Name the data arrays as the model uses
  them.
- **model**: the generative process (latent draws + likelihood) in
  language-neutral prose — no Stan/Python/JS syntax, no wire-format leakage.
- **query**: `"The marginal posterior distribution of each parameter given the
  data: <the interpretable parameters, named>."` Query the parameterization-
  invariant quantities only (§3) — e.g. `eight_schools` queries the school
  effects `theta`, never the non-centered auxiliary.

Every authored statement is **verified line-by-line against the source model**:
priors match, data claims match (e.g. n-of-k counts, standardization,
capture-recapture M/T), likelihood matches. Authoring is done with judgment, not
mechanically; the coherence gate (§4) is the automated backstop, not the primary
check.

## 3. Curation rules (what gets queried)

The answer_spec queries what `param_names(name)` exposes = the columns present in
the reference draws. Two classes of column must be curated OUT, or the item is a
bad benchmark question:

- **Parameterization-specific auxiliaries** — a non-centered model exposes
  `alpha_raw` (standardized residuals) *and* the interpretable `alpha = mu +
  sigma·alpha_raw`. Querying `alpha_raw` is a **correctness bug**: a candidate
  that writes a mathematically-equivalent *centered* model has no `alpha_raw`
  column and would wrongly fail. Query the parameterization-invariant quantities
  (`alpha`, `beta`, `mu_*`, `sigma_*`); drop every `*_raw`. Our `reference_gen`
  stores all draws columns, so this curation is explicit: filter `*_raw` keys out
  of the overlay `<name>.json`, then rebuild (param_names re-reads the overlay).
  Gold posteriordb references were already curated this way by their authors;
  self-generated overlays are not, so we must.
- **Degenerate columns** — quantities deterministic from the **data** alone
  (identical every draw, zero inference: e.g. `n_avoid`/`n_shock` counts), or
  predictive-RNG generated quantities (`y_rep`, posterior-predictive draws).
  These carry no posterior information and bloat/mislead the answer; curate to
  the model's actual `parameters`. (A *derived* quantity that is a deterministic
  function of sampled parameters — e.g. rats' `alpha0 = mu_alpha − xbar·mu_beta`
  — is interpretable and varies across draws, so it is kept.)

The coherence gate catches the tell-tales (Stan-jargon / "transformed
parameters" / generated-quantity names leaking into a statement), but curation is
proactive: inspect the model, decide the meaningful queried set, filter the
overlay, rebuild.

## 4. The gate — the arbiter (authoring-time GT verification)

`eval/gate.py`; the product claim IS that every entry passed this. No shortcuts
at scale. The five gate subcommands, in the order they apply:

- **phaseA** — multi-seed GT noise floor. Runs the GT k times; the measured
  spread (plus a candidate split-half self-noise term) *becomes* the tolerance.
  Tolerance is **measured, never authored** (`data/SCHEMA.md`). (Routes A/B/C:
  applies wherever a GT program is executed for its answer.)
- **crosscheck** — cross-language / cross-source GT agreement under symmetric
  tolerances. Route A: translated column vs WebPPL GT. Route B/C: Stan vs the
  `reference` column (gold or overlay). A pass means the realization reproduces
  the reference posterior; distances are reported as `d` vs `tol`.
- **solve** / **judge** (Route A translation campaigns) — render + submit a
  solver batch; execute + classify accept / gt_suspect / underdetermined /
  solver_failure.
- **answers** — canonical GT answers per (problem, language), composite-keyed,
  written to `data/problems/_gt_answers.jsonl` (feeds the web overlay).

Alongside the gate, the **coherence test** (`tests/test_posteriordb.py:
test_committed_dataset_coherent`, run in the pytest suite) is the committed-data
backstop: every problem has a stan + reference realization; every statement has a
non-empty given/model/query; and no markdown (`##`, `**`, list markers) or Stan
jargon (`vector[`, `int<lower`, `real<lower`, `transformed parameters`,
`parameters {`, `generated quantities`, `simplex`, `positive_ordered`) leaks into
the statement text. It runs on every commit, so a leaked statement fails CI, not
review. `test_exclusions_documented_and_absent` similarly enforces that every
retired/excluded problem carries a reason and is absent from the live corpus.

Report writers merge by key (`eval/io.py:merge_jsonl`) so partial re-runs never
clobber other rows. Triage history: `data/problems/_gate_triage.md`.

**The gate is the control, not a pre-decided policy.** Each bad entry is handled
on gate evidence, per problem: coherence caught the dogs `n_avoid` leak; crosscheck
caught the Survey `-inf`; ovarian's non-terminating crosscheck + 3072 nuisance
dims got it retired. No global rule fires — the gate surfaces the problem and the
author curates or prunes with documented reason.

## 5. Triage — diagnosing a failing run

When a realization or candidate does not pass, the failure is classified by its
**observable signature** and resolved **at its cause**. A loss is never absorbed
as an unexplained "few percent" — that compounds across stages into garbage.

**First, split by side.** Is the *ground truth* failing (a dataset defect that
blocks the problem for that language), or is a *candidate* failing (a genuine
measurement of a model)? The heuristic is decisive: a problem that is
`exec_error` for a **single weak model** is almost always a real candidate
failure — leave it. A problem that is `exec_error` for **every model** is GT-side:
the harness could not produce the reference the candidates are judged against.
*Never read a pass rate off a GT-side failure* — measuring against broken
machinery manufactures a confident wrong number. Diagnose the GT first.

The scorer stamps every failure with a stable `error_tag`
(`eval/error_tags.py`: timeout / compile / no_output / runtime / gt_side /
empty_code / corpus_miss / other). The tag routes the diagnosis:

| Signature | Diagnosis | Resolution |
|---|---|---|
| all-model `gt_side` + `Timeout` | **Budget starvation** — the GT is correct but slow (heavy MCMC, large N). *The harness's most recurring disease; check it first.* | Scale the per-run budget, **symmetrically for GT and candidate** (fairness invariant: a candidate is never given less budget than the GT that judged it): `PYRO_SEED_BUDGET_SCALE`, `GEN_SEED_BUDGET_SCALE`, Stan regime/N scaling. Never shrink or simplify the GT to beat the clock. |
| all-model `gt_side` + canonicalizer error ("label … out of space" / shape mismatch) | **Shape/protocol bug** in the realization — e.g. a draws-protocol realization returning the *whole cloud* in one run instead of one draw per seed (the harness aggregates across seeds). | Fix the realization to the protocol (`data/SCHEMA.md`), matching the reference column's shape. |
| all-model `gt_side` + `compile`/`runtime` | The GT realization itself is broken. | Repair against the reference; re-run `crosscheck` (§4). |
| `empty_code` | **Truncation, not a wrong answer** — the model exhausted its token budget before emitting a fenced block (common on small reasoners). | `benchmark regen-empty` re-generates at 2× budget; a truncation that persists at a large budget is a genuine non-answer. |
| `malformed` | The candidate emitted an out-of-space label or an ill-formed wire value; `algebra.canonicalize` rejects it. Scoring must never crash on adversarial output. | Leave — a real (failing) measurement. |
| `ill_posed` (phaseA / crosscheck) | Independent GT seeds scatter beyond the discriminability cap. Diagnose **by side** (`eval/posteriordb_diag.py`): STAN-ILL (our floor high, reference floor tiny) vs REF-ILL (the reference marginal itself trips the cap). | STAN-ILL → bake the reference's heavier regime into the bundle. REF-ILL → prune. (`data/REALIZATIONS.md` §7a.) |
| gate `gt_suspect` / `underdetermined` / `solver_failure` (Route A `judge`) | The solver batch disagrees with the GT in a way that implicates the **problem**, not the solver. | Per problem: curate the statement/spec, or retire with evidence (§6). No global rule fires — the gate surfaces, the author decides. |

**A masquerade class: environment, not data.** Some all-model `exec_error` bursts
are neither budget nor shape but the *execution environment* on a fresh box: the
subprocess cannot find its interpreter (`PPL_GYM_PYRO_PYTHON` unset where the venv
is not `./.venv`; `PPL_GYM_JULIA` unset for Gen), or `cmdstanpy` leaked the process
CWD so a relative path resolved wrong (guarded by `executor_stan._cwd_guard`).
Rule the environment out before touching a realization — the tell is the *same*
error message repeating verbatim across unrelated problems.

**Instrument before concluding.** A containment/roundtrip/sanity check that should
be ~100% by construction, returning less, is a broken *instrument*, not "coverage
to improve": fix the harness, re-measure, then read the number. Triage output is
logged, never discarded — `data/problems/_gate_triage.md` (per-problem history),
`_retired.jsonl` / `_posteriordb_excluded.jsonl` (prunes, with reason), and
fresh-run failure tags carried into the rollout export as `gt_unscorable`.

## 6. Retirement

A problem that cannot be made a good benchmark item is retired with evidence, not
silently dropped: `data/problems/_retired.jsonl` (or
`_posteriordb_excluded.jsonl`), a row stating the reason. Precedents: arma11
(non-discriminable), ovarian (horseshoe, non-terminating crosscheck, all-nuisance
query). "Don't silently drop data" — a pruned item leaves a logged reason.

## 7. Publication

After a phase lands, one refresh sweep (not per-change dribbles):
`eval.gate answers` → `eval.export_rollouts --web-out` (+ `--answers-only` capture
on the box) → HF upload (`Sheppp/ppl-gym-rollouts`) → slides. `data/web_rollouts.jsonl`
is committed (the Cloudflare build only sees the repo tree). Benchmark rerun over
new problems is a separate, deliberate step (new problems change the denominator).

## 8. Worked example — posteriordb 65 → 75 (Route C, 2026-07)

The remainder run, start to finish, as a template:

1. Inventory: 101 no-gold posteriors; filtered to feasible candidates (small N,
   clean geometry), deduped by posterior (radon centered/noncentered twins → one
   each; `seeds_model` vs `seeds_stanified` distinct priors → both).
2. refgen on 14 candidates (5 parallel workers on the box, 10-chain NUTS each).
   **Gates auto-rejected 4**: pilots (R-hat 1.033), ldaK2 (R-hat 2.63, LDA
   label-switch), prostate (10/10 timeout, regularized horseshoe), uk_drivers
   (8/10 timeout, state-space). **10 clean passers** (R-hat ≈ 1.001, ESS 8-9k).
3. Ingested the 10 (`build --ids`): corpus 65 → 75.
4. Curated: the 5 non-centered radon overlays exposed `alpha_raw`/`beta_raw` in
   the answer_spec (correctness bug — would fail centered candidates). Filtered
   `*_raw` from each overlay (e.g. 345 → 175 cols), rebuilt → answer_spec now
   queries `alpha`/`beta`/`mu_*`/`sigma_*` only.
5. Authored 10 statements from the models, verified line-by-line vs priors + data.
6. Gate: crosscheck Stan-vs-reference, coherence, answers.
7. Publish: `_gt_answers.jsonl`, web_rollouts, HF (deliberate benchmark rerun
   separate).
