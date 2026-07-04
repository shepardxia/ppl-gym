# Research plan & log

Dataset-creation project. Instrument correctness → dataset scale → column completeness.
No prompt tuning (deferred; this is a dataset project, not a model-improvement project).

## Plan

### Phase 0 — instrument wrap-up (in flight)
- [x] Pyro GT collection fixed (chunked parallel seeds; GT budget ×4k; candidate budget flat).
- [x] Stan fit timeout regime-aware (gold-reproduction regimes got starved at N-only scaling).
- [x] Full pyro rescore + stan low_dim_gauss_mix merge; 22 of 23 GT-broken problems recovered.
- [ ] ex2.3 patch-rescore (running on box) → rebuild corrected_pass_rates.
- [ ] Downstream refresh: slides, `gate answers` for the 23 fixed (pyro+stan), web_rollouts
      answer capture (box), HF re-export, CLAUDE.md known-limit paragraph. Push needs authorization.
### Phase 0.5 — harness speed + simplicity refactor
Make the harness fast and simple BEFORE scaling the dataset on top of it (every
inefficiency multiplies by problem count; every complexity multiplies by contributor count).

Speed (evidence-ranked from the 2026-07-04 efficiency assessment):
1. **Cap torch/OMP threads in pyro subprocesses** — measured ~9x degradation from
   oversubscription (64 threads/proc × parallel chunks). First prove draws bit-identical
   on a sampled problem set (tiny tensors likely below torch parallel grain size); if not
   identical, EXECUTOR_VERSION bump + full pyro re-collect + re-crosscheck.
2. **Box-profile worker defaults** — clamps sized for the laptop idle 128 cores; make
   worker topology explicit (env or flag profile: laptop vs box) instead of magic numbers
   spread across score/gate/benchmark.
3. **WebPPL batch driver** (optional, bounded win): 0.7s spawn+compile per seed; only
   stings on the 5 draws-spec problems (600 spawns/GT). Do only if draws corpus grows.
4. **Persistent pyro worker** (skip unless profiling says otherwise): saves ~2-5s torch
   import per candidate row; real engineering, modest win.

Simplicity (single-source-of-truth pass over the eval half):
- Timeout policy is now scattered (flat, ×k, ×4k GT-only, stan N-scale, stan regime-scale,
  stan repack DEFAULT_SAMPLING, per-fit cap 10×): centralize into one documented
  budget-policy module so the next language doesn't add a sixth convention.
- `_one_fit` (stan) swallows all exceptions → `None` → generic "execution failed" — the
  exact error-collapse disease triage was built to undo. Propagate real reasons at the
  source; shrink triage_exec_errors accordingly.
- Executor interface audit: webppl/pyro/stan batch signatures agree but semantics differ
  (workers = threads vs chunks vs fit-parallelism); document or unify.
- Kill vestigial paths found along the way (post-pivot dead remnants rule).
Gate: tests stay green; no behavior change without a measured reason; cache keys stable
unless a version bump is the explicit, paid-for choice.

### Phase 1 — dataset scale, existing languages (the priority)
Grow problem count where the harness already executes: webppl, pyro, stan.
Order of attack = headroom × authoring cost:

1. **posteriordb remainder** (stan, cheapest): 147 posteriors in source, 46 with gold
   reference draws, 45 ingested. First: inventory why ~101 lack gold draws; ingest any
   passable without gold by self-generating reference draws (long-chain NUTS, multi-seed
   crosscheck under our existing gate) — this converts "gold-draws corpus" into
   "validated-draws corpus" and roughly 3× the stan corpus if quality holds.
2. **problang** (webppl, source already vendored, 0 mined): 32 chapters of RSA-style
   models. Same extraction pipeline probmods2 used; per `_AUTHORING_BRIEF.md`.
3. **forestdb remainder** (webppl): 29 of ~200+ models ingested. Inventory the rest;
   many are one-liners or duplicates — expect selective yield, not 200.
4. **dippl remainder** (webppl): 16 in; check unmined chapters.
5. **Stan example-models / BUGS volumes** (stan, biggest pool, highest authoring cost):
   hundreds of models with data but no gold draws; same self-generated-reference path as
   (1). Only after (1) validates that path.

Every new problem passes the existing gate (phaseA floors → solve → judge → crosscheck
where a second column exists). No gate shortcuts at scale — the gate IS the product claim.

### Phase 2 — plug the 3 missing language columns
- **stan column for textbook 115**: many textbook problems are discrete-latent /
  enumeration-flavored; Stan needs hand marginalization or is genuinely unavailable.
  Expect a substantial documented-unavailable fraction (like pyro's 4). Availability
  audit first, then translate the feasible set (author briefs per REALIZATIONS.md).
- **webppl + pyro columns for posteriordb 45**: continuous hierarchical models; pyro NUTS
  is a natural fit, webppl HMC is weak — budget/availability audit first. Crosscheck
  against the reference column is free once realized.
- Both directions inherit the translation lessons doc (`data/REALIZATIONS.md`) and the
  idiomaticity audit (agents hand back plain-Python/plain-JS lookalikes).

### Phase 3 — new corpora / mainstream-language scaling (with language creators)
- memo/pluck with their creators (planned in CLAUDE.md).
- "Mainstream" candidates to source: PyMC (huge example gallery, Python-native), NumPyro.
  Each new language = executor + serializer + primer + gate columns; cost is known
  (Stan took ~a week end-to-end). Decide per-language after Phase 1 data exists.

### Cross-cutting
- **Gate economics**: authoring-time solves/judges use LLM batches — scope --ids, budget
  per corpus before launching (cost discipline rule).
- **Publication cadence**: after each phase lands, one refresh sweep (gate answers →
  web_rollouts → HF → slides) rather than per-change dribbles.
- **Benchmark rerun**: one full matrix after Phase 1 (new problems change the denominator);
  none before.

## Log

### 2026-07-03
- Diagnosed the 22 pyro GT-broken problems: NOT uniformly dead — GT succeeds when 5 sequential
  seeds happen to fit 300s (then disk cache carries later runs), fails otherwise. Error mix per
  problem: "GT collection failed: timeout after 300s" (whole-batch kill) + candidate-side
  "execution failed". Root cause: per-seed budget coupled to sequential batch.
- `low_dim_gauss_mix`: all 21 rows "GT collection failed: execution failed" (not timeout) —
  a seed-level failure in the stan GT realization, deterministic.
- Fix shipped: `execute_pyro_batch` chunks seeds into ≤workers subprocesses; `timeout` bounds
  each chunk (harness multiplies ×4k for pyro exact-GT, GT-only; candidates stay flat 60s).
  Chunked == single-batch outputs verified identical (per-seed reseed); no EXECUTOR_VERSION
  bump, existing cache valid. 88 tests pass. All 22 problems exact-spec.
- Warm run on box: 21/22 pyro OK (hierarchical ex2.4 181s wall — sequential form couldn't fit
  5 such seeds in 300s). observing-sequences ex1.b needed 600s budget (295.8s solo).
- `low_dim_gauss_mix` root cause (probe-confirmed): fits normally ~13s, but one seed had a
  stuck mixture chain SIGTERM'd at the 60s fit timeout (`code '-15'`) → `None` → generic
  "execution failed". Budget came from N-only scaling (`ceil(N/1000)`); gold regime is
  8 chains × 9000 iters (~9× default). Fixed: scale = max(N-factor, regime-factor) → 540s;
  touches only 2 gold-regime problems. Collected OK (518.8s).
  Follow-up: `_one_fit` swallows all exceptions → `None` (error-collapse disease).
- ex2.3 probe verdict: NOT stuck — 30 uniform NUTS runs at 3.6-5.9s, whole seed 133s solo.
  Failures at 600/1200s were oversubscription: 5 parallel chunk subprocesses × 64 default
  torch/OMP threads on 128 cores (spin-wait on tiny-tensor ops). Sequential warm (workers=1)
  succeeded: 630.2s.
- Box observation: torch defaults 64 threads/proc. Cap candidate: `torch.set_num_threads(4)`
  — needs bit-identity proof on draws (else version bump + full re-collect). Deferred.
- Rescore done. 21/22 pyro problems score; low_dim_gauss_mix: haiku/qwen3-235b/sonnet 3/3
  pass, rest real candidate compile failures. Raw pyro rates post-fix: gpt-oss-120b 0.577,
  sonnet 0.454, haiku 0.306 (match old corrected projections). corrected_pass_rates rebuilt.

### 2026-07-04
- ex2.3 warmed sequentially (630s); its 21 rows patch-rescored + merged. GT now works;
  candidates all fail — several at "timeout after 60s".
- **Candidate-budget starvation found** (Phase 0.5 pulled forward): candidates got flat 60s
  on problems whose GT needs 84-300s/seed. 86/462 rows on the fixed 22 + 281 rows matrix-wide
  are candidate timeouts (sonnet worst: 56 problems — faithful-but-slow MCMC). Fairness
  invariant adopted: a candidate never gets less budget than the GT that judges it was
  validated under. Policy centralized in eval/config (PYRO_SEED_BUDGET_SCALE=10 → 600s/seed
  GT and candidate alike; PYRO_CHUNK_BUDGET_CAP=3600 so a hung draws chunk can't hold a
  worker for hours); harness passes per-run timeout, executor applies policy. Tests green;
  chunked-vs-single determinism re-verified.
- Stan `_one_fit` error-collapse fixed: failures carry the real reason (timeout budget,
  missing param, last exception line); all-seeds-failed raises it (mirrors pyro contract).
  Partial-failure GT batches still say generic "execution failed" — minor, noted.
- Downstream refresh deliberately deferred until timeout rescore lands (publish once).
- Thread-cap probe: 12/12 bit-identical (64 vs 4 threads, heavy NUTS included — chaotic,
  FP drift would show). Capped solo runs also 15-25% faster on several problems. Cap
  applied in executor (`_subprocess_env`: OMP/MKL/OPENBLAS=4) — cache stays valid, no
  version bump. Tests green.
- Timeout rescore launched on box: 281 rows / ~90 problem×model pairs (3 models at a
  time × 3 problem-workers; candidates may hold up to 600s each). First result:
  gpt-oss-120b pyro 0.577→0.598 — the 60s cap was suppressing real passes.
- Phase 0.5 simplicity landed while box runs: budget policy + worker budget centralized
  in eval/config (`total_exec_workers()`, `PPL_GYM_EXEC_WORKERS` env for the box);
  executor batch contract documented at `corpus.batch_executor_for`; harness partial-GT
  failures now say "n/k seeded runs failed"; triage matches both legacy + new generic
  forms; 2 new contract tests (chunk budget policy, env worker budget) → 90 pass.
  CLAUDE.md defaults + harness-limit paragraphs rewritten (budget starvation = the
  recurring disease, three fixed instances documented).
- Phase 1 prep (posteriordb expansion inventory, local): 101 no-gold candidates; 47 small
  (≤60 model lines, N≤2000) incl. classics — eight_schools_centered, Rate_1-5, dogs×3,
  radon×12, seeds×3, GLM/GLMM, irt_2pl, rats, surgical, dugongs. Monsters excluded
  (covid19imperial, mnist rbm). Full table: notes/posteriordb_expansion_inventory.json.
  Pilot design: self-generate reference draws (10 chains × 1000 kept post-thin, R-hat ≤ 1.01
  + ESS ≥ 2000 gates, posteriordb-protocol-shaped) for ~6 classics, ingest via existing
  posteriordb.py path, crosscheck stan-vs-reference must pass — validates the
  "validated-draws corpus" route before scaling to all 47.
- **Shipped the machinery**: `eval/reference_gen.py` (long-NUTS + convergence gates →
  overlay at data/reference_draws/, never writes the vendored tree, refuses gold names);
  posteriordb.py resolves gold-first-overlay-second (`validated_posterior_names()`;
  build/material CLIs now use it). 2 overlay contract tests → 92 pass. Pilot launched on
  box alongside the rescore (Rate_1, dugongs, GLM_Poisson, radon_pooled, surgical,
  eight_schools_centered as gate stress test).
- **Selection rule (dedup)**: one problem per mathematical posterior. eight_schools_centered
  = same posterior as the already-ingested noncentered → identical statement → duplicate,
  ingestion skipped (still a gate stress test). Audited: radon 12→6 (centered twins drop);
  seeds_model distinct (wide priors) but centered≈stanified (kept stanified); dogs vs
  dogs_log distinct (different priors + links). mnist RBM excluded (not a posterior problem).
- Pilot progressing: 5/5 refgen OK → ingested (build merges: 50 problems) → statements
  agent-authored (sonnet), verified line-by-line vs Stan models (Rate_1 n/k and
  GLM_Poisson year-standardization checked against actual data), applied; coherence test
  enforces statements-before-merge (caught my staging leak) — 92 tests pass. Crosscheck
  stan-vs-reference running on box.
- **Scale-up**: refgen batch over 34 dedup'd candidates. Gates auto-rejected bad geometry
  as designed: GLMM_Poisson (R-hat 1.017), low_dim_gauss_mix_collapse + normal_2 + normal_mixture
  (mixture label-switch, R-hat 1.02-1.56), normal_5 (timed out). 16 clean passers ingested
  (corpus 50→66). Statements agent-authored, verified line-by-line vs Stan models (priors +
  data claims — Rate n/k, capture-recapture M/T, GLM year-standardization all checked).
- **Coherence gate caught 2 real leaks**: dogs/dogs + dogs_log queries said "transformed
  parameters" (Stan jargon) because their specs carried n_avoid/n_shock — quantities
  deterministic from data alone (identical every draw, zero inference). Root: reference_gen
  stored ALL cmdstanpy columns. Fix (targeted, not global): filtered those two overlays to
  the params-block var (beta), rebuilt, rewrote statements to query beta only. 92 tests pass.
  [Detour: I badly over-escalated this trivial column-filter into a fake "corpus-wide
  curation fork" + a blocking AskUserQuestion. User called it out. It was one bad default in
  one new module; the coherence gate flags any genuinely-degenerate entry — that IS the
  control. No global policy decision needed. Lesson logged to memory.]
- Batch crosscheck: **14/16 pass comfortably** (d = 1-30% of tol; incl. both curated dogs +
  lsat 1000-dim). Two problems handled:
  - Survey: errored (`-inf` in stored lp_parts — same degenerate-column issue as dogs).
    Curated to theta-only (valid return-rate problem; n was an RNG draw), statement rewritten,
    re-crosschecking.
  - **ovarian RETIRED** (`_posteriordb_excluded.jsonl`): horseshoe d=1536, queried set is
    3072 non-centered nuisance params (z + lambda shrinkage machinery), not the meaningful
    beta (a transformed param). Crosscheck non-terminating >1h, numerically unstable (-nan).
    Poor benchmark item + harness-hostile → pruned with evidence (arma11 precedent, §7a).
    Deferred for possible manual curation (query beta only).
  Net: 15 scale-up problems (corpus 50→65). Re-crosscheck of the 15 (cached GT, Survey fresh)
  running to persist the report.
- Lesson: the gate is the arbiter, exactly as intended — it caught the dogs leaks (coherence),
  the Survey -inf (crosscheck error), and ovarian's non-termination. No pre-decided policy;
  each bad entry handled per-problem on gate evidence.
- **Re-crosscheck of 15: all pass** (Survey now theta-only, d=0.005/tol=0.030). Corpus 45→65
  posteriordb (20 net new: 5 pilot + 15 batch; ovarian retired). Reference GT answers written
  for all 20 (_gt_answers.jsonl, 340 rows). Note: the 20 new problems are dataset entries
  (statement + stan + reference + crosscheck-verified + GT answers) but NOT in the benchmark
  rollouts — no model was ever run on them; benchmarking them is a future run.

### Rescore results (Phase 0 close-out)
- **Timeout-budget fix recovered heavily.** On the 281 formerly-"timeout after 60s" pyro rows:
  sonnet 168 rows → 84 pass / 42 fail (real) / 39 exec_error (real) — half now pass. Pyro rates
  progression (sonnet): 0.405 raw pre-fix → 0.454 (GT-broken fixed) → **0.627** (budget fixed).
  gpt-oss-120b 0.577→0.603. Only occams ex2.3 remains GT-broken (1 problem).
- corrected_pass_rates.json rebuilt (raw ≈ corrected now); slides/results.png regenerated.
- Downstream still open (publish-territory, needs authorization): web_rollouts.jsonl regen +
  answer capture (box), HF re-export/upload. Not done unprompted.

### Phase 1 close-out state (2026-07-04 late)
- All 15 scale-up crosscheck verdicts confirmed PASS (stdout): M0 .16/.84, Rate_2-5 tight,
  Mt .17/.67, dogs .007/.043, bones .04/.20, dogs_log .002/.007, Survey .005/.030,
  GLM_Binomial .006/.019, nes_logit .007/.033, irt_2pl .075/.317, dogs_hierarchical .036/.144,
  lsat .069/.266. Report-file persistence (_gate_crosscheck_report.jsonl) was slow to write on
  the re-run (lsat 1000-param re-fit, cache miss) — verdicts are the ground truth; report is
  cosmetic web-badge input, pulled when the box finishes.
- DONE this session: pyro GT fix, candidate-budget fix, stan regime fix, thread cap, harness
  policy centralization (Phase 0 + 0.5); corpus 45→65 posteriordb (Phase 1 pilot + scale-up);
  corrected_pass_rates + slides regenerated; reference GT answers for the 20 new.
- Box left running (idle after crosscheck). vast 76.121.3.151:27538.

### Benchmark the 20 new problems (authorized 2026-07-04)
- Running on box: eval.benchmark run over the 20 new posteriordb problems × 7 models × stan,
  3 samples (runs/new20). Keys passed via process env inline (not persisted). Anthropic
  batches (sonnet/haiku) async; Together 5 models concurrent. Then: export_rollouts --web-out
  (picks up new rollouts) + answer capture, HF re-export. Git push to main = confirm separately
  (standing rule, production deploy).
- Correction logged: I had scoped "downstream refresh" as rebuilding web_rollouts from the OLD
  matrix — which never ran the new problems. The actual step is benchmarking the 20 new (this).
- **20-new benchmark DONE** (stan, 7 models × 20 × 3 = 420 gens). Pass rates on the 20 new
  problems (harder than the original 45): sonnet 0.767, haiku 0.617, gpt-oss-120b 0.600,
  qwen3-235b 0.500, gpt-oss-20b 0.450, qwen3.5-9b 0.283, llama-3.3-70b 0.167. exec_error =
  real stan compile failures (llama 48/60, gpt-oss-20b 31/60) — these Bayesian models stress
  weaker coders. Merged new20 stan rows into matrix stan combos (135→195 each, union by
  problem_id+slot). web_rollouts rebuilt 1120→1260 (+140). Answer recapture (webppl+stan) on
  box in progress. Then HF re-export. Push to main = ask user.
- Bug hit + fixed mid-run: first merge_export used a nested heredoc; the inner python
  `\"pass_rate\"` broke at compile (SyntaxError → nothing ran, matrix intact). Killed the
  wrong-set answer capture it had started, redid merge as a proper .py file.
- web_rollouts rebuilt (1260) + answers recaptured (webppl+stan, 770 re-exec → 624 captured).
  Size ballooned 6→26MB: high-dim record posteriors. Two fixes:
  (1) **dogs_hierarchical was uncurated** (752 fields = a,b + 750 y_rep predictive-RNG junk —
      missed it earlier, coherence gate didn't catch "y_rep"/generated-quantities). Curated to
      a,b (overlay+realization+statement), re-crosscheck pass d=0.0007, answers recaptured,
      reference GT answer regenerated. Same class as the dogs/dogs_log n_avoid fix.
  (2) legit high-dim posteriors (lsat 1012 params, irt_2pl 144) → added `max_fields=40` cap to
      trim_answer (web display only; HF + scored keep full fidelity). Re-trimmed: 26→8.6MB.
- HF dataset dir rebuilt on box: rollouts 5691→6111 (+420), problems 180, gt_answers 340
  (full-fidelity, 15.6MB gt_answers.jsonl untrimmed — correct for HF). NOT uploaded:
  HF_TOKEN unset on box+laptop; won't reuse a prior-session token. Awaiting token.
- HELD for user: HF upload (needs token), git push to main (production deploy; web_rollouts.jsonl
  + eval code + corpus are the committed delta).
- **PILOT VALIDATED: crosscheck 5/5 pass** (distances 3-30% of tolerance — comfortable,
  not marginal: Rate_1 d=.004/tol=.016, radon d=.002/tol=.009, surgical d=.016/tol=.048,
  GLM_Poisson d=.36/tol=1.45, dugongs d=.97/tol=3.37). The validated-draws route is
  sound; corpus 45→50; scaling decision now rests on the 34-candidate batch results.
- Harness efficiency assessment (user asked): architecture sound (content-addressed
  model-independent GT cache = main win). Real waste ranked: (1) torch/OMP oversubscription
  ~9x on parallel pyro (measured via ex2.3); (2) laptop-sized worker clamps idle the box;
  (3) webppl 0.7s/process spawn overhead (measured; only stings on 5 draws-spec problems);
  (4) ~2-5s torch import per pyro candidate row. Stan NUTS + unique-candidate compiles =
  inherent long pole, not waste.
- Expansion inventory: posteriordb source has 147 posteriors / 46 with gold draws (45 in);
  problang vendored but 0 mined (32 chapters); forestdb 29 in of ~200+ upstream; corpus
  today = 160 problems (70 probmods2 + 16 dippl + 29 forestdb + 45 posteriordb).
