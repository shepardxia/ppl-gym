# REALIZATIONS — authoring verified ground truth across languages

This is the working knowledge for the realization layer: what a good realization
is, how to translate one language's column into another, how to add a new
language, and the traps we have already paid for. **Keep it current.** Every time
a language column teaches us something — a new idiom, a precision footgun, a
prompt that backfired — add it here (see §8) so we do not relearn it.

Companion docs: [`SCHEMA.md`](SCHEMA.md) is the contract (record formats, answer
algebra, the gate). [`REDESIGN.md`](REDESIGN.md) is the rationale and phase
history. [`problems/_AUTHORING_BRIEF.md`](problems/_AUTHORING_BRIEF.md) is for
authoring *problems* (statements/specs); this doc is for authoring
*realizations* (per-language ground truth).

A realization is `{problem_id, language, code}` — or, where a problem can't be
realized in a language, `{problem_id, language, available: false, reason}` (no
`code`; see Per-language availability) — in `data/realizations/<language>.jsonl`.
WebPPL is the reference column; Pyro is now **idiomatic**: 111/115 realizations
crosscheck-verified against WebPPL GT *and* proper-usage audited, 4 marked
unavailable. Stan / memo / pluck are planned.

---

## 1. What a realization must be (the bar)

**A realization expresses the problem's model through the target language's own
modeling and inference machinery, and obtains the answer by running that
inference.** Not by computing the answer some other way and dressing it up.

This is the single rule that everything else serves, and the one that is
violated over and over. The violation has a name: **handrolling** — enumerating
outcomes and multiplying/normalizing probabilities in plain Python (or JS),
precomputing a probability table and feeding it to a `Categorical`/`factor` as a
veneer, or hardcoding the known answer. The numbers pass the gate, so handrolled
realizations look fine and are worthless: they teach the model nothing about the
language and they are not what we are claiming to ship.

**Litmus test:** if deleting the inference call would leave the answer already
computed, it is handrolled. The model has to do the work.

**Why it recurs (and why a mechanical rule will not save you).** The dead
`pyro_v3` column was almost entirely handrolled. The P4 rebuild added a
*mechanical regex rule* to catch it — and the rule rewarded a thin `pyro.factor`
veneer over a precomputed table, so ~half the "fixed" column was still
handrolled, just past the regex. Agents will hand back plain-Python lookalikes
whose numbers pass whenever the idiomatic path is harder than the shortcut.
Audit the *mechanism*, not a surface pattern, and verify idiomaticity with
judgment (human or a reviewing model), not a grep.

What is **not** handrolling:
- Reading a posterior **marginal** or samples back out of a real inference run,
  then post-processing (normalize, take an expectation).
- Constructing a distribution that is *itself* the queried quantity — a prior, a
  forward/predictive distribution, or a deterministic value.

What **is** handrolling despite looking principled: a hand-written **conjugate
posterior** (`dist.Beta(a + heads, b + tails)`, a normalized Dirichlet
posterior-predictive). The math is exact, but *you* did the Bayesian update, not
Pyro — a posterior must come from running inference over a `pyro.sample` model.
(The idiomaticity audit caught several of these passing numerically; this is a
narrower rule than "closed form is fine," and the narrower one is correct.)

---

## 2. The idiomatic inference toolkit (by paradigm)

A realization picks the inference method that reproduces the reference within the
measured tolerance. The choice is per problem, not per language. **Pyro's toolkit
is far wider than the headline four — `RandomWalkKernel`, `SMCFilter`,
`Predictive`/`WeighedPredictive`, `infer_discrete`, `SVI`, Stein methods, and more
all ship in 1.9.x. Do not default to a narrow subset; pick the tool that fits.**
### Decision map: model shape → Pyro idiom (all verified this rebuild)

| The model (read from the WebPPL GT + query) | Use |
|---|---|
| Discrete latents; query is **one** variable's marginal; exact (or rare/extreme probs) | `@config_enumerate` + `TraceEnum_ELBO(max_plate_nesting=P).compute_marginals(model, lambda: None)["site"]`. *Required* (not optional) for extreme probs — Importance never draws a 1e-9 event. Needs float64. |
| Discrete latents; query is a **joint** over several (tuple-valued support) | `@config_enumerate` + `pyro.infer.infer_discrete(model, first_available_dim=-1)`; histogram the sampled tuples. `compute_marginals` gives only per-site marginals. |
| **Nested** discrete inference (RSA: L0 → speaker → L1) | one real inference per level, **unique site names across levels**, memoize each level, feed its finished distribution forward. Pyro can't run inference inside an active enumeration ("Multiple sample sites named …"). |
| Continuous latents, smooth finite density | `MCMC(NUTS(model), num_samples ~500–2000, warmup ~200–1000)` → `get_samples()["site"]`. |
| Discrete conditioning, no extreme probs | `Importance(model, num_samples ~1k–10k).run()` → `EmpiricalMarginal`. |
| **Hard** constraint / non-differentiable term on continuous latents | `Importance` + a hard-band `pyro.factor` (`0`/`-inf`). NUTS/HMC *and* `RandomWalkKernel` all fail to initialize on a measure-zero/thin set; raise the sample count for a thin band. If even that can't certify at practical cost, it may be Pyro-unavailable (see below). |
| **Mixed** discrete + continuous latents | `Importance` over the prior (the discrete latents are concrete samples), or `@config_enumerate` + NUTS. **Not** `MixtureSameFamily` as a NUTS latent — its constraint isn't transformable (`NotImplementedError`). |
| Conjugate / predictive query | run inference (sample + observe + infer). Construct a closed-form distribution **only** if the queried quantity is itself a prior/forward/deterministic — a *posterior* must be inferred (§1). |
| Sequential / particle filter | `SMCFilter` — the Pyro analog of WebPPL SMC. |

Two framing facts behind the map: Pyro's toolkit is far wider than NUTS/Importance/
enumerate (`infer_discrete`, `RandomWalkKernel`, `SMCFilter`, `Predictive`, `SVI`,
Stein, … all ship in 1.9.x), and a WebPPL `Infer` method does **not** map 1:1 to a
Pyro tool — `Infer` hides choices Pyro forces you to make (a WebPPL MH on a hard
condition is not a Pyro NUTS).

### The verified Pyro recipes

Exact discrete enumeration → marginal posterior (verified against WebPPL GT):

```python
@pyro.infer.config_enumerate
def model():
    x = pyro.sample("x", dist.Bernoulli(0.2))
    # ... derive the conditioning event as a (boolean) tensor ...
    pyro.factor("ev", torch.where(event, torch.tensor(0.0), torch.tensor(float("-inf"))))
    return x

marg = pyro.infer.TraceEnum_ELBO(max_plate_nesting=P).compute_marginals(model, lambda: None)
# marg["x"] is the exact marginal Distribution over sample site "x".
# P = number of nested pyro.plate's (0 if none).
```

`compute_marginals` gives marginals over **sample sites**, not over the model's
return value or a derived joint. Structure the model so the queried quantity is a
sample site, or use a method whose output is the return value (Importance +
`EmpiricalMarginal`).

Importance: `pyro.infer.Importance(model, num_samples=N).run()` →
`pyro.infer.EmpiricalMarginal(post)`. MCMC:
`pyro.infer.MCMC(pyro.infer.NUTS(model), num_samples=N, warmup_steps=W)` →
`mcmc.get_samples()["site"]`. Closed form: construct `dist.<Family>(...)`.

### Precision footgun: float64 is mandatory for GT

Torch defaults to **float32** (~7 significant digits). Exact enumeration over a
model with extreme probabilities silently corrupts: a conditioned posterior that
hinged on a 6e-5 vs 1e-9 competition came back **0.984** under float32 versus the
true **0.9999**. The executor preamble now sets
`torch.set_default_dtype(torch.float64)` for every realization. Any new
language's executor should run GT at its highest practical precision for the same
reason.

---

## 3. Translating from the reference column

New columns are authored by **translating the verified WebPPL realization**, not
by re-deriving from the English statement alone. The WebPPL GT is the clean,
verified specification of the exact model: priors, parameters, conditioning, and
query. The statement says *what* the answer is; the WebPPL realization pins the
*exact* model.

**What the authoring agent gets:** statement + answer_spec + WebPPL GT.
**What it must NOT get:**
- The **numeric answer** — giving it invites overfitting and hardcoding (a
  handrolled realization curve-fit to the target).
- The **existing handrolled realization** in the target language — it contaminates
  (the agent copies the precomputed tables it sees).

### WebPPL → Pyro mapping (the recurring cases)

| WebPPL | Pyro |
|---|---|
| `flip(p)` | `pyro.sample("n", dist.Bernoulli(p))` |
| `gaussian(mu, s)`, `beta(a,b)`, … | `pyro.sample("n", dist.Normal(mu, s))`, … |
| `condition(bool)` | observe an indicator (`obs=`) or `pyro.factor` with `0/-inf` |
| `observe(D, v)` | `pyro.sample("o", D, obs=v)` |
| `factor(s)` | `pyro.factor("n", s)` |
| `Infer({method:'enumerate'}, m)` | `config_enumerate` + `TraceEnum_ELBO.compute_marginals` |
| `Infer({method:'MCMC', ...}, m)` | `MCMC(NUTS(m), ...)` |
| `Infer({method:'rejection'/'forward'}, m)` | `Importance(m, num_samples=N)` |
| `mem(f)` | memoize with `functools.lru_cache` over the *latent draw*, or restructure so the shared choice is one sample site |

---

## 4. The authoring workflow

Author a whole column in stages; the slow/heavy work is centralized and the
parallel work spawns nothing heavy. **Translate every problem — do not pre-filter
for "realizability" (a statement scan over-flags; see Per-language availability).
Unavailability is decided later, from evidence.**

1. **Calibrate the hard idioms first.** Before fanning out, prove the trickiest
   recipes (exact enumeration → marginal; joint → `infer_discrete`; hard-constraint
   handling) reproduce the reference in this repo's actual executor, and put the
   *verified* recipes in the author brief. Never fan out on an unproven idiom — you
   get a uniform column of plausible-but-wrong code. Front-load this: survey the
   source corpus's inference methods and enumerate the target's hard idiom-classes
   *before* authoring, not reactively one failure-round at a time.
2. **Pure rewrite (parallel).** Agents translate *all* problems; they do **not**
   execute, test, or self-verify; import-free; one chunk each. This spawns no
   subprocesses, so it parallelizes freely (§6).
3. **Centralized crosscheck → repair (loop).** *You* run one controlled, batched,
   cached crosscheck of every candidate against the reference column; re-spawn
   agents only for failures (with the failure evidence + verified recipes); repeat
   until numerically converged. This is the only heavy-compute step and it surfaces
   the genuinely-hard cases.
4. **Proper-usage audit → repair (loop).** Numbers passing is **not** the bar. Audit
   every realization (a reviewing model armed with the full target-library API
   reference) for hand-rolled / reinvented / hand-written-conjugate inference; fix
   the flagged ones and re-audit until clean. (This rebuild went 52 → 6 → 0 flagged.)
5. **Decide availability, from evidence.** A problem that genuinely cannot be
   realized after real attempts with the right tools is marked **unavailable** with
   a documented reason (Per-language availability), in this same review round — never
   pre-filtered.
6. **Merge & stamp.** Merge accepted realizations into the durable column
   (`data/realizations/<L>.jsonl`); regenerate `gate answers --language <L>` and the
   crosscheck report.

**Roles:** subagents implement; the orchestrator holds the contract and verifies
harshly (self-written code biases self-judgment). **Stage everything in the repo,
never `/tmp`** — a cleared `/tmp` cost a whole round once.

---

## 5. Prompts: solver primer vs author brief

These are different artifacts with different audiences; do not let one leak into
the other. All live in `data/prompts/`.

- **Solver prompts** (`<lang>_system_base.txt` + `<lang>_primer.txt`) are
  *eval-facing* — fed to the models we are measuring. They must be neutral and
  self-contained: no negative priming ("do not call X" plants X, worst for small
  models), no cross-language references ("like a WebPPL realization" is
  meaningless under test), no leaked authoring agenda (our handrolling fight, our
  conventions). State what is available and how. If the harness handles something
  (seeding), say nothing about it.
- **Author briefs** (`<lang>_author_brief.md`) are *workflow-facing* — fed to the
  agents writing GT. Here the anti-handrolling bar, the inference toolkit, the
  import-free convention, and the verified recipes belong, stated plainly.

**Import-free convention.** Realizations carry no `import` statements. The
executor injects a standard toolkit preamble (mirrors WebPPL, whose deps are
`--require`'d). The solver primer documents the subset solvers rely on; the
author brief states the realization is import-free. Keep the preamble and the
prompts consistent, and bump the executor version (§6) when the preamble changes.

Treat every prompt as a serious artifact. Re-read it adversarially before it
touches an agent. (See memory: prompt-hygiene.)

---

## 6. Harness infrastructure

Each language needs an **executor**: a subprocess that runs the realization with
a serializer header (emits the wire forms `eval/algebra.py` canonicalizes) and an
import-free preamble (the standard toolkit + precision + seeding). WebPPL:
`eval/executor.py` (deps via `--require`, RNG fixed by a process-start env
override — it cannot reseed in-process). Pyro: `eval/executor_pyro.py` (preamble
imports the toolkit, sets float64, seeds via `pyro.set_rng_seed`).

- **Batch execution.** Per-seed subprocess spawning is the wrong primitive: a
  draws-protocol spec runs `k_draws × n_draws = 600` executions per side, each
  re-importing the runtime (~0.5 s of torch import). `execute_<lang>_batch(code,
  seeds, ...)` runs many seeds in **one** subprocess. Pyro truly batches (one
  torch import; per seed: `set_rng_seed` + `clear_param_store` + fresh-namespace
  `exec`, which reproduces fresh-process-per-seed exactly). WebPPL runs per-seed
  under the same interface (it cannot reseed in-process). Registered in
  `eval/corpus.py:BATCH_EXECUTORS`.
- **Persistent GT cache** (`eval/gt_cache.py`). A GT run is deterministic in
  `(language, executor_version, code, seeds)`, so raw serialized outputs are
  cached on disk under `data/.gt_cache/`, content-addressed by sha256. Transparent
  in `gate.collect_gt_answers`, so every consumer (crosscheck / phaseA / judge /
  answers / score) benefits. A run is cached only if **every** seed succeeded (a
  partial/failed run is retryable). `EXECUTOR_VERSION[<lang>]` busts the cache
  when the serializer/preamble changes — **bump it** on any executor change.
  `PPL_GYM_NO_CACHE=1` bypasses.
- **Concurrency / memory.** The OOM risk is heavy subprocesses, not agents. Each
  heavy executor subprocess (torch/node) is ~190 MB; peak ≈ agents × mc_workers.
  ~64 concurrent killed a 36 GB / 14-core laptop. So: Stage-1 rewrite agents spawn
  **nothing** and parallelize freely; the heavy step (verification) is centralized
  and batched, where you control the process count. Do not hand-tune the two
  concurrency knobs to dodge OOM — fix the primitive (batch + cache), then the
  knobs barely matter.
- **Determinism / seeding** is the harness's job. The executor sets the seed; the
  realization never seeds itself, and the solver prompt never mentions seeding.
- **Verification** is `gate crosscheck --language <L>`: the target column vs the
  reference GT under symmetric measured tolerances. **Draws protocol:** `ANSWER`
  is **one** sampled draw; the harness runs many seeds and aggregates.

---

## 7. Adding a new language (checklist)

To add language `L` (e.g. Stan, memo, pluck):

1. **Executor** `eval/executor_<L>.py`: run `L` code in a subprocess; inject a
   serializer that emits the algebra's wire forms (scalars, enumerated dist as
   `{kind:dist_enum, support, probs}`, parametric as `{kind:dist_param, family,
   params}`, samples as a list); inject an import-free preamble with the standard
   toolkit, highest practical precision, and a seeding hook. Add
   `execute_<L>(code, ...)` and `execute_<L>_batch(code, seeds, ...)`.
2. **Algebra** (`eval/algebra.py`): confirm `L`'s output forms canonicalize; add
   a parameter-name alias entry if `L` names distribution params differently.
3. **Wire it up:** `eval/corpus.py:BATCH_EXECUTORS[L]`; `gt_cache.EXECUTOR_VERSION[L]`.
4. **Prompts:** `data/prompts/<L>_system_base.txt`, `<L>_primer.txt` (solver,
   §5), `<L>_author_brief.md` (idiomatic-GT, §1–2 in `L`'s paradigm).
5. **Calibrate** `L`'s hard idioms against the WebPPL GT (§4.1) and put the
   verified recipes in the author brief.
6. **Author** the column by translating the WebPPL column (§3), via the staged
   workflow (§4).
7. **Verify:** `gate crosscheck --language L` (every realizable problem passes) +
   the proper-usage audit loop (§4.4); mark genuine non-realizables unavailable (§4.5).
8. **Stamp:** `gate answers --language L`; add the column to the web build inputs
   (the browser already renders unavailable cells as "unavailable — <reason>").
9. **Document** what `L` taught you here (§8).

Ideally co-author with the language's creators (the plan for memo/pluck): they
catch non-idiomatic realizations a translator would miss.

### 7a. Source-native languages (Stan / posteriordb) — a different pattern

The checklist above assumes you *translate the WebPPL column* into `L`. A language
can instead arrive with **its own problems and its own ground truth** — Stan via
posteriordb (`eval/posteriordb.py`, started 2026-06-18). That inverts several steps:

- **GT is given, not derived.** posteriordb ships reference draws (10 NUTS chains,
  R-hat ≈ 1, ESS ≈ 10k) for 46 posteriors. Those *are* the ground truth — model the
  GT as a **`reference` pseudo-language** whose executor replays the stored draws
  (one chain-block per seed → the gate's k-seed floor is the reference's own MC
  error). The executable column (`stan`) is then *validated against* the gold draws
  by `gate crosscheck --language stan --reference reference`, instead of against
  WebPPL. This reuses the cross-language gate verbatim — pick the reference column.
- **Posterior over many params → `record` of `dist`/`real` marginals.** `dist`/`realvec`
  is unsupported; respec to per-parameter marginals (cloud, W1). Field names must match
  the reference draws (`beta[1]`, `mu`, ...) or the columns won't align.
- **Data-parametric languages break "statement pins the answer."** Stan's data lives
  outside the model and is supplied at runtime; the prose pins the *model + data
  interface*, the supplied data pins the numbers. Keep the executor interface uniform by
  making the realization a **self-contained bundle** (`eval/stan_bundle.py`: model +
  `//@ DATA/PARAMS/SAMPLING`) — data in the string keeps the GT cache key correct and
  the realization runnable without the source clone.
- **Trim embedded data to the model's declared `data` block.** Source datasets are
  shared across models, so a data file carries columns a given model never reads (kidiq
  ships `mom_iq` for other regressions). Parse the `data` block; keep only those keys.
- **ill_posed is a real verdict, not a binding bug.** Some posteriors reproduce the gold
  posterior exactly (cross-distance ≈ 0) yet flag ill_posed: independent NUTS seeds scatter
  more than the discriminability cap because the posterior is weakly identified or
  label-switches. Triage by *side*: the diagnostic (`eval/posteriordb_diag.py`) labels each
  field STAN-ILL (our floor high, reference floor tiny → fixable by sampling) vs REF-ILL
  (the gold marginal itself trips the cap → genuinely non-discriminable).
  - STAN-ILL → bake the reference's heavier regime into the bundle's `//@ SAMPLING`
    (`reference_sampling(name)`: more chains + warmup + the reference's adapt_delta). This
    fixed low_dim_gauss_mix (mixture) and hmm_drive_0 (HMM label-switching).
  - Stays ill_posed even at the reference's *full* regime → **prune** with
    `retire(pid, reason, evidence)`. arma11: phi floor 0.53→0.26→0.21 across regimes,
    plateauing ~1.5× the cap even at 10 chains × 10k warmup; the gold reference floor
    (0.0015) is a within-run chain-block split that understates true cross-run MC variance
    ~140×. Pruned (the 1 of 46). Don't paper over, don't crank sampling forever.
- **Self-generated reference draws extend the corpus past the gold subset**
  (`eval/reference_gen.py`, 2026-07-04). posteriordb has 147 posteriors but gold draws
  for only 46; for the rest we run long NUTS (10 chains × 1000 kept post-thin-10,
  10k warmup) behind explicit gates — R-hat ≤ 1.01 AND min ESS_bulk ≥ 2000 over the
  queried params — and store passing draws as an **overlay** (`data/reference_draws/
  <name>.json` + `.info.json` w/ provenance + per-param diagnostics). The vendored
  posteriordb tree is never written; `posteriordb.py` resolves gold first, overlay
  second (`validated_posterior_names()`), so ingestion/executor/crosscheck work
  unchanged. The gate IS the provenance claim ("validated draws" vs "gold draws") —
  a posterior that fails is rejected, never stored (eight_schools_centered: funnel
  geometry, R-hat 1.011, ESS 1020 → rejected; its noncentered twin is gold).
  **Dedup rule:** one problem per mathematical posterior — centered/noncentered and
  other reparameterizations share a posterior, so they'd get identical statements;
  ingest only one (the 12 radon_mn variants ≈ 6 real models).

### 7b. Gen (Julia / Gen.jl) — exact discrete enumeration (2026-07-06)

Gen follows the *translate-the-WebPPL-column* checklist (§7), not the source-native
pattern — WebPPL stays the authoritative GT; Gen reproduces it. But Gen is **scoped
to exact discrete inference** (`enumerative_inference` → the exact posterior), the
family where it is a clean, idiomatic fit. It has no NUTS, so posteriordb-style
continuous hierarchical models are out of scope (that was the feasibility call).

- **Executor** (`eval/executor_gen.py`): a Julia subprocess runs the program; the
  injected preamble supplies `__pplgym_serialize` (recursive → the algebra wire
  forms) and `__pplgym_enum_dist(res)` (reduces an `enumerative_inference` result
  to `{__kind:distribution, support, probs}`, aggregating duplicate return values).
  Because exact inference is **deterministic given the code**, `execute_gen_batch`
  runs the program **once and replicates** across seeds — which also amortizes
  Julia's per-`@gen`-function JIT (~9 s cold, paid once, not per seed). A run
  failure is a whole-run failure → raises the real reason (batch contract).
- **Toolchain:** Julia 1.10.5 + Gen 0.4.8, **box only** (no Julia on the laptop).
  Set `PPL_GYM_JULIA` to the julia binary when it is not on PATH. `EXECUTOR_VERSION
  ["gen"]="gen1"`.
- **Realization contract** (bind a top-level `ANSWER`, like WebPPL/Pyro):
  - `dist`   → `ANSWER = __pplgym_enum_dist(enumerative_inference(model, args, obs, grid))`.
  - `record` → `ANSWER = Dict("field" => __pplgym_enum_dist(...), ...)`.
  - dist over **record-valued labels** → the `@gen` returns a `Dict("k"=>v)`; the
    serializer keeps the object as a support element (the mapping form can't — JSON
    keys are strings).
- **Idioms (validated on the pilot):**
  - Discrete latents: `{:addr} ~ bernoulli(p)` / `uniform_discrete(lo,hi)`; the
    `choice_vol_grid(...)` must list **every unobserved latent address** with its
    full support (`[false,true]`, `[1,2,3]`).
  - Hard condition (`condition(c)`): an observed deterministic channel —
    `{:c} ~ bernoulli(c ? 1.0 : 0.0)` in the model + `choicemap((:c, true))` in obs.
  - Intervention / `do(x=v)`: pass the forced value as a model arg and omit that
    address (so the grid doesn't enumerate it); conditioning: keep the address +
    observe it.
- **Validation:** compare the Gen canonical answer against the **stored WebPPL GT**
  (`_gt_answers.jsonl`) via `algebra.agreement` — the box has Julia but not WebPPL,
  and the answers are exact, so comparing to the frozen GT is faithful (pilot
  distances ~1e-17 vs tol 1e-9). No WebPPL re-execution on the box needed.
  **Caveat — forward-sampled GTs.** A few probmods problems compute their WebPPL GT
  with `Infer({method:'forward', samples:N})` (a Monte-Carlo estimate, not exact
  enumeration). There, Gen's exact enumerate is *more accurate than the frozen GT*,
  so a quick agreement-vs-frozen-GT check with tol≈0 spuriously FAILS (the distance
  is just the GT's own MC noise — e.g. gen-models ex5.b: Gen 0.4 vs GT-sample 0.3998,
  d=0.0002). Trust the analytic value / the real gate's measured floor
  (~0.005-0.01 for 10k samples), which passes them. Don't "fix" a Gen realization
  that already returns the exact truth.
- **Soft factor IS available** (an earlier note wrongly said otherwise — availability
  is empirical, not predicted). The `@gen` DSL has no `factor`, but a custom
  `Distribution` whose `logpdf` returns its argument, OBSERVED at a dummy value, adds
  that argument to the trace log-weight. This is Gen's own extension mechanism (a
  real `Gen.Distribution` subtype), not a veneer — the faithful translation of
  WebPPL's `factor()`. Shipped in the executor header as `__pplgym_factor`:
  `{:pot} ~ __pplgym_factor(w)` + `choicemap((:pot, 0.0))`. Verified exact on
  agents-as-programs ex1.a (`factor(A*3)` → P(true)=0.95257), ex1.b, ex3.
- **Nested inference (RSA / theory-of-mind) IS available** via **staged composition**:
  compute each level (literal listener L0, speaker S1, ...) as a plain Julia function
  that runs `enumerative_inference` and returns a probability vector; a higher level's
  `@gen` model consumes the lower level's log-probs through `__pplgym_factor` — no
  inference-inside-`@gen` (precompute the caches as function args). This is exactly how
  RSA is written in any PPL (WebPPL nests `Infer` in `Infer`). Verified exact: the
  3-level RSA (agents ex4.a, 4 alphas), the 4-level RSA (ex4.b, L1+L3), and the
  theory-of-mind vending-machine (social-cognition ex1.1/ex1.2). **Gotcha:** use true
  `-Inf` for `log(0)` (out-of-literal-support → speaker prob exactly 0); a `log(p+ε)`
  floor leaks probability at small α.
- **Continuous latents → the V2 sampling regime (built).** Continuous-latent models
  (Beta-Binomial, hierarchical Gaussian, Dirichlet-transition HMMs) are not exact-
  enumerable, but Gen does them via **sampling** (`mh` / `importance` / `hmc`). A
  realization declares itself stochastic with the marker **`PPLGYM_SAMPLE`** anywhere
  in its code; the executor then runs **each seed independently** (reseeded, parallel)
  instead of run-once-replicate — one run does an MCMC chain and binds `ANSWER` to the
  collected samples (a cloud) or a posterior expectation. The answer is **approximate**,
  so validation is agreement-within-a-**measured-floor** (not exact match), exactly like
  the Pyro/Stan sampling GTs. Proven on `bayesian-data-analysis/ex1.2` (mh over a Beta
  latent → posterior-predictive cloud; d=0.010 vs measured tol=0.086, seeds give distinct
  clouds). Gotchas: use Gen's own distributions (`beta`, `binom`, `normal`, …) — the box
  `Distributions.jl` may be absent; sample outside `@gen` via `Gen.random(dist, args…)`.
  ~17 more probmods/dippl continuous problems follow this pattern (not yet all authored).
- **Method-pinned still unavailable.** The **inference-algorithms** chapter
  (ex1.1/1.2/1.3/2.4) pins a specific sampler — outside the determination criterion —
  so it is **Gen-unavailable** for the same reason it is Pyro-unavailable.

---

## Per-language availability

Not every problem is realizable in every language. A realization is either
**present** (code) or **marked unavailable with a documented reason** — a
first-class state, not a failure. This keeps a valid problem in the corpus for the
languages that can express it, while honestly recording where another can't.

First case (2026-06-15): the probmods **inference-algorithms** chapter's
hard-condition method demos — `ex1.1`, `ex1.2`, `ex1.3` (the heart-curve trio) and
`ex2.4` (rejection sampling). **WebPPL stays; Pyro is unavailable for all four.**
The chapter's purpose is to demonstrate inference *algorithms* on deliberately hard
conditions, so each query pins a specific sampler + settings — outside the
determination criterion (realizing it would be transcription, not inference) — and
each target is pathological for Pyro:
- ex1.1/1.2/1.3 (heart-curve): gradient-hostile, multimodal thin-manifold
  (`x^(2/3)` gradient singular at x=0; hard `|crossSection|<0.01` band → no valid
  gradient init, so NUTS/HMC can't run; ex1.3 names HMC specifically; RandomWalkKernel
  mixes pathologically slowly across the cusp → no stable GT at practical cost).
- ex2.4 (interpolation): thin acceptance band (~2e-4) pins rejection sampling; the
  posterior over `interpolationWeight` is determinate but cannot be certified in Pyro
  at practical cost (plain Importance is accurate yet ill-posed at feasible counts and
  times out at the counts needed to clear the floor; a guide needs fragile hand-tuning).
This is evidence-based — each was genuinely attempted before being marked unavailable.

**Mechanism (implemented).** A realization record may be `{problem_id, language,
available: false, reason}` (no `code`). `eval.corpus.is_available` /
`load_unavailable` expose it, `load_corpus` excludes it from the executable corpus,
and `gate crosscheck` / `answers` fold unavailable rows into their reports as
`status: "unavailable"`. Coverage reads as available/total per language (Pyro
111/115). The web browser shows the "unavailable — <reason>" note where a
realization is absent.

**Decide availability empirically, never by a statement pre-scan.** A scan of
statements for method-pinned queries OVER-FLAGS: it cannot distinguish a determinate
posterior whose statement merely *mentions* a sampler + settings (immaterial
over-specification — translates fine) from a genuinely method-dependent target (the
heart-curve). Evidence: a method-pinned scan flagged 14 problems; 11 of them realize
and pass in Pyro. So the rule is: **always attempt the translation for every
problem; mark unavailable only from evidence** — a problem that genuinely cannot be
realized after real attempts with the right tools — and do that flagging in the
consolidated review round, alongside the proper-API / idiomaticity checks, not as a
gate before translation. (A statement that over-specifies the inference method is a
separate corpus-cleanliness issue, decoupled from availability.)

## 8. Lessons log

Append-only, dated. The point of this doc — keep adding.

- **2026-06-15** — Handrolling is the persistent failure mode and a mechanical
  (regex) anti-handrolling rule *rewards* the veneer form of it. Audit the
  mechanism with judgment. (§1)
- **2026-06-15** — Pyro exact discrete enumeration via
  `config_enumerate` + `TraceEnum_ELBO.compute_marginals` is **wrong in float32**
  on extreme-probability models (0.984 vs true 0.9999). Default GT to float64.
  Calibrating this hard idiom before the fan-out is what caught it. (§2)
- **2026-06-15** — Per-seed subprocess spawning, not concurrency tuning, was the
  real cost (600 torch imports for one draws spec). Batch execution + a persistent
  content-addressed GT cache fixed it generally; the concurrency knobs stopped
  mattering. (§6)
- **2026-06-15** — Stage-1 rewrite agents must do a *pure* rewrite (no
  self-verify) and spawn no subprocesses; centralize the heavy verification. 8
  agents each fanning out crosscheck subprocesses OOM'd the laptop. (§4, §6)
- **2026-06-15** — Solver prompts and author briefs are different artifacts.
  Leaking authoring conventions, cross-language references, or planted bad
  practices (`set_rng_seed`) into a solver primer corrupts the benchmark. (§5)
- **2026-06-15** — Realizations should be import-free via the executor preamble,
  consistent across languages (WebPPL already was, via `--require`). Scattered
  imports in a column are an inconsistency worth removing. (§5, §6)
- **2026-06-15** — "115/115 crosscheck-verified" certified *numbers*, not
  idiomaticity. The old Pyro column hit those numbers by handrolling, over-sampling,
  and hand-written MH loops — exactly the work the numeric gate let it skip. A numeric
  bar rewards the cheapest passing form; idiomaticity needs its own (judgment) bar. (§1)
- **2026-06-15** — A WebPPL reference method does not map 1:1 to a Pyro tool, and
  this is where most of the re-translation pain lived. WebPPL `Infer` papers over:
  MH-on-a-hard-condition (Pyro NUTS/RW can't initialize on a measure-zero set →
  Importance+reject, or soften to a likelihood for `RandomWalkKernel`); MCMC on
  discrete latents (Pyro NUTS has no valid init → enumerate or Importance);
  non-differentiable models (NaN gradients kill NUTS → gradient-free methods). (§2)
- **2026-06-15** — Sample count is a *range*, not an extreme. "Fewest" fails the
  tolerance / goes ill-posed just as "most" times out. Prescribe per-method ranges
  (NUTS ~500–2000+warmup, Importance ~1k–10k) and let the model's difficulty pick
  within them. (§2)
- **2026-06-15** — A heavy MCMC realization takes 15–100s for a *single* seed even
  when right-sized (per-leapfrog cost, not over-sampling), so k seeds in one batch
  subprocess need a per-seed budget (`timeout * k`), not a flat per-program one.
  Verify this with leaned samples first, so the timeout isn't masking bloat. (§6)
- **2026-06-15** — Conjugacy hides under sampling. Two "ill-posed" Importance
  realizations (Dirichlet–Categorical) had an exact closed-form posterior-predictive
  the over-sampling masked; the right fix gave *better* GT, not just passing GT. (§2)
- **2026-06-15** — Agents will invent a custom inference engine (`_EnumDist`,
  `enumerate_infer` poking `pyro.poutine.runtime`) when the real Pyro path is
  awkward — sophisticated handrolling that passes numerically. Ban it explicitly in
  the brief; the idiomaticity audit must still catch it. (§1, §5)
- **2026-06-15** — We under-used the library: the brief prescribed only
  enumeration/Importance/NUTS/closed-form while Pyro ships `RandomWalkKernel`,
  `SMCFilter`, `Predictive`, `infer_discrete`, `SVI`. Survey the *target's* full
  toolkit before authoring, not after failures force it. (§2)
- **2026-06-15** — A query that pins the inference *method* (a specific sampler +
  settings, e.g. the inference-algorithms heart-curve trio) is outside the
  determination criterion — its target is sampler-specific, so realizing it is
  transcription, not inference. Such a problem is not corpus-invalid: keep it for the
  languages that express it and mark the others **unavailable** with a documented
  reason. Per-language availability is a first-class state. (§ Per-language availability)
- **2026-06-15** — Do NOT pre-filter realizability by scanning statements
  (method-pinned, etc.): it over-flags — a method-pinned scan flagged 14 problems and
  11 of them translate and pass in Pyro (the statement merely mentioned a sampler).
  Always attempt the translation; mark unavailable only from evidence, in the review
  round alongside the API checks. Availability is empirical, not predicted.
  (§ Per-language availability)
- **2026-06-15** — Process: front-load calibration. Survey the source corpus's
  inference-method distribution, enumerate the target's hard idiom-classes
  (hard-condition-continuous, discrete-MCMC, conjugate, sequential, …), and verify
  each idiom *before* fanning out. Discovering the hard classes reactively (one
  brief-patch per failure round) is the slow, expensive way. (§4)
- **2026-06-15** — A hand-written **conjugate posterior** (`dist.Beta(a+heads, …)`,
  a normalized Dirichlet predictive) IS handrolling — the math is exact, but you did
  the Bayesian update, not Pyro. "Closed form is fine" is too loose; a *posterior*
  must be inferred. Earlier belief corrected. (§1)
- **2026-06-15** — Joint discrete posteriors need `infer_discrete`, not
  `compute_marginals` (which is per-site only). Verified: an HMM full-sequence
  posterior over a 4-tuple matched the GT via `config_enumerate` + `infer_discrete`. (§2)
- **2026-06-15** — Nested RSA: Pyro cannot run inference inside another inference's
  active trace — reusing a site name throws "Multiple sample sites named …". Compute
  each level separately and memoized, unique site names per level, chaining finished
  distributions forward. (§2)
- **2026-06-15** — Marginalizing a discrete latent into a continuous one via
  `MixtureSameFamily` is not NUTS-able (constraint not transformable →
  `NotImplementedError`). For mixed discrete/continuous models, Importance over the
  prior (discrete latents become concrete samples) is the clean fallback. (§2)
- **2026-06-15** — Crosscheck-pass and proper-usage are *two* bars. Run the
  proper-usage audit after numeric convergence and **loop** it (52 → 6 → 0 here); the
  surviving defects are usually a single hand-computed field or RSA sub-level inside
  an otherwise-fine realization. (§4)
- **2026-06-15** — Outcome of the Pyro idiomatic rebuild: **111/115** realizable
  (crosscheck-verified + proper-usage audited), **4 unavailable** (the
  inference-algorithms hard-condition method demos: ex1.1/1.2/1.3 + ex2.4). Up from a
  "115/115" column that was ~half handrolled.
- **2026-06-18** — posteriordb/Stan integration (first stab). A source-native
  language inverts the add-a-language flow: its reference draws *are* the GT, so model
  them as a `reference` pseudo-language and validate the executable `stan` column against
  them with the existing crosscheck (pick the reference column). See §7a. (§7a)
- **2026-06-18** — Stan is data-parametric (data supplied at runtime), which breaks
  "the statement pins the answer": prose pins the model + data interface, supplied data
  pins the numbers. The self-contained bundle (`eval/stan_bundle.py`, data embedded)
  keeps the uniform executor interface and GT cache key correct. (§7a)
- **2026-06-18** — Pilot crosscheck (4): eight_schools / kidiq-momhs / earnings-
  logearn_height **pass** (stan reproduces gold draws within the reference MC floor);
  arma11 **ill_posed** — genuinely multimodal (AR/MA non-identifiability), 1/5 short
  fits find a spurious mode; not a binding bug, triage it. (§7a)
- **2026-06-19** — Full posteriordb campaign: all 46 gold posteriors crosschecked →
  43 pass, 3 ill_posed, **0 fail / 0 error** (the binding reproduces every model class —
  regressions, GPs, an ODE, HMMs, a 66-dim GP, mixtures). Fix round: 2 of 3 ill_posed
  fixed by reference-mirrored sampling (mixture, HMM); arma11 pruned. **Final: 45/45
  crosscheck-pass + statement-complete, 1 excluded.** (§7a)
- **2026-06-19** — cmdstanpy's compile (make) / an errored fit can leave the *process*
  CWD changed; the gate runs problems in threads of one process, so a leaked CWD made
  concurrent relative-path ops fail intermittently — a `reference`-read crash mid-run AND a
  silent report-row drop (the crosscheck report read its prior rows from the wrong dir and
  rewrote only the new ones). Fix: a CWD guard at the Stan-executor boundary
  (`executor_stan._cwd_guard`) + repo-anchored ABSOLUTE paths for everything touched in
  worker threads (posteriordb source, gt_cache, all gate report paths). Lesson: introducing
  an in-process tool that chdir's makes every relative path in concurrent code unsafe. (§6)
- **2026-06-19** — Diagnose tiered, not monolithic. The first all-46 crosscheck gave zero
  incremental feedback (one slow ODE/GP wave blocked the report, written only at the end)
  and hid the CWD crash. Running by dimension tier (light/med/heavy) surfaced results and
  failures continuously and isolated the slow models. (§4)
- **2026-06-19** — Authoring agents over-flag the per-language harness contract ("write a
  Stan program", data block) as leakage — it is the rendered binding contract, not the
  statement. They also drifted on FORM at scale: every one of 42 used markdown, ~half quoted
  literal data values, a few leaked Stan keywords (`positive_ordered`, `simplex`), and one
  haiku agent emitted broken structured output (XML tags + its model-id in a field). A
  repair-on-majors-only gate let the minor form issues through. Lesson: give authors
  exemplars + explicit FORM rules, run a dedicated form-cleanup pass, and ALWAYS Claude-verify
  (regex form-check + spot-read complex ones + hand-fix the broken one). (§5)
- **2026-06-18** — Adversarial statement-verify agents over-flag for a source-native
  corpus: they read the per-language harness-contract paragraph ("write a Stan program")
  as program leakage (it's the binding contract, rendered per target language), and they
  demand the non-centered reparam be stated (it's a program detail; the posterior is
  identical — stating it WOULD be leakage). Real signal in the same pass: unused data
  columns embedded in bundles, and conceptual→field-name bridges (intercept→beta[1]).
  Triage agent findings against the contract; don't cave to "major" volume. (§7a, §5)
