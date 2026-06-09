# DIPPL Corpus: Overall Curation Notes

## Schema Change: `wrap_target` field (round 2)

All 17 active atoms now carry an explicit `wrap_target` field. The pipeline appends
`var ANSWER = (<wrap_target>);` instead of running the `wrap_with_answer` regex heuristic.
The heuristic remains as a fallback for atoms that omit the field, but all atoms in this
corpus declare it explicitly. The `wrap_target` must be a single WebPPL expression; it
can reference variables defined in the assembled source blocks.

## Files Skipped Entirely

**01-introduction.md** — 26 lines of prose introduction with no code blocks. Nothing to atomize.

**hmm.md** — Skipped entirely (2 atoms dropped from the first pass). The file uses a legacy
WebPPL API throughout: `Enumerate(model, maxExecs)`, `ParticleFilter(model, numSamples)`,
and `print(...)` wrapping the result. These calls are incompatible with the evaluation
runtime (confirmed: `Enumerate expected an options object but received: 100`). Adapting
hmm.md would require rewriting the GT source from scratch, defeating the purpose of
source-block assembly. Recommend flagging this file as a skip in the corpus index.

## Atoms Reformulated Beyond Adding `wrap_target`

### dippl-03-enumeration/atom-1 (cpsFactorial)

**Problem:** `print` is not defined in headless WebPPL. The first-pass GT code ended with
`cpsFactorial(print, 5)` and failed with `ReferenceError: print is not defined`.

**Fix:** `wrap_target` uses an explicit identity continuation:
`cpsFactorial(function(x){return x;}, 5)`. The GT now returns the integer `120` cleanly.
The prompt still instructs the LM to call `cpsFactorial(print, 5)` (matching the source),
but the GT is driven by the identity-continuation wrap.

### dippl-03-enumeration/atom-2 (totalCpsFactorial)

**Problem:** Same `print` issue. Additionally, the heuristic wrapped a mid-block `var`
statement rather than a call expression, producing a parse error (`Unexpected token )`).

**Fix:** `wrap_target` uses explicit identity and error continuations:
`totalCpsFactorial(function(x){return x;}, function(e){return 'err: ' + e;}, -1)`.
The GT returns the string `'err: cpsFactorial: n < 0!'` (or similar error message from
the LM's implementation). The error path is exercised as requested by the prompt.

### dippl-03-enumeration/atom-3 (three enumeration strategies)

**Problem:** The heuristic wrapped the first line of a `var maxExec = 10` block (a
statement, not an expression), producing `Unexpected token var`.

**Fix:** `wrap_target` is a record literal computing all three `Infer` calls directly:
`{depthFirst: Infer({...}), breadthFirst: Infer({...}), likelyFirst: Infer({...})}`.
This references `binomial` which the LM defines in the assembled code from block 15.
The GT is now a record of three distribution objects, matching the prompt's intent.

### dippl-04-factorseq/atom-5 (canceling heuristic factors)

**Problem:** Blocks 13 and 14 both declare `var binomial`, causing a duplicate-var
collision in the assembled GT. Additionally, the heuristic wrapped a `return` statement
inside a function body (not a top-level expression), producing `Unexpected token )`.

**Fix:** `source_block_indices` changed from `[13, 14]` to `[14]` only. Block 14 contains
the heuristic-style model (named `binomial` in source, but the LM is instructed to name
it `binomialHeuristic`). The `wrap_target` inlines the original `binomial` definition as
an anonymous function literal inside the record expression, so no named `binomial`
reference from a source block is needed. This eliminates the duplicate-var issue entirely.

### dippl-05-particlefilter/atom-2,3,4 (canvas atoms)

**Problem:** The source blocks end with `Draw(...)` and `drawLines/drawPoints(...)` calls.
The heuristic wrapped these canvas calls (which return nothing useful), producing empty
output (`output not valid JSON: Expecting value: line 1 column 1`).

**Fix:** `wrap_target` is set to the pure model computation in each case:
- atom-2: `gaussianRandomWalk(10, 2)` (reduced from source's 100 steps)
- atom-3: `semiMarkovWalk(10, 2)` (reduced from source's 80 steps)
- atom-4: `repeat(20, gaussianMixture)` (reduced from source's 100 points)

The pipeline's new stubs for `Draw`, `drawLines`, `drawPoints` make the canvas calls
no-ops, so the model computes cleanly. The wrap_target then captures the array of
positions/points as ANSWER.

### dippl-06-mcmc/atom-1 and atom-2 (design collision)

**Problem:** Both atoms listed `source_block_indices: [1]` and had no `wrap_target`,
so the heuristic assembled them to identical GT (default enumerate on skewBinomial).
atom-2's prompt asks for MCMC but the GT was the same as atom-1's.

**Fix:** Distinct `wrap_target`s per atom:
- atom-1: `Infer({ model: skewBinomial })` (default enumerate, exact)
- atom-2: `Infer({ model: skewBinomial, method: 'MCMC', samples: 1000, burn: 200 })`

## Patterns Across the 7 Files

**Algorithm-walkthrough vs. problem pattern:** Most of the corpus is structured as a
running tutorial — building up an implementation step by step, showing intermediate states
of the code, and evolving it across many blocks. This makes chunking harder: most
individual blocks are partial programs mid-construction, not standalone runnable problems.
The practical atom-worthy blocks are almost always the final, complete version in a sequence.

**JavaScript vs. WebPPL split:** Files 03-enumeration, 05-particlefilter, and 06-mcmc
contain substantial sections written in JavaScript (not WebPPL), marked
`// language: javascript`. These implement the inference algorithms themselves. They were
skipped as atoms because: (a) they require mutation and `var` reassignment not idiomatic
in WebPPL; (b) an LM without source access cannot be asked to reproduce them in any
well-defined way; (c) they are not "problems" but pedagogy.

**CPS transform sections:** Files 03-enumeration and 06-mcmc have blocks marked
`// static` showing transformation rules as pseudocode patterns, not runnable programs.
These are skipped.

**`///fold:` pattern:** Many blocks use a `///fold:` pragma to hide boilerplate in the
DIPPL web UI. When assembling multi-block atoms, the fold sections in later blocks
duplicate definitions from earlier blocks. The pipeline must strip or deduplicate these
definitions; they create duplicate `var` declarations that will fail in most JS/WebPPL
runtimes. This is flagged per atom.

**Canvas/Draw API:** Several blocks in 05-particlefilter use `Draw(...)`,
`canvas.line(...)`, `loadImage(...)`. These require a browser-side rendering environment.
The pipeline's new stubs for `viz`, `viz.<method>`, `Draw`, `drawLines`, `drawPoints`,
`drawPolygon`, `loadImage` make these no-ops. Atoms from that file focus on the pure
probabilistic model (positions array, etc.) and set `wrap_target` to the model call.

**`viz` and `viz.table` wrapping:** Many source blocks end with `viz(dist)` or
`viz.table(Infer({...}))` rather than a bare expression. All atoms in this corpus set
`wrap_target` explicitly to skip the viz wrapper and bind the inner distribution or
value directly to ANSWER.

## Judgment Calls

**02-webppl blocks 0, 1, 2 skipped:** Block 0 (`foo` function demo) is pure syntax
illustration, too abstract. Blocks 1 and 2 are single-line `sample(Bernoulli(...))` and
`viz(Bernoulli(...))` — too trivial. Block 3 (geometric) is the smallest meaningful task.

**03-enumeration CPS tutorial skipped mostly:** Only blocks 5 and 7 yield meaningful
"translate to CPS" tasks. Blocks 9–11 (JavaScript coroutine implementation) are algorithm
walkthroughs, not problems. Block 15 (enumeration strategies) is the key practical atom.

**04-factorseq blocks 4–6 skipped as standalone:** The binomial-with-factor-moved-up
sequence is a teaching illustration showing factor reordering. atom-5 (block 14 only)
bundles the comparison task.

**05-particlefilter JS implementation blocks skipped:** Blocks 7–14 are all JavaScript
implementations of likelihood weighting and particle filtering. Block 6 is the only
standalone WebPPL HMM model in this file.

**06-mcmc blocks 0, 3, 6 skipped:** These are the full JavaScript MH implementations.
Only block 1 (the skewBinomial WebPPL model) yields clean atoms.

**Same source block used for two atoms (06-mcmc):** Block 1 generates both
`mcmc/atom-1` (exact enumeration) and `mcmc/atom-2` (MCMC sampling). This is intentional
— the same model under two different inference methods gives materially different tasks.
`source_block_indices` overlap is allowed per the contract; `wrap_target` disambiguates.

## Contract Concerns for Next Round

**`wrap_target` expressions referencing LM-defined variables:** Several wrap_targets
reference variables that the LM defines (e.g., `model`, `binomialHeuristic`,
`gaussianMixture`). The pipeline appends `var ANSWER = (<wrap_target>);` after the LM's
output, so these references resolve only if the LM actually defines them. If the LM uses
a different name (e.g., names the function `heuristicBinomial` instead of
`binomialHeuristic`), the GT will be correct but the LM's ANSWER binding will fail with
a reference error. The eval pipeline should distinguish GT execution from LM execution
and not require the LM's variable names to match the wrap_target exactly.

**`///fold:` stripping:** The pipeline should strip or skip everything between `///fold:`
and `///` when assembling blocks, to avoid duplicate var declarations across adjacent
blocks. This is critical for factorseq atoms 1, 3, 4 which use multi-block assembly.

**CPS atoms GT semantics:** The GT for atom-1 and atom-2 (cpsFactorial,
totalCpsFactorial) now returns a plain value (integer or string) via the identity/error
continuation. The eval mode should be `value` not `distribution` for these atoms; the
pipeline should be updated to set `eval_mode: "value"` accordingly.

**MCMC stochasticity:** dippl-06-mcmc/atom-2 uses MCMC with 1000 samples. The GT output
is stochastic. The eval pipeline should use a distributional tolerance check (e.g.,
KL-divergence threshold) rather than exact match, or compare against the exact enumeration
GT from atom-1 with appropriate slack.

**`print` stub:** The pipeline's no-op stubs include display/canvas functions but not
`print`. If any atom's assembled source code (not just the wrap_target) contains a
`print(...)` call at the top level (e.g., block 7 defines `printError` using `print`),
the call will fail unless `print` is also stubbed. Recommend adding a `print` stub that
returns its argument (identity behavior), consistent with the CPS fix above.

---

## Calibration Pass (post-eval, post-gates)

After running the 8-config eval (haiku/sonnet × primer/noprimer × thinking/no-thinking)
on all 17 emissions and inspecting the bucket grid, four atoms failed across all or
nearly-all configs. Each failure revealed a class of curation-time bug the original
pipeline silently let through. Three of these classes are now caught by new gates in
`scripts/assemble_curated.py`; the fourth is only catchable via the eval itself.

### Gates added

**Determinism gate** — for `value`-shape atoms, re-run the GT with `random_seed=1337`
and reject if the answer differs from the seed=42 result. Catches stochastic primitives
shaped as `value` (e.g., `geometric(0.5)` → 2 at one seed, 3 at another).

**Dedup gate** — parse the assembled GT with esprima (via `scripts/_check_dup_vars.js`,
using webppl's bundled copy) and reject if any top-level `var X` is declared more than
once. Catches over-eager `source_block_indices` selection where overlapping blocks each
re-declare the same helper (the `///fold:` pattern noted above).

**Samples self-consistency gate** — for `samples`-shape atoms, re-run the GT at a
different seed and compute TV between the two empirical distributions (same metric the
eval uses to score). Reject if TV > 0.5. Catches two failure modes simultaneously:
(a) "list-as-trajectory misclassified as samples" (random walks where every continuous
float is unique), and (b) "samples drawn from a distribution whose parameters are themselves
unconditioned random variables" (the 2D mixture with random means/stds per run).

### Atom-shape override

Emissions may now include an optional `answer_shape` field (`"value"` / `"distribution"`
/ `"samples"`). When present it overrides the answer-driven heuristic in
`classify_answer`. The agent should set this explicitly whenever the wrap_target returns
a structured list that isn't IID samples (trajectories, paths, records-as-lists). The
heuristic alone can't distinguish these from samples-shape lists.

### Atoms now in `_dippl_broken.jsonl`

| Atom | Gate | Diagnosis |
|---|---|---|
| `dippl-02-webppl/atom-1` | determinism | `geometric(0.5)` shaped as value is irreducibly stochastic |
| `dippl-04-factorseq/atom-1` | dedup | `transition` / `observeState` / `hmm` declared 2–3× from `///fold:` block overlap |
| `dippl-05-particlefilter/atom-2` | samples self-consistency | random walk; trajectories at seed=42 vs seed=1337 have TV=1 |
| `dippl-05-particlefilter/atom-3` | samples self-consistency | semi-Markov walk; same problem |
| `dippl-05-particlefilter/atom-4` | samples self-consistency | mixture with random parameters per run; TV=1 between own seeds |

### Atom that passes gates but fails eval

`dippl-03-enumeration/atom-2` — `v-` across all 8 configs. The prompt says
"call err with an error message string"; the GT pins the literal
`'Error: cpsFactorial: n < 0!'`. The prompt under-specifies what the LM should emit.
This is **prompt/wrap_target drift** — the wrap_target is more specific than the prompt.
The curation pipeline can't catch this; only the eval can, by observing uniform failure
across LMs. The right fix is to tighten the prompt to pin the exact error string (or to
soften the wrap_target — e.g., wrap the error in `function(e){return 'err';}` so the
return value doesn't depend on the LM's choice of message).

### Lessons for chapters/problang/forestdb agent briefs

1. **Stochasticity discipline.** An atom whose wrap_target executes any sampling call
   (`flip`, `gaussian`, `geometric`, `categorical`, ...) without an `Infer` wrapper
   cannot be shaped as `value`. The determinism gate now enforces this, but the agent
   should choose the right shape up front rather than rely on rejection.

2. **`samples`-shape is for IID draws.** A list returned from `repeat(N, fn)` is samples.
   A list built up by recursion (random walks, trajectories, paths, sequences) is *not*
   samples-shape under the eval's TV-on-empirical metric. Such atoms should either
   (a) return a deterministic summary (final position, distance traveled), or
   (b) be wrapped in `Infer` over the relevant marginal, or (c) be retired.

3. **Prompt completeness.** Every literal value, string, or numeric constant in the
   wrap_target must either appear in the prompt or be derivable from the prompt's spec.
   The agent should re-read its own emission and ask: "could a fresh LM, seeing only the
   prompt, produce code whose output matches the wrap_target's output?" If not, the
   prompt is under-specified.

4. **Pilot eval before scaling.** For each new corpus, run a single-config eval (cheapest
   model, default settings) on the gate-passed atoms before committing to the full
   8-config sweep. Atoms that fail uniformly on the pilot are structurally broken in a
   way the gates miss (almost always the prompt-completeness issue from point 3).
