# Dataset Changes Log

> Historical log (atom era). Scripts and modules referenced below — `build_atomized_v2.py`, `rebuild_prompts.py`, `eval/harness.py`, `eval/metrics.py` — were deleted in the P2 redesign; see `data/REDESIGN.md`.

A running record of every change to `data/atomized_v2.jsonl` (the live
eval dataset) and `data/atomized.jsonl` (the v1 untouched original).

Format per entry:
- **atom_id** — what changed, why, evidence.

---

## Drops (atoms removed from v1 → v2 in `scripts/build_atomized_v2.py`)

These 4 atoms execute but their groundtruth depends on `_.now()` (wall-clock
time), so the posterior is hardware/timing-dependent and not reproducible.
TV between two runs of the same code on the same machine = 1.0.

- `probmods2-process-models/ex1`
- `probmods2-process-models/ex3`
- `probmods2-process-models/ex4`
- `probmods2-process-models/ex5`

## Prompt rewrites (v1 → v2 in `scripts/build_atomized_v2.py`)

29 atoms had their `prompt` field rewritten to be self-contained
(removed cross-atom references like "Same setup as before", inlined
data and helper code that earlier subparts had defined).

Affected atoms (hand-rolled in the `REWRITES` dict):
- conditioning/ex5.b, ex5.c
- mixture-models/ex1.a, ex1.b
- occams-razor/ex1.2, ex1.3
- social-cognition/ex2.1, ex2.2, ex2.4, ex2.5
- hierarchical-models/ex2.2, ex2.3, ex2.4, ex3.2
- observing-sequences/ex1.b, ex1.c, ex2.a, ex2.d, ex3.a
- agents-as-programs/ex2.e, ex4.b
- inference-algorithms/ex1.1, ex1.2, ex1.3, ex2.1, ex2.2, ex2.3, ex2.4, ex2.5

`groundtruth_code` was left unchanged for all of these.

---

## Post-build edits (in-place modifications to `data/atomized_v2.jsonl`)

### Round 1 — example-formatting fixes

- **`probmods2-generative-models/ex2.b, ex2.c, ex7.a`** — added `;` to
  the prompt's example code so LLMs don't mirror ASI-ambiguous code
  (`var x = mem(...) \n [a, b]` parses as `var x = mem(...)[a, b]`,
  i.e. subscript, in JS).

- **`probmods2-hierarchical-models/ex3.1`** — prompt previously described
  the data schema but didn't include the actual `data` array; non-Sonnet
  models got `ReferenceError: data is not defined`. Now inlines the
  full 24-row `data` array and the simpler-baseline BDA model.

### Round 2 — return-type ambiguity / GT mismatch

These are dataset bugs where the LLMs' natural reading of the prompt
disagreed with the groundtruth's chosen representation. Fix targets the
prompt unless noted; GT_OUTPUT cache invalidated and re-run only when
groundtruth_code changed.

- **`probmods2-conditioning/ex6.d`** — *groundtruth_code changed*. GT
  returned `true`/`false` but the prompt asked for "vowel vs consonant".
  Updated GT to `return checkVowel(letter) ? 'vowel' : 'consonant'`.
  Re-cached `groundtruth_output`. Now KL=0 / TV=0 across all 9 runs.

- **`probmods2-conditioning/ex1.b, ex1.c, ex1.d`** — prompt now
  explicitly states the return type ("(as a boolean: true=heads,
  false=tails)" and "(return the string 'fair' or 'biased')") to match
  what the GT returns. GT unchanged.

- **`probmods2-agents-as-programs/ex4.a, ex4.b`** — first attempt was a
  no-op (string-replace mismatch on "World:" vs "The world has three
  objects:"). Fixed in Round 3.

- **`probmods2-mixture-models/ex2.a`** — appended a clarification that
  outputs should use 'group1' / 'group2' labels (matching GT) rather
  than 'bonafide' / 'malingerer' (LLM's natural reading from the prose).
  Did not fully address the underlying issue (see Pending Review).

### Round 3 — RSA object representation

- **`probmods2-agents-as-programs/ex4.a, ex4.b`** — prompts now
  explicitly describe objects as records `{shape, color}` matching the
  GT's `meaningPrior`. (Previous in-place attempt missed because the
  string-replace target didn't appear verbatim in the prompts; this
  round rewrote the visible "three objects: ..." clause.)

### Round 4 — disambiguate which variable to query

- **`probmods2-conditioning/ex2.b`** — prompt previously said "construct
  a case where intervening produces a different result from conditioning,
  return distributions over the queried variable, not necessarily cough"
  which is genuinely ambiguous. LLMs picked different variables (e.g.,
  `cold` instead of `lungCancer`), so even a correct probabilistic
  programmer scored TV~0.94. Tightened: explicitly says "intervene on
  `cough = true`, condition on `cough`, return distributions over
  `lungCancer`".

---

## Harness changes (eval/ side)

### Comparator: distribution-key normalization

`_normalize_dist` in `eval/metrics.py` previously keyed the comparison
by the WebPPL-serialized dict key (e.g., `"\"true\""` vs `"true"` for
the same boolean value). Two semantically-identical distributions could
spuriously mismatch when WebPPL formatted their keys differently.

Fix: derive the comparison key from `entry["val"]` via
`json.dumps(val, sort_keys=True)` so it canonicalizes regardless of
WebPPL's outer key formatting. Object values with the same fields in
different orders also now match (handled by `sort_keys=True`).

### Architectural rewrite — explicit ANSWER binding + canonical schema

Two coupled changes:

**Extraction.** The previous protocol was "end your program with the
answer expression"; the harness used a hand-rolled JS-text splitter to
find the last expression boundary. This produced ~5 distinct splitter
bugs (ASI rules around `}`, `)`, `[`-subscript continuation, comments-
in-strings, etc.) over the course of evaluation rounds.

New protocol: programs explicitly bind `var ANSWER = <expr>;` as their
last statement. The harness wraps with a fixed serializer header and
appends `JSON.stringify(__serialize(ANSWER))`. No JS parsing.

System prompt updated to require the binding. All 76 groundtruth_codes
mechanically wrapped. Splitter (`_split_last_expression`, ~80 lines)
deleted from `eval/executor.py`.

**Serialization.** Distributions are now serialized via WebPPL's built-in
`serializeDist(d)` (which exists — was previously hand-rolling an
equivalent). Output format is `{"__kind": "distribution", "probs": [...],
"support": [...]}` — parallel arrays, language-agnostic, suitable for
cross-PPL comparison. The `_normalize_dist` Python helper now reads
parallel arrays directly instead of reconstructing keys from a WebPPL-
flavored `{key: {val, prob}}` dict.

Continuous distributions (Beta, Gaussian, ...) still fall through to a
string-repr fallback (`{__kind: "distribution_continuous", repr}`)
since `serializeDist` throws "Not implemented" on them.

**Cache invalidation.** All 76 cached `groundtruth_output` values were
re-emitted with the canonical schema. Existing LLM generations (created
under the old "bare expression" protocol) won't run under the new
harness — they need to be re-generated.

### Executor: detect silent exits

`execute_webppl` previously marked `success=True` whenever WebPPL's
subprocess exited 0, regardless of stdout content. Some LLM-generated
programs exit 0 with no stdout (e.g. `__ANSWER` ended up undefined
because of a Categorical with too many vs/ps, or a CPS issue inside an
Infer body). The harness then reported `executed=True, answer=None` —
which the comparator handled with a benign "not a distribution" error,
silently classifying the failure as a successful but uncomparable run.

Fix: if exit==0 and stdout is empty, surface as `success=False` with a
clear error message ("program exited 0 but produced no output …").
This caught at least one prior false-positive (`occams-razor/ex1.2`
across 3 runs) where `executed` was being over-counted.

## Dataset expansion — round 1

Four additional datasets extracted from the available WebPPL textbook
ecosystem, using `scripts/extract_atoms.py`. Each prose-with-code block
becomes one atom; the extractor wraps the last expression in
`var ANSWER = ...;`, runs it, and keeps blocks that produce a
distribution / value / sample list.

| dataset | source | atoms | exec rate | KL median / q75 | TV median / q75 |
|---|---|---|---|---|---|
| exercises | probmods2/exercises | 76 | 76/76 | 0 / 0 | 0 / 0 |
| chapters | probmods2/chapters | 121 | 107/121 | 16.9 / 21.9 | 1.0 / 1.0 |
| dippl | dippl/chapters | 24 | 23/24 | 20.1 / 21.9 | 1.0 / 1.0 |
| forestdb | forestdb.org/models | 67 | 58/67 | 21.3 / 22.3 | 1.0 / 1.0 |
| problang | problang/chapters | 50 | 46/50 | 20.7 / 22.1 | 1.0 / 1.0 |

**Observation.** Sonnet 4.6 + primer (no thinking) executes 80-100% of
atoms across the board, but only the *exercises* dataset gives near-zero
TV/KL. Every other dataset has TV q75 = 1.0 — meaning the LLM produces
a runnable program whose output distribution shares no support with the
groundtruth.

**Diagnosis.** Chapter / model atoms are *demonstrations* of concepts,
not graded exercises. The extracted prose context (preceding paragraph)
typically says something like "Consider this binomial example" before
showing one specific parameter setting. The LLM faithfully writes a
binomial example (different parameters), and the comparator correctly
flags it as different. The atoms aren't ill-formed — they just aren't
*tests* in the way exercises are.

**Iteration paths** (none taken yet, awaiting direction):
- Tighten extractor prompts to embed the parameter values explicitly.
- Filter atom candidates to those where the prose clearly pins down a
  unique answer (heuristics: prose mentions "show that the result is
  X", or asks "what is the probability of Y" with a specific value).
- Switch from distribution comparison to *behavioral equivalence*:
  given the LLM's program and the GT program, sample N times from each
  and check that the *marginal distributions* match within tolerance,
  rather than that the support is identical. This handles cases where
  parameter choices differ but the PPL reasoning is correct.

## Round 2 — cumulative-context extractor + per-atom failure-mode audit

After observing TV q75 = 1.0 across chapters/dippl/forestdb/problang
in round 1, dug in atom-by-atom.

### Fix applied: cumulative preceding-code as context

`scripts/extract_atoms.py` now passes earlier same-file code blocks to
the atom builder. The builder tries (a) full preamble + this block;
falls back to (b) standalone if (a) fails to execute (typically due to
duplicate `var` declarations). This recovers atoms whose code block
references definitions from earlier blocks in the same file.

Lift was modest:
- chapters: TV=0 stayed at 3, TV<0.5 went 20 → 20 (no real change)
- forestdb: TV=0 went 2 → 6 (small lift)
- problang: TV=0 stayed at 2

So cumulative context fixed *some* atoms but most remaining failures
are not "missing definitions" — they are *prose underspecification*.

### Atom-level audit, sample failures

Sampled atoms with TV=1.0 to identify what's actually breaking:

**1. Off-by-one / convention differences** (e.g., `chapters/145-non-parametric-models/block-1`):
   - Prose: *"walk down this list, deciding when to stop."*
   - GT returns 1-indexed values `{1,2,3,4}`; LLM returns 0-indexed `{0,1,2,3}`.
   - Both are correct implementations of the prose; supports are
     disjoint; TV=1.0.

**2. Different rep of the same model** (e.g., `forestdb/2025-problang-comparison-class/block-14`):
   - GT uses continuous `Gaussian(stateParams)` over `stateVals`.
   - LLM uses discrete `[1,2,3,4,5]` states.
   - Both are valid concretizations of "comparison class model".

**3. Different inference method**:
   - GT uses `repeat(5000, ...)` (returns list of samples → shape=samples).
   - LLM uses `Infer({method: 'forward', samples: 10000}, ...)` (returns Distribution).
   - The shape mismatch alone forces TV=1.

### Distribution of failure modes per dataset (worst-case TV per atom)

```
dataset       TV=0  TV<.05  TV<.5   TV<1  TV=1   failed   total
exercises       43       9      9      7     2        0      70
chapters         3       4     13     22    47       12     101
dippl            2       1      1      7     4        1      16
forestdb         6       0      6      6    19       11      48
problang         2       0      1     10    10        6      29
```

Even with cumulative context, ~50% of chapter atoms and ~40% of
forestdb/problang atoms hit TV=1.0. The bottleneck is the prose itself,
not extraction.

### What the new datasets are good for (and what they aren't)

**Good for:** "can the model write runnable WebPPL conditioned on a
prose description?" Exec rates of 80-95% across the four sources show
the model produces syntactically valid programs that execute. This is
a meaningful capability check.

**Not good for:** "can the model produce the *intended* PPL program?"
The discrete TV/KL comparator measures equivalence of returned
distributions, which requires the prose to uniquely pin down both the
model and the output convention. Pedagogical sources don't.

### Three iteration paths (none chosen yet)

1. **Tighten extractor prompts**: programmatically detect under-specified
   prose and either (a) drop the atom, (b) augment with concrete output
   conventions ("return the integer index, 0-indexed").
2. **Behavioral-equivalence comparator**: instead of comparing exact
   support, sample N times from each program; compare *marginals* or
   *summary statistics* (mean, variance, top-1 mode) rather than full
   distributions.
3. **Filter to test-shaped atoms**: only keep atoms whose prose contains
   a question ("what is the probability of...", "show that..."). Drops
   most demonstration-style content.

## Round 3 — harness bug: aggregate-samples mishandled

Per-shape per-dataset audit revealed an actual harness bug.

`scripts/extract_atoms.py` classifies an atom as `answer_shape="samples"`
whenever the GT's last expression returns a list. But there are two
fundamentally different "samples" patterns:

1. **Per-run sample**: GT returns a single bool/int/value (`flip(.4)`).
   The harness reruns N times to estimate the marginal distribution.
   *This is what the original exercises atoms use.*
2. **Aggregate samples**: GT returns a pre-collected list (e.g.
   `repeat(5000, fn)`, `_.map(...)`). The list IS the sample collection.
   *This is what most extracted chapter / forestdb / problang atoms use.*

The harness was treating both the same — running gen N=100 times for
samples-shape atoms. For an aggregate atom, this meant gen would
produce a list-of-100-lists, which the comparator compared against a
flat list of scalars. The keys (`json.dumps` of an entire list of 5000
ints vs. a single int) never matched, so TV=1.0 was guaranteed.

### Fixes applied

- `eval/harness.py:_is_aggregate_samples(answer)` — detects whether the
  cached GT answer is a flat list of scalars (treat as aggregate) vs.
  a per-run sample.
- `eval/harness._run_gen` — for aggregate-samples atoms, runs gen ONCE
  (not N times) so both sides produce a flat sample list.
- `eval/metrics._cmp_samples` — coerces a Distribution-shaped gen
  result into a sample list (deterministic expansion of support+probs)
  before histogram comparison. Handles the case where GT uses
  `repeat(N, fn)` and gen uses `Infer({forward}, ...)` — same intent,
  different realization.

### Impact

```
dataset      TV=0 before  TV=0 after  TV=1 before  TV=1 after
chapters         3            6           47           32
dippl            2            2            4            5
forestdb         6            6           19           19
problang         2            2           10            8
```

Chapters get the biggest lift (15 fewer TV=1 atoms become non-trivial
scoring). Other datasets see modest changes. The remaining TV=1 cases
are real ambiguities (off-by-one conventions, different
concretizations, prose underspecification).

## Round 4 — image scrubbing, short-list reclassification, TV clamp, dominant-failure analysis

Continued failure-mode digging across the 4 new datasets.

### Bugs found and fixed

1. **Inline base64 image data leaking into prompts.** One forestdb
   atom (`forestdb-2025-problang-politeness/block-1`) had a 167 KB
   `<img src="data:image/png;base64,...">` in the prose, leaving zero
   useful prompt content. Added `sanitize_prose` in
   `scripts/extract_atoms.py` to strip `<img>` tags and
   `data:image/...;base64` URIs (replaced with `[image]`). Wired into
   `truncate_prose` so future extractions inherit the fix; also
   applied retroactively to existing prompts (4 atoms changed: 1
   forestdb + 3 chapters). The politeness atom dropped from 167 KB →
   275 bytes — flagged for manual review (the image *was* the
   question).

2. **Floating-point TV > 1.0.** Seven atoms had TV values like
   `1.0000000000000104` from disjoint-support distributions where
   `0.5 * sum(|p - q|)` accumulated to slightly over 1.0. Clamped TV
   to `[0, 1]` in both `_tv` and `empirical_tv` in
   `eval/metrics.py`.

3. **Short fixed-size lists mis-classified as samples-shape.** GT
   code that returns an HDI `[low, up]` or a `[mean, var]` tuple is a
   structured *value*, not a sample collection — but the extractor
   was calling every list-typed answer "samples". Updated
   `classify_answer` in `scripts/extract_atoms.py`: a list of length
   ≤4 with all-scalar entries is now `value`-shape (elementwise
   `value_match` with `rtol=0.05`). Applied retroactively: 3 chapter
   atoms, 2 dippl atoms, 1 problang atom reclassified samples →
   value. (v2/exercises reverted — it has record-shape atoms from a
   different pipeline that doesn't go through this classifier.)

### Failure-mode breakdown (post-fix, sonnet-4.6 + primer)

```
dataset    TV=0  <.05  <.5  <1  =1  val-ok  val-no  shape-mm  exec-fail   total
exercises   43    9     9   7   2    4       2       0         0          76
chapters     6    4    18  27  30    0      11      14        12         122
dippl        2    1     1   5   6    2       3       3         1          24
forestdb     6    0     6   6  19    0       1      18        11          67
problang     2    0     1  10   5    0       3      24         6          51
```

### Dominant remaining failure: prose under-specification

The single largest non-trivial-TV bin in problang (24/51) and
forestdb (18/67) is now **shape mismatch** — gen returns a dict like
`{state_0: dist, ..., state_3: dist}` while GT picks one specific
marginal like `speaker(3)`. The LLM reads the same prose and
naturally produces the more comprehensive answer "for completeness".

Inspected examples:

- `problang-02-pragmatics/block-3`: prose `display("speaker's
  production probabilities for state 3:")` tags one specific state.
  GT extracts `speaker(3)`; LLM emits `{state_0...state_3: ...}`.
- `problang-01-introduction/block-0`: GT `literalListener("blue")`;
  LLM `{L0_blue, L0_circle, L0_red, L0_square: ...}`.
- `forestdb-bkmt-scalar-implicature/block-10`: GT one marginal; LLM
  `{knowledgeModel_none, jointModel_none, statePrior}`.

This is a **prose-quality limit of the source corpora**, not a
harness bug. Tutorial markdown has loose surrounding prose ("let's
see how the speaker behaves") with the specific question implied
only by a `display(...)` literal. Fixing properly would require
either (a) hoisting `display(...)` strings into the prompt as the
explicit question (covers ~11 atoms) or (b) making the comparator do
fuzzy field-extraction from dict-shaped gen results — both deferred
as out of scope for "fix the pipeline".

## Round 5 — park non-seedable atoms, recover one image-only prompt

User asked to park atoms that aren't seed-reproducible and try to
recover any LLM-scoreable atoms whose *prompt* (not GT) is genuinely
broken — without spoonfeeding the answer to the LLM.

### Parked: 3 timing atoms

GT-vs-self with the same `random_seed=42` returns different answers
across runs for these 3 atoms because they use `_.now()` (wall-clock
elapsed time) which is not under WebPPL's RNG control. Moved out of
the active dataset into `data/parked_atoms.jsonl` with
`parked_reason` annotated:

- `probmods2-chapters-inference-algorithms/block-1`
- `probmods2-chapters-inference-algorithms/block-2`
- `probmods2-chapters-inference-algorithms/block-3`

Audit run: 337/340 atoms across all 5 datasets are seed-reproducible
(survey via `execute_webppl(..., random_seed=42)` twice and comparing
canonical-JSON outputs).

### Recovered: politeness/block-1 prompt

`forestdb-2025-problang-politeness/block-1` had a 167 KB prompt that
was 100% inline base64 image data. Round 4's `sanitize_prose` reduced
it to a 275-byte prompt of just `[image]` because the retroactive
sanitizer ran on the *final assembled prompt string*, never re-running
`truncate_prose` over fresh prose paragraphs from the source.

Added `scripts/rebuild_prompts.py` that re-extracts prose from source
markdown and substitutes only the prose section of the prompt — the
"given code" preamble stays as-is so we don't drift from the GT's
actual execution context. Default policy: only rebuild prompts whose
existing prose section (excluding `[image]` markers) is shorter than
80 chars. Dry-run sweep found exactly 1 atom matching this criterion
across all 4 new datasets — the politeness one. Recovered prose:

> "Pragmatic speaker 1 (S1) is the same as the previous politeness
> model we covered in class. The speaker considers their
> *informational* (being truthful) and *social* (being kind) goals,
> and *φ* determines which goal they prioritize. Running a speaker
> who wants to convey state 1 and 0.5 *φ* returns the utterances
> with the highest probabilities on..."

That maps cleanly onto GT `speaker1(1, 0.5)`. Re-ran sonnet-4.6 on
just this atom; gen now executes (was previously shape-mm because the
LLM had nothing to go on). The two distributions still disagree
(TV=1.0) — different parameterization on the LLM's side — but the
atom now contributes a real verdict.

### Investigated but not changed

- **Prose mentions a function the LLM tries to use that isn't a
  WebPPL builtin.** Three exec-fail atoms with `ReferenceError`:
  `CRPmem` (chapters/non-parametric-models/block-11) — prose says
  "WebPPL provides CRPmem" but our WebPPL distribution doesn't ship
  it; GT defines its own `DPmem`. `Geometric` (inference-algorithms/
  block-12) — the LLM hallucinated a builtin from prose's plain-
  English mention of "geometric distribution"; not a prose bug.
  `seatCustomer` — pure LLM hallucination. Only the CRPmem case is
  genuinely a misleading prompt; left as-is for now (would require
  editing the textbook prose, deferred).
- **Display-string hoisting** (proposed in Round 4): only 9 atoms
  (7 forestdb + 2 problang) have `display(...)` strings that aren't
  already in the prompt. Hoisting these would border on spoonfeeding
  (the display string is often the precise question the GT answers).
  Skipped per user direction.

### Scoreable count now

```
dataset    TV=0  <.05  <.5  <1  =1  val+  val-  shape!  fail   total  scoreable*
exercises   43    9     9   7   2    4    2     0       0       76      76
chapters     6    4    18  31  26    0    8    14      12      119      93
dippl        2    1     1   5   6    2    3     3       1       24      20
forestdb     6    0     6   6  19    0    1    18      11       67      38
problang     2    0     1  10   5    0    3    24       6       51      21
total                                                          337     248
```

\* "scoreable" = produced a graded numeric score (TV-binned or value
match), excluding shape-mm and exec-fail.

Three counts now make sense:

- **340** total atoms (started here)
- **337** seed-reproducible (parked 3 timing atoms)
- **248** scoreable on this LLM run (can grade with a TV/value
  metric; remainder hit shape-mismatch or exec-fail in this
  particular run, but the atoms themselves are sound)

## Round 6 — output-shape spec in prompt

The shape-mismatch failure mode (LLM produces `{key: dist, ...}` when GT
is one distribution) was a *prompt under-specification* problem, not a
harness problem: the prompt only said `End your program with var ANSWER
= <expression>`, never specifying what *kind* of expression. The LLM
naturally hedged toward comprehensive dicts.

Fix: each atom already knows its `answer_shape` (we ran the GT to
determine it), so we can include a one-line type spec in the prompt.
This is task specification — like a return-type annotation — not
answer leakage. We're saying "produce a Distribution" not "produce
this specific Distribution".

### Implementation

`scripts/extract_atoms.py:format_answer_shape_hint(shape)` returns a
short English description of the expected ANSWER's type:

- `distribution` → "a single distribution object (e.g., the return of
  `Infer({...}, function() { ... })`)"
- `samples` → "a list of samples — typically the return of `repeat(N,
  function() { ... })` or equivalent"
- `value` → "a single scalar or short fixed-size tuple (number,
  string, boolean, or list of those)"
- record → recursively describes each field

The trailing instruction in `build_atom`'s prompt template now ends
with "where the value of ANSWER is {hint}." instead of the previous
"(the answer the prose is asking us to compute or produce)."

Applied retroactively to all 261 atoms in the 4 new datasets. Re-ran
gen on the 89 atoms that previously hit shape-mm or exec-fail (no need
to re-gen atoms whose previous run already produced a valid scoreable
answer — the change can only help, never hurt).

### Impact (sonnet-4.6 + primer + shape spec)

```
dataset    TV=0  <.05  <.5  <1  =1  val+  val-  shape!  fail   total  scoreable
                                                                       (before -> after)
exercises   43    9     9   7   2    4    2     0       0      76     76
chapters     7    7    21  26  41    0    8     0       9     119     93 -> 110
dippl        2    1     1   4   9    2    3     1       1      24     20 -> 22
forestdb     8    1     8  17  29    0    1     0       3      67     38 -> 64
problang     5    0     1  11  24    0    4     4       2      51     21 -> 45
TOTAL                                                          337    248 -> 317
```

**+69 atoms scoreable.** Shape-mm collapsed from 59 → 5; exec-fail
halved from 30 → 15 (LLMs no longer build ambitious comprehensive
dicts that time out on heavy MCMC). 317/337 = 94% of seed-reproducible
atoms now produce a graded score.

The remaining 20 non-scoreable atoms are LLM bugs of the kind a
different model would handle differently:
- 5 shape-mm: LLM still produced a different output type
- 15 exec-fail: hallucinated builtins (`Geometric`, `seatCustomer`),
  too-strict conditions, or genuine timeouts on heavy MCMC.

Note: TV=1 went up in absolute terms (e.g. problang 5 → 24) because
atoms that were previously *silently* shape-mm now get an explicit
TV=1 verdict. The atoms aren't worse — they were always this hard to
match exactly; the shape spec just made the disagreement visible.

## Pending review (not yet changed)

### Sparse-support / continuous-joint posteriors

When an atom's groundtruth is `Infer(MCMC, ...) → record/continuous`,
each MCMC sample is a unique high-dim point. Two chains (gen vs gt)
produce *disjoint* support sets by construction, so the discrete TV
between them collapses to ~1.0 even when both are correct. KL is at
the saturation cap (~22). My discrete-distribution comparator can't
distinguish "wrong" from "right but stochastically explored differently".

Atoms in this category (TV≥0.5 across most runs, even from the strongest
config):

- `probmods2-mixture-models/ex2.a` — 22 booleans + 2 continuous group rates
- `probmods2-mixture-models/ex1.b` — 2 mem'd prototypes × 3 props each
- `probmods2-occams-razor/ex2.2` — joint (relation, cp, b)
- `probmods2-occams-razor/ex2.3` — record of two arrays of expectations (less affected)
- `probmods2-learning-as-conditional-inference/ex2.1` — `post` is over continuous coinWeight
- `probmods2-hierarchical-models/ex2.3, ex2.4` — beta-on-beta continuous joint
- `probmods2-hierarchical-models/ex3.1, ex3.2` — gaussian joints
- `probmods2-inference-algorithms/ex4.a` — Dirichlet-distributed topic-model joint
- `probmods2-observing-sequences/ex2.c, ex3.a, ex3.b` — sentence-distribution (combinatorial space)
- `probmods2-agents-as-programs/ex2.d, ex2.e` — uniform alpha continuous

Possible fixes (each is a real engineering lift):
1. Reshape the atom to return marginals only (single scalar or finite-support dist).
2. Add a comparator for continuous Distributions that converts the support to samples and runs MMD / Wasserstein / KS.
3. Accept the limitation and mark these atoms "not scoreable by current harness".

**No change made yet; awaiting direction on (1) vs (2) vs (3).**
