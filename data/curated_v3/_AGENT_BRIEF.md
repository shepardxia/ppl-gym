# Curation agent brief — `data/curated_v3/`

This brief is the prompt template for the per-corpus curation subagent. It
codifies the lessons from the dippl pilot so chapters / problang / forestdb
agents don't repeat the same mistakes. Pass this file to the subagent
verbatim along with the source-corpus pointer and a few v2 atoms as
examples.

## Your job

Read every `.md` file under `data/sources/<corpus>/`. For each file, identify
the atom-worthy problems. Emit one JSONL row per atom with these fields:

```
{
  "id": "<corpus>-<source-stem>/atom-<N>",
  "source": "<relative path under data/sources/>",
  "source_block_indices": [<int>, ...],
  "prompt": "<self-contained problem statement>",
  "wrap_target": "<a single WebPPL expression to bind to ANSWER>",
  "answer_shape": "value" | "distribution" | "samples",
  "notes": "<free text: per-atom risks, assembly concerns, judgment calls>"
}
```

You **do not** write `groundtruth_code` or `groundtruth_output` — the pipeline
assembles the code (concatenating the listed source blocks + appending
`var ANSWER = (<wrap_target>);`) and executes it under WebPPL to capture the
ground-truth output. You **do** choose the chunking, the prompt, the
wrap_target, and the shape.

Skip whole files when the file is pure tutorial or algorithm walkthrough
with no concrete problem to solve. Note skipped files in the corpus-level
notes.

## Quality over quantity

**Producing fewer high-quality atoms is strictly better than producing
many borderline ones.** The benchmark exists to differentiate model
performance on probabilistic-programming; an atom that can't be evaluated
cleanly (e.g., LM-string-mismatch on `value` shape, MCMC sample noise on
`samples` shape, or rule-4 prompt ambiguity) is worse than no atom — it
wastes eval cost and pollutes the result distribution.

Concrete skip signals beyond pure-tutorial pages:

- **Legacy API**: files using `Enumerate(...)` / `ParticleFilter(...)` /
  `gaussianERP.score(...)` / pre-v0.7 syntax. Don't try to modernize; skip.
- **Author disclaimers**: any file whose prose contains "doesn't run",
  "experimental", "broken", "incomplete model", or that documents the
  model's wrong predictions. Skip.
- **Too-deep RSA recursion**: RSA models with `pragmaticListener(L2)` or
  deeper, especially with QUDs and valence factored in. These have so
  many design degrees of freedom (rule-4 prone) that even with the brief's
  RSA guidance, the LM and the GT will probably disagree on output shape.
  Consider skipping or atomizing only the lower-recursion layer.
- **Long stochastic walks / time series**: covered by the samples-shape
  gate, but worth flagging at chunking time.

**Skip reasons that are NOT valid:**

- "Redundant with an existing atom in the same corpus." Two atoms can use
  the same probabilistic model template (e.g., Kao 2014 metaphor RSA, RSA
  comparison-class, irony RSA) and still be distinct benchmark items if
  their parameters, state spaces, vocabulary, or evidence differ. The
  benchmark wants coverage across model variants, not just across model
  types. Curate them as separate atoms.
- "Near-identical structure to a block we already atomized." Different
  semantic phenomenon → different atom, even if the code shape is similar
  (e.g., `every-not` scope vs `two-not` scope are distinct atoms).
- "Same model family from a different paper / student group." Each
  presentation potentially clarifies different aspects of the prompt or
  evidence. If a file has self-contained complete code, atomize it.

When in doubt about whether to atomize a redundant-looking file: do
atomize it. The pilot eval will surface true duplicates (atoms that
trivially produce TV=0 with no model variance), and you can prune at
that stage.

## Strongly prefer `distribution` shape over `value`

**The benchmark scores distributions cleanly via TV; it scores values
fragilely via exact match.** When you have a choice, wrap in `Infer`.

- A "compute the expected price" answer → wrap as `Infer` over the
  bin-discretized prior conditioned on evidence, return the distribution,
  NOT just the expectation.
- A "compute the probability of X" answer → wrap as
  `Infer(function(){ return X; })` — distribution over `{false, true}`,
  NOT a single float.
- A "what is the maximum likely value" answer → return the full
  distribution and let the comparator pick out the mode.

**`value` shape is acceptable only when**:
- The answer is genuinely a single deterministic constant (e.g.,
  `factorial(5)` → `120`). These pass the determinism gate trivially.
- The answer is a record-shape combining multiple distribution sub-answers
  — but in that case use `record` shape with each field being a
  distribution.

Avoid value-shape atoms whose answer is a numeric expectation, a single
probability scalar, or a list-of-numerics. Reformulate as distribution
shape so TV scoring works.

## Hard rules (the curation pipeline enforces these; violations land in
`_<corpus>_broken.jsonl`)

### 1. Stochasticity discipline

An atom whose wrap_target executes any sampling call (`flip`, `gaussian`,
`geometric`, `categorical`, `uniform`, `discrete`, `dirichlet`, `bernoulli`,
...) **without** an `Infer` wrapper around it cannot be shaped as `value`.
The output would be different on different seeds; the comparator demands
exact equality; the atom is irreducibly untestable.

Options when the natural wrap_target is stochastic:

- Wrap in `Infer` — shape becomes `distribution`. Best for exact answers.
- Wrap in `repeat(N, function(){...})` with sufficient N — shape becomes
  `samples`. Use only when the IID samples have low-cardinality support
  (booleans, small categorical) so the empirical-TV comparison works.
- Return a deterministic summary instead (e.g., mean, max, count).
- Retire the atom.

**The determinism gate** re-runs every `value`-shape GT with a different
seed and rejects if outputs differ.

### 2. `samples` shape is for IID draws only

A list returned from `repeat(N, fn)` where `fn` draws independently is
`samples` shape. **A list built up by recursion** (random walks, paths,
trajectories, time series, Markov chains-as-sequences) is *not* samples —
it's a structured value. The eval pipeline scores `samples` by computing TV
between two empirical histograms keyed on exact value equality; for
continuous trajectories every key is unique → TV = 1 regardless of code
quality.

**The samples self-consistency gate** re-runs every `samples`-shape GT at a
different seed and rejects if TV between the two empiricals exceeds 0.5.
This catches trajectories-as-samples *and* the case where the generating
distribution itself has unconditioned random parameters (e.g.,
`var mixtureWeight = uniform(0,1);` at top level — every run draws a
different mixture).

For structured lists (trajectories etc.), either return a deterministic
summary (final state, total distance) and shape as `value`, or wrap in
`Infer` over the marginal you actually want to test.

### 3. No duplicate top-level `var` declarations

`source_block_indices` selects which fences from the source markdown to
concatenate. If two listed blocks both define `var helper = ...`, the
assembled GT contains `var helper = ...` twice. JS lets you do this but the
LM, reading only the prompt, has no way to know which definition is
canonical.

**The dedup gate** parses the assembled GT and rejects atoms with
duplicate top-level `var X`. Common cause is the `///fold:` pattern in
DIPPL / probmods sources — collapsed sections that re-state helper
definitions for display purposes. When picking block indices, treat
`///fold:` content as a phantom: it's there for the original web rendering
but creates dupes when concatenated.

### 4. Prompt must determine the wrap_target's output

Every literal value (string, number, list ordering) in your wrap_target
must either appear in the prompt or be derivable from the prompt's spec.
Before emitting an atom, **re-read your own prompt and ask**: could a fresh
LM, seeing only this prompt, produce code whose output matches the
wrap_target's output?

This rule is not enforced by a curation-time gate. It's caught later, by
the population check on the pilot eval — atoms that fail uniformly across
all LMs almost always violate this rule. The canonical example from dippl
is `dippl-03-enumeration/atom-2`: the prompt said "call err with an error
message string"; the GT pinned `'Error: cpsFactorial: n < 0!'`. Every LM
generated a *valid* error string and got `v-` (value mismatch). The fix is
to either pin the string in the prompt or design the wrap_target so its
output doesn't depend on the LM's choice (e.g., have the error
continuation return a constant that ignores its argument).

**Self-check pattern**: scan your wrap_target for any string literal, any
numeric constant, any record-field name, any function-name reference that
the LM would have to invent or pin. If found, either move it into the
prompt or abstract it out of the wrap_target.

**Subtle traps to watch for under rule (4):**

- **Recursive base case semantics.** "Base case (n==1) starts with state
  X" is ambiguous — does `f(1)` return 1 item or 2 items (the initial X
  plus one step)? Pick one interpretation and write it out as code in the
  prompt, e.g. "base case: `n == 1` returns `{states: [true]}`".
- **Length conventions.** "A walk of N steps" could mean N+1 positions
  (start + N transitions) or N positions. Specify exactly which.
- **Index conventions.** 0-indexed vs 1-indexed; inclusive vs exclusive
  bounds; whether the "current" position is included in a return list.
- **Function-name choice.** If the wrap_target calls a function the LM
  must define (`Infer({model: gaussianMixtureComponent})`), the LM's
  binding name has to exactly match. Either inline the function literal
  in the wrap_target, or pin the binding in the prompt with code like
  `var ANSWER = Infer({model: gaussianMixtureComponent})`.

When in doubt, **inline the full model in the wrap_target** so the GT
doesn't depend on the LM's naming/structure at all. The LM still has the
problem to solve (define the helpers, structure the recursion); the GT
just doesn't ride on the LM's specific choices.

**RSA-style models are especially rule-4 prone.** The chain
`literalListener → speaker1 → pragmaticListener` has many degrees of
freedom:

- What does the literal listener *return* — a distribution over states?
  full-state objects with QUD slots? a single quality dimension?
- What does the speaker *marginalize over* — just utterances? utterances ×
  QUD? utterances × QUD × valence?
- Recursion depth: is `pragmaticListener` an L1 or an L2?
- The QUD prior — uniform, biased, conditional on something else?

The brief's general rule (4) says "every literal in wrap_target must be
in prompt"; for RSA add: **every field of every returned object must be
named and computed by code that's either pinned in the prompt or
deterministically derivable from prompt prose**. If your wrap_target says
`pragmaticListener(10000)` and that returns a distribution over
`{price, valence, qud}` triples, the prompt must specify *exactly* that
shape — not "infer the price and stuff." Alternative: inline the entire
RSA chain in the wrap_target as one big expression; ask the LM to
demonstrate they understand by reproducing the helpers (`meaning`,
`utterancePrior`, etc.) but pin the inference structure in the binding.

The forestdb pilot found 4/10 atoms hit rule (4) on first emission, all
of them RSA-flavored. See `_forestdb_overall_notes.md` "Rule-4 Prevalence
by Corpus" for empirical rate.

## Chunking heuristics

- An atom should ask for **one coherent answer**. A record `{a, b, c}` is
  fine when `a`, `b`, `c` come from the same model under different
  conditioning. A record bundling two genuinely different models is two
  atoms.
- An atom should not require the LM to write a chapter end-to-end. If a
  multi-block running example is too big, split into two with deliberate
  helper overlap (later atom inlines the earlier atom's helper in the
  prompt as a `Helper:` block).
- An atom should not be so small that the LM writes one line. If the
  helpers are too much of the work, raise the boundary so the LM still
  has something meaningful to produce.

## Helper handling

The eval pipeline does **not** prepend any source-side context at eval
time. The atom's `prompt` is the entire user message. So whatever helper
context the GT relies on must either be inlined in the prompt under a
`Helper:` block, or be naturally re-derivable from prose.

Inline a helper when it's mechanical / domain-specific / boilerplate.
Describe in prose when re-derivable. Two atoms in the same source needing
the same helper: both atoms inline it (overlap in `source_block_indices`
is fine).

## Inference method prescription

Specify the inference method in the prompt when it materially affects the
answer (`Infer({method: 'enumerate'})` for exact, `'MCMC'` with specific
burn/samples for noisy). Don't blanket-prescribe; only when the answer
depends on it.

## Output files

The pipeline produces three files per corpus run:

- `<corpus>.jsonl` — atoms that passed all gates (determinism, dedup,
  samples self-consistency).
- `_<corpus>_broken.jsonl` — emissions that failed assembly or a gate,
  with the specific error message for triage.
- `_<corpus>_overall_notes.md` — corpus-level pattern notes (yours, plus
  any pipeline observations).

After the pipeline runs, run a single-config pilot eval (cheapest model,
default settings) before the full 8-config sweep. Atoms that fail
uniformly on the pilot violate rule 4 above and need re-curation; don't
spend the full eval budget on them.
