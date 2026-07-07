# Problem authoring brief

> Written for P1 (the legacy-atom migration, now complete — its inputs live in
> `archive/redesign/`). The rules below — determination criterion, emission format,
> spec authoring, calibration-then-scale process — are the standing contract for
> authoring problems from any new source material. Merge emissions with
> `scripts/build_problems.py <emissions.jsonl>`.

Contract: `data/SCHEMA.md` (read it first). Per source item you produce **one problem
record with an embedded WebPPL realization**, emitted as one JSONL line.

## Emission format (one line per problem)

```json
{
  "problem_id": "<legacy atom id, unchanged>",
  "provenance": {"source": "<atom.source>", "origin_language": "webppl",
                 "collection": "<source jsonl path>"},
  "statement": {"given": "...", "model": "...", "query": "..."},
  "answer_spec": { ... },
  "action": "keep" | "respec" | "retire",
  "realization": {"language": "webppl", "code": "<groundtruth_code>"},
  "status": {"review": "draft", "notes": "<decisions you made and why, 1-3 sentences>"}
}
```

## Statement: the determination criterion

A competent solver reading ONLY the statement must converge to the GT answer within
tolerance — and must NOT be able to reconstruct the GT program's structure from it.

- **given** — every quantity the answer depends on: priors and their parameters,
  observed data values, dataset contents, discretization grids, anything numeric.
  Read the GT code line by line: any constant in the code that affects the answer and
  is not implied by `model` prose MUST be stated here. Embedded data blobs (word
  vectors, tables) are stated as data, e.g. fenced blocks of values.
- **model** — the generative story in prose/notation. FORBIDDEN: function names,
  decomposition, helper structure, `mem`/`cache` directives, variable names from the
  code, inference method.
- **query** — the quantity or distribution requested (a posterior, marginal,
  expectation, or exact value). Name the *quantity*, never the inference method or
  sample count: for a **determinate** target — nearly all of them — the method is
  immaterial (the measured tolerance absorbs estimator noise, and `answer_spec` already
  encodes exact-vs-sampled). Pin a method ONLY when the target is genuinely
  method-dependent — the rare case where the inference procedure *is* the object of
  study and different correct methods would disagree (a truncated enumeration under a
  fixed execution budget, a comparison of exploration strategies). See the
  sampler-prescription principle in `data/REALIZATIONS.md` (§Per-language availability).

Hard bans anywhere in the statement: code blocks copied from the GT, `var `/`ANSWER`/
`Infer(`/WebPPL API names, serializer internals (`__kind`, probs/support dicts).
The statement is language-neutral; the harness adds the per-language binding contract
at render time.

Rules added after the calibration round (each one a real failure that occurred):

- **No answer leakage.** Never state the answer's support, probabilities, or any
  property the solver should derive ("the support consists of exactly two outcomes,
  each with probability 0.5" gives the answer away). Describe the process; query the
  quantity.
- **No library idioms.** `_.range(0.01, 1, 0.025)` is code — state the values (or the
  start/stop/step rule in words).
- **No wire-shape dictation.** Never describe how the output is structured ("each
  support element is a record {prevalence: ...}"). Record field *names* live in
  `answer_spec`; the query refers to them semantically. If the GT's natural output
  wraps values in records the solver couldn't anticipate, that's a respec (marginalize
  in the ANSWER binding), not a statement instruction.
- **Internal consistency.** No sentence may contradict another ("draws uniformly"
  followed by non-uniform probabilities).
- **Inference details don't belong in the statement.** Method names + settings (MCMC,
  HMC leapfrog steps, forward-sampling N, drift-kernel width, "computed by exact
  enumeration") are *program* details; for a determinate target they change nothing the
  solver must hit, so they appear nowhere — not `given`, `model`, or `query`. The only
  exception is a genuinely method-pinned target, whose load-bearing parameter (a
  truncated-enumeration execution budget, an exploration strategy) is stated in `query`.
  (2026-07-07 sweep: stripped 32 determinate problems; kept 3 truncated-enum/strategy.)
- **State soft-conditioning semantics exactly.** "Prefers utterances the listener
  would interpret correctly" underdetermines a speaker; say "chooses utterances with
  probability proportional to the literal listener's probability of the intended
  referent".
- **Retired records**: statement fields are empty strings; the reason lives in
  `status.notes` only.
- **Spec must canonicalize the realization's actual output.** If the code emits a
  joint and the spec says record-of-marginals, the action is respec with an ANSWER
  patch — never ship the mismatch.

**Underdetermined atoms** (worklist flag, or your own judgment): the GT code defines
the intended answer. Extract the missing rule or parameter from the code and state it
in `given` (e.g. "the two causes of smiling combine as an OR of independent events").
Never carry the ambiguity forward.

## answer_spec: authored, from the algebra

`{kind: value|dist|record, domain: bool|finite|int|real|realvec}` plus
`protocol: "draws"` iff one program run produces one draw (the harness collects N
seeded runs), and `estimated: true` for value statistics computed by sampling.
Correct the bootstrap draft where it is wrong — typical fixes: numeric supports that
drafted as `finite` but are ordered quantities → `int`/`real`; values that are
posterior statistics → `estimated: true`.

**Finite domains must declare their vocabulary.** Small closed label spaces (≤ ~25)
get `support: [...]` on the spec — the renderer enumerates it in the contract and the
canonicalizer rejects out-of-space mass as malformed. The declared support is the
label SPACE the statement's surface defines (include zero-probability labels), never
the realized support of a run and never the output of reachability/semantic analysis —
if deriving the set requires solving any part of the problem, declaring it leaks the
answer (then declare the larger surface space, or pin the format in `query` prose for
patterned/large spaces). Record-shaped labels use `labels: {record: {...}}`; both can
combine.

## action

- **keep** — realization code = `groundtruth_code` verbatim.
- **respec** — the *query* must change: trajectories, joint distributions over large
  structured supports, and hollow answers (ANSWER doesn't witness the modeled work)
  get re-queried as marginals / summaries / per-step quantities that the model code
  must actually compute. Change ONLY the final `var ANSWER = ...;` binding (plus
  minimal marginalization code); the model code stays identical. Statement and spec
  describe the new query.
- **retire** — no salvageable task (duplicate, degenerate, unfixably hollow). Still
  emit the full record; `status.notes` carries the reason. Nothing is deleted.

## Worked example (illustrative numbers)

```json
{
  "problem_id": "probmods2-conditioning/ex5.b",
  "statement": {
    "given": "It rains on a given day with probability 0.3. Each of two lawns has its own sprinkler; each sprinkler runs on a given day with probability 0.5, independently of everything else.",
    "model": "A lawn is wet if it rained that day or if that lawn's sprinkler ran. Rain affects both lawns; each sprinkler affects only its own lawn.",
    "query": "The posterior distribution over whether it rained, given that both lawns are wet."
  },
  "answer_spec": {"kind": "dist", "domain": "bool"},
  "action": "keep"
}
```

Note what the statement does NOT contain: no `flip`, no function names, no instruction
to enumerate, no output format. A solver in any PPL can answer it; only the GT answer
distribution constrains them.
