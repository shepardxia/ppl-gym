# SCHEMA — problems, answers, and the gate

Single source of truth for the problem-centric dataset. `eval/algebra.py` implements
the answer algebra; everything else (executors, gates, scoring, web) consumes it.
Design rationale and migration plan: `data/REDESIGN.md`.

## Problem record

`data/problems/<corpus>.jsonl`, one per line:

```json
{
  "problem_id": "probmods2-conditioning/ex5.b",
  "provenance": {"source": "probmods2/chapters/conditioning.md", "origin_language": "webppl"},
  "statement": {
    "given": "Rain falls with probability 0.3. Each of two sprinklers ...",
    "model": "The lawn is wet if it rained or its sprinkler ran ...",
    "query": "The posterior probability distribution over rain given both lawns are wet."
  },
  "answer_spec": {"kind": "dist", "domain": "bool"},
  "status": {"review": "draft|reviewed|retired", "notes": ""}
}
```

- **statement** is the complete task semantics, 3-field prose:
  - `given` — every parameter, prior, observed data value. Omitting one makes the task guessing.
  - `model` — the generative story in prose/notation. **No program structure**: no function
    names, no decomposition, no `mem`/`cache` directives, no inference method — except when
    the answer is itself a realized empirical posterior, in which case method + sample count
    belong in `query` (they pin the answer, not the program).
  - `query` — the quantity or distribution requested.
- **Determination criterion**: the statement must pin the answer up to the problem's
  tolerance, and must not pin the program. Enforced by the gate (solver convergence),
  not by eyeballing.
- Prompts shown to solvers are **rendered**: statement + the language's harness-contract
  sentence (how to bind `ANSWER`). One renderer, shared by eval and web. Serializer wire
  formats never appear in prompts.
- Retired problems move to `data/problems/_retired.jsonl` with `status.notes` saying why.
  Nothing is deleted.

## Realization record

`data/realizations/<language>.jsonl`:

```json
{
  "problem_id": "probmods2-conditioning/ex5.b",
  "language": "webppl",
  "code": "var model = ... var ANSWER = Infer(...);",
  "gate": {"passed": true, "floor": 0.0, "details": "..."}
}
```

A realization's code must produce `ANSWER` **via the language's own modeling/inference
machinery**. Serializer limitations are binding bugs, never prompt workarounds.

## Answer algebra

Public surface of `eval/algebra.py`:

| function | purpose |
|---|---|
| `parse_spec(d)` / `spec_to_dict(s)` | spec ↔ dict serialization |
| `canonicalize(raw, spec)` | wire JSON → canonical representation; raises `AlgebraError` on structural failures |
| `distance(a, b, spec)` | `Distance(value, metric, diagnostics, fields)` between two canonical answers |
| `noise_floor(answers, spec)` | max pairwise distance in a list of canonical answers; used by the gate for GT floor and solver-scatter detection |
| `self_noise(answer, spec)` | split-half self-noise of a single canonical answer (0 for exact/parametric, W1 between first and second half for Cloud, worst-case over fields for Rec) |
| `agreement(a, b, spec, margin=2.0)` | do two canonical answers agree relative to their own measured noise? `tol = max(margin × max(self_noise(a), self_noise(b)), eps)`; returns `{"agree": bool, "distance": float, "tol": float, "metric": str}` |
| `verdict(cand, gts, spec, margin=2.0)` | judge canonical `cand` against k≥2 canonical GT runs; raises `AlgebraError` if fewer than 2 GTs |
| `judge(cand_raw, gts, spec, margin=2.0)` | combined canonicalize + verdict; returns `{"status": "malformed"\|"ill_posed"\|"pass"\|"fail", ...}` |

`judge()` is the primary entry point for harnesses and the web app. `ill_posed` status wins over `pass`/`fail`. A `malformed` result carries `{"status": "malformed", "error": <str>}`.

An answer is a mathematical object:

| object | meaning |
|---|---|
| `{"kind": "value", "domain": D}` | a deterministic quantity in D |
| `{"kind": "dist",  "domain": D}` | a probability distribution over D |
| `{"kind": "record", "fields": {name: spec, ...}}` | finite product of the above |

Domains: `bool` | `finite` (opaque labels — strings, ints used as labels, JSON objects) |
`int` | `real` | `realvec`. There is no trajectory object and no distribution over
unbounded structured supports — problems whose natural answer is one of those are
respecified to marginals / summaries / per-step queries.

**Structured finite labels** (`dist` over `finite` whose support elements are objects)
must declare the label shape, or solvers cannot know the field names and equivalent
answers fail on spelling:

```json
{"kind": "dist", "domain": "finite",
 "labels": {"record": {"sneeze": "bool", "fever": "bool"}}}
```

- field domains are atomic (`bool` | `int` | `real` | `string`);
- the renderer states the field names and types in the harness-contract paragraph;
- the canonicalizer validates and normalizes every support element against the schema
  (per-field bool/int coercion; unknown or missing fields are malformed). A `labels`
  declaration on a non-finite or non-dist spec is invalid.

Optional spec fields:

- `"protocol": "object" | "draws"` (dist only, default `"object"`) — how realizations
  expose the distribution. `object`: one run returns the whole distribution (e.g. WebPPL
  `Infer`). `draws`: one run returns a single draw; the harness collects N seeded runs
  into a sample cloud. Protocol is *observation mechanics*, never part of the answer.
- `"estimated": true` (value only, default false) — the value is a statistic computed by
  sampling (e.g. a posterior mean from MCMC); compared within measured noise instead of
  exactly.
- `"support": [...]` (value or dist with `domain="finite"` only) — declares the label
  **space** for the finite domain: all possible labels including zero-probability ones.
  **Never use it to encode the realized/surviving support** (the labels that actually
  received mass in a particular run) — that would leak the answer. Valid on both `value`
  and `dist` specs with `domain="finite"`. The canonicalizer rejects any answer that
  carries positive mass outside the declared support as malformed. The renderer enumerates
  the declared labels in the harness-contract paragraph. Authoring guidance: declare it
  for small, closed vocabularies (e.g. `["H", "T"]`, `["red", "green", "blue"]`).
  Patterned or large spaces (e.g. hundreds of generated labels) should be pinned in
  statement prose instead.

### Representations (orthogonal to the object)

How an executed program happens to expose an answer:

| representation | canonical JSON (native wire) |
|---|---|
| exact | raw JSON scalar / list |
| enumerated | `{"kind": "dist_enum", "support": [...], "probs": [...]}` |
| parametric | `{"kind": "dist_param", "family": "beta", "params": {"a": 2, "b": 5}}` |
| sample cloud | list of draws (program-internal or harness reruns) |

Legacy wire forms (`__kind: distribution`, `__kind: distribution_continuous` with a
`repr` string, `__kind: tensor`) are accepted by the canonicalizer until the serializers
are updated; new serializers must emit the native forms. Family and parameter names are
canonicalized inside the algebra (one alias table: Pyro `concentration1/0` → `a/b`,
`loc/scale` → `mu/sigma`, ...). Bool/int normalization (`0.0/1.0` ↔ `false/true`) is part
of canonicalization, driven by the spec's domain.

### Comparison

Defined on the object, between **any** pair of representations:

| object | primary metric | notes |
|---|---|---|
| `dist` over `bool`/`finite` | TV | clouds are histogrammed; KL reported as diagnostic |
| `dist` over `int`/`real` | Wasserstein-1 | parametric sides are sampled when the other side is a cloud (N=16384 seeded draws, common-random-numbers keyed by family+params — `algebra._PARAM_SAMPLE_N`); family+param match is a fast path when both are parametric, never a wall; TV/KS reported as diagnostics when available |
| `value`, exact | equality | float jitter epsilon only |
| `value`, estimated | absolute difference | judged against the noise floor |
| `record` | per-field | passes iff every field passes |

`dist` over `realvec` is not supported in v1 (respec those problems).

### Tolerance: measured, not authored

The gate runs every GT realization at k seeds (k≥2 required; k=2 exact, k=5 sampled).
Calling `verdict()` with fewer than 2 GT runs raises `AlgebraError`. The problem's
**GT noise floor** = max pairwise distance among GT runs (`noise_floor(gts, spec)`).
The **candidate self-noise** = W1 between the first and second half of a Cloud candidate
(split-half test); for non-Cloud representations this is 0. Acceptance for a candidate:

```
gt_floor   = noise_floor(gts, spec)          # max pairwise GT distance
cand_floor = split-half W1 of candidate      # 0 for exact/parametric
distance   = median over GT runs of d(candidate, gt_i)
tol        = max(margin * max(gt_floor, cand_floor), eps)   # margin default 2.0
pass       = distance <= tol
```

- Exact representations: floor 0, eps = numerical jitter (1e-9, scale-relative for W1).
- Parametric-vs-parametric GT runs: when both sides are identical `ParamDist` objects the
  fast path returns distance 0 (common-random-numbers variance reduction — deliberate,
  not a bug). The candidate self-noise then provides the full tolerance budget.
- Non-finite probabilities (NaN, Inf) and non-finite sample values (NaN, ±Inf) are rejected
  as malformed at canonicalization time; they never reach comparison.
- KL divergence is a diagnostic only: it equals +∞ when q has zero probability where p is
  positive (disjoint support), not a large finite approximation.
- Finite-domain labels normalize whole-number floats to integers before key comparison, so
  WebPPL integer label `1` and Pyro float label `1.0` are the same label cross-PPL.
- **Ill-posed flag**: TV floor > 0.3, or W1 floor > 0.5 × spread of pooled GT samples,
  or relative floor > 0.5 for estimated values → the problem cannot discriminate and goes
  to triage (this subsumes the old determinism and samples-self-consistency gates).

## Gate: re-derivation agreement

For each (problem, language): k independent solver generations from the rendered prompt,
plus the multi-seed GT runs above, all compared under the spec distance.

- solvers match GT → accept
- solvers agree with each other but not GT → GT suspect → triage.
  "Agree with each other" is defined as `agreement(s0, s1, spec, margin).agree` being
  `True`, where `tol = max(margin × max(self_noise(s0), self_noise(s1)), eps)` — no
  fixed cutoff.
- solvers scatter (not agreement()) → statement underdetermined → triage
- solver matches GT with near-verbatim code → overdetermination / memorization flag.
  Defined as token-set Jaccard > 0.6 between solver and GT code, computed after
  stripping comments and collapsing whitespace (`gate.code_jaccard`).

Triage is a queue (JSONL), reviewed by humans/collaborators through the web app.
Nothing is silently dropped.

## Status vocabulary

Every status string a consumer (web `tones.ts`, reports, scored rows) can see:

| layer | statuses |
|---|---|
| `algebra.judge` / `eval.score` rows | `pass` \| `fail` \| `ill_posed` \| `malformed` (+ `exec_error` from score when code doesn't run) |
| gate phase A (`_gate_report.jsonl`) | `ok` \| `ill_posed` \| `error` |
| gate phase B (`_gate_solver_report.jsonl`) | `accept` \| `gt_suspect` \| `underdetermined` \| `solver_failure` |
| problem `status.review` | `draft` \| `reviewed` \| `retired` |

Adding a status means updating the emitting module, this table, and
`web/src/lib/tones.ts`.
