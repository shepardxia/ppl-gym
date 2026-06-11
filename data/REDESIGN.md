# REDESIGN: problem-centric ppl-gym

Scoped 2026-06-09. Per-atom audit verdicts (full coverage of the 163 legacy atoms, pre-P1 count; the canonical post-P1 corpus is 115 problems): `archive/redesign/redesign_worklist.jsonl`.

## Why (verified defects in the current shape)

1. **The Pyro column isn't Pyro.** 33/40 GTs are plain Python (no `pyro`/`dist`/`torch`); 31/40 prompts instruct the solver to hand-build the serializer wire format (`__kind`/`probs`/`support`). The translation gate demanded exact output match, the serializer couldn't capture a Pyro posterior, so the translator satisfied the gate by abandoning Pyro inference. The 40 "successes" are more damaged than the 36 in `_probmods_broken.jsonl`.
2. **Pyro execution is unseeded** (verified empirically: same seed → different outputs). `executor_pyro.py` sets `PPL_GYM_PYRO_SEED` but nothing reads it; `pyro.set_rng_seed` was never written. Masked by defect 1; the 4 genuine-Pyro atoms are all stochastic `samples`-shape, exactly where this bites.
3. **Prompt pathologies at scale** (full audit): only 8/163 prompts are clean. code_verbatim 46, prose_dictation 119, realization_pinned 75, underdetermined 6, hollow_witness 7 (ANSWER doesn't depend on the requested work).
4. **Answer semantics are post-hoc.** `output_spec` is inferred from output values by heuristics; three parallel type systems (`answer_shape`, `output_spec`, serializer `__kind` tags) live across two comparators, two serializers, and the web port.
5. **Parametric cross-PPL comparison is string-repr mimicry.** `executor_pyro._continuous_repr` formats Pyro params to look like WebPPL's `toString` so `spec_metrics` can regex-parse it back.

Root cause: an atom's task semantics — what is asked, what counts as correct — live nowhere explicit, so every component (extractor, translator, gates, comparators) reconstructs them independently and divergently.

## Design

### Problem schema (the unit of the dataset)

`data/problems/<corpus>.jsonl`, one problem per line:

```json
{
  "problem_id": "probmods2-conditioning/ex5.b",
  "provenance": {"source": "...", "origin_language": "webppl", "blocks": []},
  "statement": {
    "given": "every parameter, prior, observation, data value",
    "model": "the generative story, prose/notation — no program structure",
    "query": "the quantity/distribution requested"
  },
  "answer_spec": {"...": "authored, from the algebra below"},
  "status": {"review": "...", "gate": {}}
}
```

`data/realizations/<language>.jsonl`: `{problem_id, language, code, gate}`. GT outputs cached per (problem, language, seed) as today. The pyro↔v2 id linkage is a pure rename (verified), so v2 ids carry over as problem ids.

**Determination criterion (the prompt contract):** the statement must pin the answer up to the spec's tolerance, and must not pin the program. Priors/data/query: required. Function names, decomposition, `mem`/`cache` directives, inference method: forbidden — except where the answer *is* a realized empirical posterior, in which case method + sample count are part of the query.

Prompts are **rendered**: statement + per-language harness contract (the ANSWER binding sentence), from one renderer shared by eval and web. Serializer wire format never appears in a prompt.

### Answer algebra (`eval/algebra.py` — single module)

**An answer is a mathematical object** — the type system is just:

- `Value(domain)` — a deterministic quantity
- `Dist(domain)` — a distribution
- `Record{name: answer}` — finite product

with domains `bool | finite(labels) | int | real | real^n`. Structured supports are not domains: problems whose natural answer is a trajectory or a joint distribution over objects get respecced to marginals/summaries/per-step queries (~17 atoms).

**Representation is orthogonal to the object** — how a realization happens to expose it:

- exact (closed form)
- enumerated `{support, probs}`
- parametric `{family, params}` (canonical param names; the cross-PPL alias table lives only here — replaces repr-string mimicry/parsing)
- sample cloud — program-internal draws, or harness N-seeded reruns; *which* protocol applies is part of the realization contract, never part of the answer

**Comparison is defined on the object, between any pair of representations**: `Dist(finite)` → TV (KL diagnostic); `Dist(real)` → W1 (KS diagnostic), sampling from a parametric side when the other side is a cloud; family+param match is a fast path when both are parametric, never a wall — a NUTS sample cloud must be comparable against a parametric Beta, which is exactly the cross-PPL posterior case. `Value` → exact, or within measured noise for sampled estimates. Bool/int normalization (Pyro 0.0/1.0 ↔ WebPPL true/false) is part of canonicalization here.

**Tolerance is measured, not authored.** The gate runs every GT at k seeds anyway; a problem's acceptance threshold is its own GT-vs-GT noise floor × margin (exact representations: zero floor). No hand-picked TV/rtol constants. A noise floor too large to discriminate flags the problem as ill-posed — which is precisely what the old bespoke gates were hand-built to catch (determinism gate = "floor is zero"; samples self-consistency = "TV floor < 0.5"). Dissolved by this design: `_is_aggregate_samples` list-sniffing in harness.py, distribution↔samples coercion, the parametric-vs-empirical comparison wall, support-aligned TV on floats, and per-atom `equiv` blocks.

Audit census → algebra: dist_enum 83 → `Dist(finite)`; record 32 → `Record`; dist_structured 28 → `Dist(finite)` where enumerable, ~10 respec; value 9 → `Value`; samples 8 → `Dist(·)` observed via reruns; dist_empirical_1d 2 → `Dist(real)`; trajectory 1 → respec.

**Authority flip:** during migration, draft specs may be derived from existing GT outputs — but that derivation is a bootstrap only. Once a problem's spec is reviewed, the spec is authoritative and the GT must conform to it (the gate enforces this direction). A spec that is silently adjusted to match whatever the GT emits is the old post-hoc classifier reborn.

### One gate (`eval/gate.py`): re-derivation agreement

For each (problem, language): k independent solver generations from the rendered prompt + multi-seed reruns of the GT, all compared under the spec distance. The GT-vs-GT runs do double duty: they are the determinism/consistency check *and* they measure the problem's noise floor, which sets its acceptance tolerance (see algebra).

- solvers match GT → accept
- solvers agree with each other but not GT → GT suspect → triage
- solvers scatter → statement underdetermined → triage
- solver matches with near-verbatim code (jaccard > θ) → overdetermination/memorization flag

Replaces the determinism gate, samples-self-consistency gate, and translation comparator gate (all special cases of "derive twice, compare"). `_check_dup_vars.js` stays as a static lint. Triage queue is a JSONL consumed by the web review UI; nothing is auto-deleted — kills go to `_retired.jsonl` with reasons.

### Language bindings

A binding = executor + serializer (to canonical wire) + harness-contract text + primer. **Binding rule: ANSWER must be produced by the language's own modeling/inference machinery. Serializer limitations are binding bugs, never prompt workarounds.**

- **WebPPL**: exists; serializer updated to structured `dist_param`.
- **Pyro**: rebuild — (a) actually seed (`pyro.set_rng_seed` in header); (b) serialize `pyro.infer` artifacts (Importance/EmpiricalMarginal, MCMC samples, Predictive) into the algebra; (c) re-derive realizations for **all 115 problems** (probmods2 + dippl + forestdb) under the gate, not the previous 40. Validate the binding with ~3 hand-written pilot realizations (enum / empirical / parametric) before any LM translation.
- **Stan**: next column (cmdstanpy; posterior draws → dist_empirical_1d, generated quantities → value/samples). Opens ARM, BPA, example-models, posteriordb as problem sources.
- **memo / pluck**: with the language creators; the review loop is their entry point.

## Migration worklist (sized, full coverage)

**Every problem gets a freshly authored statement; no existing prompt survives as-is.** Prompts are rendered from statements by design, so there is no "this prompt is fine, skip it" path — the old prompt is raw material for the statement, nothing more. The table below is therefore a *pathology census* (how damaged is the existing material, what to watch for while re-authoring), **not** a work gate. The worklist verdicts are single-pass sonnet judgments: the mechanically checkable labels were verified against independent scans, the per-atom difficulty calls were not — treat them as triage hints, and re-judge each atom during authoring.

| corpus | total | trivial | prompt_rewrite | respec | redesign | kill |
|---|---|---|---|---|---|---|
| probmods_v2 | 76 | 11 | 48 | 14 | 3 | 0 |
| dippl | 17 | 0 | 14 | 1 | 1 | 1 |
| forestdb | 30 | 0 | 28 | 1 | 0 | 1 |
| pyro | 40 | 1 | 31 | 1 | 5 | 2 |

(The pyro row is mostly moot — that column is re-derived from problems in P4; its records contribute only their v2 linkage.)

Notables (full lists in the worklist JSONL):

- **underdetermined (6)**: conditioning/ex4.b (smile-combination rule unstated), agents-as-programs/ex1.b (factor magnitude), inference-algorithms/ex4.a (corpus documents elided by a `/* 6 short documents */` placeholder), generative-models/ex1.c (open-ended by design — candidate for a future modeling-judgment track, not this dataset), 2 pyro seeding artifacts.
- **hollow_witness (7)**: dippl-04-factorseq/atom-5, dippl-05-particlefilter/atom-4, forestdb-social-meaning/atom-1, 4 pyro.
- **near-duplicate**: forestdb-kids-scope atom-1/atom-2 produce identical probabilities (different support label) — merge candidate.

## File-level map

- **RIP**: `eval/metrics.py` + `eval/spec_metrics.py` (→ `eval/algebra.py`), `scripts/classify_atom_specs.py`, `scripts/validate_spec_metrics.py`, `data/atom_specs.jsonl`, `data/_unclassifiable.md`, `answer_shape` everywhere.
- **REWRITE — outcomes (P2)**: `eval/score.py` rewritten problem-centric; `eval/harness.py` and `scripts/cache_groundtruth_outputs.py` ripped instead of rewritten (the gate/score paths subsumed them); `eval/prompt.py` kept as system-prompt assembly with the renderer split into `eval/render.py`; `eval/io.py` trimmed in place; `scripts/assemble_curated.py` kept legacy (comparator rewired to algebra); `scripts/translate_to_pyro.py` kept legacy with a do-not-extend header (P4 replaces it); web: `atoms.ts`→`problems.ts`, `buckets.ts`→`tones.ts`, `AtomDetail.astro`→`ProblemDetail.astro`, `types.ts` deleted, `render.ts` reduced to code highlighting.
- **KEEP**: executors (modulo serializer updates + pyro seeding), `scripts/scrape_*.py`, web feedback API, `data/prompts/*.txt` (renderer inputs; they live only there — no copies under curated_v3), historical broken/emissions sidecars.
- **DECIDED (P2)**: `scripts/render_atoms_html.py` ripped; the web app is the sole renderer.

## Phases

- **P0 — DONE** (2026-06-09) — algebra + schema: `data/SCHEMA.md` + `eval/algebra.py` (96 tests; adversarially reviewed, 3 critical fixes incl. candidate self-noise in tolerance).
- **P1 — DONE** (2026-06-09) — all 123 WebPPL atoms re-authored → 115 problems in `data/problems/*.jsonl` + `data/realizations/webppl.jsonl`; 8 retired with rationale. Calibration round (10, full-read) then scale (113, self-checked + verified).
- **P3 — DONE, ran before P2** (2026-06-09/11) — `eval/gate.py`: phase A (multi-seed GT floors; 115/115 ok after 7 ill-posed fixed/retired) + phase B (2× solver re-derivation via `solve`/`judge`; `--model` flag for stronger-model gates). Final: **115/115 solver-verified** (superseded by the P2 v2 re-gate: 114 sonnet-gated, 1 opus-gated — see P2 and `_gate_triage.md`) — full history + evidence in `data/problems/_gate_triage.md`. Structural fixes along the way: label-schema spec extension, vocabulary/form pinning rules, primer dialect patch, one prior-vs-kernel transcription class, one textbook off-by-one GT correction (occams ex1.2/ex1.3, deviation documented in the realization code).
- **P2 — DONE** (2026-06-11) — harness collapse + web rewrite. `eval/corpus.py` (single dataset loader), `eval/score.py` (problem-centric: generations → judge-verdict rows), `eval/generate_batch.py` (problems in, batch out); ripped: harness.py, metrics.py, spec_metrics.py, answer_shape, atom-spec classifier/validator, GT cache script, legacy HTML renderer. Generality fixes folded in: finite specs declare `support` (label space; renderer enumerates it, canonicalizer rejects out-of-space mass as malformed — 39 problems declared, leak rule enforced: space from statement surface, never realized/reachability); gate `gt_suspect` uses `algebra.agreement()` (self-noise-derived tolerance, no fixed cutoff); every judge row stamped gate_model/timeout/n_solvers; phaseA merges by problem_id; solve --dry-run writes a .dry.json sidecar. Web: problem-centric `/p/<slug>` pages off problems/realizations/gate reports; legacy bucket vocabulary dead. Completed 2026-06-11: full re-gate under the new pipeline → report v2, 115/115 accept (112 first-pass sonnet; 2 statement fixes from convergent-miss investigations; 1 opus-gated row), uniformly stamped.
- **P4** — Pyro binding rebuild (seeding, pyro.infer serialization) + re-derivation of all probmods problems + fresh baseline.
- **P5** — Stan binding; corpus growth via translation sourcing.

## Open decisions

1. Noise-floor mechanics: the margin multiplier, and how many GT seeds the floor estimate needs (k=2 is itself noisy; analytic bounds on TV/W1 between two n-sample empiricals can tighten it).
2. Records: keep as product type (32 atoms) but forbid joint structured-empirical answers — confirm the respec policy.
3. ~~Statement format~~ — DECIDED (P1): structured 3-field prose.
4. ~~Retired atoms~~ — DECIDED (P1): parked in `_retired.jsonl`; loaders skip them.
5. ~~`render_atoms_html.py`~~ — DECIDED (P2): ripped; the web app is the sole renderer.
6. Open-ended modeling-judgment problems (e.g. generative-models/ex1.c): separate future track with different scoring — out of scope here.
