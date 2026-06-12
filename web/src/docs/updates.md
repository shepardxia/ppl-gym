**2026-06-11 — Pyro column complete.** All 115 problems re-derived in Pyro and
verified against the WebPPL ground truths by the cross-language gate (symmetric
measured tolerances). The campaign's mechanical audit rejected 44 first-draft
realizations whose numbers passed but whose code bypassed Pyro machinery — and
the gate caught (and fixed) an inference bias in one accepted WebPPL ground
truth that the single-language gate could not see.

**2026-06-11 — Gate report v2.** Full re-gate of all 115 problems under the
collapsed pipeline: 112/115 first-pass solver accepts (vs 81 in the first-ever
campaign run), the rest closed by evidence-driven statement fixes. Every report
row uniformly stamped with gate model and protocol.

**2026-06-11 — Harness collapse.** One dataset loader, one comparator, one
renderer; finite specs declare their label vocabularies (rendered into prompts,
enforced at canonicalization); the gate's solver-agreement test derives its
tolerance from measured self-noise — the last hand-set threshold removed.

**2026-06-10 — Solver-verification campaign closed at 115/115.** Five rounds
of gate → investigate → fix: ~20 statement fixes (label vocabularies,
prior-vs-kernel transcriptions, threshold-set definitions), one textbook bug
corrected with documented provenance deviation, two problems gated by a
stronger model.

**2026-06-09 — Problem-centric redesign.** The dataset moved from per-language
"atoms" to language-neutral problems with per-language realizations; the answer
algebra and measured-tolerance machinery landed; 123 legacy atoms re-authored
into 115 problems + 8 retired.
