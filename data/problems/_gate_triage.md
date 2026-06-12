# Gate phase-B triage

## P4 cross-language campaign (Pyro) — CLOSED 2026-06-11: 115/115 pass

Full independent crosscheck (`eval.gate crosscheck --language pyro`): every Pyro
realization agrees with the WebPPL GT within symmetric measured tolerances
(report: `_gate_crosscheck_report.jsonl`). Campaign shape: 3 hand-written pilots
→ 2 authoring waves (6 agents × ~10, self-verified via crosscheck_problem) →
mechanical audit → 1 rework wave. Notable:

- **Audit caught 44 means-violations**: realizations whose numbers passed but
  whose code bypassed pyro machinery (pure-Python enumeration) or built
  serializer wire dicts in-program — the failure mode that killed the old
  pyro_v3 corpus, reproduced by agents gaming their own pass criterion. All
  reworked with the rule enforced mechanically (regex on pyro.sample/factor/
  infer/plate; ban on support/probs dict construction).
- **mixture-models/ex2.a: cross-language gate caught a real WebPPL GT bias** —
  10k unburned MH samples put g1_p at 0.980–0.986 across seeds; converged value
  is 0.991 (Pyro NUTS agreed). Invisible to the single-language solver gate
  (solvers shared the GT's inference). WebPPL realization upgraded to
  50k+10k-burn (floor 0.0044), both sides now pass at d=0.0022.
- **Crosscheck made symmetric mid-campaign**: tolerance from
  margin × max(target floor, reference floor) — a noisy WebPPL reference
  (observing-sequences/ex1.c, floor 0.145) no longer fails a tight Pyro target.
- The label "null" gotcha: mapping-form dict keys are JSON-parsed, so the
  STRING label "null" must be emitted as json.dumps("null"); one realization
  fixed for this.


**Canonical tally: 115 accept / 115 — report v2 (re-gate under the collapsed P2
pipeline, 2026-06-11).** Every row uniformly stamped (gate_model, timeout=60,
n_solvers=2). 114 sonnet-gated; 1 opus-gated (inference-algorithms/ex1.3 —
sonnet keeps choosing a soft-Gaussian score for a hard constraint).

## Report v2 (final): full re-gate under the new pipeline
Fresh prompts (39 `support` declarations + primer fix) via batch
msgbatch_01VVNvo4XhmePKXNBGWXn6o1 (230 sonnet requests): **112/115 first-pass
accept** (vs 81 in the original campaign's first run — the statement fixes,
vocabulary declarations, and primer repair account for the difference; the
formerly opus-only occams ex1.2/ex1.3 now pass sonnet). The 3 stragglers:

- **inference-algorithms/ex1.3** — opus-gated as designed (opus 2/2 within the
  floor-derived tolerance; batch msgbatch_01XEJzCrJCLhEyKwXJu8fCrv).
- **observing-sequences/ex3.b** — convergent miss (sonnet+opus×2 all at
  d=0.515, answering `chases: 1.0`). Investigation: GT (matching its textbook
  source) generates a FRESH second sentence after conditioning — the grammar
  has no shared parameters, so the observation teaches nothing and ~50/50 is
  the chapter's intended lesson, not a GT bug (investigator's gt_bug/retire
  verdict overridden on provenance + pedagogy). The defect was the query's
  ambiguous noun phrases ("a newly generated sentence" vs "a generated
  sentence"); query now spells out the two-sentence structure. Re-gate: accept.
- **2025-problang-teasing/atom-1** — opus×2 convergent at d=0.389 from
  misreading "the expected value of the state" as the value function applied to
  the TRUE state (verified: switching to the L0-expectation with the identity
  value function → 9.5e-16). Statement now pins whose expectation and the
  unrescaled value function. Re-gate: **2/2 exact (d=0.0)**.

Round-2 batch msgbatch_01XUJD3wuRknnaY9wciNsw1r. Previous campaign history below
(report v1, superseded).

## Round 5 (final): the last 5, resolved by opus re-gate + evidence-mandatory investigations
Opus re-gate of the 5 open problems (batch msgbatch_012UL7zkZtecVuYcYDCsDJFZ) flipped
kachakeche (opus d=0.0 exact) and inference-algorithms/ex1.3 (2/2 within floor-derived
tolerance) immediately, and supplied convergent-answer evidence for the rest. Four
investigations (one per remaining problem) closed every divergence with
predicted≈measured experiments, all re-verified by hand:

- **kachakeche** — statement gap: "meaning threshold for state x is x−0.1" read as a
  meaning function rather than two offset threshold *sets* to sample from; all 4 sonnet
  solver runs shared the misread. Fix: given defines both 30-element sets; model pins
  sampling + pass-through. Modified solver → 1.2e-15.
- **hlms** — statement gap: missing "the same threshold pair is passed to the speaker and
  through to the literal listener"; solvers let L0 marginalize its own threshold (also the
  cause of the 121 s timeout: 18×18×18 internal enumeration). Plus unpinned label
  vocabulary (opus matched GT probs to 1e-13 but answered 'sub'/'super'). Fix: pass-through
  sentence + query pins 'superordinate'/'subordinate'. Modified solver → 1.2e-15.
- **occams ex1.2** — entire d=0.089 was one label string: GT 'multiples_of_1' vs solver
  'multiples_1' (query pinned interval labels but not rule labels). Fix: query pins all
  rule-label formats. Label rename alone → d=0.0.
- **occams ex1.3** — **GT bug (first GT semantic edit of the campaign)**: the textbook
  starter code's ranges put 0 into every multiples concept and dropped 20 from evens/odds,
  contradicting the statement's "integers 1 through 20" AND the source's own interval
  hypotheses ([1,20]) — an internal inconsistency of the source, not a modeling choice, so
  the provenance rule (GT-matches-source ⇒ authoritative) does not protect it. Fixed ranges
  in BOTH ex1.2 and ex1.3 realizations (shared helpers; ex1.2's answer verified unchanged —
  only multiples_of_1 survives [3,10] and it has size 20 under both readings). Corrected GT
  matches both opus solvers to 1.7e-16. Deviation marked by a comment in the realization
  code — do not "restore" it to source. Statements also pin the powers-of-N convention
  (exponents start at 0, so 1 ∈ every powers concept).

Re-gate (sonnet batch msgbatch_01DVMKfThm9ZrxnUbhB6XZAK + opus batch
msgbatch_01R3cJpjoDB5jB878zzm8Qs7): every accept at d=0.0 exact — kachakeche 4/4
generations exact across both models, hlms sonnet 2/2 exact, ex1.2 sonnet 2/2 exact
(sonnet's old cpsInnerStatement failure gone), ex1.3 opus 2/2 exact. No memorization
flags. Phase A re-passed on both edited realizations (floors 0.0).

## Round 4: ex1.a/ex1.c Dirichlet fix, codenames pair-format, ex1.b inference pin
All four flipped to accept. ex1.a/ex1.c root cause (found analytically, gate-confirmed):
the statement transcribed `dirichletDrift`'s `concentration: 10` (an MCMC kernel parameter)
as the Dirichlet *prior* concentration; true prior is alpha=1. Predicted TV 0.118, measured
0.120.

## Round 3 (2026-06-10): full 27-problem investigation → fixes → re-gate
All 27 non-accepts root-caused by investigator agents (evidence-mandatory), verified, fixed
(14 statement/spec fixes incl. 3 overturned gt_bug verdicts — provenance keeps GT, statement
describes it; consolidated primer patch for WebPPL-dialect crash modes), re-gated with fresh
generations: 18 flipped to accept. Batch msgbatch_015mfMAsEVgpPpkpUE3iqWmE.

Remaining 9, with next-step notes:
- observing-sequences/ex1.a (d≈0.12), ex1.c (d≈0.3) — solvers agree with each other across
  TWO rounds; primer-MCMC explanation now disproven. Something structural in statement vs GT
  (memoized-randomness MCMC). PRIORITY for close reading.
- codenames/atom-1 — still d=[1.0,1.0]: likely residual answer-form mismatch; compare
  solver vs GT supports directly.
- mixture-models/ex1.b (d=0.057), kachakeche (d=0.044), hlms (d=0.045 + one timeout),
  occams ex1.3 (d≈0.12, now runs post-primer) — near-misses with solver agreement; either
  residual statement gaps or tolerance-margin questions.
- occams ex1.2 (cpsInnerStatement persists), inference-algorithms ex1.3 (solvers keep
  choosing soft-Gaussian for a hard constraint) — hard-for-sonnet problems; consider a
  stronger gate model or accept as ungated-hard.

## First full run (historical)

Batch `msgbatch_01NM44SLE1Z91waWGYs1Skyt`, 2× sonnet-4-6 solvers per problem.
Result: **81 accept / 17 gt_suspect / 12 underdetermined / 5 solver_failure** (115 gated).
Per-problem rows: `_gate_solver_report.jsonl`. Solver code retrievable from the batch via
`eval.generate_batch.collect_results` (keys `<problem_id with / -> __, . -> _dot_>__s{0,1}`).

## Root-cause classes (next session's worklist)

### A. Label-schema contract gap — RESOLVED (11/13 recovered)
Fix landed in three layers: (1) spec `labels` declaration for record-shaped finite labels
(SCHEMA.md + algebra validation + renderer contract sentence) — 10 problems authored;
(2) query-pinned label forms for string/list-shaped labels (ex5.b 'h'/'t', ex7.a/ex7.b
two-element lists); (3) query-pinned string *vocabularies* one level deeper (lai-irony goal
labels, occams ex1.2 hypothesis-id format). Re-gated with fresh generations: 11 accept.
Reassigned with evidence: ccgn-metaphor → class B (solver probs genuinely differ, solvers
agree with each other at d=0.286); occams ex1.2 → class C (both solvers cpsInnerStatement);
adjectives-qud stays class D. Batches: msgbatch_017iD8gNb4bDTw1dm8L4zYnp (13),
msgbatch_017EzG6aRgUr56GiPkKodWn6 (2). Original analysis below kept for the record.

### A-original. Label-schema contract gap — CONFIRMED, structural, fixes ~7 at once
`d=[1.0,1.0]` gt_suspects: ex4.b verified — solver probs identical to GT to the last digit;
TV=1 purely because the solver named a record-label field `sneezes` vs GT `sneeze`. The spec
cannot express structured-label naming (`finite` is opaque), so the rendered contract can't
state it, so solvers guess.
**Fix (right altitude):** extend answer_spec for finite domains with structured labels —
optional `label_schema` (field names + atomic domains); renderer emits the names in the
contract paragraph; canonicalize validates labels against it. SCHEMA.md + algebra + render +
respec the affected specs. Suspects in this class (verify each before assuming):
generative-models/ex4.b, ex4.c, ex5.b, ex7.a, ex7.b; forestdb-dickson-speaker-cost/atom-1;
forestdb-lai-irony/atom-1.

### B. Near-miss at zero floor — per-problem statement audits
Both solvers agree exactly with each other, tiny-but-nonzero d vs exact GT (tol=eps at floor 0):
agents-as-programs/ex3 (d=7e-4), learning-as-cond-inf/ex2.1 (d=0.0036), kids-scope/atom-1
(d=0.005), cnqr-comparison-class (d=0.019). Likely statement↔GT constant mismatches or a
legitimately ambiguous parameter — each needs a number-fidelity read (statement vs GT code).
Larger same-shape cases: social-cognition/ex2.5 (d=1/6), observing-sequences/ex1.a (0.11),
ex1.c (0.33), overinf (0.21), agents-as-programs/ex2.b/ex2.e (d≈2.7 absdiff) — these may be
genuine statement defects or GT bugs; solvers agreeing exactly with each other is strong
evidence the statement reads coherently but pins something different from the GT.

### C. Solver WebPPL-dialect failures — primer/contract gaps, not statement bugs
Both-crashed (solver_failure): occams-razor/ex1.2+ex1.3 (`cpsInnerStatement` — likely `while`
loops), inference-algorithms/ex1.3 (`score argument is not a number`), kachakeche +
zhu-antonyms (`Categorical ps should be vector`). One-crashed (lands in underdetermined):
adjectives-qud, ccgn-metaphor, codenames (`address.split`), keysar/atom-2 (`mean` undefined),
schizophrenia-urns, observing-sequences/ex3.a (`_.range is not a function` — check shim load).
**These say "sonnet can't write this WebPPL reliably," not "the problem is broken."** Options:
primer additions (vector/ps, no while-loops, no lodash in model code), k>2 solvers, or a
stronger gate model for these. Do NOT weaken problems to accommodate solver limitations.

### D. True underdetermination candidates — read closely
2025-problang-adjectives-qud (d=[0.11, 0.03] — solvers disagree with each other),
2025-problang-teasing (0.13/0.20), mixture-models/ex1.b (0.58/0.64 — label-switching
convention may be insufficiently pinned), hlms-comparison-class, agents-as-programs/ex4.b.

### E. hierarchical-models/ex3.2 (d≈16-22 on W1) — solvers agree with each other; check
whether GT's respec'd marginal (`diff`) matches what the statement asks.

## Process notes
- n_pass>=1 counts as accept: 81 includes some 1/2 passes — the report has n_pass per row;
  1/2-pass problems deserve a skim too.
- memorization_suspect flags: none raised at jaccard>0.6.
- After class-A fix lands, re-judge ONLY affected problems (the batch results are cached;
  re-running judge re-executes solver code locally without a new LLM batch).
