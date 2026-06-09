# Forestdb Corpus: Overall Curation Notes

## Files Skipped Entirely

**example.md** — Single line `flip(.5)`. Trivially stochastic, no inference problem. Skip.

**habituals-cogsci2016.md** — Both blocks use the legacy `Enumerate(function(){...})` API and `gaussianERP.score([mu,sigma],b)` (also legacy). These are incompatible with the evaluation runtime. Would require rewriting from scratch. Skip.

**tug-o-war-explanations.md** — Uses `Enumerate(...)` (legacy API) and `vizPrint(...)` throughout all four blocks. Every meaningful block ends with `Enumerate(inference)`. Skip.

**plural-predication-webppl.md** — The file's own header says "It doesn't run." The first model creates zero-probability events in the distributive semantics. The revised model (block 3 with soft semantics) technically could run but the file documents that it predicts the wrong pattern — it is a debugging/research-notes page rather than a finished model. The `pre-v0.7` API flag is also a concern. Skip.

## Files Yielding Atoms

**actually-good.md** (1 atom): Single block, full RSA chain (listener0/speaker1/listener1/speaker2/listener2). `expectation(listener2(...))` is deterministic. Captures the "actually good" microaggression inference.

**keysar.md** (2 atoms): Three blocks, each self-contained. Block 0 = baseline model (alpha=3, uttCost = word count, L2 uses only shared context for S1). Block 1 = uncertainty model (alpha=4, fractional uttCost, S1 marginalizes over hidden distractor). These are meaningfully different models. Block 2 (deception model) skipped — the file says "more experimental" and the model is structurally more complex; better to keep the two cleaner models.

**incremental.md** (1 atom): Single block. All `incrementalUtteranceSpeaker(utt, state)` calls return deterministic scalars (products of exp(log-probabilities)). Nine values bundled into a record.

**codenames.md** (1 atom): Blocks 0–4 are incremental build-up with ///fold: duplications of the vectors object. Block 5 is the full self-contained model. Use block 5 only to avoid dedup issues.

**zhu-antonyms.md** (1 atom): Block 3 is the full model. Expectation of price under enumerate inference is deterministic.

**adjectives-qud.md** (1 atom): Block 4 is the complete QUD+adjectives model. Blocks 0–3 are build-up fragments. Block 5 is a reference model from Chapter 4. Block 6 is the "better model" from Chapter 5 — skipped because it introduces a much larger dataset (5 items × 40–80 bins each) and is not primarily the paper's main contribution. Block 4's QUDPrior is degenerate (always returns "what is the price?") — this is a simplification the authors chose.

**2025-problang-irony.md** (1 atom): 12 blocks. Blocks 0–9 are piece-by-piece build-up of the Kao & Goodman irony model (many re-declaring the same vars). Block 10 is the London/Canada variation with reversed statePrior — skipped to avoid redundancy with lai-irony. Block 11 is the Kao et al. hyperbole model — used as atom-1.

**lai-irony.md** (1 atom): 9 blocks, piece-by-piece build-up. Block 8 is the complete model. This is the canonical Kao & Goodman (2015) irony model with California prior. Preferred over the irony blocks in 2025-problang-irony to avoid redundancy.

**gl-polite-irony.md** (1 atom): 9 blocks. Block 7 is the full polite-irony extension with uniform state prior and utterance "okay". Block 8 uses a skewed prior (bad baker) — skipped to avoid a second near-identical atom.

## Patterns Observed

**Legacy API prevalence**: Three files (habituals, tug-o-war, plural-predication) use the pre-v0.7 `Enumerate(function(){...})` API exclusively. These must be skipped. Any future forestdb curation should screen for `Enumerate(` at the start.

**///fold: pattern**: Very common in forestdb, especially codenames (5 of 6 blocks have `///fold: vectors`). When a single block is used, fold content within that block is needed for execution. The assembler must NOT strip `///fold:` content from single-block atoms, or must at least preserve the content (strip only the pragma line itself). Two atoms (codenames block 5, gl-polite-irony block 7) have functional code inside fold sections that would be lost if the assembler strips them.

**`/// fold:` vs `///fold:`**: The 2025-problang-irony hyperbole block uses `/// fold:` (with a space). It's unclear whether the assembler's `///fold:` stripping regex catches this variant. Flag for manual check.

**Paper-page style vs tutorial style**: As expected, each file is a self-contained model. The right grain is one atom per file. Two exceptions: keysar.md yields 2 atoms (two distinct models with different parameters), and 2025-problang-irony was considered for 2 atoms but reduced to 1 to avoid irony-model redundancy with lai-irony.

**RSA model uniformity**: Most files are Kao-style RSA (L0/S1/L1 or L0/S1/L2) with enumerate inference. The models differ primarily in: (a) the meaning function (vector distance vs threshold semantics vs exact equality), (b) the state space (objects, prices, weather, cookies), (c) the presence of QUDs, valence, arousal.

**`marginalize` availability**: Several wrap_targets use `marginalize(dist, key)`. This helper is not always defined in the source block. For zhu-antonyms block 3 and gl-polite-irony block 7, marginalize is called but not defined in the block. In adjectives-qud block 4, it IS redefined at the top of the block. The eval runtime needs to provide marginalize globally, or the assembler needs to prepend it. This is the same issue as in probmods; the probmods-deps package likely provides it. Flag for verification.

**Dedup risks**: No atoms use multiple blocks from the same file except implicitly through fold. Single-block atoms avoid the typical dedup problem.

## Scale Pass Pilot Eval (sonnet-46-primer, 12 new atoms)

Pipeline yield from agent's 13 emissions across 56 viable files (43 skipped
under quality-over-quantity guidance):
- 12 emissions passed all three gates after minor fix-ups (3 had wrong
  block indices; 1 had a malformed wrap_target; 1 was dropped due to
  duplicate `var pl` declarations from overlapping source blocks in the
  source file itself).
- LM pilot eval on the 12 new production atoms:

| Atom | TV | Bucket |
|---|---|---|
| forestdb-dickson-speaker-cost/atom-1 | 0.0000 | clean |
| forestdb-generics/atom-1 | 0.0000 | clean |
| forestdb-generics/atom-2 | 0.0000 | clean |
| forestdb-kids-scope/atom-1 | 0.0000 | clean |
| forestdb-lxz-chinese-scope/atom-1 | 0.0000 | clean |
| forestdb-prior-inference/atom-1 | 0.0000 | clean |
| forestdb-scalar-implicature-qud/atom-1 | 0.0000 | clean |
| forestdb-singh-uyeda-pronouns/atom-1 | 0.0000 | clean |
| forestdb-schizophrenia-urns/atom-1 | 0.1174 | TV<0.2 |
| forestdb-social-meaning/atom-1 | 0.2000 | LM RSA-divergence |
| forestdb-overinf/atom-1 | 0.2069 | LM RSA-divergence |
| forestdb-blm/atom-1 | 0.5000 | LM RSA-divergence |

**8/12 = 67% at TV=0** (vs pilot's 6/10 = 60%). The 3 atoms at TV≥0.2 are
structurally sound (clean GTs, specific prompts) — the LM made real RSA
modeling mistakes on them. We leave them in production as legitimate
benchmark signal: distinguishing models on RSA semantics is valuable.

## Redundancy Re-Curation Pilot Eval (sonnet-46-primer, 12 new atoms)

After agent re-curated 15 files previously skipped only for "redundant
with existing atom," 12 emissions passed gates (3 failed: 2 jmss-irony
syntax errors at line 27 of assembled GT, 1 hii-generics undefined `bins`
from fold-strip).

| Atom | TV | Verdict |
|---|---|---|
| forestdb-cnqr-comparison-class/atom-1 | 0.0000 | clean |
| forestdb-hlms-comparison-class/atom-1 | 0.0000 | clean |
| forestdb-jmr-irony-extension/atom-1 | 0.0000 | clean |
| forestdb-kachakeche-comparison-class/atom-1 | 0.0000 | clean |
| forestdb-kids-scope/atom-2 (two-not scope) | 0.0000 | clean |
| forestdb-lxz-chinese-scope/atom-3 | 0.0000 | clean |
| forestdb-lxz-chinese-scope/atom-2 | 0.0379 | TV<0.05 |
| forestdb-lxz-chinese-scope/atom-4 | 0.0867 | TV<0.2 |
| forestdb-2025-problang-adjectives-qud/atom-1 | 0.2510 | LM mistake |
| forestdb-2025-problang-teasing/atom-1 | 0.4463 | LM mistake |
| forestdb-ccgn-metaphor/atom-1 | 1.0000 | LM mistake (metaphor RSA rule-4) |
| forestdb-astt-metaphor/atom-1 | EXEC ERR | LM produced malformed model |

**8/12 at TV<0.1 (67%), 9/12 at TV<0.2 (75%).** Comparable to the scale
pass's rates — the redundancy claim was wrong; these atoms ARE distinct
benchmark items and produce meaningful signal.

**Confirmed: Kao 2014 metaphor RSA is rule-4-prone.** Both ccgn and astt
metaphor atoms failed despite specific prompts. The threshold-prior +
feature-set design has enough freedom that constrained prompts still
can't pin the LM into the GT structure. Worth keeping as a known-hard
problem class.

## Final forestdb dataset (30 atoms)

Pilot atoms (6): keysar/atom-1, keysar/atom-2, codenames/atom-1,
zhu-antonyms/atom-1 (value), adjectives-qud/atom-1 (value), lai-irony/atom-1

Scale atoms (12): blm/atom-1, dickson-speaker-cost/atom-1, generics/atom-1,
generics/atom-2, kids-scope/atom-1, lxz-chinese-scope/atom-1, overinf/atom-1,
prior-inference/atom-1, scalar-implicature-qud/atom-1, schizophrenia-urns/atom-1,
singh-uyeda-pronouns/atom-1, social-meaning/atom-1

Shape distribution: 16 distribution-shape, 2 value-shape (both
expectations from the pilot, scored exact). The new brief discourages
value-shape; the scale pass produced 0 value atoms.

## Brief-update validation

The scale pass tested the updated brief ("quality over quantity",
"discourage value-shape", more aggressive skipping). Outcomes:
- Skip rate: 43/56 = 77% (vs pilot 3/13 = 23% — much more aggressive)
- 0 of 13 emissions chose value-shape (vs pilot's 3/10 = 30%)
- Rule-4 violation rate (TV≥0.2 on pilot eval): 3/12 = 25% (vs pilot's 4/10 = 40% — modestly improved)

The discipline shift was effective. Remaining RSA mistakes (blm, overinf,
social-meaning) are the irreducible long tail — LMs occasionally
miscompute Bayesian recursion regardless of how specific the prompt is.

## Original Pilot Eval Results (sonnet-46-primer, all 10 atoms)

| Atom | Result | Diagnosis |
|---|---|---|
| forestdb-keysar/atom-1 | TV=0.000 | clean |
| forestdb-keysar/atom-2 | TV=0.000 | clean |
| forestdb-codenames/atom-1 | TV=0.000 | clean |
| forestdb-zhu-antonyms/atom-1 | exact | clean |
| forestdb-adjectives-qud/atom-1 | exact | clean |
| forestdb-lai-irony/atom-1 | TV=0.000 | clean |
| forestdb-actually-good/atom-1 | EXEC ERR (60s timeout) | rule-4: LM's listener2 recursion deeper than GT's. GT block 0 output is also suspect — both utterances yield identical posterior, suggesting block 0 is pre-microaggression baseline, not the full model |
| forestdb-incremental/atom-1 | EXEC ERR (all paths zero) | rule-4: LM's tokenization/transition-table builds an unreachable inference target. RSA increment-by-prefix semantics has fragile design space |
| forestdb-2025-problang-irony/atom-1 | TV=0.610 | rule-4: LM's pragmaticListener returns price×valence pairs; GT returns price×valence×qud triples. Prompt didn't pin return-record shape |
| forestdb-gl-polite-irony/atom-1 | EXEC ERR (all paths zero) | rule-4: polite-irony RSA has tight constraint chains; small structural differences from GT produce infeasible models |

**6 atoms moved to production `forestdb.jsonl`; 4 moved to `_forestdb_broken.jsonl` for re-curation.**

## Rule-4 Prevalence by Corpus

- dippl (tutorials): 1/17 ≈ 6% rule-4 violation
- forestdb (paper-style RSA): 4/10 = 40% rule-4 violation

RSA models have high design-space dimensionality (literal listener shape, speaker marginalization, QUD definition, recursion depth), and the brief alone isn't enough for the agent to constrain the LM into the exact GT structure. For future forestdb (or problang) atoms, prompts may need to inline the full L0/S1/L1 scaffold and ask the LM to fill in specific pieces, rather than asking for the whole RSA chain from scratch.

## Assembly Concerns Summary

| Atom | Concern |
|---|---|
| codenames/atom-1 | Block 5 has ///fold: vectors — assembler must preserve fold content |
| zhu-antonyms/atom-1 | Block 3 calls marginalize — must be globally available |
| 2025-problang-irony/atom-1 | Block 11 has `/// fold:` (with space) containing valencePrior/meaning/qudFns |
| gl-polite-irony/atom-1 | Block 7 has `///` fold wrapping speaker1 — must be preserved |
| gl-polite-irony/atom-1 | Calls marginalize twice in wrap_target — must be globally available |

## Scale Pass (56 files)

### Files Assessed

All 56 files in `/tmp/forestdb_scale_files.txt` were read and assessed. 13 atoms were emitted to `_forestdb_scale_emissions.jsonl`. Details follow.

### Files Yielding Atoms

**schizophrenia-urns.md** (1 atom): Single block, v0.9.9. Social urn Bayesian inference: participant infers pRed from own binomial data + social signals modeled via hypergeometric likelihood. `Infer({method:'enumerate'})`. Distribution over pRed.

**scalar-implicature-qud.md** (1 atom): Single block. Scalar implicature with explicit QUD parameter. States [0,1,2,3], utterances ['all','some','none'], QUDs ['all?','any?']. `pragmaticListener('some','any?')`. Clean, fully enumerate.

**generics.md** (2 atoms): Block 3 = priorModel with DiscreteBeta mixture prior (potential, prevalenceWhenPresent, concentrationWhenPresent). Block 4 = literal listener with threshold uncertainty over discretized prevalence space. Both are enumerate-compatible.

**overinf.md** (1 atom): Block 1 = relaxed semantics overinformativeness model. Soft meaning function with `size_semvalue=0.8`, `color_semvalue=0.99`. Speaker distribution for `{size:'small',color:'blue'}`.

**prior-inference.md** (1 atom): Block 0 = vanilla RSA relativized to listener preferences. `preferenceTable` maps preference → salience weights over 3 objects. `pragmaticListener('square','blue_things')`.

**blm.md** (1 atom): Block 0 = simple RSA with {black,white} state space and ['blm','nblm'] utterances. Uniform state prior. Clean and fully enumerate.

**social-meaning.md** (1 atom): Single block = Burnett (2019) personae-based sociolinguistic RSA. Conditionalization / speaker / valueSpeaker / valueInformedListener / naiveListener. `conditionalization('ng')` = distribution over personae.

**dickson-speaker-cost.md** (1 atom): Block 6 = full model where pragmatic listener jointly infers `{obj, costParameter}`. costParameterPrior = uniformDraw(_.range(0.05,5,0.5)). `pragmaticListener('blue')`.

**kids-scope.md** (1 atom): Block 0 = every-not scope ambiguity model. States [0,1,2], scopes surface/inverse, QUDs, pragmatic speaker. `pragmaticSpeaker(1)`.

**singh-uyeda-pronouns.md** (1 atom): Block 3 = pragmatic listener for pronoun resolution. Utterances ['him','Fred','John'] with weights [2,1,1]. Strategies ['Subject','Parallel']. `pragmaticListener('him')`.

**bkmt-scalar-implicature.md** (1 atom): Block 9 = full joint inference model with hypergeometric likelihood over speaker access and observed apples. `marginalize(pragmaticListener('some'),'state')`. Depends on `marginalize` being globally available.

**lxz-chinese-scope.md** (1 atom): Block 10 = English Experiment 1 full model. States [0,1,2], 'not-two' utterance, scopes, QUDs. `pragmaticSpeaker(1)` = endorsement of ambiguous utterance for surface-scope-true state.

### Files Skipped

**adj-order-appendix.md**: Parameter exploration (epsBins, subjCheck) with viz.scatter/viz.table output only. No clean inference target.

**astt-metaphor.md**: Student re-implementation of Kao (2014) metaphor RSA — identical structure to `ccgn-metaphor.md` and `2025-problang-metaphor.md`. Redundant with metaphor atoms; all three student groups wrote the same model. SKIP.

**bkmt-scalar-implicature.md** blocks 0–8: Build-up fragments. Block 9 used.

**ccgn-metaphor.md**: Third student re-implementation of same Kao metaphor model. Redundant. SKIP.

**cnqr-comparison-class.md**: Student comparison-class extension (Tessler framework). Redundant with `adjectives-qud` atom already in production. SKIP.

**cushman-generics.md**: Generics model extension with L2 speaker (speaker2). Too deep RSA. SKIP.

**dickson-speaker-cost.md** blocks 0–5, 7: Build-up fragments and 2-object variant. Block 6 used.

**elephants.md / elephants_continuized.md**: Formal semantics continuation-based model. Complex grammar machinery (kindChecker, featureChecker). Not standard RSA, assembler-hostile structure. SKIP.

**false-cognates.md**: Bigram string-similarity function inside `translate()` is fragile and hard to pin precisely. SKIP.

**generic-id.md**: Hidden, v0.9.7. Generic vs specific sense model with complex binomial context. SKIP.

**generics-conjunction.md**: Hidden, v0.9.7. Double-threshold conjunction inference. Unusual model structure; uncertain enumerate-compatibility. SKIP.

**generics-intergenerational.md**: Hidden, v0.9.6. Multi-generation learner simulation. Time-series structure, MCMC-flavored. SKIP.

**gonzalez-zhang-irony.md**: Irony replication + S2 speaker. Redundant with `lai-irony` atom; deeper RSA. SKIP.

**hii-generics.md**: Same threshold-prior generics extension as `cushman-generics`. print-based output only. SKIP.

**hlms-comparison-class.md**: Student comparison-class (Tessler framework). Redundant. SKIP.

**jmr-irony-extension.md**: Irony extension with 5 states and continuous arousal bins. Redundant with `lai-irony`. SKIP.

**jmss-irony.md**: Full irony model replication (identical structure to `lai-irony`). SKIP.

**kachakeche-comparison-class.md**: Student comparison-class (Tessler framework) for "heavy" adjective. Redundant. SKIP.

**kids-scope.md** block 1: `two-not` model. Near-identical structure to block 0 (every-not). One atom is sufficient to represent this model family.

**liquid_physics.md**: Pre-v0.7, requires WebGL LiquidFun library. SKIP.

**luong-extensional-generics.md**: Uses `beta()` primitive directly (MCMC continuous), stochastic world generation. SKIP.

**lxz-chinese-scope.md** blocks 11+: English Expt 2 and Mandarin models. Near-identical structure to block 10. One atom sufficient.

**multi-agent-lda.md**: Pre-v0.7, uses legacy `discrete()`. SKIP.

**negatron.md**: RSA speaker with negation, complex utterance structure, no clean single distribution wrap_target. SKIP.

**new-webppl.md**: Hidden, v0.9.6, MCMC-only linear regression. SKIP.

**overinf.md** block 0: Vanilla boolean semantics model. Superceded by block 1 (relaxed semantics) which is the paper's main contribution.

**PLU-2019-projects.md**: Multi-author student RSA models; `cost[utterance]` access issue (cost is a function not an object in some sections), fragile design space. SKIP.

**plural-predication.md**: File's own header says "It doesn't run." SKIP (from pilot notes).

**politeness.md**: L0→S1→L1→S2 RSA, too deep. SKIP.

**politeness-qud.md**: Politeness + QUD + phi bins. Too many degrees of freedom. SKIP.

**progressive-shift.md**: Powerset state space × 2 thresholds. Extremely large, MCMC-needed. SKIP.

**pronouns.md**: QUDfun samples from categorical inside the function (stochastic design choice). Rule-4 risk. SKIP.

**questions-answers.md**: KL-optimized Q&A model, multiple complex blocks. Too many design DOFs. SKIP.

**schizophrenia-urns.md**: Full atom. Done.

**singh-uyeda-pronouns.md** blocks 4+: Noise extension adding erf function. More complex, lower rule-4 safety. Block 3 used.

**social-meaning.md**: Full atom. Done.

**spanish-gender.md**: Generics extension with two asymmetric thresholds (thresholdM, thresholdF). Complex free parameters. SKIP.

**spector-rsa.md**: Uses `dp.cache` (unclear if available in runtime). Homogeneity in plural definites — complex recursive utility with QUDs. Too risky. SKIP.

**torabian-politeness-QUDs.md**: Politeness + arousal + QUD. Too many stacked degrees of freedom. SKIP.

**upadhye-aspect.md**: Event interpretation with object-specific Beta params. params object is complex; unclear enumerate-compatibility; likely MCMC. SKIP.

**2025-problang-adjectives-qud.md**: Student project with commented-out parameters. Redundant with `adjectives-qud` atom. SKIP.

**2025-problang-bilingualism.md**: Shirley Temple ambiguity with noise parameters not deterministically pinnable. SKIP.

**2025-problang-comparison-class.md**: Student comparison-class tutorial. Redundant. SKIP.

**2025-problang-irony.md**: Already assessed in pilot. SKIP (pilot notes cover it).

**2025-problang-metaphor.md**: Build-up style only; no single self-contained block with all vars defined. The three student-group metaphor files (astt, ccgn, 2025-problang) all implement the same Kao (2014) model. SKIP all as redundant.

**2025-problang-politeness.md**: File too large to read (>188k tokens). Likely student politeness tutorial. SKIP.

**2025-problang-teasing.md**: Irony extension redundant with `lai-irony`. SKIP.

### Patterns Observed

**Metaphor model saturation**: Three student groups (astt, ccgn, 2025-problang) each re-implemented the exact same Kao (2014) whale/person metaphor RSA model with identical featureSetPrior weights. None yields an atom distinct from the others. The canonical `lai-irony` approach (use the cleanest single-block version) would apply, but all three are student tutorial write-ups with multi-block build-up — none has a single self-contained complete block. Skip all three.

**Comparison class saturation**: Five files (cnqr, hlms, kachakeche, cushman, hii) extend the same Tessler comparison-class / generics framework. The canonical version is already captured by `adjectives-qud` in the pilot. Skip all extensions.

**Irony model saturation**: Four files (jmr, jmss, gonzalez-zhang, 2025-problang-teasing) re-implement or extend the Kao irony model already covered by `lai-irony`. Skip all.

**bkmt-scalar-implicature**: The joint inference model (access+observed+state) is non-trivially different from vanilla RSA. Uses hypergeometric distribution — the most technically sophisticated model in this batch. Moderate rule-4 risk since the return shape {state,access,observed} must be pinned. Mitigated by using `marginalize(...,'state')` in the wrap_target and pinning the field name in the prompt.

**`marginalize` dependency**: `bkmt-scalar-implicature/atom-1` requires `marginalize` globally. Same dependency as `zhu-antonyms/atom-1` and others from the pilot. The probmods-deps runtime provides it.

**lxz-chinese-scope**: Unlike `kids-scope` (which is from problang), this is a student project applying scope ambiguity to Mandarin/English cross-linguistic differences. The English Experiment 1 model is structurally identical to `kids-scope/atom-1` but with a 'not-two' utterance instead of 'every-not'. Two scope atoms in the benchmark provides useful coverage diversity.

### Assembly Concerns

| Atom | Concern |
|---|---|
| schizophrenia-urns/atom-1 | wrap_target inlines the full trialModel — no LM naming issues |
| generics/atom-1 | Block 3 has ///fold: with DiscreteBeta — assembler must preserve fold |
| generics/atom-2 | Block 4 has ///fold: with DiscreteBeta and priorModel — assembler must preserve fold |
| dickson-speaker-cost/atom-1 | Block 6 — verify this is the 3-object model, not the 2-object variant (block 7) |
| bkmt-scalar-implicature/atom-1 | Uses marginalize — must be globally available |
| singh-uyeda-pronouns/atom-1 | Block 3 has ///fold: with literalListener+speaker — assembler must preserve fold |

## Redundancy Re-Curation Pass (15 files)

The scale pass incorrectly flagged 15 files as "redundant with existing atom" — a skip reason that the updated agent brief now explicitly disallows (same model family from a different paper/student group is NOT a valid skip). This pass re-examines all 15 to extract atoms or identify a legitimate skip reason.

Source files: astt-metaphor.md, ccgn-metaphor.md, 2025-problang-metaphor.md, cnqr-comparison-class.md, hlms-comparison-class.md, kachakeche-comparison-class.md, 2025-problang-comparison-class.md, jmss-irony.md, jmr-irony-extension.md, 2025-problang-teasing.md, gonzalez-zhang-irony.md, hii-generics.md, 2025-problang-adjectives-qud.md, kids-scope.md (block 1), lxz-chinese-scope.md (blocks 11–13).

### Files Yielding Atoms

**astt-metaphor.md** (1 atom): 10 blocks. Block 9 = full complete model — pragmaticListener("whale") and pragmaticListener("person") both called. goalPrior weights [1,1,1] (uniform). Atom-1: block [9], wrap_target `pragmaticListener("whale")`, returns {category, large, graceful, majestic}. Different from scale-pass atoms because the metaphor RSA is a structurally distinct model family (feature-based categorical semantics) not represented elsewhere in the benchmark.

**ccgn-metaphor.md** (1 atom): 9 blocks. Block 7 = full complete model with goalPrior [5,1,1] (biased toward "large"). Meaningfully different prior from astt-metaphor (uniform goalPrior). Atom-1: block [7], wrap_target `pragmaticListener("whale")`, returns {category, large, graceful, majestic}.

**cnqr-comparison-class.md** (1 atom): 19 blocks. Block 13 = first complete pragmaticListener block. Uses Tessler comparison-class framework with subParams["basketballPlayers"]. Atom-1: block [13], wrap_target `marginalize(pragmaticListener("tall", subParams["basketballPlayers"]), "comparisonClass")`. Different from adjectives-qud (which uses price/valence QUDs); the comparison-class framework integrates a subordinate-vs-superordinate prior, a distinct model structure.

**hlms-comparison-class.md** (1 atom): 11 blocks. Block 10 = "Final model" section, same Tessler comparison-class structure as cnqr. Atom-1: block [10], wrap_target `marginalize(pragmaticListener("tall", subParams["basketballPlayers"]), "comparisonClass")`. Structurally parallel to cnqr, providing a second implementation of the same framework; valid for benchmark coverage.

**kachakeche-comparison-class.md** (1 atom): 4 blocks. Block 3 = final complete model with "heavy"/"light" adjectives and child/adult/bodybuilder comparison classes. Soft meaning function via flip(). thresholdHeavy and thresholdLight are separate. comparisonClassPrior conditioned on `whoSaidIt`. Atom-1: block [3], wrap_target `pragmaticListener("heavy", "child")`, returns distribution over state (box weight in {10, 20, 30, 40, 50}).

**jmss-irony.md** (2 atoms): 14 blocks. Block 12 = Full Model with arousal. Block 13 = Arousal Removed variant. Two structurally distinct atoms from the same file. Atom-1: block [12], wrap_target `pragmaticListener("terrible")`, returns {state, valence, arousal}. Atom-2: block [13], same wrap_target, returns {state, valence}. The arousal dimension adds non-trivial structure difference.

**jmr-irony-extension.md** (1 atom): 6 blocks. Blocks 0–4 = comparative snippets referencing earlier models; Block 5 = first self-contained complete model with 5 states (1–5 weather scale), continuous arousal bins [0.1,0.3,0.5,0.7,0.9], statePrior=[1,5,40,40,40] (skewed toward bad weather). Atom-1: block [5], wrap_target `pragmaticListener("terrible")`, returns {state, valence, arousal}.

**2025-problang-teasing.md** (1 atom): 8 blocks. Block 7 = "Final compiled teasing model" with lambda=-1.25, alpha=10, phi drawn from uniformDraw(_.range(0.05,0.95,0.05)), literalSemantics lookup table, antisocial utility combining epistemic and antisocial components. Atom-1: block [7], wrap_target `pragmaticListener("dumb as rocks")`, returns {state, phi, goal, valence, arousal}.

**hii-generics.md** (1 atom): 6 four-tilde blocks (file uses mixed 3-tilde/4-tilde delimiters; assembler uses only 4-tilde). Block 5 = full model with sigPrior=categorical([5,10,5],...) over three threshold-prior types and uniform prevalence prior. `var prior = priorModel({potential:1, prevalenceWhenPresent:0.5, concentrationWhenPresent:2})` is defined within block 5. Atom-1: block [5], wrap_target `marginalize(pragmaticListener("generic", prior), "prevalence")`.

**2025-problang-adjectives-qud.md** (1 atom): 14 blocks. Block 13 = "Full adjectives + QUD model" with prices=[50,500,1000,5000,10000], utterances=["expensive","notExpensive"], cost(notExpensive)=1, alpha=1, qudPrior uniform over ["price","valence","priceValence"]. Atom-1: block [13], wrap_target `pragmaticListener("expensive")`, returns {price, valence}.

**kids-scope.md** (1 atom): Block 1 = "two-not" model (numHorses=2, meaning: surface scope → state==0, inverse scope → state<numHorses). Distinct from block 0 (every-not) which is already in production as kids-scope/atom-1. Atom-2 (continuing from atom-1): block [1], wrap_target `pragmaticSpeaker(1)`, returns distribution over ["null","two-not"].

**lxz-chinese-scope.md** (3 atoms): Block 10 already in production as atom-1. Three additional blocks:
- Block 11 = English Expt 2 (4-object, utterancePrior ps:[1,10], meaning: surface→state<2, inverse→state<3). Atom-2: block [11], wrap_target `pragmaticSpeaker(2)`.
- Block 12 = Chinese Expt 1 (2-object, utterances include "none", scopePrior [100,1], meaning: none→state==0, not-two inverse→state==0). Atom-3: block [12], wrap_target `pragmaticSpeaker(1)`.
- Block 13 = Chinese Expt 2 (4-object, utterances include "none", scopePrior [100,1], meaning: not-two inverse→state<3). Atom-4: block [13], wrap_target `pragmaticSpeaker(2)`.

### Files Skipped (Legitimate Reasons)

**2025-problang-metaphor.md** — Build-up tutorial style throughout all blocks; no single self-contained block with all dependent variables defined. No single-block atom is extractable. Legitimate skip: no self-contained complete block.

**gonzalez-zhang-irony.md** — Implements a sarcasm-extension RSA with speaker2va → pragmaticListener → speaker1 → literalListener call chain. This is L2 RSA recursion (the pragmaticListener calls speaker1, which calls literalListener). Too-deep RSA recursion is a legitimate skip signal per the brief. Legitimate skip: L2+ RSA depth.

**2025-problang-comparison-class.md** — Tutorial build-up style. Each block redefines the same variables with incremental additions; no single block is self-contained without the prior blocks. Legitimate skip: no self-contained complete block.

### Summary

| File | Decision | Atoms | Block(s) | Legitimate skip reason |
|------|----------|-------|----------|----------------------|
| astt-metaphor.md | EMIT | 1 | [9] | — |
| ccgn-metaphor.md | EMIT | 1 | [7] | — |
| 2025-problang-metaphor.md | SKIP | 0 | — | No self-contained complete block (build-up style only) |
| cnqr-comparison-class.md | EMIT | 1 | [13] | — |
| hlms-comparison-class.md | EMIT | 1 | [10] | — |
| kachakeche-comparison-class.md | EMIT | 1 | [3] | — |
| 2025-problang-comparison-class.md | SKIP | 0 | — | No self-contained complete block (build-up style only) |
| jmss-irony.md | EMIT | 2 | [12],[13] | — |
| jmr-irony-extension.md | EMIT | 1 | [5] | — |
| 2025-problang-teasing.md | EMIT | 1 | [7] | — |
| gonzalez-zhang-irony.md | SKIP | 0 | — | L2+ RSA depth (sarcasm extension introduces speaker2va → pragmaticListener recursion) |
| hii-generics.md | EMIT | 1 | [5] | — |
| 2025-problang-adjectives-qud.md | EMIT | 1 | [13] | — |
| kids-scope.md (block 1) | EMIT | 1 | [1] | — |
| lxz-chinese-scope.md (blocks 11–13) | EMIT | 3 | [11],[12],[13] | — |

**Yield: 15 atoms from 12 files; 3 files legitimately skipped.** Emissions written to `_forestdb_redo_emissions.jsonl`.
