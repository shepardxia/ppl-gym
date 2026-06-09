# Pyro Translation of v2 ProbMods Atoms

## Summary

Translated 40/76 v2 ProbMods exercises (`data/atomized_v2.jsonl`) to Pyro
via `scripts/translate_to_pyro.py`. The remaining 36 atoms hit one of five
failure modes documented below.

- **Production**: `data/pyro_v3/probmods.jsonl` (40 atoms)
- **Broken**: `data/pyro_v3/_probmods_broken.jsonl` (36 atoms, with diagnostics)

## How translation works

Each WebPPL atom is sent to Claude Sonnet 4.6 along with its prompt, GT code,
GT output, and the Pyro primer. The model emits a JSON block containing:

- A rewritten prompt instructing an LM to produce a Pyro program ending in
  `ANSWER = <expression>`.
- Pyro GT code that produces the same answer as the WebPPL GT.

The translator script then:

1. Executes the Pyro GT via `eval.executor_pyro`.
2. Compares the Pyro output to the WebPPL GT output using shape-aware
   semantic equivalence (TV ≤ 0.05 for distributions, empirical TV ≤ 0.15
   for samples, numeric tolerance for value-shape lists, exact-match for
   discrete value scalars).
3. Atoms that pass become production; the rest land in the broken file
   with a categorized failure reason.

## Translation infrastructure

New code added to support Pyro alongside WebPPL:

- `eval/executor_pyro.py` — Pyro subprocess executor. Wraps user code with
  a serializer header that emits the cross-PPL JSON schema (`{__kind:
  "distribution", probs, support}` for discrete; `{__kind:
  "distribution_continuous", repr: ...}` for continuous; tensors get their
  own tag).
- `data/prompts/pyro_primer.txt` — Pyro analog of `webppl_primer.txt`.
- `eval/prompt.py` — `system_prompt(language=...)` and `format_messages`
  dispatch on `atom.language`. WebPPL atoms (no language field) unchanged.
- `eval/harness.py` — `_execute_for(atom)` routes to the right executor.
  `_run_mc` accepts an `executor` parameter.
- `scripts/translate_to_pyro.py` — Batch translation script.

## Failure categories (36 broken)

| Category | Count | Cause | Fixability |
|---|---|---|---|
| LM ran out of output budget (max_tokens) | 11 | Complex MCMC mixtures and hierarchical models require >16k tokens for the LM to emit a complete translation. | Possible with Opus-level models or by chunking the translation; not attempted in this pass. |
| MCMC support size mismatch (high-cardinality posteriors) | 9 | WebPPL's MCMC posteriors deduplicate samples into supports of 500-25000 unique points. Pyro's NUTS at modest sample counts produces a different support size. The supports themselves often only weakly overlap because both are continuous-flavored discrete posteriors. | Fundamentally hard cross-PPL; would require either matched sample counts or a TV metric that doesn't require support alignment. |
| Pyro syntax errors from LM | 5 | LM-generated code uses an undefined `pyro.sample` site name, or passes a non-scalar tensor where a scalar is expected. Translator bug. | Fixable with targeted retries that include the executor error in the prompt; not attempted. |
| Distribution TV > 0.05 (real divergence) | 5 | Translator's Pyro implementation diverges from the WebPPL GT — either different inference method (Pyro's NUTS converges differently than WebPPL's MH) or subtle structural differences. | Some fixable by manual review; others reflect genuine cross-PPL semantic gaps. |
| Value-shape numeric list differs | 4 | Computations that should produce closed-form analytical results but the LM's translation has slightly different numerics (Beta posterior parameters off-by-one, etc.). | Fixable with manual review. |
| Other (samples too small, samples TV high) | 2 | Tiny edge cases (atom with N=10 samples, atom with non-trivial mixture sampling). | Marginal. |

## What's in the production 40

By v2 source category (rough; some atoms span multiple):

- conditioning: 10/13
- agents-as-programs: 7/9
- social-cognition: 6/6
- generative-models: 6/9
- conditional-dependence: 2/2 (record-shape)
- learning-as-conditional-inference: 0/4 (all failed numerically)
- mixture-models: 0/4 (all MCMC-derived posteriors)
- hierarchical-models: 2/6
- observing-sequences: 2/8
- occams-razor: 2/4
- bayesian-data-analysis: 0/1
- inference-algorithms: 0/8 (all MCMC/SMC structural mismatches)

## Translation prompt highlights (from `scripts/translate_to_pyro.py`)

The system prompt instructs:

1. **No prose** — emit JSON only. Verbosity is a failure mode under
   max_tokens.
2. **Imports pre-injected** — don't re-import `pyro`, `torch`, `dist`.
3. **Manual enumeration over MCMC** — when the latent space is finite,
   enumerate analytically rather than running NUTS.
4. **Cross-PPL distribution schema** — for non-integer-support
   distributions, emit a literal `{"__kind": "distribution", ...}` dict
   instead of trying to coerce a Pyro `Categorical`.
5. **Modest sample counts for MCMC** — when MCMC is genuinely required,
   500 samples + 200 warmup, not 5000 + 500.
6. **Match WebPPL Bernoulli support representation** — Pyro returns
   `1.0/0.0` for Bernoulli; my comparator normalizes bool↔int.

## Cost

Three Anthropic Batch API calls (50% discount):
- Pilot batch (10 atoms): ~$0.03
- Pilot retry (4 atoms): ~$0.02
- Main batch (65 atoms): ~$0.30
- Main retry (20 atoms): ~$0.15

Total: ~$0.50.

## Validation

GT-vs-self eval (each atom's Pyro GT scored against itself) passes for
all 40 production atoms with TV=0.

## LM Eval Results (sonnet-46-primer)

| Bucket | Count | % |
|---|---|---|
| TV=0 | 13 | 33% |
| TV<.05 | 15 | 38% |
| TV<.2 | 2 | 5% |
| TV≥.2 | 2 | 5% |
| samples (TV<.2) | 3 | 8% |
| samples_off | 1 | 3% |
| exact (value) | 1 | 3% |
| val- (value mismatch) | 1 | 3% |
| EXEC_ERR (timeout) | 2 | 5% |

**Strong passes (TV<.05 + exact): 29/40 = 73%**
**Loose passes (TV<.2 + samples_ok): 34/40 = 85%**

The two EXEC_ERR atoms (`occams-razor/ex1.2`, `observing-sequences/ex3.a`)
are LM-generated translations that take >60s — the GT itself runs in <1s.
These would pass with a higher timeout or a more concise model.

The two TV≥.2 atoms (`hierarchical-models/ex2.3`, `observing-sequences/ex3.b`)
are genuine LM mistakes — the model produced a structurally different
inference than the GT.

## Comparator changes made during this work

`eval/metrics.py:_looks_like_distribution` — accepts any dict with `probs`
and `support` keys as a distribution, regardless of the `__kind` tag.
This handles Pyro LMs that emit `__kind: 'Categorical'` or
`__kind: 'joint_distribution'` instead of the canonical `'distribution'`.
The canonical schema is still preferred; this is permissive recovery for
LM noise.
