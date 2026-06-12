The full contract lives in
[`data/SCHEMA.md`](https://github.com/shepardxia/ppl-gym/blob/main/data/SCHEMA.md);
this page is the short version of the four ideas that carry the design.

## The determination criterion

A problem's statement must *determine the answer* — and must *not* determine
the program. Priors, observed data, and the queried quantity are required;
function names, code structure, and inference method are banned (inference
details are allowed only in the query, and only when the answer is itself a
realized empirical posterior, where method and sample counts pin what "the
answer" means).

This is enforced by machinery, not by eyeballing: if independent solvers can't
re-derive the answer from the statement alone, the problem doesn't ship.

## One answer algebra

Every comparison in the pipeline goes through a single comparator
(`eval/algebra.py`). An answer is a mathematical object — a value, a
distribution, or a record of these — over domains `bool | finite | int | real |
realvec`. How a program *represents* that object (exact value, enumerated
distribution, parametric family, sample cloud, or a plain outcome→probability
mapping) is orthogonal: comparison is defined on the object, between any pair
of representations. Distributions over finite domains compare by total
variation; numeric domains by Wasserstein-1; parametric families are
canonicalized through one alias table (so Pyro's `Beta(concentration1,
concentration0)` and WebPPL's `Beta(a, b)` are the same object).

## Measured tolerance — no hand-set thresholds

Nothing in the pipeline carries an authored numeric threshold. A problem's
tolerance is derived from its own measured noise: the ground truth runs at k
independent seeds, the **noise floor** is the maximum pairwise distance among
those runs, and a candidate passes iff its distance is within
`margin × max(GT floor, candidate self-noise)` (plus a numerical epsilon).
Deterministic problems therefore demand near-exact answers; stochastic
problems get exactly as much slack as their own sampling noise justifies. A
floor too large to discriminate answers flags the problem itself as ill-posed.

## Two gates

**Solver re-derivation** (per language): independent LLM solvers get the
rendered statement and nothing else. If a solver matches the ground truth, the
problem is *accepted*. If solvers agree with each other but not the ground
truth, the ground truth is *suspect* — every such case is investigated to a
measured root cause (these investigations have caught statement ambiguities,
unpinned label vocabularies, and two genuine textbook bugs). If solvers
scatter, the statement is underdetermined.

**Cross-language consistency** (per new column): the new language's
realization runs at k seeds, the reference's at k seeds, and the two must agree
within `margin × max(both floors)`. Two languages agreeing is a stronger check
than two solvers agreeing — solvers share the ground truth's blind spots. This
gate caught a real bias in an accepted WebPPL ground truth (an under-burned
MCMC chain) that the solver gate was structurally unable to see, because the
solvers ran the same biased inference.

Acceptance into a language column also requires the code to express its model
through the language's own machinery (e.g. `pyro.sample` / `pyro.infer`),
audited mechanically — matching numbers alone don't qualify a realization.

## Provenance

Ground-truth code that matches its textbook source is authoritative;
statements are the rewritable layer. Overriding a source requires evidence the
source is internally inconsistent, and the deviation is documented in the
realization code itself. Retired problems are parked with reasons, never
deleted.
