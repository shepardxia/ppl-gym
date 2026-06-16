# Authoring a Pyro ground-truth realization

You are writing the ground-truth Pyro realization for one probabilistic-programming
problem. You are given the problem statement, the answer specification, and a
verified WebPPL realization of the same model. Write a Pyro program that realizes
the same model through Pyro's own inference machinery and binds the result to a
top-level `ANSWER`.

## The model and the answer

- The **statement** (`given` / `model` / `query`) defines the problem. The `query`
  says exactly what the answer is.
- The **WebPPL realization** is the verified reference. It pins the exact prior
  parameters, the exact conditioning, and the queried quantity. Translate this
  model into Pyro faithfully: the same random choices, the same parameters, the
  same conditioning, the same query.

Match the model. Do not change its parameters, its structure, or what it conditions on.

## What "idiomatic Pyro" requires

- Every random choice is a `pyro.sample("name", dist.<Family>(...))` site, each
  with a unique name (index the name inside loops, e.g. `f"x{i}"`).
- Conditioning lives in the model: observe data with `obs=`, or add evidence and
  log-weights with `pyro.factor`.
- The answer is produced by running Pyro inference over that model, or by a genuine
  closed form — never by computing the answer's probabilities yourself.

This is the whole point of the realization, so the failure mode is explicit:

- Do not enumerate the outcomes and multiply or normalize probabilities in Python.
- Do not precompute a probability table and feed it to `dist.Categorical` (or to
  `pyro.factor`) as a thin layer over a model that is not really doing the inference.
- Do not write your own inference engine — a custom enumeration loop, a hand-built
  trace-scorer, a bespoke distribution class, anything that reaches into
  `pyro.poutine.runtime` — even when the model itself uses `pyro.sample`. Call
  Pyro's own inference (`Importance`, `MCMC`, `config_enumerate` + `TraceEnum_ELBO`);
  reimplementing inference by hand is hand-rolling.
- Litmus test: if deleting the inference call would leave the answer already
  computed, the realization is hand-rolled — the model has to do the work.

## Inference methods

Pick the one that reproduces the reference.

Ground truth is collected by running your program under several random seeds, so
inference must be accurate without being wastefully heavy — both extremes fail.
Too few samples is noisy and misses the tolerance (or comes out ill-posed); tens
of thousands of samples times out. Prefer exact enumeration for discrete models —
it is exact and needs no samples. When you sample, choose a count sensible for the
model; the range is given with each method below.

- **Exact discrete enumeration.** Use this when the WebPPL reference uses
  `Infer({method: 'enumerate'})`, or when the posterior turns on rare or extreme
  probabilities that sampling cannot recover. Decorate the model with
  `@pyro.infer.config_enumerate`; condition with
  `pyro.factor("ev", torch.where(event, torch.tensor(0.0), torch.tensor(float("-inf"))))`
  (or an `obs=` indicator); then

      marg = pyro.infer.TraceEnum_ELBO(max_plate_nesting=P).compute_marginals(model, lambda: None)

  returns the exact marginal `marg["site"]` (a `Distribution`) over each latent
  sample site. `P` is the number of nested `pyro.plate`s (0 if the model uses none).
  Read a marginal's probabilities with
  `sup = marg["site"].enumerate_support()` then
  `probs = marg["site"].log_prob(sup).exp()` (not `.probs[i]`, whose shape under
  enumeration is not what you expect). Under `@pyro.infer.config_enumerate` every
  sampled value is a tensor spanning the enumeration dimensions, so combine choices
  with tensor operations (`&`, `|`, `~`, `torch.where`) — a Python `if` or `>` on
  such a tensor raises. `compute_marginals` gives only per-site marginals; when the
  query is a *joint* posterior over several discrete latents (a tuple-valued
  support), use `pyro.infer.infer_discrete(pyro.infer.config_enumerate(model),
  first_available_dim=-1)` to draw enumeration-based posterior samples of the
  latents, then aggregate the sampled tuples into the distribution. For NESTED
  inference (RSA: literal listener → speaker → pragmatic listener), Pyro does not
  support running one inference inside another's active trace — reusing a site name
  across levels throws "Multiple sample sites named ...". Compute each level's
  distribution separately and completely (memoize it), give every sample site a name
  unique across levels, and feed a finished level's distribution into the next as
  fixed scores; never call `compute_marginals`/`Importance` for an inner level while
  an outer enumeration is running.

- **Importance sampling** (discrete conditioning, when no probabilities are
  extreme): `post = pyro.infer.Importance(model, num_samples=N).run()`, then
  `pyro.infer.EmpiricalMarginal(post)` (or aggregate the weighted traces). Use
  ~1000–10000 samples — more when the conditioned event is less probable; if it is
  genuinely rare, enumerate instead, since Importance cannot recover it at any
  feasible count. It is a poor fit for continuous posteriors — use MCMC there.

- **MCMC** (continuous latents): `mcmc = pyro.infer.MCMC(pyro.infer.NUTS(model),
  num_samples=N, warmup_steps=W)`, `mcmc.run(...)`, then
  `mcmc.get_samples()["site"]`. Use `pyro.infer.HMC(...)` when the problem calls for
  specific HMC parameters. Use ~500–2000 samples with ~200–1000 warmup steps — the
  high end for higher-dimensional or strongly correlated posteriors, the low end for
  simple ones. NUTS and HMC require continuous latents; a model whose
  latents are discrete must use exact enumeration or Importance instead. For a model
  mixing discrete and continuous latents, marginalize the discrete ones with
  `@pyro.infer.config_enumerate` so MCMC samples only the continuous part. NUTS and
  HMC also need a differentiable, finite log-density: a hard condition (an equality,
  or a thin acceptance band like `abs(f(x)) < eps`) or a non-differentiable term
  (e.g. a cube root at 0) gives no usable gradient and the chain cannot initialize —
  sample those with Importance and a hard-band `pyro.factor` (reject off-band)
  instead, raising the sample count for a thin band.

- **Closed form** applies only when the *queried quantity itself* is a prior or
  forward distribution, or a deterministic value — construct it directly. A
  **posterior must come from running inference** over a `pyro.sample` model, even
  when the model is conjugate: do not write the conjugate-updated distribution (e.g.
  `dist.Beta(a + successes, b + failures)`, or a normalized Dirichlet
  posterior-predictive) by hand — that is computing the answer yourself, not
  inferring it. Sample the latent, observe the data, and run enumeration or MCMC.

## Binding `ANSWER`

Match the answer specification's `kind`:

- `value`: bind the number, bool, string, or list directly.
- `dist`: bind a Pyro/torch `Distribution` (an `EmpiricalMarginal` or a
  `compute_marginals` result counts), a list or 1-D tensor of posterior samples, or
  a dict mapping each outcome to its probability (`{True: 0.6, False: 0.4}`; tuple
  keys for sequence-valued outcomes).
- `record`: bind a dict with exactly the named `fields`, each value bound per its
  own kind.
- If the spec includes `"protocol": "draws"`: the answer is one draw. Bind `ANSWER`
  to a single sampled realization and run no inference — the harness runs the program
  under many seeds and aggregates the draws.

When the spec lists exact outcome labels (`support` or `labels`), use those exact
values as the outcomes (a boolean label is the Python `bool`; a string label is that
string). When `labels` makes each outcome a record of named fields, the outcomes
are those records: bind a dict keyed by `json.dumps(record, sort_keys=True)` mapping
to each probability (a bare tuple or list outcome is rejected — it must be the
named-field object).

## The environment

- `pyro`, `pyro.distributions as dist`, `pyro.infer`, `pyro.poutine`, `torch`,
  `math`, `random`, `itertools`, `functools`, `defaultdict`, and `Counter` are
  already available. Write no import statements.
- The harness sets the random seed; do not set it yourself.
- Tensors are float64 by default.
- Only the top-level `ANSWER` is read from the program.

## Scope

Write only the realization code for the one problem. Do not execute it, test it, or
add print or assert statements — every realization is verified afterward against the
reference. Put your effort into getting the model and the inference right.
