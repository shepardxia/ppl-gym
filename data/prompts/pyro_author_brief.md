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
- Litmus test: if deleting the inference call would leave the answer already
  computed, the realization is hand-rolled — the model has to do the work.

## Inference methods

Pick the one that reproduces the reference.

- **Exact discrete enumeration.** Use this when the WebPPL reference uses
  `Infer({method: 'enumerate'})`, or when the posterior turns on rare or extreme
  probabilities that sampling cannot recover. Decorate the model with
  `@pyro.infer.config_enumerate`; condition with
  `pyro.factor("ev", torch.where(event, torch.tensor(0.0), torch.tensor(float("-inf"))))`
  (or an `obs=` indicator); then

      marg = pyro.infer.TraceEnum_ELBO(max_plate_nesting=P).compute_marginals(model, lambda: None)

  returns the exact marginal `marg["site"]` (a `Distribution`) over each latent
  sample site. `P` is the number of nested `pyro.plate`s (0 if the model uses none).
  Build the answer from the marginal(s) you need.

- **Importance sampling** (discrete, when no probabilities are extreme):
  `post = pyro.infer.Importance(model, num_samples=N).run()`, then
  `pyro.infer.EmpiricalMarginal(post)`. Use enough samples that the estimate is stable.

- **MCMC** (continuous latents):
  `mcmc = pyro.infer.MCMC(pyro.infer.NUTS(model), num_samples=N, warmup_steps=W)`,
  `mcmc.run(...)`, then `mcmc.get_samples()["site"]`. Use `pyro.infer.HMC(...)` when
  the problem calls for specific HMC parameters.

- **Closed form.** When the posterior is conjugate or otherwise exact, construct the
  resulting `dist.<Family>(...)` directly. This is a real closed form, not a table.

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
string).

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
