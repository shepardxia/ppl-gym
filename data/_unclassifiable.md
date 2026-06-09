# Atom Spec Calibration — Review List

Generated from `scripts/classify_atom_specs.py`.

**Total atoms across collections**: 163.
**Items flagged for human review**: 39.

Each item is either:
- `role=unknown` — the script could not assign a role.
- has a `review_note` — assigned but with a flagged concern (metric placeholder, curation-time misclassification, ambiguous trajectory vs samples, etc.).

## Role distribution (after classification)

| Role | Count |
|---|---|
| `distribution` | 114 |
| `record.distribution` | 69 |
| `samples` | 8 |
| `deterministic` | 7 |
| `summary` | 3 |
| `record.samples` | 3 |
| `record.deterministic` | 2 |
| `record.summary` | 2 |
| `trajectory` | 1 |

## Review items, grouped by issue

### empirical (>500) support  (9)

- `probmods2-bayesian-data-analysis/ex1.2` (support=2498, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `probmods2-mixture-models/ex1.b` (support=537, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `probmods2-occams-razor/ex2.1` (support=7978, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `probmods2-occams-razor/ex2.2` (support=9277, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `probmods2-hierarchical-models/ex3.2` (support=4098, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `probmods2-inference-algorithms/ex1.1` (support=1910, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `probmods2-inference-algorithms/ex2.2` (support=2882, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `probmods2-inference-algorithms/ex2.4` (support=1000, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder
- `forestdb-2025-problang-teasing/atom-1` (support=864, dtype=structured, role=distribution)  — empirical (>500) support — TV with support alignment is inappropriate; wasserstein is placeholder

### numeric vector classified as deterministic  (7)

- `probmods2-learning-as-conditional-inference/ex1.1` (dtype=float, role=deterministic)  — numeric vector classified as deterministic — verify GT code is not stochastic
- `probmods2-learning-as-conditional-inference/ex2.2` (dtype=float, role=deterministic)  — numeric vector classified as deterministic — verify GT code is not stochastic
- `probmods2-occams-razor/ex1.3` (dtype=float, role=deterministic)  — numeric vector classified as deterministic — verify GT code is not stochastic
- `probmods2-occams-razor/ex2.3.cpValues` (dtype=float, role=deterministic)  — numeric vector classified as deterministic — verify GT code is not stochastic
- `probmods2-occams-razor/ex2.3.csValues` (dtype=float, role=deterministic)  — numeric vector classified as deterministic — verify GT code is not stochastic
- `probmods2-inference-algorithms/ex2.3` (dtype=float, role=deterministic)  — numeric vector classified as deterministic — verify GT code is not stochastic
- `pyro-occams-razor/ex1.3` (dtype=float, role=deterministic)  — numeric vector classified as deterministic — verify GT code is not stochastic

### per-run output is fixed-length list  (6)

- `probmods2-generative-models/ex2.b` (role=samples)  — per-run output is fixed-length list — ambiguous between vector-valued sample and repeated trajectory
- `probmods2-generative-models/ex2.c` (role=samples)  — per-run output is fixed-length list — ambiguous between vector-valued sample and repeated trajectory
- `probmods2-generative-models/ex7.a` (role=samples)  — per-run output is fixed-length list — ambiguous between vector-valued sample and repeated trajectory
- `pyro-generative-models/ex2.b` (role=samples)  — per-run output is fixed-length list — ambiguous between vector-valued sample and repeated trajectory
- `pyro-generative-models/ex2.c` (role=samples)  — per-run output is fixed-length list — ambiguous between vector-valued sample and repeated trajectory
- `pyro-generative-models/ex7.a` (role=samples)  — per-run output is fixed-length list — ambiguous between vector-valued sample and repeated trajectory

### scalar float  (5)

- `probmods2-conditioning/ex1.a` (dtype=float, role=summary)  — scalar float — could be a deterministic closed-form value or a posterior summary; default tolerance assumes summary
- `forestdb-adjectives-qud/atom-1` (dtype=float, role=summary)  — scalar float — could be a deterministic closed-form value or a posterior summary; default tolerance assumes summary
- `forestdb-zhu-antonyms/atom-1.expensivePrice` (dtype=float, role=summary)  — scalar float — could be a deterministic closed-form value or a posterior summary; default tolerance assumes summary
- `forestdb-zhu-antonyms/atom-1.notInexpensivePrice` (dtype=float, role=summary)  — scalar float — could be a deterministic closed-form value or a posterior summary; default tolerance assumes summary
- `pyro-conditioning/ex1.a` (dtype=float, role=summary)  — scalar float — could be a deterministic closed-form value or a posterior summary; default tolerance assumes summary

### large-support float-valued distribution  (5)

- `probmods2-learning-as-conditional-inference/ex2.1.post` (support=499, dtype=float, role=distribution)  — large-support float-valued distribution — likely MCMC posterior; ks_marginal is a placeholder metric
- `probmods2-hierarchical-models/ex3.1` (support=947, dtype=float, role=distribution)  — large-support float-valued distribution — likely MCMC posterior; ks_marginal is a placeholder metric
- `probmods2-agents-as-programs/ex2.d` (support=23926, dtype=float, role=distribution)  — large-support float-valued distribution — likely MCMC posterior; ks_marginal is a placeholder metric
- `probmods2-inference-algorithms/ex2.1.point2` (support=2430, dtype=float, role=distribution)  — large-support float-valued distribution — likely MCMC posterior; ks_marginal is a placeholder metric
- `probmods2-inference-algorithms/ex2.1.interpolationWeight` (support=895, dtype=float, role=distribution)  — large-support float-valued distribution — likely MCMC posterior; ks_marginal is a placeholder metric

### parametric continuous distribution  (1)

- `probmods2-learning-as-conditional-inference/ex2.1.prior` (dtype=float, role=distribution)  — parametric continuous distribution — comparator needs family-aware param match; cross-PPL param-name remap required

### large enumerated support (257)  (1)

- `probmods2-mixture-models/ex1.a` (support=257, dtype=structured, role=distribution)  — large enumerated support (257) — TV with permissive alignment; verify metric choice

### large enumerated support (92)  (1)

- `probmods2-mixture-models/ex2.a` (support=92, dtype=structured, role=distribution)  — large enumerated support (92) — TV with permissive alignment; verify metric choice

### per-run outputs are variable-length lists  (1)

- `probmods2-observing-sequences/ex2.c` (role=trajectory)  — per-run outputs are variable-length lists — likely trajectory; metric is placeholder

### large enumerated support (405)  (1)

- `probmods2-inference-algorithms/ex1.2` (support=405, dtype=structured, role=distribution)  — large enumerated support (405) — TV with permissive alignment; verify metric choice

### large enumerated support (57)  (1)

- `probmods2-inference-algorithms/ex1.3` (support=57, dtype=structured, role=distribution)  — large enumerated support (57) — TV with permissive alignment; verify metric choice

### large enumerated support (216)  (1)

- `probmods2-inference-algorithms/ex2.5` (support=216, dtype=structured, role=distribution)  — large enumerated support (216) — TV with permissive alignment; verify metric choice

