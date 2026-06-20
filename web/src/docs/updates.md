**2026-06-20 — posteriordb corpus online (Stan).** A second corpus of 45 Bayesian-inference problems from posteriordb — applied regression, hierarchical, time-series, GP and ODE models — each realized in Stan and cross-checked against posteriordb's gold reference posterior draws (45/45 pass). The browser renders them in the same two-pane explorer: the reference column holds the stored gold posterior, the Stan column the program that reproduces it, and the answer overlay shows each parameter's posterior marginal on shared bins. One posterior (arma11) was pruned as non-discriminable, with documented evidence.

**2026-06-16 — Pyro column rebuilt to idiomatic Pyro.** Every realization now expresses its model through Pyro's own inference — enumeration, MCMC, importance sampling, `infer_discrete` — and is both cross-verified against WebPPL and audited for proper library use. 111/115 realize this way; the 4 inference-algorithms hard-condition method demos are marked Pyro-unavailable with documented reasons, a new per-language availability state the browser now shows.

**2026-06-11 — Pyro column complete.** 115/115 cross-verified against WebPPL; the gate also caught and fixed a biased, previously-accepted WebPPL ground truth.

**2026-06-11 — Gate report v2.** Full re-gate under the rebuilt pipeline: 112/115 first-pass solver accepts (vs 81 in the first campaign), rest closed by evidence-driven statement fixes.

**2026-06-11 — Harness collapse.** One loader, one comparator, one renderer; finite label vocabularies declared and enforced; last hand-set threshold removed.

**2026-06-10 — Solver campaign closed at 115/115.** Five rounds of gate → investigate → fix; ~20 statement fixes, one documented textbook-bug correction.

**2026-06-09 — Problem-centric redesign.** 123 per-language "atoms" re-authored into 115 language-neutral problems + measured-tolerance answer algebra.
