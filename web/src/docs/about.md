**ppl-gym** is a benchmark dataset of probabilistic-programming *problems*:
language-neutral statements with per-language, machine-verified ground-truth
solutions. It exists to evaluate how well language models write probabilistic
programs — especially in low-resource PPLs — and to provide clean ground truths
for constrained-generation research.

The unit of the dataset is a **problem**, not a program. Each problem is a prose
statement in three fields — *given* (every parameter, prior, and observation),
*model* (the generative story), *query* (the quantity requested) — plus a typed
`answer_spec` describing the mathematical object the answer is. The statement
must pin the answer without pinning the program: any competent solver, in any
PPL, reading only the statement, should converge to the same answer.

That property is not assumed; it is **gated**. Every problem's ground truth is
verified two independent ways:

- **Solver re-derivation**: independent LLM solvers, reading only the rendered
  statement, must reproduce the ground-truth answer within measured tolerance.
- **Cross-language consistency**: each new language column's realization must
  agree with the reference column's ground truth, with tolerances derived from
  both sides' measured sampling noise.

Current state: **115 problems** sourced from
[probmods](http://probmods.org), [dippl](http://dippl.org), and
[forestdb](http://forestdb.org), with two verified realization columns —
WebPPL (the reference) and Pyro — and a Stan column planned. The
[problem browser](/) shows every statement, realization, and gate verdict;
collaborators can leave per-problem feedback directly on the pages.

The project is part of the CHI-PPL effort; the corpus and harness live at
[shepardxia/ppl-gym](https://github.com/shepardxia/ppl-gym).
