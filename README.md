# ppl-gym (CHI PPL Hub)

A benchmark dataset of probabilistic-programming **problems**: language-neutral
statements with per-language ground-truth realizations, for evaluating LLMs on
probabilistic programming.

- **Dataset**: `data/problems/{probmods2,dippl,forestdb}.jsonl` — 115 problems,
  each a `(given, model, query)` statement plus an authored `answer_spec`;
  WebPPL realizations in `data/realizations/webppl.jsonl` (1:1, all
  solver-verified by the re-derivation gate). Pyro and Stan columns are planned;
  memo and pluck with the language creators (stub directories: `webppl/`,
  `memo/`, `pluck/`).
- **Contract**: `data/SCHEMA.md` — problem/realization records, the answer
  algebra (one comparator for all answer types and representations), measured
  tolerances, and the gate protocol. Design history: `data/REDESIGN.md`.
- **Pipeline**: `eval/` — render prompts, generate solutions via the Anthropic
  batch API, execute, and judge (`eval/algebra.py` is the single comparator).
  Working notes for agents/contributors: `CLAUDE.md`.
- **Web app**: `web/` — problem browser + collaborator review/feedback, live at
  <https://pplgym.kingdomofends.org>.
