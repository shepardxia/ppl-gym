# Archive — redesign (P1/P2) campaign artifacts

Working files from closed campaigns, kept for provenance only. Nothing here is
read by code or load-bearing for the pipeline.

- `_calib_*.jsonl`, `_scale_*.jsonl` — P1 authoring-agent emissions, already
  merged into the canonical `data/problems/*.jsonl` + `data/realizations/`.
  Do not re-merge: the canonical files have since accumulated gate-round fixes.
- `_spec_drafts.jsonl` — P1 bootstrap spec drafts from legacy GT outputs.
- `redesign_worklist.jsonl` — pre-P1 audit verdicts over the 163 legacy atoms.
- `_gate_solve_batch_{opus,r5_opus,r5_sonnet}.json`, `_gate_solver_report_opus.jsonl`
  — manifests/partial reports from closed gate rounds (results merged into
  `data/problems/_gate_solver_report.jsonl`; history in `_gate_triage.md`).
- `_triage_solver_code/` — solver programs fetched for the round-3/5 gate
  investigations.
- `atoms_v2.html` — output of the deleted `scripts/render_atoms_html.py`.
