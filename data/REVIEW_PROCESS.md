# Dataset review process

When changes are made to problem statements, GT realizations, or pipeline
logic, run a review pass before declaring the round complete. This file
documents that process so it's not reinvented each iteration.

(Written in the atom era; the principles carry over unchanged to the
problem-centric dataset. Mechanics updated for the current pipeline.)

## Why automated cleanup isn't enough

Source-derived text pulls prose verbatim from textbook markdown. Pattern-based
sanitizers fix specific known artifacts (Liquid templates, Pandoc citations,
image data, etc.) but miss issues that require *judgment*:

- prose that asks for "the result shown above" but the figure isn't visible
- the spec says "dist" but the prose actually describes a single sample
- the GT's ANSWER expression doesn't match what the prose asks for
- prose only makes sense in the context of an earlier chapter not in the statement
- GT computes one specific tuple but prose is general ("show how the speaker behaves")

These need a reader, not a regex.

## Review rounds

Each modification round (statement change, GT change, harness change) followed by:

1. **Rebuild the web app** (`cd web && npm run build`) so the modified state
   is browsable, or render prompts directly via `eval.render.render_problem`.
2. **Subagent review pass** (manual `Agent` calls, one per corpus, run
   in parallel). Reviews **every problem** in the corpus — no sampling.
   Pin `model: "sonnet"` to keep cost reasonable; reviewers don't need
   Opus-grade reasoning to read statements and flag issues.
3. **Findings file** (JSONL) with one row per flagged problem:
   `{problem_id, category, severity, finding, suggested_action}`.
4. **Triage**: read findings, decide which are real issues, which are
   false positives, which to defer. Decisions logged with evidence
   (the gate campaign used `data/problems/_gate_triage.md`).
5. **Apply fixes** that are clear enough to be safe; defer judgment calls
   to the user.

## Finding categories

Reviewers should classify each issue:

- `prose-broken` — statement references content not in the prompt (figures,
  values, prior context the LLM can't see)
- `prose-vague` — statement is genuinely underspecified about what to compute
- `spec-mismatch` — declared `answer_spec` doesn't match the GT's
  actual answer or the statement's intent
- `gt-mismatch` — GT code's `ANSWER` doesn't match what the statement asks for
- `template-leak` — markdown / template / citation syntax leaked through
- `dead-reference` — statement mentions a function name (e.g. `CRPmem`) that
  isn't defined in our WebPPL distribution
- `other` — anything else worth flagging

Severity:
- `block` — problem is unscoreable in current state
- `warn` — problem has a quality issue but might still produce a useful score
- `info` — minor, FYI

## Anti-patterns to avoid

- **Don't sample.** "Review 25 of 119" is the lazy default we're trying
  to escape. The whole point is full coverage; if it's too slow,
  parallelize across more agents instead of cutting the count.
- **Don't inherit Opus.** Subagents inherit the parent model unless
  pinned. For routine reading-and-flagging, set `model: "sonnet"`.
- **Don't act on subagent findings without reading them.** They are
  recommendations, not commands. Each finding is one judgment call by one
  reader; some will be wrong.
- **Don't fix everything at once.** Fix one category at a time and
  re-review; otherwise it's hard to attribute changes.
- **Don't regenerate every prompt.** Track which problems actually changed
  and re-generate only those (`--ids` on `eval.generate_batch` / `eval.gate solve`).
