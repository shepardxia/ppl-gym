"""Single source for classifying a candidate/GT failure into a coarse, stable tag.

Both the production scorer (eval.score, at scoring time) and the legacy triage
tool (eval.triage_exec_errors, re-executing already-collapsed matrix rows) call
``classify`` on the SAME real error string, so a fresh run and a re-executed
legacy row land in the same bucket. The tag is intentionally coarse and stable;
the full reason lives beside it in the row's ``error`` field for drill-down.

Precondition: the real reason must reach here. The executors surface each failed
seed's ``error_message`` (aligned errors channel), and the harness raises with it
instead of a generic count — so ``classify`` sees the actual cause, not the old
collapsed "execution failed" / "n/k seeded runs failed" placeholder.
"""

from __future__ import annotations

import re

# Stable vocabulary. Coarse on purpose: a finer taxonomy over three languages'
# free-text errors is brittle; the exact reason is kept in the row's `error`.
TAGS = (
    "timeout",      # ran out of budget (candidate or, via gt_side, the GT)
    "compile",      # parse/semantic failure before execution (Stan compile, Python SyntaxError)
    "no_output",    # program ran but produced no ANSWER / empty / non-JSON output
    "runtime",      # code executed and raised a real exception (the common candidate crash)
    "gt_side",      # OUR ground-truth collection failed, not the candidate
    "empty_code",   # the model returned no code
    "corpus_miss",  # problem/realization not found (harness/data issue)
    "other",        # unclassified — inspect the reason string
)

# The pre-fix collapsed placeholders. If one of these still reaches classify, the
# real reason was lost upstream (legacy row) — tag it `other` so it is visible as
# "needs re-exec" rather than silently bucketed.
_COLLAPSED = re.compile(r"^(execution failed|\d+/\d+ seeded runs failed)$", re.I)


def is_collapsed(error: str | None) -> bool:
    """True if ``error`` is a pre-fix placeholder whose real reason was lost.

    These are the rows the triage tool re-executes to recover a cause; fresh
    runs no longer produce them (the executors surface per-seed reasons).
    """
    return bool(_COLLAPSED.match((error or "").strip().lower()))


def classify(error: str | None, language: str = "") -> str:
    """Map a real failure reason to one stable tag. Language is advisory."""
    e = (error or "").strip()
    el = e.lower()
    if not e or is_collapsed(e):
        return "other"
    if "gt collection failed" in el:
        return "gt_side"
    if "empty code" in el:
        return "empty_code"
    if "not found" in el and ("problem" in el or "realization" in el):
        return "corpus_miss"
    if "timeout" in el:
        return "timeout"
    # Compile / parse failures: Stan compiler messages + Python SyntaxError.
    if ("compile" in el or "syntax error" in el or "semantic error" in el
            or "parsing error" in el or "syntaxerror" in el
            or "indentationerror" in el):
        return "compile"
    if ("did not define answer" in el or "produced no output" in el
            or "not valid json" in el or "non-json output" in el
            or "no error_message" in el or "wrong number of results" in el):
        return "no_output"
    return "runtime"


def join_reasons(errors) -> str:
    """Collapse a per-seed errors list into one deduped reason string.

    Used by the harness when >=1 seed failed: surface the distinct real reasons
    (order-preserving) rather than a generic count. None entries (succeeded
    seeds) are skipped.
    """
    seen: list[str] = []
    for e in errors:
        if e is None:
            continue
        s = str(e).strip()
        if s and s not in seen:
            seen.append(s)
    return "; ".join(seen) if seen else "execution failed"
