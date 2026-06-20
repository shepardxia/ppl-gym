"""System-prompt assembly and response parsing for problem-centric eval.

`eval.render.render_problem` produces the user message; this module owns the
language-specific system prompt (base + primer) and extracts the fenced code
block from the LLM response. The primer levels the playing field across models
that may not have seen much of the target PPL in pretraining.
"""

from __future__ import annotations

import re
from pathlib import Path


# Source of truth for the prompt text lives in data/prompts/.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"

WEBPPL_SYSTEM_BASE = (_PROMPTS_DIR / "webppl_system_base.txt").read_text().rstrip("\n")
PYRO_SYSTEM_BASE = (_PROMPTS_DIR / "pyro_system_base.txt").read_text().rstrip("\n")
STAN_SYSTEM_BASE = (_PROMPTS_DIR / "stan_system_base.txt").read_text().rstrip("\n")
WEBPPL_PRIMER = (_PROMPTS_DIR / "webppl_primer.txt").read_text().rstrip("\n")
PYRO_PRIMER = (_PROMPTS_DIR / "pyro_primer.txt").read_text().rstrip("\n")

_SYSTEM_BASE = {"webppl": WEBPPL_SYSTEM_BASE, "pyro": PYRO_SYSTEM_BASE, "stan": STAN_SYSTEM_BASE}
# Primers level the playing field for PPLs models rarely saw in pretraining. Only
# webppl/pyro have one; stan (posteriordb) has none.
_PRIMER = {"webppl": WEBPPL_PRIMER, "pyro": PYRO_PRIMER}


def _base_for(language: str) -> str:
    if language not in _SYSTEM_BASE:
        raise ValueError(f"no system base for language {language!r}")
    return _SYSTEM_BASE[language]


def _primer_for(language: str) -> str:
    return _PRIMER.get(language, "")


def system_prompt(*, with_primer: bool = True, language: str = "webppl") -> str:
    base = _base_for(language)
    primer = _primer_for(language) if with_primer else ""
    return f"{base}\n\n{primer}" if primer else base


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Matches ```js ... ```, ``` ... ```, ```webppl ... ```, etc.
_FENCE_RE = re.compile(
    r"```(?:[A-Za-z0-9_+-]*)?\s*\n(.*?)```",
    re.DOTALL,
)


def parse_response(text: str) -> tuple[str, list[str]]:
    """Extract a single WebPPL program from the LLM response.

    Returns (code, warnings). If no fence is found, returns the raw
    response trimmed and a warning. If multiple fences are found, the last
    one wins (model's final answer).
    """
    warnings: list[str] = []
    matches = _FENCE_RE.findall(text)
    if not matches:
        warnings.append("no fenced code block; using raw response")
        return text.strip(), warnings
    if len(matches) > 1:
        warnings.append(f"{len(matches)} fenced blocks; using the last one")
    code = matches[-1].rstrip()
    return code, warnings
