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
WEBPPL_PRIMER = (_PROMPTS_DIR / "webppl_primer.txt").read_text().rstrip("\n")
PYRO_PRIMER = (_PROMPTS_DIR / "pyro_primer.txt").read_text().rstrip("\n")


def _primer_for(language: str) -> str:
    if language == "pyro":
        return PYRO_PRIMER
    return WEBPPL_PRIMER


def _base_for(language: str) -> str:
    if language == "pyro":
        return PYRO_SYSTEM_BASE
    return WEBPPL_SYSTEM_BASE


def system_prompt(*, with_primer: bool = True, language: str = "webppl") -> str:
    base = _base_for(language)
    if with_primer:
        return base + "\n\n" + _primer_for(language)
    return base


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
