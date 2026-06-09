"""Prompt formatting and response parsing for atom-based eval.

Each atom has its own self-contained prompt; the LLM responds with one
fenced WebPPL block. The system prompt optionally includes a small WebPPL
primer to level the playing field across models that may not have seen
much WebPPL in pretraining.
"""

from __future__ import annotations

import re
from pathlib import Path


PROMPT_VERSION = "v2-atom"

# Source of truth for the prompt text lives in data/prompts/. The web UI
# reads the same files at build time so the model and the browser see
# byte-identical strings.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"

SYSTEM_PROMPT_BASE = (_PROMPTS_DIR / "system_base.txt").read_text().rstrip("\n")
WEBPPL_PRIMER = (_PROMPTS_DIR / "webppl_primer.txt").read_text().rstrip("\n")
PYRO_PRIMER = (_PROMPTS_DIR / "pyro_primer.txt").read_text().rstrip("\n")

# The base prompt is WebPPL-flavored ("emit one fenced code block ending in
# var ANSWER = ..."). For Pyro we override the format guidance.
PYRO_SYSTEM_BASE = (
    "You are a Pyro code generator. Given an exercise, produce a single Python program "
    "using Pyro that binds the answer to a top-level variable named `ANSWER`.\n\n"
    "Answer format (strict): emit exactly one fenced code block.\n\n"
    "```python\n"
    "<your Pyro program ending with: ANSWER = <expression>>\n"
    "```\n\n"
    "The last statement of your program MUST be `ANSWER = <expression>` where "
    "`<expression>` is the answer the prompt asks for — typically a "
    "`pyro.distributions.Distribution` for a distribution, a numeric/list/dict value, "
    "or a Python literal.\n\n"
    "Do not write prose, explanations, or multiple code blocks."
)


def _primer_for(language: str) -> str:
    if language == "pyro":
        return PYRO_PRIMER
    return WEBPPL_PRIMER


def _base_for(language: str) -> str:
    if language == "pyro":
        return PYRO_SYSTEM_BASE
    return SYSTEM_PROMPT_BASE


def system_prompt(*, with_primer: bool = True, language: str = "webppl") -> str:
    base = _base_for(language)
    if with_primer:
        return base + "\n\n" + _primer_for(language)
    return base


def format_messages(atom: dict, *, with_primer: bool = True) -> list[dict]:
    """Return [system, user] messages for one atom.

    The atom's optional `language` field (defaults to "webppl") selects the
    appropriate system prompt + primer.
    """
    language = atom.get("language", "webppl")
    return [
        {"role": "system", "content": system_prompt(with_primer=with_primer, language=language)},
        {"role": "user", "content": atom["prompt"]},
    ]


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
