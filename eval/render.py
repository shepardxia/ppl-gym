"""Shared prompt renderer: problem statement → user-message text.

One renderer consumed by both the eval pipeline and the web UI.  Contract:

    render_problem(problem, language="webppl") -> str

The returned string is the *user* message.  Callers must prepend a system
message via eval.prompt.system_prompt(); this module does not embed the
system prompt.

Harness-contract paragraph rules (from data/SCHEMA.md §prompts):
- value spec         → bind the computed value to ``var ANSWER = <expression>;``
- dist + protocol=object  → bind the distribution object (result of Infer) to ANSWER
- dist + protocol=draws   → bind ONE sampled draw to ANSWER; the program is run
                            many times with fresh seeds to collect the distribution
- record             → bind an object with exactly the spec's field names;
                       each field described recursively per the rules above

The contract paragraph is the ONLY place ANSWER/binding/API language appears.
No wire formats, no ``__kind``, no ``probs``, no ``support`` tokens.
"""

from __future__ import annotations

from eval.algebra import Spec, _has_draws_field, parse_spec


# ---------------------------------------------------------------------------
# Contract paragraph generation
# ---------------------------------------------------------------------------

def _labels_contract(spec: Spec) -> str:
    """Return an additional sentence describing declared label fields, or empty string."""
    if not spec.labels:
        return ""
    parts = []
    for name, fdomain in spec.labels:
        parts.append(f"`{name}` ({fdomain})")
    fields_list = ", ".join(parts)
    return (
        f"Each support element is an object with exactly these fields: {fields_list}."
    )


def _support_contract(spec: Spec) -> str:
    """Return a sentence enumerating declared support labels, or empty string."""
    if not spec.support:
        return ""
    # spec.support holds canonical JSON label keys; render them verbatim so the
    # enumeration is language-neutral (true, not True; "a", not 'a').
    labels_str = ", ".join(f"`{k}`" for k in spec.support)
    if spec.kind == "dist":
        if spec.protocol == "object":
            return f"The distribution's support elements are exactly: {labels_str}. Use these exact values."
        else:  # draws
            return f"Each draw must be exactly one of: {labels_str}."
    # value
    return f"The answer is exactly one of: {labels_str}."


def _contract_paragraph(spec: Spec) -> str:
    """Return the harness-contract sentence(s) for a spec."""
    return _contract_for(spec)


def _contract_for(spec: Spec) -> str:
    """Recursively produce the contract text fragment for a spec node."""
    if spec.kind == "value":
        if spec.domain == "realvec":
            binding = (
                f"End your program with `var ANSWER = <expression>;` where "
                f"`<expression>` evaluates to the computed value. "
                f"For a real-valued vector, ANSWER should be a list of numbers."
            )
        else:
            binding = (
                f"End your program with `var ANSWER = <expression>;` where "
                f"`<expression>` evaluates to the computed value."
            )
        support_text = _support_contract(spec)
        if support_text:
            return binding + " " + support_text
        return binding

    if spec.kind == "dist":
        labels_text = _labels_contract(spec)
        support_text = _support_contract(spec)
        if spec.protocol == "object":
            base = (
                f"End your program with `var ANSWER = <expression>;` where "
                f"`<expression>` is the distribution object returned by `Infer`."
            )
        else:  # draws
            base = (
                f"End your program with `var ANSWER = <expression>;` where "
                f"`<expression>` evaluates to **one** sampled draw from the process. "
                f"Your program will be run many times with different random seeds; "
                f"each run contributes one draw to the collected distribution."
            )
        extras = " ".join(s for s in [labels_text, support_text] if s)
        if extras:
            return base + " " + extras
        return base

    # record
    assert spec.kind == "record"
    field_descriptions = []
    for name, fspec in spec.fields:
        field_descriptions.append(_field_contract(name, fspec))

    fields_text = "\n".join(f"  - {d}" for d in field_descriptions)

    return (
        f"End your program with `var ANSWER = <expression>;` where "
        f"`<expression>` is an object with exactly these fields:\n"
        f"{fields_text}"
    )


def _field_contract(name: str, spec: Spec) -> str:
    """One-line description for a record field."""
    if spec.kind == "value":
        if spec.domain == "realvec":
            base = f"`{name}`: the computed value (a list of numbers)."
        else:
            base = f"`{name}`: the computed value."
        support_text = _support_contract(spec)
        if support_text:
            return base + " " + support_text
        return base

    if spec.kind == "dist":
        labels_text = _labels_contract(spec)
        support_text = _support_contract(spec)
        if spec.protocol == "object":
            base = f"`{name}`: the distribution object returned by `Infer`."
        else:  # draws
            base = (
                f"`{name}`: one sampled draw from the process "
                f"(the program will be run many times to collect the distribution)."
            )
        extras = " ".join(s for s in [labels_text, support_text] if s)
        if extras:
            return base + " " + extras
        return base

    # nested record
    assert spec.kind == "record"
    inner_fields = "; ".join(
        f"`{n}`: {_field_contract_inline(n, f)}"
        for n, f in spec.fields
    )
    return f"`{name}`: an object with fields {{{inner_fields}}}."


def _field_contract_inline(name: str, spec: Spec) -> str:
    """Ultra-compact description used for nested record fields."""
    if spec.kind == "value":
        return "a computed value" + (" (list of numbers)" if spec.domain == "realvec" else "")
    if spec.kind == "dist":
        if spec.protocol == "draws":
            return "one draw from the process"
        return "a distribution object"
    return "a record"


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------

def render_problem(problem: dict, language: str = "webppl") -> str:
    """Render a problem dict into the user-message text for the LLM.

    Parameters
    ----------
    problem:
        A problem record as stored in ``data/problems/<corpus>.jsonl``.
        Must contain ``statement`` (with keys ``given``, ``model``, ``query``)
        and ``answer_spec``.
    language:
        The target PPL (default ``"webppl"``).  Currently only the
        harness-contract paragraph is language-sensitive (future: language-
        specific syntax hints could be added here without touching the spec).

    Returns
    -------
    str
        The user message text.  The caller is responsible for building the
        full ``[system, user]`` message list.
    """
    stmt = problem["statement"]
    spec = parse_spec(problem["answer_spec"])

    given = stmt.get("given", "").strip()
    model = stmt.get("model", "").strip()
    query = stmt.get("query", "").strip()

    contract = _contract_paragraph(spec)

    parts = []
    if given:
        parts.append(f"## Given\n{given}")
    if model:
        parts.append(f"## Model\n{model}")
    if query:
        parts.append(f"## Task\n{query}")
    parts.append(f"## Answer format\n{contract}")

    return "\n\n".join(parts)
