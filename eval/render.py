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
# Language-specific wording table
# ---------------------------------------------------------------------------

# All contract text lives here. Keys are language identifiers; each sub-dict
# provides the fragments consumed by _contract_for/_field_contract below.
_LANG: dict[str, dict[str, str]] = {
    "webppl": {
        # prefix for `var ANSWER = <expression>;`
        "binding_prefix": "End your program with `var ANSWER = <expression>;`",
        # realvec variant suffix
        "realvec_suffix": "For a real-valued vector, ANSWER should be a list of numbers.",
        # dist object description (used after "where `<expression>` is ")
        "dist_object": "the distribution object returned by `Infer`",
        # full dist-object sentence (used when generating the standalone sentence)
        "dist_object_sentence": (
            "End your program with `var ANSWER = <expression>;` where "
            "`<expression>` is the distribution object returned by `Infer`."
        ),
        # dist-draws full sentence
        "dist_draws_sentence": (
            "End your program with `var ANSWER = <expression>;` where "
            "`<expression>` evaluates to **one** sampled draw from the process. "
            "Your program will be run many times with different random seeds; "
            "each run contributes one draw to the collected distribution."
        ),
        # record intro
        "record_intro": (
            "End your program with `var ANSWER = <expression>;` where "
            "`<expression>` is an object with exactly these fields:"
        ),
        # field: dist object
        "field_dist_object": "the distribution object returned by `Infer`",
        # field: dist draws
        "field_dist_draws": (
            "one sampled draw from the process "
            "(the program will be run many times to collect the distribution)"
        ),
        # field: value
        "field_value": "the computed value",
        # field: value realvec
        "field_value_realvec": "the computed value (a list of numbers)",
    },
    "pyro": {
        "binding_prefix": "End your program with a top-level assignment `ANSWER = <expression>`",
        "realvec_suffix": "a list of numbers (a 1-D tensor is also accepted)",
        "dist_object": (
            "the posterior distribution — either a dict mapping each outcome to its "
            "probability, a `pyro.distributions`/`torch.distributions` object, or a "
            "list of (many) posterior samples"
        ),
        "dist_object_sentence": (
            "End your program with a top-level assignment `ANSWER = <expression>` "
            "where `<expression>` is the posterior distribution — either a dict "
            "mapping each outcome to its probability, a "
            "`pyro.distributions`/`torch.distributions` object, or a list of (many) "
            "posterior samples."
        ),
        "dist_draws_sentence": (
            "End your program with a top-level assignment `ANSWER = <expression>` "
            "where `<expression>` evaluates to **one** sampled draw from the process. "
            "Your program will be run many times with different random seeds; "
            "each run contributes one draw to the collected distribution."
        ),
        "record_intro": (
            "End your program with a top-level assignment `ANSWER = <expression>` "
            "where `<expression>` is a dict with exactly these keys:"
        ),
        "field_dist_object": (
            "the posterior distribution — either a dict mapping each outcome to its "
            "probability, a `pyro.distributions`/`torch.distributions` object, or a "
            "list of (many) posterior samples"
        ),
        "field_dist_draws": (
            "one sampled draw from the process "
            "(the program will be run many times to collect the distribution)"
        ),
        "field_value": "the computed value",
        "field_value_realvec": "the computed value (a list of numbers; a 1-D tensor is also accepted)",
    },
}


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


def _contract_paragraph(spec: Spec, language: str = "webppl") -> str:
    """Return the harness-contract sentence(s) for a spec."""
    return _contract_for(spec, language)


def _contract_for(spec: Spec, language: str = "webppl") -> str:
    """Recursively produce the contract text fragment for a spec node."""
    if language not in _LANG:
        raise ValueError(f"unknown language: {language!r}")
    lang = _LANG[language]

    if spec.kind == "value":
        prefix = lang["binding_prefix"]
        if spec.domain == "realvec":
            realvec = lang["realvec_suffix"]
            binding = f"{prefix} where `<expression>` evaluates to the computed value. {realvec}"
        else:
            binding = f"{prefix} where `<expression>` evaluates to the computed value."
        support_text = _support_contract(spec)
        if support_text:
            return binding + " " + support_text
        return binding

    if spec.kind == "dist":
        labels_text = _labels_contract(spec)
        support_text = _support_contract(spec)
        if spec.protocol == "object":
            base = lang["dist_object_sentence"]
        else:  # draws
            base = lang["dist_draws_sentence"]
        extras = " ".join(s for s in [labels_text, support_text] if s)
        if extras:
            return base + " " + extras
        return base

    # record
    assert spec.kind == "record"
    field_descriptions = []
    for name, fspec in spec.fields:
        field_descriptions.append(_field_contract(name, fspec, language))

    fields_text = "\n".join(f"  - {d}" for d in field_descriptions)
    intro = lang["record_intro"]
    return f"{intro}\n{fields_text}"


def _field_contract(name: str, spec: Spec, language: str = "webppl") -> str:
    """One-line description for a record field."""
    if language not in _LANG:
        raise ValueError(f"unknown language: {language!r}")
    lang = _LANG[language]

    if spec.kind == "value":
        if spec.domain == "realvec":
            base = f"`{name}`: {lang['field_value_realvec']}."
        else:
            base = f"`{name}`: {lang['field_value']}."
        support_text = _support_contract(spec)
        if support_text:
            return base + " " + support_text
        return base

    if spec.kind == "dist":
        labels_text = _labels_contract(spec)
        support_text = _support_contract(spec)
        if spec.protocol == "object":
            base = f"`{name}`: {lang['field_dist_object']}."
        else:  # draws
            base = f"`{name}`: {lang['field_dist_draws']}."
        extras = " ".join(s for s in [labels_text, support_text] if s)
        if extras:
            return base + " " + extras
        return base

    # nested record
    assert spec.kind == "record"
    inner_fields = "; ".join(
        f"`{n}`: {_field_contract_inline(n, f, language)}"
        for n, f in spec.fields
    )
    return f"`{name}`: an object with fields {{{inner_fields}}}."


def _field_contract_inline(name: str, spec: Spec, language: str = "webppl") -> str:
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

    if language not in _LANG:
        raise ValueError(f"unknown language: {language!r}")
    contract = _contract_paragraph(spec, language)

    parts = []
    if given:
        parts.append(f"## Given\n{given}")
    if model:
        parts.append(f"## Model\n{model}")
    if query:
        parts.append(f"## Task\n{query}")
    parts.append(f"## Answer format\n{contract}")

    return "\n\n".join(parts)
