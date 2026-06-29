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

import json

from eval.algebra import Spec, _has_draws_field, parse_spec


# ---------------------------------------------------------------------------
# Language-specific wording table
# ---------------------------------------------------------------------------

# All contract text lives here. Keys are language identifiers; each sub-dict
# provides the fragments consumed by _contract_for/_field_contract below.
_LANG: dict[str, dict[str, str]] = {
    "webppl": {
        # full top-level value sentence
        "value_sentence": "End your program with `var ANSWER = <expression>;` where `<expression>` evaluates to the computed value.",
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
        "value_sentence": "End your program with a top-level assignment `ANSWER = <expression>` where `<expression>` evaluates to the computed value.",
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
    # Stan's binding is structural, not an `ANSWER` expression: the program *is*
    # the model. The harness supplies the data (matching the model's `data`
    # block), draws from the posterior with NUTS, and reports the marginal
    # posterior of each named parameter.
    "stan": {
        "value_sentence": (
            "Write a complete Stan program for this model. Declare a `data` block "
            "matching the inputs described above; the harness supplies their values "
            "and draws from the posterior with NUTS. Compute the queried quantity in "
            "a `generated quantities` block; its posterior is what is reported."
        ),
        "realvec_suffix": "",
        "dist_object": "a parameter whose marginal posterior is reported",
        "dist_object_sentence": (
            "Write a complete Stan program for this model. Declare a `data` block "
            "matching the inputs described above; the harness supplies their values "
            "and draws from the posterior with NUTS. Expose the queried quantity as "
            "a parameter (in `parameters` or `transformed parameters`); its marginal "
            "posterior distribution is what is reported."
        ),
        "dist_draws_sentence": (
            "Write a complete Stan program for this model. Declare a `data` block "
            "matching the inputs described above; the harness supplies their values "
            "and draws from the posterior with NUTS. Expose the queried quantity as "
            "a parameter; its marginal posterior distribution is what is reported."
        ),
        "record_intro": (
            "Write a complete Stan program for this model. Declare a `data` block "
            "matching the inputs described above; the harness supplies their values "
            "and draws from the posterior with NUTS. Your `parameters` (or "
            "`transformed parameters`) block must expose, under exactly these names, "
            "each quantity whose marginal posterior is reported:"
        ),
        "field_dist_object": "a parameter; its marginal posterior distribution is reported",
        "field_dist_draws": "a parameter; its marginal posterior distribution is reported",
        "field_value": "a quantity computed in `generated quantities`",
        "field_value_realvec": "a vector computed in `generated quantities`",
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
        binding = lang["value_sentence"]
        if spec.domain == "realvec" and lang["realvec_suffix"]:
            binding = binding + " " + lang["realvec_suffix"]
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


def _has_object_labels(spec: Spec) -> bool:
    """True if any finite leaf has object-valued labels (record labels, or
    support elements that are JSON objects). Python cannot hash these, so the
    solver needs an explicit encoding instruction."""
    if spec.kind == "record":
        return any(_has_object_labels(f) for _, f in spec.fields)
    if spec.domain == "finite":
        if spec.labels:
            return True
        for k in (spec.support or ()):
            try:
                if isinstance(json.loads(k), dict):
                    return True
            except (ValueError, TypeError):
                pass
    return False


# Python-only: object outcomes are unhashable, so a solver cannot key a dict by
# the object. Tell it the two encodings the canonicalizer accepts.
_OBJECT_LABEL_CLAUSE = (
    "Note on object-valued outcomes: each outcome is an object (a dict of the "
    "named fields), which is not hashable in Python — do NOT use the object, a "
    "tuple, or a frozenset as a dict key. Return EITHER a list of many sampled "
    "outcome objects (each a dict with the named fields), OR a dict whose keys "
    "are `json.dumps(outcome, sort_keys=True)` and whose values are the probabilities."
)


def _data_interface_section(realization: dict) -> str:
    """The Stan data-block signature section: pins the input I/O interface."""
    from eval.stan_bundle import data_block
    block = data_block(realization.get("code", ""))
    if not block:
        return ""
    return (
        "## Data interface\n"
        "The harness supplies these inputs; your `data` block must declare them "
        "exactly, by name and type. Values arrive **raw** — compute any transforms "
        "(logs, standardization, interactions, …) inside your program.\n\n"
        "```stan\n" + block + "\n```"
    )


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------

def render_problem(problem: dict, language: str = "webppl",
                   realization: dict | None = None) -> str:
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
    # Python solvers need the object-key encoding spelled out (unhashable outcomes).
    if language == "pyro" and _has_object_labels(spec):
        contract = contract + "\n\n" + _OBJECT_LABEL_CLAUSE

    parts = []
    if given:
        parts.append(f"## Given\n{given}")
    if model:
        parts.append(f"## Model\n{model}")
    if query:
        parts.append(f"## Task\n{query}")
    # Stan: pin the data-block I/O interface from the GT bundle (inputs the
    # harness binds by name), so the solver's program actually runs.
    if language == "stan" and realization is not None:
        section = _data_interface_section(realization)
        if section:
            parts.append(section)
    parts.append(f"## Answer format\n{contract}")

    return "\n\n".join(parts)
