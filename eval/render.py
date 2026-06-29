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
        "value_sentence": "`ANSWER` is the computed value.",
        "realvec_suffix": "For a real-valued vector, `ANSWER` is a list of numbers.",
        "dist_object": "the distribution returned by `Infer`",
        "dist_object_sentence": "`ANSWER` is the distribution returned by `Infer`.",
        "dist_draws_sentence": (
            "`ANSWER` is a single sampled draw from the process; the program is run "
            "repeatedly with different random seeds, each run contributing one draw."
        ),
        "record_intro": "`ANSWER` is an object with exactly these fields:",
        "field_dist_object": "the distribution returned by `Infer`",
        "field_dist_draws": "a single sampled draw from the process",
        "field_value": "the computed value",
        "field_value_realvec": "the computed value (a list of numbers)",
    },
    "pyro": {
        "value_sentence": "`ANSWER` is the computed value.",
        "realvec_suffix": "a list of numbers (a 1-D tensor is also accepted)",
        "dist_object": (
            "the posterior distribution: a dict from each outcome to its probability, "
            "a `pyro.distributions`/`torch.distributions` object, or a list of posterior samples"
        ),
        "dist_object_sentence": (
            "`ANSWER` is the posterior distribution: a dict from each outcome to its "
            "probability, a `pyro.distributions`/`torch.distributions` object, or a "
            "list of posterior samples."
        ),
        "dist_draws_sentence": (
            "`ANSWER` is a single sampled draw from the process; the program is run "
            "repeatedly with different random seeds, each run contributing one draw."
        ),
        "record_intro": "`ANSWER` is a dict with exactly these keys:",
        "field_dist_object": (
            "the posterior distribution: a dict from each outcome to its probability, "
            "a `pyro.distributions`/`torch.distributions` object, or a list of posterior samples"
        ),
        "field_dist_draws": "a single sampled draw from the process",
        "field_value": "the computed value",
        "field_value_realvec": "the computed value (a list of numbers, or a 1-D tensor)",
    },
    # Stan binds structurally: the program defines the model; queried quantities
    # are exposed as named parameters. (Binding convention is in the system base;
    # the data inputs are listed in the rendered Data interface section.)
    "stan": {
        "value_sentence": "Compute the queried quantity in a `generated quantities` block.",
        "realvec_suffix": "",
        "dist_object": "a parameter (in `parameters` or `transformed parameters`)",
        "dist_object_sentence": (
            "Expose the queried quantity as a parameter (in `parameters` or "
            "`transformed parameters`)."
        ),
        "dist_draws_sentence": "Expose the queried quantity as a parameter.",
        "record_intro": "Expose these quantities as parameters, under exactly these names:",
        "field_dist_object": "a parameter",
        "field_dist_draws": "a parameter",
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


# Object-valued outcomes: state the accepted return forms (not the why).
_OBJECT_LABEL_CLAUSE = (
    "Each outcome is an object with the named fields. Give the distribution as a "
    "list of sampled outcome objects, or as a mapping from each outcome's JSON "
    "string to its probability."
)


def _data_interface_section(realization: dict) -> str:
    """The Stan data-block signature: the exact inputs the program must declare."""
    from eval.stan_bundle import data_block
    block = data_block(realization.get("code", ""))
    if not block:
        return ""
    return (
        "## Data interface\n"
        "Your `data` block must declare exactly these inputs:\n\n"
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
