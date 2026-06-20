"""Tests for eval/render.py — render_problem contract-paragraph generation.

One function per promise the renderer makes; per-(spec-kind × language) detail is
parametrized. Does not pin exact prose beyond the load-bearing tokens.
"""

from __future__ import annotations

import pytest

from eval.render import render_problem


def _make_problem(answer_spec: dict, given="G.", model="M.", query="Q.") -> dict:
    return {
        "problem_id": "test/dummy",
        "statement": {"given": given, "model": model, "query": query},
        "answer_spec": answer_spec,
    }


# ---------------------------------------------------------------------------
# Structure + statement passthrough
# ---------------------------------------------------------------------------

def test_sections_and_statement_passthrough():
    p = _make_problem({"kind": "value", "domain": "real"},
                      given="The sky is blue.", model="A model.", query="What is the answer?")
    text = render_problem(p)
    for header in ("## Given", "## Model", "## Task", "## Answer format"):
        assert header in text
    for s in ("The sky is blue.", "A model.", "What is the answer?"):
        assert s in text


# ---------------------------------------------------------------------------
# Per-spec-kind contract wording (webppl)
# ---------------------------------------------------------------------------

def test_value_contract():
    """value: binds var ANSWER, no distribution/sampling language."""
    text = render_problem(_make_problem({"kind": "value", "domain": "real"}))
    assert "var ANSWER" in text
    assert "Infer" not in text and "draw" not in text.lower()


@pytest.mark.parametrize("domain, expect_list", [("realvec", True), ("bool", False)])
def test_value_realvec_list_mention(domain, expect_list):
    text = render_problem(_make_problem({"kind": "value", "domain": domain}))
    assert ("list of numbers" in text.lower()) is expect_list


def test_dist_object_contract():
    """dist/object: names Infer, binds ANSWER, no repeated-runs language."""
    text = render_problem(_make_problem({"kind": "dist", "domain": "bool"}))
    assert "Infer" in text and "ANSWER" in text
    assert "many times" not in text


def test_dist_draws_contract():
    """dist/draws: one draw + many-runs semantics, returns a value (no Infer)."""
    text = render_problem(_make_problem({"kind": "dist", "domain": "finite", "protocol": "draws"}))
    assert "ANSWER" in text and "Infer" not in text
    assert "one" in text.lower() or "single" in text.lower()
    assert "many times" in text.lower() or "runs" in text.lower()


def test_record_recursion():
    """record: every field name appears; dist field carries Infer, value field present."""
    text = render_problem(_make_problem({"kind": "record", "fields": {
        "mean": {"kind": "value", "domain": "real"},
        "post": {"kind": "dist", "domain": "real"}}}))
    assert "mean" in text and "post" in text and "Infer" in text


def test_labels_enumeration():
    """A labeled dist (incl. nested in a record) enumerates field names AND types."""
    text = render_problem(_make_problem({"kind": "record", "fields": {
        "joint": {"kind": "dist", "domain": "finite",
                  "labels": {"record": {"color": "string", "size": "int"}}}}}))
    assert "color" in text and "size" in text and "string" in text and "int" in text


# ---------------------------------------------------------------------------
# Declared support enumeration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec, marker, labels", [
    ({"kind": "dist", "domain": "finite", "support": ["H", "T"]},
     "support elements are exactly", ["H", "T"]),
    ({"kind": "dist", "domain": "finite", "protocol": "draws", "support": ["r", "g", "b"]},
     "Each draw must be exactly one of", ["r", "g", "b"]),
    ({"kind": "value", "domain": "finite", "support": ["yes", "no"]},
     "The answer is exactly one of", ["yes", "no"]),
])
def test_support_wording_per_kind(spec, marker, labels):
    text = render_problem(_make_problem(spec))
    assert marker in text
    for label in labels:
        assert label in text


def test_support_coexists_with_labels_and_absent_when_undeclared():
    p = _make_problem({"kind": "dist", "domain": "finite",
                       "labels": {"record": {"color": "string"}},
                       "support": [{"color": "red"}, {"color": "blue"}]})
    text = render_problem(p)
    assert "color" in text and "support elements are exactly" in text
    assert text.index("color") < text.index("support elements are exactly")
    # undeclared support → no enumeration sentence
    assert "support elements are exactly" not in render_problem(
        _make_problem({"kind": "dist", "domain": "finite"}))


# ---------------------------------------------------------------------------
# No wire-format leak (the richest spec, both languages)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language", ["webppl", "pyro"])
def test_no_wire_format_leak(language):
    text = render_problem(_make_problem({"kind": "record", "fields": {
        "joint": {"kind": "dist", "domain": "finite",
                  "labels": {"record": {"sneeze": "bool"}}},
        "n": {"kind": "value", "domain": "int"}}}), language=language)
    assert "__kind" not in text and '"probs"' not in text and '"support"' not in text
    assert "probs" not in text.split()


# ---------------------------------------------------------------------------
# Pyro wording differs from webppl
# ---------------------------------------------------------------------------

def test_pyro_contract():
    """pyro: `ANSWER = <expression>` (never `var ANSWER`), dict/distribution language,
    no WebPPL Infer; record uses dict wording + field names."""
    val = render_problem(_make_problem({"kind": "value", "domain": "real"}), language="pyro")
    assert "ANSWER = <expression>" in val and "var ANSWER" not in val
    dist = render_problem(_make_problem({"kind": "dist", "domain": "bool"}), language="pyro")
    assert "Infer" not in dist and ("dict" in dist.lower() or "distribution" in dist.lower())
    rec = render_problem(_make_problem({"kind": "record", "fields": {
        "rain": {"kind": "dist", "domain": "bool"}, "n": {"kind": "value", "domain": "int"}}}),
        language="pyro")
    assert "rain" in rec and "n" in rec and "var ANSWER" not in rec


# ---------------------------------------------------------------------------
# Golden: default language is webppl; unknown language raises
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec_key", [
    {"kind": "value", "domain": "real"},
    {"kind": "record", "fields": {"p": {"kind": "dist", "domain": "real"}}},
])
def test_default_equals_explicit_webppl(spec_key):
    p = _make_problem(spec_key)
    assert render_problem(p) == render_problem(p, language="webppl")


def test_unknown_language_raises():
    with pytest.raises(ValueError, match="unknown language"):
        render_problem(_make_problem({"kind": "value", "domain": "real"}), language="church")
