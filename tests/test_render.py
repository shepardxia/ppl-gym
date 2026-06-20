"""Tests for eval/render.py — render_problem contract-paragraph generation.

Covers:
  - value spec contract (including realvec list-of-numbers mention)
  - dist + protocol=object contract
  - dist + protocol=draws contract
  - record spec recursion (fields described individually)
  - no-leak property: rendered text must not contain wire-format tokens
    (__kind, "probs", "support") for dist problems
"""

from __future__ import annotations

import pytest

from eval.render import render_problem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem(answer_spec: dict, given="G.", model="M.", query="Q.") -> dict:
    return {
        "problem_id": "test/dummy",
        "statement": {"given": given, "model": model, "query": query},
        "answer_spec": answer_spec,
    }


# ---------------------------------------------------------------------------
# Contract: value spec
# ---------------------------------------------------------------------------

class TestValueContract:
    def test_value_real_mentions_answer_binding(self):
        p = _make_problem({"kind": "value", "domain": "real"})
        text = render_problem(p)
        assert "ANSWER" in text
        assert "var ANSWER" in text

    def test_value_real_no_distribution_language(self):
        p = _make_problem({"kind": "value", "domain": "real"})
        text = render_problem(p)
        # Must not suggest Infer or sampling mechanics for a value spec
        assert "Infer" not in text
        assert "seeded" not in text.lower()
        assert "draw" not in text.lower()

    def test_value_realvec_mentions_list_of_numbers(self):
        p = _make_problem({"kind": "value", "domain": "realvec"})
        text = render_problem(p)
        assert "list of numbers" in text.lower() or "list-of-numbers" in text.lower() or "list" in text.lower()

    def test_value_bool_no_list_mention(self):
        p = _make_problem({"kind": "value", "domain": "bool"})
        text = render_problem(p)
        # bool is exact — should not say "list of numbers"
        assert "list of numbers" not in text.lower()


# ---------------------------------------------------------------------------
# Contract: dist + protocol=object
# ---------------------------------------------------------------------------

class TestDistObjectContract:
    def test_dist_object_mentions_infer(self):
        p = _make_problem({"kind": "dist", "domain": "bool"})
        text = render_problem(p)
        assert "Infer" in text
        assert "ANSWER" in text

    def test_dist_object_no_draws_language(self):
        p = _make_problem({"kind": "dist", "domain": "bool"})
        text = render_problem(p)
        # Must not mention repeated runs for object protocol
        assert "many times" not in text
        assert "seeded" not in text.lower()

    def test_dist_object_no_leak(self):
        """Wire-format tokens must not appear in the rendered prompt."""
        for domain in ("bool", "finite", "int"):
            p = _make_problem({"kind": "dist", "domain": domain})
            text = render_problem(p)
            assert "__kind" not in text, f"__kind leaked for domain={domain}"
            assert '"probs"' not in text, f'\"probs\" leaked for domain={domain}'
            assert '"support"' not in text, f'\"support\" leaked for domain={domain}'
            # Also check unquoted forms
            assert "probs" not in text.split(), f"'probs' token leaked for domain={domain}"
            assert "support" not in text.split(), f"'support' token leaked for domain={domain}"


# ---------------------------------------------------------------------------
# Contract: dist + protocol=draws
# ---------------------------------------------------------------------------

class TestDistDrawsContract:
    def test_draws_mentions_single_draw(self):
        p = _make_problem({"kind": "dist", "domain": "finite", "protocol": "draws"})
        text = render_problem(p)
        assert "one" in text.lower() or "single" in text.lower()
        assert "ANSWER" in text

    def test_draws_mentions_many_runs(self):
        p = _make_problem({"kind": "dist", "domain": "finite", "protocol": "draws"})
        text = render_problem(p)
        assert "many times" in text.lower() or "multiple" in text.lower() or "runs" in text.lower()

    def test_draws_no_infer_mandate(self):
        """Draws protocol: user returns a value, not a distribution object."""
        p = _make_problem({"kind": "dist", "domain": "finite", "protocol": "draws"})
        text = render_problem(p)
        # Should NOT tell the LM to call Infer
        assert "Infer" not in text

    def test_draws_no_leak(self):
        p = _make_problem({"kind": "dist", "domain": "finite", "protocol": "draws"})
        text = render_problem(p)
        assert "__kind" not in text
        assert '"probs"' not in text
        assert '"support"' not in text


# ---------------------------------------------------------------------------
# Contract: record spec recursion
# ---------------------------------------------------------------------------

class TestRecordContract:
    def _rain_problem(self):
        return _make_problem({
            "kind": "record",
            "fields": {
                "rain": {"kind": "dist", "domain": "bool"},
                "sprinkler": {"kind": "dist", "domain": "bool"},
            }
        })

    def test_record_mentions_field_names(self):
        p = self._rain_problem()
        text = render_problem(p)
        assert "rain" in text
        assert "sprinkler" in text

    def test_record_mentions_answer_binding(self):
        p = self._rain_problem()
        text = render_problem(p)
        assert "ANSWER" in text

    def test_record_dist_fields_say_infer(self):
        p = self._rain_problem()
        text = render_problem(p)
        # Each dist field should mention Infer
        assert "Infer" in text

    def test_record_value_field_described_correctly(self):
        p = _make_problem({
            "kind": "record",
            "fields": {
                "mean": {"kind": "value", "domain": "real"},
                "dist": {"kind": "dist", "domain": "real"},
            }
        })
        text = render_problem(p)
        assert "mean" in text
        assert "dist" in text
        # value field should not mention Infer individually but dist field should
        assert "Infer" in text

    def test_record_draws_field_contract(self):
        """A draws field inside a record should say 'one draw'."""
        p = _make_problem({
            "kind": "record",
            "fields": {
                "sample": {"kind": "dist", "domain": "finite", "protocol": "draws"},
            }
        })
        text = render_problem(p)
        assert "sample" in text
        # Should mention draw semantics
        assert "draw" in text.lower() or "run" in text.lower()

    def test_record_no_leak(self):
        p = self._rain_problem()
        text = render_problem(p)
        assert "__kind" not in text
        assert '"probs"' not in text
        assert '"support"' not in text


# ---------------------------------------------------------------------------
# Structural: sections present
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Contract: dist/finite with labels
# ---------------------------------------------------------------------------

class TestLabelsContract:
    def _labeled_problem(self):
        return _make_problem({
            "kind": "dist",
            "domain": "finite",
            "labels": {"record": {"sneeze": "bool", "fever": "bool"}},
        })

    def test_labels_field_names_in_contract(self):
        """Both declared field names must appear in the rendered contract."""
        text = render_problem(self._labeled_problem())
        assert "sneeze" in text, "field 'sneeze' not found in contract"
        assert "fever" in text, "field 'fever' not found in contract"

    def test_labels_types_in_contract(self):
        """The field types (domains) should appear in the rendered contract."""
        text = render_problem(self._labeled_problem())
        assert "bool" in text, "type 'bool' not found in contract"

    def test_labels_no_wire_format_leak(self):
        """Wire-format tokens (__kind, quoted probs/support) must not appear."""
        text = render_problem(self._labeled_problem())
        assert "__kind" not in text
        assert '"probs"' not in text
        assert '"support"' not in text
        # Also check unquoted forms as standalone tokens (probs is never natural English)
        assert "probs" not in text.split()

    def test_labels_infer_still_mentioned(self):
        """Object-protocol labeled dist still says Infer."""
        text = render_problem(self._labeled_problem())
        assert "Infer" in text

    def test_labels_on_record_field(self):
        """A record field that is dist/finite/labeled should also name its fields."""
        p = _make_problem({
            "kind": "record",
            "fields": {
                "joint": {
                    "kind": "dist",
                    "domain": "finite",
                    "labels": {"record": {"color": "string", "size": "int"}},
                },
            },
        })
        text = render_problem(p)
        assert "color" in text
        assert "size" in text
        assert "__kind" not in text


class TestSections:
    def test_sections_present(self):
        p = _make_problem({"kind": "value", "domain": "real"}, given="G.", model="M.", query="Q.")
        text = render_problem(p)
        assert "## Given" in text
        assert "## Model" in text
        assert "## Task" in text
        assert "## Answer format" in text

    def test_given_text_in_output(self):
        p = _make_problem({"kind": "value", "domain": "real"}, given="The sky is blue.")
        text = render_problem(p)
        assert "The sky is blue." in text

    def test_query_text_in_output(self):
        p = _make_problem({"kind": "value", "domain": "real"}, query="What is the answer?")
        text = render_problem(p)
        assert "What is the answer?" in text


# ---------------------------------------------------------------------------
# Contract: support enumeration
# ---------------------------------------------------------------------------

class TestSupportContract:
    """Tests for _support_contract wording for the three cases."""

    def test_dist_object_support_wording(self):
        """dist + protocol=object + support → 'distribution's support elements are exactly'."""
        p = _make_problem({
            "kind": "dist",
            "domain": "finite",
            "support": ["H", "T"],
        })
        text = render_problem(p)
        assert "distribution's support elements are exactly" in text
        assert "'H'" in text or "H" in text
        assert "'T'" in text or "T" in text
        assert "Use these exact values" in text

    def test_dist_draws_support_wording(self):
        """dist + protocol=draws + support → 'Each draw must be exactly one of'."""
        p = _make_problem({
            "kind": "dist",
            "domain": "finite",
            "protocol": "draws",
            "support": ["red", "green", "blue"],
        })
        text = render_problem(p)
        assert "Each draw must be exactly one of" in text
        assert "red" in text
        assert "green" in text
        assert "blue" in text

    def test_value_support_wording(self):
        """value + support → 'The answer is exactly one of'."""
        p = _make_problem({
            "kind": "value",
            "domain": "finite",
            "support": ["yes", "no"],
        })
        text = render_problem(p)
        assert "The answer is exactly one of" in text
        assert "yes" in text
        assert "no" in text

    def test_support_and_labels_coexist(self):
        """When both labels and support are declared, both sentences appear; support is last."""
        p = _make_problem({
            "kind": "dist",
            "domain": "finite",
            "labels": {"record": {"color": "string"}},
            "support": [{"color": "red"}, {"color": "blue"}],
        })
        text = render_problem(p)
        # Labels sentence
        assert "color" in text
        assert "string" in text
        # Support sentence
        assert "distribution's support elements are exactly" in text
        # Support comes after labels — check ordering
        labels_pos = text.index("color")
        support_pos = text.index("distribution's support elements are exactly")
        assert labels_pos < support_pos

    def test_support_no_wire_format_leak(self):
        """Support contract must not leak wire-format tokens."""
        p = _make_problem({
            "kind": "dist",
            "domain": "finite",
            "support": ["A", "B"],
        })
        text = render_problem(p)
        assert "__kind" not in text
        assert '"probs"' not in text

    def test_no_support_sentence_when_not_declared(self):
        """When support is not declared, none of the support sentence wordings appear."""
        p = _make_problem({"kind": "dist", "domain": "finite"})
        text = render_problem(p)
        assert "distribution's support elements are exactly" not in text
        assert "Each draw must be exactly one of" not in text
        assert "The answer is exactly one of" not in text


# ---------------------------------------------------------------------------
# Language-aware contract: Pyro
# ---------------------------------------------------------------------------

class TestPyroContract:
    """Pyro-specific contract wording for every spec kind."""

    def test_pyro_value_binding(self):
        """Pyro value spec: ANSWER = <expression>, no 'var' prefix."""
        p = _make_problem({"kind": "value", "domain": "real"})
        text = render_problem(p, language="pyro")
        assert "ANSWER = <expression>" in text
        assert "var ANSWER" not in text

    def test_pyro_value_realvec_mentions_tensor(self):
        """Pyro realvec value mentions 1-D tensor as an accepted form."""
        p = _make_problem({"kind": "value", "domain": "realvec"})
        text = render_problem(p, language="pyro")
        assert "ANSWER" in text
        assert "list" in text.lower() or "tensor" in text.lower()

    def test_pyro_dist_object_mentions_dict_or_distribution(self):
        """Pyro dist/object contract describes dict/distribution/samples — not Infer."""
        p = _make_problem({"kind": "dist", "domain": "bool"})
        text = render_problem(p, language="pyro")
        assert "ANSWER" in text
        # The pyro description mentions dict or distribution
        assert "dict" in text.lower() or "distribution" in text.lower()

    def test_pyro_dist_object_no_infer(self):
        """Pyro contract must not mention WebPPL's Infer."""
        p = _make_problem({"kind": "dist", "domain": "bool"})
        text = render_problem(p, language="pyro")
        assert "Infer" not in text

    def test_pyro_dist_draws_single_draw(self):
        """Pyro draws contract mentions one draw and many-runs semantics."""
        p = _make_problem({"kind": "dist", "domain": "finite", "protocol": "draws"})
        text = render_problem(p, language="pyro")
        assert "ANSWER" in text
        assert "one" in text.lower() or "single" in text.lower()
        assert "many times" in text.lower() or "runs" in text.lower()

    def test_pyro_dist_draws_no_infer(self):
        """Pyro draws contract must not mention Infer."""
        p = _make_problem({"kind": "dist", "domain": "finite", "protocol": "draws"})
        text = render_problem(p, language="pyro")
        assert "Infer" not in text

    def test_pyro_record_uses_dict_language(self):
        """Pyro record contract says 'dict' not 'object'."""
        p = _make_problem({
            "kind": "record",
            "fields": {
                "rain": {"kind": "dist", "domain": "bool"},
                "n": {"kind": "value", "domain": "int"},
            }
        })
        text = render_problem(p, language="pyro")
        assert "ANSWER" in text
        assert "dict" in text.lower()
        assert "rain" in text
        assert "n" in text

    def test_pyro_no_var_in_record(self):
        """Pyro record contract must not use 'var ANSWER'."""
        p = _make_problem({
            "kind": "record",
            "fields": {"x": {"kind": "value", "domain": "real"}},
        })
        text = render_problem(p, language="pyro")
        assert "var ANSWER" not in text


# ---------------------------------------------------------------------------
# Golden assertion: webppl output is byte-identical across language= calls
# ---------------------------------------------------------------------------

class TestWebpplGolden:
    """render_problem(p) and render_problem(p, language='webppl') must be identical."""

    def test_default_equals_explicit_webppl(self):
        p = _make_problem({"kind": "dist", "domain": "bool"})
        assert render_problem(p) == render_problem(p, language="webppl")

    def test_value_default_equals_explicit_webppl(self):
        p = _make_problem({"kind": "value", "domain": "real"})
        assert render_problem(p) == render_problem(p, language="webppl")

    def test_record_default_equals_explicit_webppl(self):
        p = _make_problem({
            "kind": "record",
            "fields": {
                "rain": {"kind": "dist", "domain": "bool"},
                "n": {"kind": "value", "domain": "int"},
            }
        })
        assert render_problem(p) == render_problem(p, language="webppl")

    def test_webppl_dist_object_contains_infer(self):
        """Golden: webppl dist/object contract explicitly names Infer."""
        p = _make_problem({"kind": "dist", "domain": "bool"})
        text = render_problem(p, language="webppl")
        assert "var ANSWER = <expression>;" in text
        assert "Infer" in text

    def test_webppl_value_contains_var(self):
        """Golden: webppl value contract uses 'var ANSWER'."""
        p = _make_problem({"kind": "value", "domain": "real"})
        text = render_problem(p, language="webppl")
        assert "var ANSWER" in text


# ---------------------------------------------------------------------------
# Unknown language raises
# ---------------------------------------------------------------------------

class TestUnknownLanguage:
    def test_unknown_language_raises_value_error(self):
        p = _make_problem({"kind": "value", "domain": "real"})
        with pytest.raises(ValueError, match="unknown language"):
            render_problem(p, language="bugs")

    def test_unknown_language_raises_for_dist(self):
        p = _make_problem({"kind": "dist", "domain": "bool"})
        with pytest.raises(ValueError, match="unknown language"):
            render_problem(p, language="church")
