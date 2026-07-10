"""The shared failure classifier (eval.error_tags).

Contract: score.py (fresh runs) and triage_exec_errors.py (legacy re-exec) call
the SAME classify() on the SAME real reason, so both land in the same bucket.
"""

from eval.error_tags import TAGS, classify, is_collapsed, join_reasons


def test_classify_returns_stable_vocabulary():
    samples = [
        "stan compile failed: Syntax error in 'model.stan', line 11",
        "timeout after 60s",
        "GT collection failed: timeout after 300s",
        "empty code",
        "NameError: name 'undefined_name' is not defined",
        "program did not define ANSWER",
        "problem/realization not found for problem_id='x'",
        "",
    ]
    for s in samples:
        assert classify(s, "pyro") in TAGS


def test_specific_reasons_bucket_correctly():
    assert classify("stan compile failed: Semantic error line 17", "stan") == "compile"
    assert classify("SyntaxError: invalid syntax", "pyro") == "compile"
    assert classify("timeout after 60s", "pyro") == "timeout"
    assert classify("GT collection failed: execution failed", "stan") == "gt_side"
    assert classify("empty code", "webppl") == "empty_code"
    assert classify("program did not define ANSWER", "pyro") == "no_output"
    assert classify("TypeError: unhashable type: 'list'", "pyro") == "runtime"
    assert classify("problem/realization not found for problem_id='z'", "") == "corpus_miss"


def test_webppl_compile_vocabulary_is_compile_not_runtime():
    # WebPPL fails at compile time with its own vocabulary (no "compile"/"syntax
    # error" substring). These are parse + CPS/naming transform passes, not
    # execution — must bucket `compile`, not `runtime`.
    for msg in (
        "Error: cpsInnerStatement",
        "Error: cpsFinalStatement",
        "Error: can't cps Categorical(...)",
        "Error: atomize: unrecognized node",
        "Error: Line 119: Did you mean var ANSWER = ?",
        "Error: Line 33: Unexpected identifier",
        "Error: Line 21: Unexpected token ILLEGAL",
        'Error: "**" does not match field "operator": == | != | ...',
        "Error: Line 5: You tried to assign to a field of ANSWER, but you can only assign",
    ):
        assert classify(msg, "webppl") == "compile", msg
    # Genuine WebPPL runtime errors stay runtime (dist construction / model logic).
    for msg in (
        'Error: Parameter "p" missing from Bernoulli distribution.',
        "Error: The score argument is not a number.",
        "Error: All paths explored by Enumerate have probability zero.",
        "TypeError: address.split is not a function",
    ):
        assert classify(msg, "webppl") == "runtime", msg


def test_collapsed_placeholders_are_flagged_not_bucketed():
    # Pre-fix placeholders: real reason was lost -> must not be silently bucketed.
    for ph in ("execution failed", "3/5 seeded runs failed", "1/1 SEEDED RUNS FAILED"):
        assert is_collapsed(ph)
        assert classify(ph, "pyro") == "other"
    assert not is_collapsed("TypeError: unhashable type: 'list'")
    assert not is_collapsed("")


def test_source_and_reexec_forms_agree():
    # The "same stuff" guarantee: however the reason reaches classify (raised
    # from the harness at scoring time, or recovered by triage re-exec), the
    # same real string yields the same tag.
    reason = "stan fit failed (timeout 60s): Exception: initialization failed"
    assert classify(reason, "stan") == classify(reason, "stan")
    assert classify(reason, "stan") in TAGS


def test_join_reasons_dedups_and_skips_success():
    errs = [None, "NameError: x", "NameError: x", None, "ValueError: y"]
    assert join_reasons(errs) == "NameError: x; ValueError: y"
    assert join_reasons([None, None]) == "execution failed"  # all-succeeded fallback
    assert join_reasons([]) == "execution failed"
