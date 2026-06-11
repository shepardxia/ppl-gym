"""End-to-end smoke test for eval/score.py (problem-centric path).

Uses problem probmods2-generative-models/ex4.b (finite enumeration — fast,
deterministic, no MCMC) with its own realization code as the candidate
(gt-vs-self). Expects status == "pass" and pass_rate == 1.0.

WebPPL executions are expected (~seconds each).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.corpus import load_corpus
from eval.score import run_scoring


TARGET_ID = "probmods2-generative-models/ex4.b"


def _get_realization_code(tmp_path: Path) -> str:
    """Look up the GT code for the target problem from the corpus."""
    problems, realizations = load_corpus({TARGET_ID})
    assert problems, f"problem {TARGET_ID!r} not found in corpus"
    assert realizations, f"realization for {TARGET_ID!r} not found"
    return realizations[0]["code"]


def test_gt_vs_self_pass(tmp_path: Path) -> None:
    """Scoring GT code against itself must yield status=pass and pass_rate=1.0."""
    gt_code = _get_realization_code(tmp_path)

    # Build a minimal generations JSONL: one row with the GT code.
    gen_path = tmp_path / "generations.jsonl"
    gen_path.write_text(
        json.dumps({"problem_id": TARGET_ID, "code": gt_code}) + "\n"
    )

    output_path = tmp_path / "scored.jsonl"
    summary = run_scoring(
        generations_path=gen_path,
        output_path=output_path,
        language="webppl",
        problem_ids={TARGET_ID},
        workers=1,
    )

    # Summary assertions
    assert summary["n"] == 1, f"expected 1 scored row, got {summary['n']}"
    assert summary["pass_rate"] == 1.0, (
        f"expected pass_rate=1.0, got {summary['pass_rate']}"
    )
    assert summary["pass"] == 1, f"expected 1 pass, got {summary['pass']}"

    # Row-level assertions
    rows = [
        json.loads(line)
        for line in output_path.read_text().splitlines()
        if line.strip() and not json.loads(line).get("summary")
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "pass", f"expected status=pass, got {row['status']!r}"
    assert row["problem_id"] == TARGET_ID
    # GT vs self: distance should be very small (algebra tolerance allows small noise floor).
    assert row["distance"] is not None
    assert row["distance"] < 0.5, f"distance too high for gt-vs-self: {row['distance']}"
    # code_jaccard of code vs itself must be 1.0
    assert row["code_jaccard"] == pytest.approx(1.0)
