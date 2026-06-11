"""Unit tests for gate report mechanics (no WebPPL, no LLMs)."""

from pathlib import Path

from eval.gate import _merge_report
from eval.io import load_jsonl


def test_merge_report_updates_without_dropping(tmp_path: Path):
    report = tmp_path / "report.jsonl"

    n = _merge_report(report, [
        {"problem_id": "b", "status": "accept"},
        {"problem_id": "a", "status": "gt_suspect"},
    ])
    assert n == 2

    # partial re-judge: updates one row, adds one, drops none
    n = _merge_report(report, [
        {"problem_id": "a", "status": "accept"},
        {"problem_id": "c", "status": "accept"},
    ])
    assert n == 3

    rows = load_jsonl(report)
    assert [r["problem_id"] for r in rows] == ["a", "b", "c"]  # sorted
    assert {r["problem_id"]: r["status"] for r in rows} == {
        "a": "accept", "b": "accept", "c": "accept",
    }
