"""The status vocabulary (SCHEMA.md §Status vocabulary) must be covered by
web/src/lib/tones.ts — this is the drift-catcher for the cross-language
mapping that can't be import-checked."""

from pathlib import Path

# One entry per status a consumer can see; mirrors the SCHEMA.md table.
JUDGE_STATUSES = ["pass", "fail", "ill_posed", "malformed", "exec_error"]
PHASE_A_STATUSES = ["ok", "ill_posed", "error"]
PHASE_B_STATUSES = ["accept", "gt_suspect", "underdetermined", "solver_failure"]
REVIEW_STATUSES = ["draft", "reviewed", "retired"]

ALL_STATUSES = sorted(
    set(JUDGE_STATUSES + PHASE_A_STATUSES + PHASE_B_STATUSES + REVIEW_STATUSES)
)

_TONES_TS = Path(__file__).resolve().parent.parent / "web" / "src" / "lib" / "tones.ts"


def test_tones_ts_covers_the_documented_vocabulary():
    src = _TONES_TS.read_text()
    missing = [s for s in ALL_STATUSES if f"case '{s}':" not in src]
    assert not missing, f"tones.ts missing status cases: {missing}"


def test_schema_documents_the_same_vocabulary():
    schema = (_TONES_TS.parents[3] / "data" / "SCHEMA.md").read_text()
    section = schema.split("## Status vocabulary")[1]
    missing = [s for s in ALL_STATUSES if f"`{s}`" not in section]
    assert not missing, f"SCHEMA.md §Status vocabulary missing: {missing}"
