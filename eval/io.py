"""Shared JSONL IO helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path | str) -> list[dict]:
    """Read a JSONL file. Skips blank and unparseable lines."""
    p = Path(path)
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_jsonl(path: Path | str, records: list[dict], *, append: bool = False):
    """Write records to JSONL. Creates parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def iter_scored(path: Path | str):
    """Yield non-summary scored records from a scored.jsonl, tolerating
    partial writes (mid-write reads of in-flight scoring)."""
    for rec in load_jsonl(path):
        if rec.get("summary") or "id" not in rec:
            continue
        yield rec


DEFAULT_ATOM_SPECS = Path("data/atom_specs.jsonl")


def load_atom_specs(path: Path | str | None = None) -> dict[str, dict]:
    """Load `atom_specs.jsonl` into `{atom_id: spec}`. Returns {} if missing."""
    p = Path(path) if path is not None else DEFAULT_ATOM_SPECS
    if not p.exists():
        return {}
    return {r["atom_id"]: r["spec"] for r in load_jsonl(p) if "atom_id" in r and "spec" in r}
