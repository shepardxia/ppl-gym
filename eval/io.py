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


