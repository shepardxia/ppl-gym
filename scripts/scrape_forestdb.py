"""
Scrape ForestDB models into structured JSONL.

Filters to WebPPL-only models. Preserves full document structure
as ordered prose/code sections. Source files in data/sources/forestdb.org/
are the lossless layer — this script produces a parsed view.

Usage:
    python scripts/scrape_forestdb.py
"""

import json
import re
from pathlib import Path

SOURCES_DIR = Path("data/sources/forestdb.org/models")
OUTPUT_FILE = Path("data/raw/forestdb.jsonl")


def parse_frontmatter(text: str):
    """Extract YAML frontmatter and remaining body from markdown."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    yaml_str, body = match.group(1), match.group(2)

    # Simple YAML parser — frontmatter is flat key-value pairs
    frontmatter = {}
    for line in yaml_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.strip().strip("'\"")
            frontmatter[key.strip()] = value

    return frontmatter, body


def parse_sections(body: str) -> list[dict]:
    """Parse markdown body into ordered prose/code sections.

    ForestDB uses indented code blocks (tab or 4+ spaces).
    Also handles fenced code blocks (``` or ~~~~) as a fallback.
    """
    sections = []
    current_type = None
    current_lines = []

    def flush():
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"type": current_type, "content": content})

    in_fenced = False
    fence_marker = None

    for line in body.split("\n"):
        # Handle fenced code blocks
        if not in_fenced:
            fence_match = re.match(r"^(````|~~~~|```|~~~)", line)
            if fence_match:
                flush()
                current_lines = []
                current_type = "code"
                in_fenced = True
                fence_marker = fence_match.group(1)
                continue
        else:
            if line.strip().startswith(fence_marker):
                flush()
                current_lines = []
                current_type = None
                in_fenced = False
                fence_marker = None
                continue
            current_lines.append(line)
            continue

        # Handle indented code blocks (tab or 4+ spaces)
        is_code_line = line.startswith("\t") or line.startswith("    ")
        is_blank = line.strip() == ""

        if is_code_line:
            # Strip one level of indentation
            if line.startswith("\t"):
                stripped = line[1:]
            else:
                stripped = line[4:]

            if current_type != "code":
                flush()
                current_lines = []
                current_type = "code"
            current_lines.append(stripped)

        elif is_blank:
            # Blank lines: keep in current section to preserve structure
            current_lines.append("")

        else:
            if current_type != "prose":
                flush()
                current_lines = []
                current_type = "prose"
            current_lines.append(line)

    flush()
    return sections


def parse_tags(tag_string: str) -> list[str]:
    """Parse comma-separated tags."""
    if not tag_string:
        return []
    return [t.strip() for t in tag_string.split(",") if t.strip()]


def scrape_model(filepath: Path):
    """Parse a single ForestDB model file into a structured record."""
    text = filepath.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)

    # Filter: WebPPL only
    language = frontmatter.get("model-language", "").lower()
    if language != "webppl":
        return None

    # Filter: must have actual code
    status = frontmatter.get("model-status", "")
    if status in ("stub", "link"):
        return None

    sections = parse_sections(body)

    # Check that at least one code section exists
    has_code = any(s["type"] == "code" for s in sections)
    if not has_code:
        return None

    return {
        "id": f"forestdb/{filepath.stem}",
        "source": "forestdb",
        "source_file": str(filepath),
        "frontmatter": frontmatter,
        "title": frontmatter.get("title", filepath.stem),
        "language_version": frontmatter.get("model-language-version", ""),
        "category": frontmatter.get("model-category", ""),
        "tags": parse_tags(frontmatter.get("model-tags", "")),
        "sections": sections,
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not SOURCES_DIR.exists():
        print(f"ERROR: {SOURCES_DIR} not found. Clone forestdb.org first:")
        print("  git clone https://github.com/forestdb/forestdb.org.git data/sources/forestdb.org")
        return

    model_files = sorted(SOURCES_DIR.glob("*.md"))
    print(f"Found {len(model_files)} model files")

    records = []
    # scrape_model returns None for both stub/link and no-code models; main()
    # can't tell them apart, so both fall under no_code.
    skipped = {"non_webppl": 0, "no_code": 0}

    for filepath in model_files:
        text = filepath.read_text(encoding="utf-8")
        frontmatter, _ = parse_frontmatter(text)
        lang = frontmatter.get("model-language", "").lower()

        if lang != "webppl":
            skipped["non_webppl"] += 1
            continue

        record = scrape_model(filepath)
        if record is None:
            skipped["no_code"] += 1
            continue

        records.append(record)

    with open(OUTPUT_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nResults:")
    print(f"  WebPPL models extracted: {len(records)}")
    print(f"  Skipped (non-WebPPL):    {skipped['non_webppl']}")
    print(f"  Skipped (no code):       {skipped['no_code']}")
    print(f"  Output: {OUTPUT_FILE}")

    # Print category distribution
    categories = {}
    for r in records:
        cat = r["category"] or "(uncategorized)"
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
