"""Extract atoms from prose-with-code WebPPL sources.

Works for any markdown source where each "atom candidate" is a code
fence preceded by prose: probmods2 chapters, DIPPL chapters, ProbLang
chapters, ForestDB models. The extractor:

  1. Walks each .md file, splits into (preceding_prose, code_block) pairs.
  2. Wraps each code block so its last expression is bound to ANSWER.
  3. Runs it. If it produces a Distribution / value, emits an atom.
  4. Otherwise skips with a reason logged.

Pedagogical sources (chapters, models) won't all atomize cleanly.
Iterate per source.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.executor import execute_webppl
from eval.io import write_jsonl


CODE_FENCE = re.compile(r"^~~~~?\s*$|^```\s*(?:js|webppl|norun)?\s*$", re.MULTILINE)


# Per-source extraction config.
SOURCES = {
    "probmods_chapters": {
        "dir": "data/sources/probmods2/chapters",
        "id_prefix": "probmods2-chapters",
        "label": "probmods2 chapter",
    },
    "dippl": {
        "dir": "data/sources/dippl/chapters",
        "id_prefix": "dippl",
        "label": "DIPPL chapter",
    },
    "problang": {
        "dir": "data/sources/problang/chapters",
        "id_prefix": "problang",
        "label": "ProbLang chapter",
    },
    "forestdb": {
        "dir": "data/sources/forestdb.org/models",
        "id_prefix": "forestdb",
        "label": "ForestDB model",
    },
}


def split_blocks(text: str):
    """Yield (preceding_prose, code) pairs from a chapter markdown.

    Detects ~~~~ / ~~~ / ``` fences. Skips fences flagged `norun`.
    """
    lines = text.split("\n")
    in_fence = False
    fence_buf: list[str] = []
    prose_buf: list[str] = []
    skip_block = False
    for ln in lines:
        stripped = ln.strip()
        is_fence = (
            stripped.startswith("~~~~") or stripped.startswith("~~~")
            or stripped.startswith("```")
        )
        if is_fence:
            if not in_fence:
                in_fence = True
                fence_buf = []
                skip_block = "norun" in stripped.lower()
            else:
                in_fence = False
                if not skip_block and fence_buf:
                    prose = "\n".join(prose_buf).strip()
                    code = "\n".join(fence_buf)
                    yield prose, code
                # Reset prose accumulator AFTER yielding so prior prose
                # isn't reused for the next block (it's already attached).
                prose_buf = []
                fence_buf = []
                skip_block = False
            continue
        if in_fence:
            fence_buf.append(ln)
        else:
            prose_buf.append(ln)


def looks_like_full_program(code: str) -> bool:
    """Cheap filter: program should have either Infer or a clear expression
    that returns something interesting (not just a fragment)."""
    s = code.strip()
    if not s:
        return False
    # Must contain something that produces an answer-like value
    has_infer = "Infer(" in s or "viz(" in s or "viz." in s
    has_print = "print(" in s
    has_repeat = "repeat(" in s
    return has_infer or has_print or has_repeat


def strip_viz_print(code: str) -> str:
    """Remove final viz/print wrappers so we can rebind to ANSWER."""
    s = code.rstrip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    # Common patterns: `viz(X)` `viz.table(X)` `print(X)`
    m = re.match(r"^(.*?)(viz(?:\.[a-zA-Z]+)?\s*\(\s*)(.+)\)\s*$", s, re.DOTALL)
    if m:
        prefix, _wrapper, inner = m.groups()
        return prefix + inner
    m = re.match(r"^(.*?)print\s*\(\s*(.+)\)\s*$", s, re.DOTALL)
    if m:
        prefix, inner = m.groups()
        return prefix + inner
    return s


def find_last_expression(code: str) -> tuple[str, str]:
    """Best-effort split: everything except the last top-level expression
    goes in 'before'; the last expression goes in 'last'.

    For chapter blocks the last statement is usually a viz/print or a
    bare expression (Infer call, repeat call, etc.).
    """
    code = code.rstrip()
    if code.endswith(";"):
        code = code[:-1].rstrip()
    # Find last top-level boundary (semicolon or `}` followed by newline)
    depth = 0
    in_str = False
    sc = None
    last = 0
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == sc:
                in_str = False
            i += 1
            continue
        if c in '"\'':
            in_str = True
            sc = c
        elif c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
            if depth == 0 and c in ")}]":
                # ASI-ish: treat as boundary if followed by newline + statement-start
                j = i + 1
                while j < n and code[j] in " \t\r":
                    j += 1
                if j < n and code[j] == "\n":
                    while j < n and code[j] in " \t\r\n":
                        j += 1
                    if j < n and code[j] not in ".[(,?:":
                        last = i + 1
        elif c == ";" and depth == 0:
            last = i + 1
        i += 1
    if last == 0:
        return "", code.strip()
    return code[:last].rstrip(), code[last:].strip()


def wrap_with_answer(code: str) -> str | None:
    """Return code with `var ANSWER = (last_expr);` appended.

    Returns None if we couldn't identify a sensible last expression.
    """
    cleaned = strip_viz_print(code)
    before, last = find_last_expression(cleaned)
    if not last:
        return None
    return f"{before}\nvar ANSWER = ({last});\n" if before else f"var ANSWER = ({last});\n"


def format_answer_shape_hint(shape) -> str:
    """One-line spec of the expected ANSWER output type for the prompt.

    The atom's `answer_shape` is determined by what the GT actually
    returns (we know it because we ran it). Telling the LLM the shape
    is task spec — like a return-type annotation — not answer leakage.
    """
    if shape == "distribution":
        return ("a single distribution object (e.g., the return of "
                "`Infer({...}, function() { ... })`)")
    if shape == "samples":
        return ("a list of samples — typically the return of "
                "`repeat(N, function() { ... })` or equivalent")
    if shape == "value":
        return "a single scalar or short fixed-size tuple (number, string, boolean, or list of those)"
    return "the answer"


def classify_answer(answer) -> tuple[str, object]:
    """Return (answer_shape, eval_mode) by inspecting the executed answer.

    Short lists (≤4 entries) of scalars are treated as structured *values*
    (e.g. an HDI `[low, up]`, a (mean, var) tuple) rather than samples,
    since two-point empirical distributions don't carry meaningful TV.
    """
    if isinstance(answer, dict) and answer.get("__kind") == "distribution":
        return "distribution", "distribution"
    if isinstance(answer, dict) and answer.get("__kind") == "distribution_continuous":
        # Hard to compare; treat as distribution-repr (will mostly mismatch).
        return "distribution", "distribution"
    if isinstance(answer, list):
        if len(answer) <= 4 and all(not isinstance(x, (list, dict)) for x in answer):
            return "value", "value"
        return "samples", "samples"
    return "value", "value"


_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
_DATA_URI_RE = re.compile(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
# Closed HTML comments. Note: prose can be truncated mid-comment by
# `truncate_prose`, leaving an unclosed `<!--`; we strip those too.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_COMMENT_OPEN_RE = re.compile(r"<!--[\s\S]*$")
# Jekyll Liquid templates: `{% include %}`, `{{ site.baseurl }}`. The
# plugin renders these to URLs / partials; raw they're noise.
_LIQUID_BLOCK_RE = re.compile(r"\{%[\s\S]*?%\}")
_LIQUID_VAR_RE = re.compile(r"\{\{[^}]*\}\}")
# Citation tokens — both Jekyll plugin syntax (problang) and Pandoc
# bracket-free citations (probmods).
_REFT_RE = re.compile(r"\breft:[A-Za-z0-9_:-]+")
_PANDOC_CITE_RE = re.compile(r"@[A-Z][A-Za-z]+\d{4}\w*")
# Cross-chapter markdown links `[Chapter 2](02-pragmatics.html)`. The
# referenced page isn't in our context, so we drop the URL and keep
# only the visible link text.
# Match `[text](path/file.html)`, `[text](file.html#anchor)`,
# `[text](/foo/bar.html#anchor)` — local relative-path markdown links.
# Excludes http:// / https:// (we leave external URLs alone).
_CHAPTER_LINK_RE = re.compile(r"\[([^\]]+)\]\((?!https?:)[^)]*?\.html(?:#[^)]*)?\)")
# Wiki-style cross-chapter links `[[Section Title#anchor|display]]` or
# `[[plain]]`. Use the post-pipe text if present, else the inner.
_WIKI_LINK_RE = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")


def sanitize_prose(prose: str) -> str:
    """Strip noise that rendered to nothing in the source toolchain but
    leaks into our prompts as raw template/citation/comment syntax.
    """
    prose = _IMG_TAG_RE.sub("[image]", prose)
    prose = _DATA_URI_RE.sub("[image]", prose)
    prose = _HTML_COMMENT_RE.sub("", prose)
    prose = _HTML_COMMENT_OPEN_RE.sub("", prose)
    prose = _LIQUID_BLOCK_RE.sub("", prose)
    prose = _LIQUID_VAR_RE.sub("", prose)
    prose = _REFT_RE.sub("(citation)", prose)
    prose = _PANDOC_CITE_RE.sub("(citation)", prose)
    prose = _CHAPTER_LINK_RE.sub(lambda m: m.group(1), prose)
    prose = _WIKI_LINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), prose)
    return prose


def truncate_prose(prose: str, max_chars: int = 1200) -> str:
    """Keep the last few paragraphs as the prompt context."""
    prose = sanitize_prose(prose)
    paragraphs = re.split(r"\n\s*\n", prose.strip())
    out = []
    total = 0
    for p in reversed(paragraphs):
        p = p.strip()
        if not p or p.startswith("#"):
            continue
        if total + len(p) > max_chars and out:
            break
        out.insert(0, p)
        total += len(p)
    return "\n\n".join(out)


def build_atom(chapter_name: str, idx: int, prose: str, code: str,
               *, id_prefix: str, label: str, source_subpath: str,
               preceding_code_blocks: list[str] = None,
               timeout: int = 30) -> dict | None:
    """Build an atom from a single code block.

    `preceding_code_blocks` is the list of all earlier code blocks in the
    same file. They're (a) prepended to the GT code so it actually runs,
    and (b) shown to the LLM as "given code" context so it can reference
    definitions established earlier in the chapter / model.
    """
    preceding_code_blocks = preceding_code_blocks or []

    # Try with full preamble first. If concatenation produces errors
    # (duplicate var, conflicting definitions), fall back to standalone.
    def _try(preamble_blocks):
        preamble = "\n\n".join(preamble_blocks).rstrip()
        full = (preamble + "\n\n" + code) if preamble else code
        wrapped = wrap_with_answer(full)
        if wrapped is None:
            return None, None, preamble
        res = execute_webppl(wrapped, timeout=timeout, random_seed=42)
        if res.success and res.answer is not None:
            return wrapped, res.answer, preamble
        return None, None, preamble

    wrapped, answer, preamble = _try(preceding_code_blocks)
    if wrapped is None and preceding_code_blocks:
        wrapped, answer, preamble = _try([])
    if wrapped is None:
        return None
    shape, mode = classify_answer(answer)

    prompt_prose = truncate_prose(prose)
    if not prompt_prose:
        return None

    # Show preceding code blocks to the LLM verbatim.
    prefix_section = ""
    if preamble:
        prefix_section = (
            f"The following code is given (definitions established "
            f"earlier in the same source):\n\n```\n{preamble}\n```\n\n"
        )

    atom_id = f"{id_prefix}-{chapter_name}/block-{idx}"
    return {
        "id": atom_id,
        "source": source_subpath,
        "task_type": "write_from_scratch",
        "eval_mode": mode,
        "answer_shape": shape,
        "prompt": (
            f"{prompt_prose}\n\n"
            f"{prefix_section}"
            f"Write a WebPPL program that demonstrates what the prose "
            f"asks for."
            + (" You may rely on the given code above." if preamble else "")
            + f" End your program with `var ANSWER = <expression>;` "
            f"where the value of ANSWER is {format_answer_shape_hint(shape)}."
        ),
        "groundtruth_code": wrapped,
        "groundtruth_output": answer,
    }


def process_chapter(path: Path, *, id_prefix: str, label: str,
                    source_subpath_prefix: str,
                    timeout: int = 30, max_blocks: int | None = None,
                    workers: int = 1):
    try:
        text = path.read_text()
    except Exception:
        return [], 0, 0
    blocks = list(split_blocks(text))
    if max_blocks:
        blocks = blocks[:max_blocks]
    chapter_name = path.stem
    source_subpath = f"{source_subpath_prefix}/{path.name}"
    # We'll attempt each candidate with a cumulative preamble of all
    # earlier code blocks in the same file (so block N has access to
    # definitions in blocks 0..N-1).
    candidates = []
    code_history = []
    for i, (prose, code) in enumerate(blocks):
        if looks_like_full_program(code):
            candidates.append((i, prose, code, list(code_history)))
        code_history.append(code)

    atoms: list[dict] = []

    def _build(args):
        i, prose, code, preceding = args
        return build_atom(
            chapter_name, i, prose, code,
            id_prefix=id_prefix, label=label,
            source_subpath=source_subpath,
            preceding_code_blocks=preceding,
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_build, c): c for c in candidates}
        for fut in as_completed(futs):
            try:
                atom = fut.result()
            except Exception:
                atom = None
            if atom is not None:
                atoms.append(atom)
    atoms.sort(key=lambda a: a["id"])
    return atoms, len(blocks), len(candidates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(SOURCES.keys()), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-blocks-per-chapter", type=int, default=None)
    p.add_argument("--chapters", nargs="+", default=None,
                   help="Restrict to specific stems")
    args = p.parse_args()

    cfg = SOURCES[args.source]
    src = Path(cfg["dir"])
    chapters = sorted(src.glob("*.md"))
    if args.chapters:
        wanted = set(args.chapters)
        chapters = [c for c in chapters if c.stem in wanted]

    # Source-subpath prefix used in `source` field of each atom
    src_prefix = src.relative_to("data/sources").as_posix() if str(src).startswith("data/sources") else cfg["id_prefix"]

    print(f"Processing {len(chapters)} files from {args.source}, workers={args.workers}")
    all_atoms: list[dict] = []
    t0 = time.time()
    for chap in chapters:
        t_chap = time.time()
        atoms, n_blocks, n_runnable = process_chapter(
            chap, id_prefix=cfg["id_prefix"], label=cfg["label"],
            source_subpath_prefix=src_prefix,
            timeout=args.timeout, max_blocks=args.max_blocks_per_chapter,
            workers=args.workers,
        )
        all_atoms.extend(atoms)
        print(f"  {chap.stem:40s} blocks={n_blocks:3d} runnable={n_runnable:3d} "
              f"atoms={len(atoms):3d} ({time.time() - t_chap:.1f}s)")

    write_jsonl(args.output, all_atoms)
    print(f"\nWrote {len(all_atoms)} atoms to {args.output} "
          f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
