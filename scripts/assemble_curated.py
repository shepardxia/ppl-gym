"""Assemble curated atoms from agent emissions.

Agent emissions JSONL: each row has
  {id, source, source_block_indices, prompt,
   wrap_target (optional but recommended),
   answer_shape (optional override: "value" | "distribution" | "samples"),
   notes (optional)}

When `answer_shape` is provided, it overrides the heuristic in
`classify_answer`. The heuristic is purely answer-driven ("long list →
samples") and can't distinguish a list-of-IID-samples from a
list-as-data-structure (e.g., a random-walk trajectory). The agent has the
context to make that call; the pipeline shouldn't second-guess it.

For each row, this script:
  1. Reads the source markdown file and splits into code blocks.
  2. Concatenates the listed blocks (in the order the agent listed them,
     deduped while preserving first-seen order).
  3. Prepends display/canvas/print stubs so source code referencing
     viz/Draw/drawLines/drawPoints/print runs cleanly in a headless
     executor (those calls become no-ops; the value of ANSWER is unaffected).
  4. If `wrap_target` is provided, appends `var ANSWER = (<wrap_target>);`
     literally. Otherwise falls back to `wrap_with_answer` (which uses a
     regex-based last-expression scan; fragile when source has multiple
     top-level statements).
  5. Runs via `execute_webppl(seed=42)`.
  6. On success: emits the full atom record to --output.
     On failure: emits the broken record to --broken with the error and
     the agent's notes for triage.

Agents own judgment (chunking, prompt, wrap_target, notes). Pipeline owns
mechanics (concat, stub, wrap, execute, classify).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.executor import execute_webppl
from eval.io import write_jsonl
from eval.algebra import canonicalize, distance, parse_spec
from scripts.extract_atoms import (
    classify_answer, find_last_expression, split_blocks,
    strip_viz_print, wrap_with_answer,
)

def empirical_tv(gen_samples: list, gt_samples: list) -> float | None:
    """TV between two empirical sample sets, via the canonical comparator."""
    if not gen_samples or not gt_samples:
        return None
    spec = parse_spec({"kind": "dist", "domain": "finite"})
    return distance(
        canonicalize(gen_samples, spec), canonicalize(gt_samples, spec), spec
    ).value



# Second seed used by the determinism gate; chosen far from 42 so any
# `flip`/`gaussian`/... call that leaks into a `value`-shape GT produces a
# different sample stream.
_DETERMINISM_SEED = 1337
_DUP_VARS_SCRIPT = Path(__file__).resolve().parent / "_check_dup_vars.js"
# TV threshold above which two seeded runs of a samples-shape GT are
# considered "the GT can't agree with itself," meaning the eval comparison
# (which uses the same metric) can't ever score it well. 0.5 is the same
# cutoff as the eval's worst non-degenerate bucket ("X").
_SAMPLES_SELF_TV_MAX = 0.5


# Display/canvas/print stubs are provided by the executor as preloaded
# WebPPL packages (probmods-draw, probmods-viz-stub) — see eval/executor.py.
# WebPPL forbids top-level field assignments like `viz.table = ...`, so
# the stubs must be host-side globals, not in-program declarations.


def _dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _resolve_source(repo_root: Path, src: str) -> Path:
    p = Path(src)
    if p.is_absolute():
        return p
    candidate = repo_root / p
    if candidate.exists():
        return candidate
    candidate = repo_root / "data" / "sources" / p
    if candidate.exists():
        return candidate
    return repo_root / p


def _assemble_program(picked_codes: list[str], wrap_target: str | None) -> tuple[str | None, str | None]:
    """Return (wrapped_program, error). wrapped_program ends with var ANSWER = ...;"""
    body = "\n\n".join(picked_codes).strip()
    if not body and not wrap_target:
        return None, "all listed blocks are empty and no wrap_target provided"
    if wrap_target:
        wt = wrap_target.strip().rstrip(";").strip()
        # If the source's trailing expression is literally the same as the
        # agent's wrap_target, strip it so we don't double-evaluate (which
        # would consume random bits twice and yield a different sample
        # than the LM gets when faithfully following the prompt). This is
        # a conservative, literal match — if the trailing expression
        # differs, leave the body alone (the side-effect call is usually
        # a viz/Draw no-op via the stubs, or a deterministic enumerate
        # Infer; either way it's harmless).
        cleaned_body = _strip_trailing_match(body, wt)
        wrapped_body = f"{cleaned_body}\n\nvar ANSWER = ({wt});\n"
    else:
        guess = wrap_with_answer(body)
        if guess is None:
            return None, "wrap_target not provided and wrap_with_answer heuristic returned None"
        wrapped_body = guess
    return wrapped_body, None


def _strip_trailing_match(body: str, wrap_target: str) -> str:
    """Strip wrap_target's literal text from the end of body, if present.

    Conservative: only strips when the source's trailing expression is
    EXACTLY the wrap_target (modulo whitespace and an optional trailing
    semicolon). Doesn't try to strip viz/print wrappers or smart-match.
    """
    body = body.rstrip()
    target = wrap_target.rstrip().rstrip(";").rstrip()
    # Trim a trailing standalone `;` from body before comparing.
    body_no_semi = body[:-1].rstrip() if body.endswith(";") else body
    if body_no_semi.endswith(target):
        return body_no_semi[: -len(target)].rstrip()
    return body


def _find_duplicate_top_vars(code: str) -> list[dict] | None:
    """Return a list of {name, lines} for top-level `var X` declared >1 time.

    Returns None when the parse helper can't run (no node, missing script, or
    parser error) — callers should treat that as "gate unavailable" and let
    the atom through rather than rejecting on a tooling fault.
    """
    if not _DUP_VARS_SCRIPT.exists():
        return None
    try:
        r = subprocess.run(
            ["node", str(_DUP_VARS_SCRIPT)],
            input=code, capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        payload = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    if not payload.get("ok"):
        return None
    return payload.get("dupes") or []


def _check_samples_self_consistency(wrapped: str, first_answer, *, timeout: int) -> tuple[bool, str | None]:
    """Re-run a samples-shape GT at a different seed; True iff TV between
    the two empirical distributions is below `_SAMPLES_SELF_TV_MAX`.

    Catches atoms where the eval pipeline can't score generated samples
    against the GT because the GT itself is too noisy across seeds — either
    because the generating distribution has unconditioned random parameters
    (dippl-05-particlefilter/atom-4: random mixture per run) or because the
    "samples" are actually a structured sequence whose values almost never
    collide on exact keys (atom-2/atom-3: continuous random walks).
    """
    if not isinstance(first_answer, list):
        return True, None  # nothing to compare; let it through
    result = execute_webppl(wrapped, timeout=timeout, random_seed=_DETERMINISM_SEED)
    if not (result.success and result.answer is not None):
        return False, f"samples self-consistency re-run failed: {result.error_message or 'no answer'}"
    if not isinstance(result.answer, list):
        return False, f"samples self-consistency: second run returned non-list ({type(result.answer).__name__})"
    tv = empirical_tv(first_answer, result.answer)
    if tv is None:
        return True, None
    if tv > _SAMPLES_SELF_TV_MAX:
        return False, (
            f"samples self-consistency failed: TV(seed=42, seed={_DETERMINISM_SEED}) "
            f"= {tv:.3f} > {_SAMPLES_SELF_TV_MAX} (GT's own seeded runs can't agree; "
            f"eval against generated code can't score better than this)"
        )
    return True, None


def _check_value_determinism(wrapped: str, first_answer, *, timeout: int) -> tuple[bool, str | None]:
    """Re-run the GT with a different seed; True iff the answer is byte-equal.

    Only meaningful for `value`-shape atoms — sampled/distribution-shape GTs
    legitimately depend on the seed. On executor failure, returns
    (False, error) so the atom lands in broken with a clear reason.
    """
    result = execute_webppl(wrapped, timeout=timeout, random_seed=_DETERMINISM_SEED)
    if not (result.success and result.answer is not None):
        return False, f"determinism re-run failed: {result.error_message or 'no answer'}"
    if result.answer == first_answer:
        return True, None
    return False, f"value differs across seeds (seed=42 vs seed={_DETERMINISM_SEED}): {first_answer!r} vs {result.answer!r}"


def assemble(emissions_path: Path, output_path: Path, broken_path: Path,
             *, timeout: int = 60) -> tuple[int, int]:
    repo_root = Path(__file__).resolve().parent.parent
    emissions = [
        json.loads(l) for l in emissions_path.read_text().splitlines() if l.strip()
    ]

    atoms: list[dict] = []
    broken: list[dict] = []
    block_cache: dict[str, list[tuple[str, str]]] = {}

    t0 = time.time()
    for em in emissions:
        em_id = em.get("id", "<no-id>")
        src = em.get("source", "")
        src_path = _resolve_source(repo_root, src)
        if not src_path.exists():
            broken.append({**em, "error": f"source file not found: {src}"})
            continue

        key = str(src_path)
        if key not in block_cache:
            block_cache[key] = list(split_blocks(src_path.read_text()))
        blocks = block_cache[key]

        idxs_raw = em.get("source_block_indices", [])
        if not isinstance(idxs_raw, list) or not all(isinstance(i, int) for i in idxs_raw):
            broken.append({**em, "error": f"source_block_indices must be a list of ints, got {idxs_raw!r}"})
            continue
        idxs = _dedupe_keep_order(idxs_raw)
        try:
            picked_codes = [blocks[i][1] for i in idxs]
        except IndexError:
            broken.append({**em, "error": f"block index out of range (file has {len(blocks)} blocks, got {idxs})"})
            continue

        wrap_target = em.get("wrap_target")
        wrapped, err = _assemble_program(picked_codes, wrap_target)
        if wrapped is None:
            broken.append({**em, "error": err})
            continue

        result = execute_webppl(wrapped, timeout=timeout, random_seed=42)
        if not (result.success and result.answer is not None):
            broken.append({
                **em,
                "error": result.error_message or "execution failed (no answer)",
                "stderr_tail": (result.stderr or "")[-500:],
                "wrapped_code": wrapped,
            })
            continue

        # Gate: duplicate top-level `var X` declarations. Overlapping
        # source_block_indices stitch together GTs that compile by accident
        # (JS allows duplicate `var`), but the prompt can't tell the LM
        # which definition is canonical — see dippl-04-factorseq/atom-1.
        dupes = _find_duplicate_top_vars(wrapped)
        if dupes:
            broken.append({
                **em,
                "error": f"duplicate top-level var declarations: {dupes}",
                "wrapped_code": wrapped,
            })
            continue

        shape_override = em.get("answer_shape")
        if shape_override is not None:
            if shape_override not in ("value", "distribution", "samples"):
                broken.append({
                    **em,
                    "error": f"invalid answer_shape override: {shape_override!r} (expected value/distribution/samples)",
                    "wrapped_code": wrapped,
                })
                continue
            shape, mode = shape_override, shape_override
        else:
            shape, mode = classify_answer(result.answer)

        # Gate: a `value`-shape answer must be deterministic — otherwise the
        # comparator demands exact-equality on a stochastic output, which is
        # untestable by design (see dippl-02-webppl/atom-1: `geometric(0.5)`
        # shaped as value).
        if shape == "value":
            ok_det, det_err = _check_value_determinism(wrapped, result.answer, timeout=timeout)
            if not ok_det:
                broken.append({
                    **em,
                    "error": det_err,
                    "wrapped_code": wrapped,
                    "answer_at_seed_42": result.answer,
                })
                continue

        # Gate: a `samples`-shape answer must agree with itself across seeds
        # under the same TV metric the eval uses. Otherwise the eval can't
        # score generated code better than the GT's own self-noise — see
        # dippl-05-particlefilter/{atom-2,3,4}.
        if shape == "samples":
            ok_s, s_err = _check_samples_self_consistency(wrapped, result.answer, timeout=timeout)
            if not ok_s:
                broken.append({
                    **em,
                    "error": s_err,
                    "wrapped_code": wrapped,
                    "answer_at_seed_42": result.answer,
                })
                continue

        atom = {
            "id": em_id,
            "source": src,
            "source_block_indices": idxs,
            "task_type": "write_from_scratch",
            "eval_mode": mode,
            "answer_shape": shape,
            "prompt": em.get("prompt", ""),
            "groundtruth_code": wrapped,
            "groundtruth_output": result.answer,
        }
        if wrap_target:
            atom["wrap_target"] = wrap_target
        if em.get("notes"):
            atom["notes"] = em["notes"]
        atoms.append(atom)

        elapsed = time.time() - t0
        print(f"  [{len(atoms) + len(broken)}/{len(emissions)}] {em_id:50s} OK ({elapsed:.1f}s)", flush=True)

    write_jsonl(output_path, atoms)
    write_jsonl(broken_path, broken)
    return len(atoms), len(broken)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emissions", required=True, help="Agent JSONL: {id, source, source_block_indices, prompt, wrap_target, notes}")
    p.add_argument("--output", required=True, help="Output JSONL of fully-assembled atoms")
    p.add_argument("--broken", required=True, help="Output JSONL of emissions that failed assembly")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    n_ok, n_broken = assemble(
        Path(args.emissions), Path(args.output), Path(args.broken), timeout=args.timeout,
    )
    print(f"\nDone: {n_ok} OK, {n_broken} broken")
    print(f"  → {args.output}")
    print(f"  → {args.broken}")


if __name__ == "__main__":
    main()
