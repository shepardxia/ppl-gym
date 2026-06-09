"""Translate WebPPL atoms to Pyro via Anthropic Batch API.

For each input atom:
  1. Send (WebPPL prompt + WebPPL GT code + WebPPL GT output) to the model
     with a translation instruction. Response is a JSON object
     {prompt, groundtruth_code}.
  2. Execute the new GT code via execute_pyro to capture groundtruth_output.
  3. Compare to the original WebPPL GT output (semantic equivalence: same
     probs, support up to bool↔int representation).
  4. Emit success/broken records to separate JSONL files.

Usage:
  PYTHONPATH=. .venv/bin/python -m scripts.translate_to_pyro \
    --input data/atomized_v2.jsonl \
    --ids probmods2-conditioning/ex1.a probmods2-conditioning/ex5.b \
    --output data/pyro_v3/probmods.jsonl \
    --broken data/pyro_v3/_probmods_broken.jsonl \
    --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import Anthropic

from eval.executor_pyro import execute_pyro
from eval.io import load_jsonl, write_jsonl


PYRO_PRIMER = (Path(__file__).resolve().parent.parent / "data" / "prompts" / "pyro_primer.txt").read_text()


TRANSLATION_SYSTEM = """You are translating WebPPL probabilistic-programming atoms into Pyro (Python). \
You are given one WebPPL atom — its prompt, its ground-truth WebPPL code, and the output that code produced. \
Your job is to produce a Pyro version with the same semantics.

**Output format is strict**: emit exactly ONE fenced JSON block. Do NOT prefix with explanation or analysis. Do NOT add trailing prose. Your entire output must be parseable as JSON via a regex on the first ```json … ``` block. If you find yourself running long, you are over-explaining — get to the JSON.

```json
{
  "prompt": "<rewritten prompt suitable for instructing an LM to write Pyro code>",
  "groundtruth_code": "<Pyro code that produces the same answer as the WebPPL GT>"
}
```

Rules for the rewritten prompt:
- Keep the problem statement identical — same Bayesian model, same evidence, same answer.
- Change ALL syntactic instructions from WebPPL to Pyro. WebPPL's `var ANSWER = (Infer({method:'enumerate'}, model));` becomes Pyro's `ANSWER = dist.Categorical(probs=...)` or `ANSWER = {"__kind": "distribution", "probs": [...], "support": [...]}` for non-integer support.
- The prompt MUST instruct the LM to bind a top-level variable named `ANSWER`.
- The prompt should mention that `pyro`, `pyro.distributions as dist`, and `torch` are pre-imported.

Rules for the groundtruth_code:
- Imports are pre-injected: do NOT include `import pyro`, `import torch`, or `import pyro.distributions as dist` lines — assume they are already in scope. You MAY include `import itertools`, `import math`, or other stdlib imports.
- The code MUST end with a top-level `ANSWER = <expression>` assignment.
- For atoms whose WebPPL GT uses `Infer({method:'enumerate'},...)`, translate via manual enumeration over the discrete latent space, or analytical Bayesian inference, OR by constructing the posterior as a closed-form `dist.Categorical`. **Avoid MCMC for translatable-by-enumeration atoms** — exact enumeration is always preferable when the latent space is finite.
- For atoms whose answer has non-integer support (string labels, list-of-things, dict support items), construct the answer as a literal dict matching the cross-PPL schema: `ANSWER = {"__kind": "distribution", "probs": [...], "support": [...]}`. The support list must be SORTED canonically (alphabetical for strings, numeric for numbers, JSON-serialized for dicts).
- For `value`-shape atoms, bind ANSWER to a scalar / list / dict. The shape must match the WebPPL GT output exactly.
- For `samples`-shape atoms, bind ANSWER to a list of samples (length N matching the WebPPL GT). Boolean-valued WebPPL samples (`true`/`false`) translate to Python `True`/`False`. The comparator uses empirical TV, not element-wise equality — your samples need not be in the same order as WebPPL's, but the empirical distribution must match.
- For `record`-shape atoms (where the WebPPL GT returns `{key: distribution, ...}`), bind ANSWER to a dict whose values are the appropriate distribution objects or literal-dicts. Each value in the record must independently match the corresponding WebPPL distribution.
- For atoms whose WebPPL GT uses MCMC (`method: 'MCMC'`) and the answer is a `distribution`, if the underlying problem is exactly solvable, do exact enumeration instead. If MCMC is genuinely required (continuous posteriors, large discrete spaces), use Pyro's `MCMC`+`NUTS` with **modest** sample counts (≤500 samples + 200 warmup) and target the same marginal as the WebPPL atom.
- For atoms with continuous posteriors that need to be returned as a `distribution` shape, discretize the support to a small number of bins (≤30) and return as a `{__kind: "distribution"}` literal with `probs` and `support` being the bin centers. The WebPPL GT for such atoms typically already discretizes; mirror that bin scheme exactly.
- **Performance budget**: the GT must execute in under 120s. Avoid loops over thousands of samples in pure Python; prefer torch-vectorized operations.
- The Pyro program should be self-contained, deterministic (given the same seed via `pyro.set_rng_seed(seed)` if any randomness is used), and produce output byte-equal-modulo-floating-point with the WebPPL GT.

Pyro primer (for reference; do not include in your output):

""" + PYRO_PRIMER


def build_user_message(atom: dict) -> str:
    return (
        f"### WebPPL atom\n\n"
        f"id: {atom['id']}\n"
        f"answer_shape: {atom['answer_shape']}\n\n"
        f"WebPPL prompt:\n{atom['prompt']}\n\n"
        f"WebPPL groundtruth_code:\n```js\n{atom['groundtruth_code']}\n```\n\n"
        f"WebPPL groundtruth_output (the answer your Pyro code must reproduce):\n"
        f"```json\n{json.dumps(atom['groundtruth_output'], indent=2)[:4000]}\n```\n\n"
        f"Emit the JSON block now."
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def parse_response(text: str) -> dict | None:
    m = _JSON_FENCE_RE.search(text)
    if not m:
        # Try to parse the whole text as JSON.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _approx_distribution_match(gt_webppl, gt_pyro, *, tv_threshold: float = 0.05) -> tuple[bool, str]:
    """Return (ok, reason). Compares two distribution-shaped outputs."""
    if not (isinstance(gt_webppl, dict) and isinstance(gt_pyro, dict)):
        return False, f"shape mismatch: {type(gt_webppl).__name__} vs {type(gt_pyro).__name__}"
    if gt_webppl.get("__kind") != "distribution" or gt_pyro.get("__kind") != "distribution":
        return False, f"not both distributions: {gt_webppl.get('__kind')} vs {gt_pyro.get('__kind')}"
    sw, sp = gt_webppl["support"], gt_pyro["support"]
    pw, pp = gt_webppl["probs"], gt_pyro["probs"]
    if len(sw) != len(sp):
        return False, f"support size mismatch: {len(sw)} vs {len(sp)}"
    # Normalize support equivalence (bool↔int, sort by canonical key).
    def key(v):
        if isinstance(v, bool): return ('b', int(v))
        if isinstance(v, (int, float)): return ('n', float(v))
        if isinstance(v, str): return ('s', v)
        return ('o', json.dumps(v, sort_keys=True))
    pairs_w = sorted(zip([key(s) for s in sw], pw))
    pairs_p = sorted(zip([key(s) for s in sp], pp))
    if [k for k,_ in pairs_w] != [k for k,_ in pairs_p]:
        return False, f"support sets differ: {[k for k,_ in pairs_w]} vs {[k for k,_ in pairs_p]}"
    # Compute TV.
    tv = 0.5 * sum(abs(a - b) for (_, a), (_, b) in zip(pairs_w, pairs_p))
    if tv > tv_threshold:
        return False, f"TV={tv:.4f} > {tv_threshold} (distributions differ in mass)"
    return True, f"TV={tv:.4f}"


def _approx_value_match(gt_webppl, gt_pyro, *, rtol: float = 1e-3, atol: float = 1e-4) -> tuple[bool, str]:
    if isinstance(gt_webppl, dict) and isinstance(gt_pyro, dict):
        if set(gt_webppl) != set(gt_pyro):
            return False, f"record keys differ: {set(gt_webppl)} vs {set(gt_pyro)}"
        for k in gt_webppl:
            ok, reason = compare_values(gt_webppl[k], gt_pyro[k])
            if not ok:
                return False, f"record[{k}]: {reason}"
        return True, "record values match"
    # Lists: element-wise with numeric tolerance.
    if isinstance(gt_webppl, list) and isinstance(gt_pyro, list):
        if len(gt_webppl) != len(gt_pyro):
            return False, f"list length differs ({len(gt_webppl)} vs {len(gt_pyro)})"
        for i, (a, b) in enumerate(zip(gt_webppl, gt_pyro)):
            ok, reason = compare_values(a, b)
            if not ok:
                return False, f"list[{i}]: {reason}"
        return True, f"list of {len(gt_webppl)} matches"
    if isinstance(gt_webppl, bool) and isinstance(gt_pyro, bool):
        if gt_webppl == gt_pyro: return True, "bool equal"
        return False, f"bool differ ({gt_webppl} vs {gt_pyro})"
    # int/float (bool excluded above; note bool is subclass of int in Python).
    if isinstance(gt_webppl, (int, float)) and isinstance(gt_pyro, (int, float)):
        if math.isclose(float(gt_webppl), float(gt_pyro), rel_tol=rtol, abs_tol=atol):
            return True, f"close ({gt_webppl} ≈ {gt_pyro})"
        return False, f"differ ({gt_webppl} vs {gt_pyro})"
    if gt_webppl == gt_pyro:
        return True, "equal"
    return False, f"differ ({gt_webppl!r} vs {gt_pyro!r})"


def _empirical_tv_samples(samples_a: list, samples_b: list) -> float:
    """TV between two empirical histograms (for samples-shape comparison)."""
    def key(s):
        if isinstance(s, (list, dict)):
            return json.dumps(s, sort_keys=True)
        return s
    counts_a, counts_b = {}, {}
    for s in samples_a: counts_a[key(s)] = counts_a.get(key(s), 0) + 1
    for s in samples_b: counts_b[key(s)] = counts_b.get(key(s), 0) + 1
    n_a, n_b = len(samples_a), len(samples_b)
    keys = set(counts_a) | set(counts_b)
    return 0.5 * sum(abs(counts_a.get(k, 0)/n_a - counts_b.get(k, 0)/n_b) for k in keys)


def compare_values(gt_webppl, gt_pyro, *, samples_tv_threshold: float = 0.15) -> tuple[bool, str]:
    """Semantic equivalence with bool↔int support normalization."""
    if isinstance(gt_webppl, dict) and gt_webppl.get("__kind") == "distribution":
        return _approx_distribution_match(gt_webppl, gt_pyro)
    if isinstance(gt_webppl, dict) and not gt_webppl.get("__kind"):
        if not isinstance(gt_pyro, dict) or gt_pyro.get("__kind"):
            return False, f"record vs non-record: {gt_webppl.keys()} vs {type(gt_pyro).__name__}"
        if set(gt_webppl) != set(gt_pyro):
            return False, f"record keys differ: {set(gt_webppl)} vs {set(gt_pyro)}"
        for k in gt_webppl:
            ok, reason = compare_values(gt_webppl[k], gt_pyro[k])
            if not ok:
                return False, f"record[{k}]: {reason}"
        return True, "record matches"
    return _approx_value_match(gt_webppl, gt_pyro)


def compare_for_shape(gt_webppl, gt_pyro, shape) -> tuple[bool, str]:
    """Shape-aware comparison entry point.

    For samples-shape atoms, do empirical TV instead of element-wise — two
    seeded sample sequences from the same distribution will not be
    element-wise equal but should have low TV.
    """
    if shape == "samples":
        if not (isinstance(gt_webppl, list) and isinstance(gt_pyro, list)):
            return False, f"samples shape but inputs aren't lists ({type(gt_webppl).__name__}, {type(gt_pyro).__name__})"
        # Boolean lists in WebPPL serialize as True/False; in Pyro as 1.0/0.0.
        # Normalize both via JSON-encoding for the empirical key.
        def norm(s):
            if isinstance(s, list):
                return [norm(x) for x in s]
            if isinstance(s, bool):
                return bool(s)
            if isinstance(s, (int, float)):
                # Coerce 0.0/1.0 to bool if the matching side is bool — done
                # via canonicalizing 1↔True and 0↔False when paired.
                return s
            return s
        # Simple cross-rep: bools and 0/1 floats are equivalent for our atoms.
        # Treat all bools as ints (0/1) for the histogram key.
        def to_int_if_bool_like(v):
            if isinstance(v, bool): return int(v)
            return v
        def deep_normalize(v):
            if isinstance(v, list): return [deep_normalize(x) for x in v]
            return to_int_if_bool_like(v)
        a = [deep_normalize(s) for s in gt_webppl]
        b = [deep_normalize(s) for s in gt_pyro]
        tv = _empirical_tv_samples(a, b)
        if tv <= 0.15:
            return True, f"samples TV={tv:.4f}"
        return False, f"samples TV={tv:.4f} > 0.15"
    return compare_values(gt_webppl, gt_pyro)


def translate_batch(client: Anthropic, atoms: list[dict], model: str,
                    poll_interval: int = 30, timeout: int = 3600) -> dict[str, dict]:
    """Submit translation as a single batch; return {id: parsed_response_or_None}."""
    requests = []
    for a in atoms:
        requests.append({
            "custom_id": a["id"].replace("/", "__").replace(".", "-"),
            "params": {
                "model": model,
                "max_tokens": 16384,
                "system": TRANSLATION_SYSTEM,
                "messages": [{"role": "user", "content": build_user_message(a)}],
            },
        })

    print(f"[batch] submitting {len(requests)} translation requests…")
    batch = client.messages.batches.create(requests=requests)
    bid = batch.id
    print(f"[batch] id={bid}")

    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(bid)
        st = b.processing_status
        cs = b.request_counts
        print(f"  [{time.time()-t0:5.0f}s] status={st} processing={cs.processing} "
              f"succeeded={cs.succeeded} errored={cs.errored}", flush=True)
        if st == "ended":
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Batch {bid} not done after {timeout}s")
        time.sleep(poll_interval)

    out: dict[str, dict] = {}
    cid_to_id = {a["id"].replace("/", "__").replace(".", "-"): a["id"] for a in atoms}
    for line in client.messages.batches.results(bid):
        atom_id = cid_to_id[line.custom_id]
        if line.result.type != "succeeded":
            out[atom_id] = {"error": f"batch result type={line.result.type}"}
            continue
        msg = line.result.message
        text = "\n".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        parsed = parse_response(text)
        out[atom_id] = {"raw": text, "parsed": parsed,
                        "tokens_in": msg.usage.input_tokens,
                        "tokens_out": msg.usage.output_tokens}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--ids", nargs="+", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--broken", required=True)
    p.add_argument("--model", default="claude-sonnet-4-6")
    args = p.parse_args()

    atoms = load_jsonl(args.input)
    if args.ids:
        atoms = [a for a in atoms if a["id"] in set(args.ids)]
    print(f"loaded {len(atoms)} atoms to translate")

    client = Anthropic()
    results = translate_batch(client, atoms, args.model)

    ok_atoms: list[dict] = []
    broken_atoms: list[dict] = []

    for a in atoms:
        r = results.get(a["id"])
        if not r or r.get("error") or not r.get("parsed"):
            broken_atoms.append({**a, "translation_error": (r or {}).get("error", "no response or unparseable"),
                                  "raw": (r or {}).get("raw", "")[:1000]})
            print(f"  [BROKEN] {a['id']}  no parsed response")
            continue
        parsed = r["parsed"]
        new_prompt = parsed.get("prompt", "")
        new_gt = parsed.get("groundtruth_code", "")
        if not new_prompt or not new_gt:
            broken_atoms.append({**a, "translation_error": "missing prompt/groundtruth_code in response",
                                  "raw_parsed": parsed})
            print(f"  [BROKEN] {a['id']}  missing fields")
            continue
        # Execute the new GT (longer timeout for MCMC/SMC translations).
        exec_r = execute_pyro(new_gt, timeout=180, random_seed=42)
        if not exec_r.success:
            broken_atoms.append({**a, "translation_error": f"pyro exec failed: {exec_r.error_message}",
                                  "pyro_prompt": new_prompt, "pyro_groundtruth_code": new_gt,
                                  "stderr_tail": exec_r.stderr[-400:]})
            print(f"  [BROKEN] {a['id']}  exec: {exec_r.error_message[:100]}")
            continue
        # Compare to WebPPL GT (shape-aware).
        ok, reason = compare_for_shape(a["groundtruth_output"], exec_r.answer, a["answer_shape"])
        if not ok:
            broken_atoms.append({**a, "translation_error": f"output mismatch: {reason}",
                                  "pyro_prompt": new_prompt, "pyro_groundtruth_code": new_gt,
                                  "pyro_groundtruth_output": exec_r.answer})
            print(f"  [BROKEN] {a['id']}  mismatch: {reason[:100]}")
            continue
        ok_atom = {
            "id": "pyro-" + a["id"].split("-", 1)[1] if a["id"].startswith("probmods2-") else "pyro-" + a["id"],
            "language": "pyro",
            "task_type": a.get("task_type", "write_from_scratch"),
            "eval_mode": a.get("eval_mode"),
            "answer_shape": a["answer_shape"],
            "prompt": new_prompt,
            "groundtruth_code": new_gt,
            "groundtruth_output": exec_r.answer,
            "source_atom": a["id"],
            "translation_check": reason,
        }
        ok_atoms.append(ok_atom)
        print(f"  [OK    ] {a['id']:50s}  {reason}")

    write_jsonl(Path(args.output), ok_atoms)
    write_jsonl(Path(args.broken), broken_atoms)
    print(f"\n{len(ok_atoms)} OK, {len(broken_atoms)} broken")
    print(f"  → {args.output}")
    print(f"  → {args.broken}")


if __name__ == "__main__":
    main()
