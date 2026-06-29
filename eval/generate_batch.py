"""Submit problem generations as an Anthropic Message Batch (50% discount, async).

Public API consumed by eval.gate and eval.benchmark:
  build_requests(problems, language, model, n_solvers, max_tokens, temperature)
  problem_id_to_cid(problem_id, slot)
  cid_to_problem_slot(cid)
  submit_batch(client, requests)
  wait_for_batch(client, batch_id, ...)
  collect_results(client, batch_id)
  write_generation_rows(problems, results, output_path, *, model, language, n_solvers)

CLI:
  PYTHONPATH=. .venv/bin/python -m eval.generate_batch \\
      --model <model> \\
      --output <path/to/generations.jsonl> \\
      [--ids ID ...] \\
      [--language webppl] \\
      [--n-samples 1] \\
      [--no-poll]
      [--collect BATCH_ID]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from anthropic import Anthropic

from eval.prompt import parse_response, system_prompt
from eval.render import render_problem


DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0
POLL_INTERVAL = 30
POLL_TIMEOUT = 3600  # 1 hour

_DEFAULT_N_SOLVERS = 2


# ---------------------------------------------------------------------------
# custom_id encoding
# ---------------------------------------------------------------------------

def problem_id_to_cid(problem_id: str, slot: int) -> str:
    """Encode problem_id + slot → custom_id satisfying ^[a-zA-Z0-9_-]{1,64}$.

    Encoding: '/' → '__', '.' → '_dot_'
    These are invertible and safe: '__' does not appear in raw problem_ids,
    '_dot_' does not conflict with the slot suffix '__s{n}'.
    """
    safe = problem_id.replace(".", "_dot_").replace("/", "__")
    return f"{safe}__s{slot}"


def cid_to_problem_slot(cid: str) -> tuple[str, int]:
    """Reverse of problem_id_to_cid."""
    m = re.match(r"^(.+)__s(\d+)$", cid)
    if not m:
        raise ValueError(f"cannot parse custom_id: {cid!r}")
    problem_id = m.group(1).replace("__", "/").replace("_dot_", ".")
    return problem_id, int(m.group(2))


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def build_requests(
    problems: list[dict],
    language: str = "webppl",
    model: str = "claude-sonnet-4-6",
    n_solvers: int = _DEFAULT_N_SOLVERS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict]:
    """Build Anthropic batch request dicts (n_solvers per problem).

    Returns a list of request dicts ready for submit_batch.
    """
    sys_text = system_prompt(with_primer=True, language=language)
    system_blocks = [{
        "type": "text",
        "text": sys_text,
        "cache_control": {"type": "ephemeral"},
    }]

    # Stan pins its data-block interface from the GT bundle; other languages
    # render from the problem alone. Load realizations once for the join.
    from eval.corpus import load_realizations
    real_by_id = {r["problem_id"]: r for r in load_realizations(language)} if language == "stan" else {}

    requests = []
    for prob in problems:
        user_text = render_problem(prob, language=language,
                                   realization=real_by_id.get(prob["problem_id"]))
        for slot in range(n_solvers):
            cid = problem_id_to_cid(prob["problem_id"], slot)
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_blocks,
                    "messages": [{"role": "user", "content": user_text}],
                },
            })
    return requests


# ---------------------------------------------------------------------------
# Batch mechanics
# ---------------------------------------------------------------------------

def submit_batch(client: Anthropic, requests: list[dict]) -> str:
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(
    client: Anthropic,
    batch_id: str,
    *,
    poll_interval: int = POLL_INTERVAL,
    timeout: int = POLL_TIMEOUT,
    verbose: bool = True,
) -> object:
    t0 = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        elapsed = time.time() - t0
        status = batch.processing_status
        counts = batch.request_counts
        if verbose:
            print(
                f"  [{elapsed:5.0f}s] status={status} "
                f"processing={counts.processing} succeeded={counts.succeeded} "
                f"errored={counts.errored} canceled={counts.canceled} "
                f"expired={counts.expired}",
                flush=True,
            )
        if status == "ended":
            return batch
        if elapsed > timeout:
            raise TimeoutError(f"Batch {batch_id} not done after {timeout}s")
        time.sleep(poll_interval)


def collect_results(client: Anthropic, batch_id: str) -> dict:
    """Returns {custom_id: {ok, code, raw, warnings, meta}}."""
    out = {}
    for line in client.messages.batches.results(batch_id):
        cid = line.custom_id
        result = line.result
        if result.type == "succeeded":
            msg = result.message
            text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            text = "\n".join(text_parts)
            code, warnings = parse_response(text)
            out[cid] = {
                "ok": True,
                "code": code,
                "raw": text,
                "warnings": warnings,
                "meta": {
                    "stop_reason": msg.stop_reason,
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(
                        msg.usage, "cache_creation_input_tokens", 0) or 0,
                    "cache_read_input_tokens": getattr(
                        msg.usage, "cache_read_input_tokens", 0) or 0,
                },
            }
        else:
            err = getattr(result, "error", None)
            err_msg = str(err) if err is not None else f"result_type={result.type}"
            out[cid] = {
                "ok": False,
                "code": "",
                "raw": "",
                "warnings": [f"batch error: {err_msg}"],
                "meta": {"error": err_msg, "result_type": result.type},
            }
    return out


# ---------------------------------------------------------------------------
# Generation row writer
# ---------------------------------------------------------------------------

def write_generation_rows(
    problems: list[dict],
    results: dict,
    output_path: Path,
    *,
    model: str,
    language: str,
    n_solvers: int,
) -> None:
    """Write generation rows to output_path (JSONL, no summary trailer)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for prob in problems:
            pid = prob["problem_id"]
            for slot in range(n_solvers):
                cid = problem_id_to_cid(pid, slot)
                r = results.get(cid)
                if r is None:
                    row = {
                        "problem_id": pid,
                        "slot": slot,
                        "model": model,
                        "language": language,
                        "code": "",
                        "warnings": ["missing from batch results"],
                    }
                else:
                    row = {
                        "problem_id": pid,
                        "slot": slot,
                        "model": model,
                        "language": language,
                        "code": r["code"],
                        "warnings": r["warnings"],
                        **r["meta"],
                    }
                f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Submit/collect an Anthropic Message Batch for problem-centric generation."
    )
    p.add_argument("--model", required=True, help="Model to use for generation.")
    p.add_argument("--output", required=True, help="Output generations JSONL path.")
    p.add_argument(
        "--ids", nargs="+", default=None, metavar="ID",
        help="Restrict to specific problem IDs.",
    )
    p.add_argument(
        "--language", default="webppl",
        help="Language (default: webppl). Selects prompt primer and realization.",
    )
    p.add_argument(
        "--n-samples", type=int, default=1,
        help="Generations per problem (default: 1, i.e. one slot).",
    )
    p.add_argument(
        "--no-poll", action="store_true",
        help="Submit the batch and print batch_id without waiting for results.",
    )
    p.add_argument(
        "--collect", default=None, metavar="BATCH_ID",
        help="Resume: collect results for an already-completed batch_id.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per request (default {DEFAULT_MAX_TOKENS}).",
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE}).",
    )
    p.add_argument(
        "--poll-interval", type=int, default=POLL_INTERVAL,
        help=f"Poll interval in seconds (default {POLL_INTERVAL}).",
    )
    p.add_argument(
        "--timeout", type=int, default=POLL_TIMEOUT,
        help=f"Max wait time in seconds (default {POLL_TIMEOUT}).",
    )
    args = p.parse_args()

    # Lazy import to avoid circular dependency at module load time.
    from eval.corpus import load_problems

    id_set = set(args.ids) if args.ids else None
    problems = load_problems(id_set)
    if not problems:
        print("No matching problems found.")
        return

    output_path = Path(args.output)
    client = Anthropic()
    n_solvers = args.n_samples

    # --collect mode: fetch results for an existing batch
    if args.collect:
        batch_id = args.collect
        print(f"[generate_batch] collecting results for batch {batch_id}...")
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            print(f"[generate_batch] batch not ended yet (status={batch.processing_status}), waiting...")
            wait_for_batch(client, batch_id, poll_interval=args.poll_interval, timeout=args.timeout)
        results = collect_results(client, batch_id)
        write_generation_rows(
            problems, results, output_path,
            model=args.model, language=args.language, n_solvers=n_solvers,
        )
        print(f"[generate_batch] wrote {len(problems) * n_solvers} rows to {output_path}")
        return

    # Normal mode: build + submit
    requests = build_requests(
        problems,
        language=args.language,
        model=args.model,
        n_solvers=n_solvers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"[generate_batch] submitting {len(requests)} requests ({len(problems)} problems × {n_solvers} slots)...")
    batch_id = submit_batch(client, requests)
    print(f"[generate_batch] batch_id={batch_id}")

    if args.no_poll:
        print(f"[generate_batch] --no-poll: exiting. Resume with: --collect {batch_id}")
        return

    print(f"[generate_batch] polling every {args.poll_interval}s (timeout {args.timeout}s)...")
    wait_for_batch(client, batch_id, poll_interval=args.poll_interval, timeout=args.timeout)

    print("[generate_batch] streaming results...")
    results = collect_results(client, batch_id)
    write_generation_rows(
        problems, results, output_path,
        model=args.model, language=args.language, n_solvers=n_solvers,
    )
    print(f"[generate_batch] wrote {len(problems) * n_solvers} rows to {output_path}")


if __name__ == "__main__":
    main()
