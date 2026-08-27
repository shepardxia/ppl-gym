"""Generation backends: turn provider-agnostic prompts into solver code.

A prompt is {custom_id, system, user}. A backend takes the prompts + a model and
returns {custom_id: {ok, code, raw, warnings, meta}} — the shape the scorer reads.

  anthropic — Message Batch API (50% off, async), optional extended thinking.
  together  — OpenAI-compatible chat completions, run concurrently; optional
              reasoning_effort (gpt-oss) and a large token budget for reasoners.

The model registry (MODELS) names each model's backend, provider id, token budget
and thinking config; `benchmark` resolves short names through it. Together reads
its key from $TOGETHER_API_KEY (base url $TOGETHER_BASE_URL or the default).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from eval.corpus import load_realizations
from eval.generate_batch import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, problem_id_to_cid
from eval.prompt import parse_response, system_prompt
from eval.render import render_problem

TOGETHER_BASE_URL = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")


@dataclass
class ModelConfig:
    name: str                 # short label (also the run-dir slug)
    backend: str              # "anthropic" | "together"
    model_id: str             # provider model id
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    thinking_budget: int | None = None   # anthropic extended-thinking budget tokens
    adaptive_thinking: bool = False       # anthropic adaptive thinking; rejects temperature
    effort: str | None = None             # anthropic output_config effort level
    reasoning_effort: str | None = None  # together gpt-oss reasoning level


# A reasoner can spend its whole budget thinking before emitting the answer, so
# reasoning models get a large max_tokens. Thinking is on where it is a lever.
_REASONER_MAX = 24000

MODELS: dict[str, ModelConfig] = {
    "opus":         ModelConfig("opus", "anthropic", "claude-opus-5",
                                max_tokens=32000, adaptive_thinking=True, effort="high"),
    "sonnet":       ModelConfig("sonnet", "anthropic", "claude-sonnet-4-6",
                                max_tokens=8192, thinking_budget=4096),
    "haiku":        ModelConfig("haiku", "anthropic", "claude-haiku-4-5-20251001",
                                max_tokens=8192, thinking_budget=4096),
    "gpt-oss-20b":  ModelConfig("gpt-oss-20b", "together", "openai/gpt-oss-20b",
                                max_tokens=_REASONER_MAX, reasoning_effort="medium"),
    "gpt-oss-120b": ModelConfig("gpt-oss-120b", "together", "openai/gpt-oss-120b",
                                max_tokens=_REASONER_MAX, reasoning_effort="medium"),
    "llama-3.3-70b": ModelConfig("llama-3.3-70b", "together",
                                 "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "qwen3-235b":   ModelConfig("qwen3-235b", "together",
                                "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
    "qwen3.5-9b":   ModelConfig("qwen3.5-9b", "together", "Qwen/Qwen3.5-9B",
                                max_tokens=_REASONER_MAX),  # reasoner by default
}


def resolve(name: str) -> ModelConfig:
    if name in MODELS:
        return MODELS[name]
    raise ValueError(f"unknown model {name!r}; known: {sorted(MODELS)}")


def resolve_loose(model: str) -> ModelConfig | None:
    """Registry entry matching a short name or a provider model id, else None.

    Callers that accept a raw model id must route through this: a model's
    request-shape contract (adaptive thinking, effort, sampling params) lives on
    its registry entry, and a hand-built ModelConfig silently loses it.
    """
    if model in MODELS:
        return MODELS[model]
    for cfg in MODELS.values():
        if cfg.model_id == model:
            return cfg
    return None


# ---------------------------------------------------------------------------
# Provider-agnostic prompt building
# ---------------------------------------------------------------------------

def build_prompts(problems, language, *, n_solvers, with_primer=True,
                  verbose_primer=False) -> list[dict]:
    """[{custom_id, system, user}] — n_solvers per problem. Single render path."""
    sys_text = system_prompt(with_primer=with_primer, language=language, verbose=verbose_primer)
    real_by_id = ({r["problem_id"]: r for r in load_realizations(language)}
                  if language == "stan" else {})
    out: list[dict] = []
    for prob in problems:
        user = render_problem(prob, language=language,
                              realization=real_by_id.get(prob["problem_id"]))
        for slot in range(n_solvers):
            out.append({"custom_id": problem_id_to_cid(prob["problem_id"], slot),
                        "system": sys_text, "user": user})
    return out


# ---------------------------------------------------------------------------
# Anthropic request packaging (batch submission lives in generate_batch)
# ---------------------------------------------------------------------------

def anthropic_requests(prompts, cfg: ModelConfig) -> list[dict]:
    """Pack prompts into Anthropic Message Batch requests (+ thinking if set)."""
    requests = []
    for p in prompts:
        params = {
            "model": cfg.model_id,
            "max_tokens": cfg.max_tokens,
            "system": [{"type": "text", "text": p["system"],
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": p["user"]}],
        }
        if cfg.adaptive_thinking:
            params["thinking"] = {"type": "adaptive"}
            if cfg.effort:
                params["output_config"] = {"effort": cfg.effort}
        else:
            params["temperature"] = cfg.temperature
        if cfg.thinking_budget and not cfg.adaptive_thinking:
            # Extended thinking forces temperature=1 and needs max_tokens > budget.
            params["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
            params["temperature"] = 1.0
            params["max_tokens"] = max(cfg.max_tokens, cfg.thinking_budget + 4096)
        requests.append({"custom_id": p["custom_id"], "params": params})
    return requests


# ---------------------------------------------------------------------------
# Together (OpenAI-compatible) concurrent backend
# ---------------------------------------------------------------------------

def together_generate(prompts, cfg: ModelConfig, *, workers: int = 16,
                      max_retries: int = 4) -> dict:
    """Run prompts concurrently through Together's chat-completions API.

    Returns {custom_id: {ok, code, raw, warnings, meta}}. Reasoning content (when
    a model emits it) is dropped — only the answer (message.content) is parsed.
    """
    from openai import OpenAI

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=TOGETHER_BASE_URL)

    extra = {"reasoning_effort": cfg.reasoning_effort} if cfg.reasoning_effort else {}

    def one(p: dict):
        last = None
        for attempt in range(max_retries):
            try:
                r = client.chat.completions.create(
                    model=cfg.model_id,
                    messages=[{"role": "system", "content": p["system"]},
                              {"role": "user", "content": p["user"]}],
                    max_tokens=cfg.max_tokens, temperature=cfg.temperature, **extra,
                )
                choice = r.choices[0]
                text = choice.message.content or ""
                code, warnings = parse_response(text)
                if not text.strip():
                    warnings = warnings + [f"empty content (finish={choice.finish_reason})"]
                return p["custom_id"], {
                    "ok": True, "code": code, "raw": text, "warnings": warnings,
                    "meta": {"finish_reason": choice.finish_reason,
                             "output_tokens": getattr(r.usage, "completion_tokens", None)},
                }
            except Exception as e:  # transient (rate limit / timeout) → backoff + retry
                last = e
                time.sleep(2 ** attempt)
        return p["custom_id"], {"ok": False, "code": "", "raw": "",
                                "warnings": [f"api error: {last}"],
                                "meta": {"error": str(last)[:200]}}

    results: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for cid, res in ex.map(one, prompts):
            results[cid] = res
    return results
