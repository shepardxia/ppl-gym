"""Mechanical handroll-triage for realization columns — a cheap pre-filter that
narrows the set an LLM idiomaticity audit must actually read.

The handroll signature (a required inference level faked in host-language
arithmetic) is largely mechanical: a problem whose WebPPL GT **conditions** (uses
condition / observe / factor / an Infer that isn't pure forward) needs real
inference; if the target realization makes **no real inference call**, the
inference is faked. A secondary, noisier signal flags realizations that DO call
inference but also carry a lot of manual probability arithmetic (hand softmax /
normalize / weight-resample), which is where PARTIAL handrolls hide (one faked
helper beside real inference).

Classes: forward_ok (GT is pure forward → raw sampling is legitimate),
handroll (GT conditions, 0 real-inference calls — high confidence),
partial_suspect (real inference present but heavy manual prob-math — needs a read),
clean.

Only handroll + partial_suspect need an LLM read. CLI:
  PYTHONPATH=. .venv/bin/python -m eval.idiom_triage --language gen [--ids ...]
"""

from __future__ import annotations

import argparse
import re

from eval.corpus import load_realizations

# Real-inference call tokens, per target language.
INFER_TOKENS = {
    "gen": ["enumerative_inference", "mh(", "importance", "generate(", "metropolis", "hmc("],
    "pyro": ["config_enumerate", "compute_marginals", "infer_discrete", "Importance",
             "MCMC(", "NUTS(", "SMCFilter", "SVI(", "TraceEnum"],
}
# Manual probability-arithmetic signals (hand-rolled inference smells): explicit
# resampling, log-sum-exp / softmax, and hand-normalization (divide by a
# parenthesized sum, e.g. `w_tall/(w_tall+w_null)`).
MANUAL_MATH = [r"cumsum", r"searchsorted", r"logsumexp", r"softmax",
               r"\./\s*sum\(", r"/\s*sum\(", r"normalize\(", r"\bexp\.\(",
               r"/\s*\([^)]*\+[^)]*\)"]
# WebPPL conditioning signals (the problem needs inference).
COND_TOKENS = ["condition(", "observe(", "factor(", "mapData"]


def _real_infer_calls(code: str, language: str) -> int:
    return sum(code.count(t) for t in INFER_TOKENS.get(language, []))


def _manual_math(code: str) -> int:
    return sum(len(re.findall(p, code)) for p in MANUAL_MATH)


def _gt_conditions(webppl_code: str) -> bool:
    return any(t in webppl_code for t in COND_TOKENS)


def _gt_infer_levels(webppl_code: str) -> int:
    """How many inference levels the GT runs (each Infer/Enumerate is one)."""
    return webppl_code.count("Infer(") + webppl_code.count("Enumerate(")


def _is_rejection(code: str) -> bool:
    # accept-reject Monte Carlo: a loop that conditionally push!es accepted draws,
    # with no hand-computed distribution — a legitimate inference method, not a table.
    return ("while" in code or "for " in code) and "push!" in code and _manual_math(code) == 0


def classify(realization_code: str, webppl_gt: str, language: str) -> dict:
    real = _real_infer_calls(realization_code, language)
    manual = _manual_math(realization_code)
    conditions = _gt_conditions(webppl_gt)
    levels = _gt_infer_levels(webppl_gt)
    # Core signal: the realization should run as many inference levels as the GT.
    # Fewer real-inference calls than GT Infer levels ⇒ a level was faked (or the
    # method replaced by rejection, which is legitimate).
    if not conditions:
        cls = "forward_ok" if real == 0 else "clean"          # pure forward → raw sampling fine
    elif real == 0 and levels <= 2 and _is_rejection(realization_code):
        cls = "rejection_ok"                                  # single conditioned model via draw+filter MC
        # (a multi-level RSA with 0 inference calls can't be one rejection loop → falls through to handroll)
    elif real < levels:
        cls = "handroll" if real == 0 else "partial_suspect"  # missing whole vs some levels
    elif manual >= max(4, 3 * real):
        cls = "partial_suspect"                               # level count ok but heavy manual math
    else:
        cls = "clean"
    return {"class": cls, "real_infer": real, "gt_levels": levels,
            "manual_math": manual, "gt_conditions": conditions}


def run(language: str, ids: set[str] | None = None) -> list[dict]:
    reals = {r["problem_id"]: r for r in load_realizations(language) if r.get("code")}
    wppl = {r["problem_id"]: r.get("code", "") for r in load_realizations("webppl")}
    out = []
    for pid, r in reals.items():
        if ids and pid not in ids:
            continue
        c = classify(r["code"], wppl.get(pid, ""), language)
        out.append({"problem_id": pid, **c})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Mechanical handroll-triage of a realization column.")
    p.add_argument("--language", required=True)
    p.add_argument("--ids", nargs="+", default=None)
    a = p.parse_args()
    rows = run(a.language, set(a.ids) if a.ids else None)
    from collections import Counter
    counts = Counter(r["class"] for r in rows)
    print(f"[{a.language}] {len(rows)} realizations: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    for r in sorted(rows, key=lambda x: (x["class"] != "handroll", x["class"] != "partial_suspect", x["problem_id"])):
        if r["class"] in ("handroll", "partial_suspect"):
            print(f"  {r['class']:16} {r['problem_id']:48} real_infer={r['real_infer']} gt_levels={r['gt_levels']} manual_math={r['manual_math']}")


if __name__ == "__main__":
    main()
