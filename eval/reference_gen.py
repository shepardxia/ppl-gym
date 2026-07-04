"""Self-generated reference posteriors for posteriordb models without gold draws.

posteriordb ships gold reference draws (10 chains x 1000, R-hat ~ 1, ESS ~ 10k)
for only 46 of its 147 posteriors. This module produces reference draws of the
same shape for the rest: long NUTS runs behind explicit convergence gates. A
posterior that fails the gates is rejected, not stored — the gate IS the
provenance claim ("validated draws", vs posteriordb's "gold draws").

Storage is an OVERLAY: draws land in data/reference_draws/<name>.json (chains
list, same schema as posteriordb's .json.zip payload) + <name>.info.json
(sampler args, per-parameter diagnostics, provenance). The vendored posteriordb
tree is never written. eval/posteriordb.py resolves gold first, overlay second,
so everything downstream (executor_reference, ingestion, crosscheck) works
unchanged for overlay posteriors.

Heavy: run on the box, never the laptop (10 chains x long warmup per model).

    PYTHONPATH=. python -m eval.reference_gen --names eight_schools-eight_schools_centered
    PYTHONPATH=. python -m eval.reference_gen --names ... --chains 10 --iter-sampling 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.posteriordb import (
    data_block_vars,
    gold_posterior_names,
    model_code,
    model_data,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = _REPO_ROOT / "data/reference_draws"

# Convergence gates. posteriordb's own reference protocol targets R-hat ~ 1 and
# ESS ~ 10k over 10 chains; we gate slightly looser but still reference-grade.
RHAT_MAX = 1.01
ESS_MIN = 2000.0

# Sampler diagnostics columns (cmdstanpy draws_pd) — everything else is a
# model parameter / transformed parameter and becomes a queried param.
_DIAG_SUFFIX = "__"


def overlay_names() -> list[str]:
    """Posteriors with self-generated (overlay) reference draws."""
    if not OVERLAY_DIR.exists():
        return []
    return sorted(p.name[: -len(".json")] for p in OVERLAY_DIR.glob("*.json")
                  if not p.name.endswith(".info.json"))


def generate_reference(
    name: str,
    *,
    chains: int = 10,
    iter_warmup: int = 10000,
    iter_sampling: int = 1000,
    thin: int = 10,
    adapt_delta: float = 0.9,
    max_treedepth: int = 12,
    seed: int = 4711,
    timeout: int | None = 1800,
) -> dict:
    """Run long NUTS for ``name``; gate on R-hat/ESS; write overlay on pass.

    Returns a report dict {name, status, rhat_max, ess_min, runtime_sec, ...};
    status is "ok" (written), "rejected" (gates failed, nothing written) or
    "error". ``iter_sampling`` draws per chain are KEPT post-thinning: the
    sampler runs iter_sampling * thin iterations and thins by ``thin``, which
    is how posteriordb's references decorrelate their kept draws.
    """
    if name in gold_posterior_names():
        return {"name": name, "status": "error",
                "error": "gold draws already exist; overlay refused"}

    from cmdstanpy import CmdStanModel
    from eval.executor_stan import _compiled_model, _cwd_guard

    t0 = time.time()
    try:
        model_src = model_code(name)
        declared = set(data_block_vars(model_src))
        data = {k: v for k, v in model_data(name).items() if k in declared}
        with _cwd_guard():
            model = _compiled_model(model_src)
            fit = model.sample(
                data=data,
                seed=seed,
                chains=chains,
                parallel_chains=chains,
                iter_warmup=iter_warmup,
                iter_sampling=iter_sampling * thin,
                thin=thin,
                adapt_delta=adapt_delta,
                max_treedepth=max_treedepth,
                show_progress=False,
                show_console=False,
                timeout=timeout,
            )
            summary = fit.summary()
            df = fit.draws_pd()
    except Exception as e:
        msg = str(e).strip().splitlines()
        return {"name": name, "status": "error",
                "error": (msg[-1][:300] if msg else type(e).__name__),
                "runtime_sec": round(time.time() - t0, 1)}

    params = [c for c in df.columns
              if not c.endswith(_DIAG_SUFFIX) and c != "chain__"]
    # cmdstanpy summary indexes by variable name incl. lp__; gate on the
    # queried params only (lp__ mixing is not part of the claim).
    diag = summary.loc[[i for i in summary.index if i in params]]
    rhat_max = float(diag["R_hat"].max())
    ess_min = float(diag["ESS_bulk"].min())
    report = {
        "name": name, "rhat_max": round(rhat_max, 5), "ess_min": round(ess_min, 1),
        "chains": chains, "kept_draws": iter_sampling, "thin": thin,
        "runtime_sec": round(time.time() - t0, 1),
    }
    if rhat_max > RHAT_MAX or ess_min < ESS_MIN:
        return {**report, "status": "rejected",
                "error": f"gates: R-hat {rhat_max:.4f} (max {RHAT_MAX}), "
                         f"ESS {ess_min:.0f} (min {ESS_MIN:.0f})"}

    # chains list, same schema as posteriordb's payload: [{param: [draws]}, ...]
    n_per = iter_sampling
    chain_dicts = []
    for c in range(chains):
        lo, hi = c * n_per, (c + 1) * n_per
        chain_dicts.append({p: df[p].iloc[lo:hi].astype(float).tolist()
                            for p in params})

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    (OVERLAY_DIR / f"{name}.json").write_text(json.dumps(chain_dicts))
    info = {
        "provenance": "self-generated (eval/reference_gen.py); NOT posteriordb gold",
        "inference": {"method_arguments": {
            "chains": chains, "warmup": iter_warmup,
            "iter_sampling": iter_sampling * thin, "thin": thin,
            "control": {"adapt_delta": adapt_delta, "max_treedepth": max_treedepth},
            "seed": seed}},
        "diagnostics": {"rhat_max": rhat_max, "ess_min": ess_min,
                        "per_param": {p: {"rhat": float(diag.loc[p, "R_hat"]),
                                          "ess_bulk": float(diag.loc[p, "ESS_bulk"])}
                                      for p in diag.index}},
    }
    (OVERLAY_DIR / f"{name}.info.json").write_text(json.dumps(info, indent=1))
    return {**report, "status": "ok"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-generate reference draws (BOX, not laptop).")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--chains", type=int, default=10)
    ap.add_argument("--iter-warmup", type=int, default=10000)
    ap.add_argument("--iter-sampling", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--adapt-delta", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=4711)
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if an overlay already exists.")
    args = ap.parse_args()

    existing = set(overlay_names())
    for name in args.names:
        if name in existing and not args.force:
            print(f"[refgen] {name}: overlay exists, skipping (--force to redo)", flush=True)
            continue
        r = generate_reference(
            name, chains=args.chains, iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling, thin=args.thin,
            adapt_delta=args.adapt_delta, seed=args.seed)
        line = f"[refgen] {name}: {r['status']}"
        if r["status"] == "ok":
            line += f"  rhat_max={r['rhat_max']} ess_min={r['ess_min']} ({r['runtime_sec']}s)"
        else:
            line += f"  {r.get('error', '')}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
