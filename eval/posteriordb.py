"""Ingest posteriordb (stan-dev/posteriordb) into the ppl-gym corpus.

posteriordb ships, per *posterior* (a model + data pairing), the Stan model
code, the input data, and — for a curated subset — reference posterior draws
(10 NUTS chains x 1000 draws, R-hat ~ 1, ESS ~ 10k). Those reference draws are
gold ground truth: the answer to "what is the posterior of these parameters
given this data" is solved to reference grade and need not be recomputed.

This module maps each gold-draw posterior onto the ppl-gym contract:

  problem  (data/problems/posteriordb.jsonl)
    answer_spec = record{param: dist/real}   (a posterior over real parameters
    is respec'd to per-parameter marginals; SCHEMA forbids dist/realvec)
    statement describes the model and the data interface; the literal data
    values are supplied to the program at runtime (Stan is data-parametric), so
    the answer is pinned by model-semantics + supplied data, not prose alone.

  realization language "stan"      (data/realizations/stan.jsonl)
    a self-contained Stan bundle (model + //@ DATA/PARAMS/SAMPLING, data
    embedded so it runs without posteriordb present) executed by
    eval.executor_stan — the binding, validated by crosscheck against...

  realization language "reference" (data/realizations/reference.jsonl)
    code = posterior name; eval.executor_reference replays the stored gold draws
    as the ground-truth column.

`gate crosscheck --language stan --reference reference` then verifies that our
cmdstanpy execution reproduces the gold posterior.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

from eval.io import load_jsonl, merge_jsonl, write_jsonl
from eval.stan_bundle import pack

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute, repo-anchored paths. cmdstanpy's compile (make) transiently chdir's
# the *process*, so any relative path resolved in a concurrent worker thread
# (the reference executor, GT-cache writes) can hit the wrong CWD — an
# intermittent FileNotFoundError. Absolute paths are CWD-independent.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PDB_ROOT = _REPO_ROOT / "data/sources/posteriordb/posterior_database"

PROBLEMS_OUT = _REPO_ROOT / "data/problems/posteriordb.jsonl"
STAN_OUT = _REPO_ROOT / "data/realizations/stan.jsonl"
REFERENCE_OUT = _REPO_ROOT / "data/realizations/reference.jsonl"

CORPUS_PREFIX = "posteriordb"


def _pdb_root() -> Path:
    if not _PDB_ROOT.exists():
        raise FileNotFoundError(
            f"posteriordb not found at {_PDB_ROOT}. Clone it:\n"
            f"  git clone --depth 1 https://github.com/stan-dev/posteriordb "
            f"data/sources/posteriordb")
    return _PDB_ROOT


def _read_zip_json(path: Path) -> object:
    """posteriordb stores data/draws as single-entry .json.zip archives."""
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        return json.loads(zf.read(name).decode("utf-8"))


# ---------------------------------------------------------------------------
# Posterior discovery + metadata
# ---------------------------------------------------------------------------

def gold_posterior_names() -> list[str]:
    """Posteriors that ship full reference draws (the gold-GT subset)."""
    draws = _pdb_root() / "reference_posteriors" / "draws" / "draws"
    return sorted(p.name[: -len(".json.zip")] for p in draws.glob("*.json.zip"))


@lru_cache(maxsize=None)
def posterior_info(name: str) -> dict:
    return json.loads((_pdb_root() / "posteriors" / f"{name}.json").read_text())


def model_code(name: str) -> str:
    model_name = posterior_info(name)["model_name"]
    return (_pdb_root() / "models" / "stan" / f"{model_name}.stan").read_text()


@lru_cache(maxsize=None)
def model_data(name: str) -> dict:
    data_name = posterior_info(name)["data_name"]
    return _read_zip_json(_pdb_root() / "data" / "data" / f"{data_name}.json.zip")


_DATA_BLOCK_RE = re.compile(r"\bdata\s*\{([^}]*)\}", re.DOTALL)


def data_block_vars(model: str) -> list[str]:
    """The variable names declared in a Stan model's ``data`` block.

    posteriordb datasets are shared across several models, so a data file often
    carries columns a given model never declares (e.g. kidiq ships mom_iq for
    other regressions). Only the declared inputs belong in the bundle.
    """
    m = _DATA_BLOCK_RE.search(model)
    if not m:
        return []
    names: list[str] = []
    for decl in m.group(1).split(";"):
        decl = re.sub(r"//.*", "", decl)            # line comments
        decl = re.sub(r"<[^>]*>", "", decl)         # <lower=..,upper=..>
        decl = re.sub(r"\[[^\]]*\]", "", decl)      # [N], [N,K], array[J]
        toks = re.findall(r"[A-Za-z_]\w*", decl)
        if toks:
            names.append(toks[-1])                   # name is the last identifier
    return names


@lru_cache(maxsize=None)
def reference_chains(name: str) -> list[dict]:
    """The gold draws as a list of chains; each chain is {param: [draws]}."""
    path = (_pdb_root() / "reference_posteriors" / "draws" / "draws"
            / f"{name}.json.zip")
    chains = _read_zip_json(path)
    if not isinstance(chains, list):
        raise ValueError(f"unexpected draws shape for {name}: {type(chains)}")
    return chains


def reference_info(name: str) -> dict:
    path = (_pdb_root() / "reference_posteriors" / "draws" / "info"
            / f"{name}.info.json")
    return json.loads(path.read_text())


def param_names(name: str) -> list[str]:
    """Queried parameters: the names the reference draws expose (model
    parameters and transformed parameters, excluding sampler diagnostics)."""
    return list(reference_chains(name)[0].keys())


def reference_sampling(name: str, *, warmup_cap: int = 4000,
                       sampling: int = 2000, chains: int = 4) -> dict:
    """A cmdstanpy sampling config derived from how the gold draws were made.

    posteriordb documents each reference's sampler arguments (warmup, thin,
    adapt_delta). Mirroring adapt_delta (and a capped warmup) is what a noisy /
    weakly-identified target needs to reproduce the gold posterior — the default
    config is too short for some. Draw counts are capped for cost; raise the
    caps for a stubborn posterior.
    """
    args = reference_info(name).get("inference", {}).get("method_arguments", {})
    adapt = args.get("control", {}).get("adapt_delta", 0.8)
    warmup = min(int(args.get("warmup", 1000)), warmup_cap)
    return {"chains": chains, "iter_warmup": max(warmup, 1000),
            "iter_sampling": sampling, "adapt_delta": adapt}


# ---------------------------------------------------------------------------
# Reference draws -> seed-blocked GT answers (used by executor_reference)
# ---------------------------------------------------------------------------

def reference_blocks(name: str, n_blocks: int) -> list[dict]:
    """Partition the gold chains into ``n_blocks`` record answers.

    Each block concatenates a disjoint group of chains; per block we emit a
    record {param: [draws]} (the cloud representation). With 10 high-ESS,
    R-hat~1 chains, inter-block Wasserstein-1 is the reference posterior's own
    Monte-Carlo noise floor — exactly the floor the gate measures from k seeds.
    """
    chains = reference_chains(name)
    n_blocks = max(1, min(n_blocks, len(chains)))
    params = list(chains[0].keys())
    # contiguous chain groups, as even as possible
    out: list[dict] = []
    base, extra = divmod(len(chains), n_blocks)
    i = 0
    for b in range(n_blocks):
        take = base + (1 if b < extra else 0)
        group = chains[i:i + take]
        i += take
        rec = {p: [v for ch in group for v in ch[p]] for p in params}
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def problem_id(name: str) -> str:
    info = posterior_info(name)
    return f"{CORPUS_PREFIX}-{info['data_name']}/{info['model_name']}"


def answer_spec(name: str) -> dict:
    """record{param: dist/real} — per-parameter posterior marginals."""
    return {
        "kind": "record",
        "fields": {p: {"kind": "dist", "domain": "real"} for p in param_names(name)},
    }


def stan_realization(name: str, sampling: dict | None = None) -> dict:
    model = model_code(name)
    declared = set(data_block_vars(model))
    data = {k: v for k, v in model_data(name).items() if k in declared}
    bundle = pack(model, data, param_names(name), sampling)
    return {"problem_id": problem_id(name), "language": "stan", "code": bundle}


def reference_realization(name: str) -> dict:
    # `code` is the posterior name; executor_reference replays its stored draws.
    return {"problem_id": problem_id(name), "language": "reference", "code": name}


def problem_record(name: str, statement: dict | None = None) -> dict:
    info = posterior_info(name)
    minfo_path = _pdb_root() / "models" / "info" / f"{info['model_name']}.info.json"
    minfo = json.loads(minfo_path.read_text()) if minfo_path.exists() else {}
    return {
        "problem_id": problem_id(name),
        "provenance": {
            "source": f"posteriordb/{name}",
            "origin_language": "stan",
            "title": minfo.get("title", ""),
            "references": minfo.get("references", []),
        },
        "statement": statement or {
            "given": "",
            "model": "",
            "query": "",
        },
        "answer_spec": answer_spec(name),
        "status": {"review": "draft", "notes": "ingested from posteriordb"},
    }


# ---------------------------------------------------------------------------
# Build CLI: emit problem/realization rows (merge by problem_id, never clobber)
# ---------------------------------------------------------------------------

def cmd_build(args) -> None:
    all_names = gold_posterior_names()
    # accept either posteriordb posterior names or problem_ids
    by_pid = {problem_id(n): n for n in all_names}
    resolved = [by_pid.get(n, n) for n in (args.ids or all_names)]

    # preserve existing authored statements when rebuilding
    existing_stmt: dict[str, dict] = {}
    if PROBLEMS_OUT.exists():
        for r in load_jsonl(PROBLEMS_OUT):
            existing_stmt[r["problem_id"]] = r.get("statement", {})

    problems, stans, refs = [], [], []
    skipped: list[tuple[str, str]] = []
    for name in resolved:
        pid = problem_id(name)
        try:
            stan = stan_realization(name)
            ref = reference_realization(name)
            stmt = existing_stmt.get(pid)
            prob = (problem_record(name, stmt)
                    if stmt and any(stmt.get(k) for k in ("given", "model", "query"))
                    else problem_record(name))
        except Exception as exc:  # one bad posterior must not sink the batch
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        problems.append(prob)
        stans.append(stan)
        refs.append(ref)
    if skipped:
        print(f"[posteriordb build] skipped {len(skipped)} (ingestion error):")
        for n, e in skipped:
            print(f"    {n}: {e[:100]}")

    n_p = merge_jsonl(PROBLEMS_OUT, problems)
    n_s = merge_jsonl(STAN_OUT, stans)
    n_r = merge_jsonl(REFERENCE_OUT, refs)
    print(f"[posteriordb build] {len(resolved)} posterior(s) -> "
          f"problems={n_p} stan={n_s} reference={n_r}")


def authoring_material(name: str) -> dict:
    """Everything a statement author (human or agent) needs for one posterior:
    the reference Stan model (ground truth for what the model IS), a compact data
    summary, the queried parameters, and the title/description/references."""
    info = posterior_info(name)
    minfo_p = _pdb_root() / "models" / "info" / f"{info['model_name']}.info.json"
    minfo = json.loads(minfo_p.read_text()) if minfo_p.exists() else {}
    model = model_code(name)
    declared = set(data_block_vars(model))
    summary = {}
    for k, v in model_data(name).items():
        if k not in declared:
            continue
        summary[k] = (f"array[{len(v)}] e.g. {v[:4]}" if isinstance(v, list) else v)
    return {
        "problem_id": problem_id(name),
        "name": name,
        "title": minfo.get("title", ""),
        "description": minfo.get("description", ""),
        "references": minfo.get("references", []),
        "params": param_names(name),
        "model_code": model,
        "data_summary": summary,
    }


def cmd_material(args) -> None:
    all_names = gold_posterior_names()
    by_pid = {problem_id(n): n for n in all_names}
    names = [by_pid.get(x, x) for x in args.ids] if args.ids else all_names
    mat = [authoring_material(n) for n in names]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mat))
    print(f"[posteriordb material] wrote {len(mat)} -> {out}")


EXCLUDED_OUT = _REPO_ROOT / "data/problems/_posteriordb_excluded.jsonl"


def retire(pid: str, reason: str, *, evidence: dict | None = None) -> None:
    """Remove a problem from the live posteriordb corpus, with an audit record.

    Drops the problem + its stan/reference realizations and appends a row to
    `_posteriordb_excluded.jsonl` (problem_id, reason, evidence) so the exclusion
    is documented, never silent. For posteriors the gate finds non-discriminable
    (the gold marginal itself trips the floor) or impossible to bind.
    """
    for path in (PROBLEMS_OUT, STAN_OUT, REFERENCE_OUT):
        if path.exists():
            write_jsonl(path, [r for r in load_jsonl(path) if r["problem_id"] != pid])
    led = load_jsonl(EXCLUDED_OUT) if EXCLUDED_OUT.exists() else []
    led = [r for r in led if r["problem_id"] != pid]
    led.append({"problem_id": pid, "status": "excluded", "reason": reason,
                "evidence": evidence or {}})
    write_jsonl(EXCLUDED_OUT, sorted(led, key=lambda r: r["problem_id"]))


def cmd_list(args) -> None:
    for name in gold_posterior_names():
        print(f"{problem_id(name):<55s} params={len(param_names(name))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="posteriordb ingestion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="emit problem/realization rows")
    b.add_argument("--ids", nargs="*", help="posterior names or problem_ids "
                   "(default: all gold-draw posteriors)")
    b.set_defaults(func=cmd_build)
    li = sub.add_parser("list", help="list gold-draw posteriors")
    li.set_defaults(func=cmd_list)
    mt = sub.add_parser("material", help="dump statement-authoring material JSON")
    mt.add_argument("--ids", nargs="*", help="posterior names or problem_ids (default: all gold)")
    mt.add_argument("--out", default="data/.rework/posteriordb_material.json")
    mt.set_defaults(func=cmd_material)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
