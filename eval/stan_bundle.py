"""Self-contained Stan realization bundle: model code + harness directives.

A Stan model is data-parametric by design — the ``data`` block declares names,
the values live in a separate file, and the run extracts named parameters. The
rest of the eval machinery, however, threads a single ``code`` string through a
uniform executor interface ``(code, seeds, timeout, workers)`` and caches GT
runs by the hash of that string. To keep both invariants, a Stan realization is
a *bundle*: the model code followed by three single-line comment directives that
carry the data, the queried parameters, and the sampler configuration.

    data { ... } parameters { ... } model { ... }

    //@ DATA {"J": 8, "y": [...], "sigma": [...]}
    //@ PARAMS ["mu", "tau", "theta[1]", ...]
    //@ SAMPLING {"chains": 4, "iter_warmup": 1000, "iter_sampling": 1000}

The directives are valid Stan line comments, so the bundle still reads (and
compiles) as a Stan program; the executor strips them to recover the run inputs.
Embedding data in the string means a different dataset yields a different cache
key automatically — correctness for free.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_DIRECTIVE_RE = re.compile(r"^//@\s+(DATA|PARAMS|SAMPLING)\s+(.*)$")

DEFAULT_SAMPLING = {"chains": 4, "iter_warmup": 1000, "iter_sampling": 1000}


@dataclass(frozen=True)
class StanBundle:
    model: str
    data: dict
    params: list[str]
    sampling: dict


def pack(model: str, data: dict, params: list[str],
         sampling: dict | None = None) -> str:
    """Assemble a bundle string from its parts."""
    s = dict(DEFAULT_SAMPLING)
    if sampling:
        s.update(sampling)
    lines = [
        model.rstrip(),
        "",
        f"//@ DATA {json.dumps(data, separators=(',', ':'))}",
        f"//@ PARAMS {json.dumps(params, separators=(',', ':'))}",
        f"//@ SAMPLING {json.dumps(s, separators=(',', ':'))}",
    ]
    return "\n".join(lines) + "\n"


def data_block(bundle: str) -> str:
    """The verbatim `data { ... }` block from a bundle's model.

    This is the input interface the harness binds by name — emitted into the
    Stan solver prompt so a solver declares exactly the supplied inputs (it
    pins the I/O signature, never the model body). Empty string if absent.
    """
    model = unpack(bundle).model
    m = re.search(r"data\s*\{[^}]*\}", model)
    return m.group(0) if m else ""


def repack_model(bundle: str, model: str, sampling: dict | None = None) -> str:
    """Return a new bundle with `model` swapped in, keeping the original's
    data and params.

    Used to score a solver's bare Stan model: a solver writes only the program
    (data block declarations + parameters + model), so it is repacked around the
    GT bundle's embedded data values and queried params — it runs against the
    same inputs as the ground-truth realization. ``sampling`` overrides the
    bundle's sampler config (None keeps it); candidates pass DEFAULT_SAMPLING so
    a solver's fit never inherits a GT's heavy gold-reproduction regime.
    """
    b = unpack(bundle)
    return pack(model, b.data, b.params, sampling if sampling is not None else b.sampling)


def unpack(bundle: str) -> StanBundle:
    """Recover (model, data, params, sampling) from a bundle string.

    The model is everything above the first directive line; directives may
    appear in any order. Missing DATA/PARAMS raise; SAMPLING defaults.
    """
    model_lines: list[str] = []
    found: dict[str, object] = {}
    for line in bundle.splitlines():
        m = _DIRECTIVE_RE.match(line.strip())
        if m:
            found[m.group(1)] = json.loads(m.group(2))
        elif not found:   # model is everything before the first directive
            model_lines.append(line)
    if "DATA" not in found:
        raise ValueError("Stan bundle missing //@ DATA directive")
    if "PARAMS" not in found:
        raise ValueError("Stan bundle missing //@ PARAMS directive")
    sampling = dict(DEFAULT_SAMPLING)
    if isinstance(found.get("SAMPLING"), dict):
        sampling.update(found["SAMPLING"])  # type: ignore[arg-type]
    return StanBundle(
        model="\n".join(model_lines).strip() + "\n",
        data=found["DATA"],          # type: ignore[arg-type]
        params=list(found["PARAMS"]),  # type: ignore[arg-type]
        sampling=sampling,
    )
