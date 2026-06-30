"""Grouped bar chart of pass rate for the update slide.

Groups = datasets (WebPPL / Pyro / Stan); bars within a group = models.
Pass rates are the CORRECTED rates (unscorable GT-broken problems dropped),
read live from runs/matrix/triage/corrected_pass_rates.json — single source,
no hardcoded numbers. Regenerate that file with eval.triage_exec_errors.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

langs = ["WebPPL", "Pyro", "Stan"]
_KEYS = ["webppl", "pyro", "stan"]
# Display order of models (most-capable first).
_ORDER = ["sonnet", "haiku", "qwen3-235b", "gpt-oss-120b",
          "llama-3.3-70b", "gpt-oss-20b", "qwen3.5-9b"]

_DATA = json.loads(
    (Path(__file__).resolve().parents[1]
     / "runs/matrix/triage/corrected_pass_rates.json").read_text())
_BY = {}
for _r in _DATA["corrected"]:
    _BY.setdefault(_r["model"], {})[_r["lang"]] = _r["corrected"]
models = [(m, [_BY[m][k] for k in _KEYS]) for m in _ORDER]
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(models))]

x = np.arange(len(langs))
n = len(models)
w = 0.8 / n

fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200)
for i, (name, vals) in enumerate(models):
    off = (i - (n - 1) / 2) * w
    ax.bar(x + off, vals, w, label=name, color=colors[i], edgecolor="none")

ax.set_ylabel("pass rate", fontsize=11)
ax.set_ylim(0, 0.95)
ax.set_xticks(x)
ax.set_xticklabels(langs, fontsize=12)
ax.tick_params(axis="y", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#e3ddd0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.16))

fig.tight_layout()
fig.savefig("results.png", bbox_inches="tight", facecolor="white")
print("saved results.png")
