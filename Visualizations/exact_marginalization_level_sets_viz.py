"""
exact_marginalization_level_sets_viz.py

Caption (put in LaTeX, not rendered in the figure): Level sets of the
banana target p(z) (left, blue), the Exact Marginalization fit q(z)
(middle, purple) from exact_marginalization_baseline.py (MC_samples=4096,
n_steps=10000, seed=3, K=5, eigenvaluedecomp, final KL(q||p)=0.0198), and
the five individual weighted mixture components w_i q_i(z) (right, one
color per component) that q(z) sums to. Style follows the original
banana_level_sets.png (Test.ipynb): filled density + contour lines, no
peak-normalization (real, correctly-scaled densities throughout).

Loads the trained parameters from exact_marginalization_baseline.json --
does not retrain.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gmvi.targets.distributions import make_target
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.utils.visualization import _make_grid

OUTDIR = os.path.dirname(__file__)

TEXTWIDTH_IN = 418 / 72.27
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
})

TARGET_COLOR = "#457B9D"      # steelblue-ish, matches original "Target" column
MARG_COLOR   = "#6A4C93"      # Exact Marginalization's established color
COMPONENT_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#6A4C93"]

XLIM, YLIM = (-6, 6), (-6, 9)
RESOLUTION = 200

# ── Load trained parameters (no retraining) ─────────────────────────────────
with open(os.path.join(OUTDIR, "exact_marginalization_baseline.json")) as f:
    report = json.load(f)

cfg = report["config"]
target = make_target("banana")
ref = make_reference("normal", dim=2)
model = GeneralizedMixture(n_components=cfg["n_components"], dim=2, reference=ref,
                           param_type=cfg["param_type"], init_scale=2.5)

weights = torch.tensor(report["final_parameters"]["weights"])
with torch.no_grad():
    model.log_weights.copy_(torch.log(weights))
    for comp, saved in zip(model.components, report["final_parameters"]["components"]):
        comp.a.copy_(torch.tensor(saved["mean_a"]))
        comp.log_diag.copy_(torch.tensor(saved["log_diag"]))
        comp.skew_raw.copy_(torch.tensor(saved["skew_raw"]))
model.eval()

# ── Grid & densities ─────────────────────────────────────────────────────────
xx, yy, grid = _make_grid(XLIM, YLIM, RESOLUTION)
with torch.no_grad():
    p_density = target.log_prob(grid).exp().numpy().reshape(xx.shape)
    q_density = model.log_prob(grid).exp().numpy().reshape(xx.shape)
    comp_log_probs = model.component_log_probs(grid)          # (N, K)
    comp_densities = (comp_log_probs.exp() * weights[None, :]).numpy()  # weighted, (N, K)
    comp_densities = [comp_densities[:, k].reshape(xx.shape) for k in range(cfg["n_components"])]


def _panel(ax, density, cmap, line_color, title):
    ax.contourf(xx, yy, density, levels=12, cmap=cmap, alpha=0.85)
    ax.contour(xx, yy, density, levels=12, colors=line_color, linewidths=0.5, alpha=0.7)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb")
    ax.set_title(title, fontsize=11, fontweight="bold", color=line_color, pad=6)


fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.4),
                          constrained_layout=True)

blues = LinearSegmentedColormap.from_list("target", ["#ffffff", TARGET_COLOR])
purples = LinearSegmentedColormap.from_list("marg", ["#ffffff", MARG_COLOR])

_panel(axes[0], p_density, blues, TARGET_COLOR, "target")
_panel(axes[1], q_density, purples, MARG_COLOR, "Exact Marg.")

LEVELS_FRAC = [0.10, 0.30, 0.60, 0.90]  # fractions of each component's own peak

ax3 = axes[2]
ax3.contour(xx, yy, p_density, levels=6, colors="#999999", linewidths=0.6,
            linestyles="--", alpha=0.6)
for k in range(cfg["n_components"]):
    d = comp_densities[k]
    levels = [f * d.max() for f in LEVELS_FRAC]
    ax3.contour(xx, yy, d, levels=levels, colors=COMPONENT_COLORS[k],
                linewidths=1.1, alpha=0.9)
ax3.set_xlim(XLIM); ax3.set_ylim(YLIM); ax3.set_aspect("equal")
ax3.set_xticks([]); ax3.set_yticks([])
for sp in ax3.spines.values():
    sp.set_edgecolor("#bbbbbb")
ax3.set_title("5 components", fontsize=11, fontweight="bold", pad=6)

for k in range(cfg["n_components"]):
    w = report["final_parameters"]["weights"][k]
    ax3.plot([], [], color=COMPONENT_COLORS[k], label=f"$w_{k}$={w:.3f}")
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
           fontsize=6.5, frameon=False)

for ext in ("png", "pdf"):
    out_path = os.path.join(OUTDIR, f"exact_marginalization_level_sets.{ext}")
    fig.savefig(out_path, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {out_path}")
