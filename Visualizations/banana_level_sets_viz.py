"""
banana_level_sets_viz.py

Caption (put in LaTeX, not rendered in the figure): Level sets of the fitted
q(z) against the banana target p(z) (dashed contours, overlaid on every
panel) over training, for four gradient estimators: Straight-Through, Score
Function, ODE Transport (linear path) and ODE Transport (geometric path).
Rows are training iterations; columns are estimators. Hyperparameters:
K=5 components, param_type=eigenvaluedecomp, MC_samples=256, Adam lr=5e-3
with cosine annealing over 1600 steps, gradient clipped to norm 1.0,
ODE Transport integrated with 8 RK4 steps (both paths), seed=3.

Adapted from the exploratory version in Test.ipynb: drops the redundant
"Target" column (the dashed target contour on every panel already shows it)
and adds the geometric-path ODE Transport estimator alongside the linear one.

Trains real gmvi models -- this is not a closed-form/no-training theory
sketch like the other Visualizations/*.py scripts.
"""

import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from gmvi.targets.distributions import make_target
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.estimators.gradient_estimators import make_estimator
from gmvi.utils.visualization import _make_grid

OUTDIR = os.path.dirname(__file__)

# ── LaTeX-matched fonts (no working latex/dvipng on this machine; mathtext
#    Computer-Modern fallback reads the same at these sizes). Figure width is
#    pinned to \textwidth (418pt) so these point sizes read correctly once
#    placed at full width in the thesis, instead of shrinking to near-illegible
#    size. ─────────────────────────────────────────────────────────────────────
TEXTWIDTH_IN = 418 / 72.27

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# ── Config ───────────────────────────────────────────────────────────────────
SEED           = 3
SNAPSHOT_STEPS = [200, 400, 800, 1600]
COMPONENTS     = 5
PARAM_TYPE     = "eigenvaluedecomp"
MC_SAMPLES     = 256
LR             = 5e-3
ODE_STEPS      = 8
XLIM, YLIM     = (-6, 6), (-6, 9)
RESOLUTION     = 200

torch.manual_seed(SEED)
target_banana = make_target("banana")

# ── Training helper that saves model state at given steps and timing ───────────
def train_with_snapshots(model, estimator, log_target, lr, max_step, snapshot_steps):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_step)
    snaps = {}
    t0 = time.perf_counter()
    for step in range(max_step + 1):
        opt.zero_grad()
        loss, _ = estimator.loss(model, log_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if step in snapshot_steps:
            snaps[step] = copy.deepcopy(model.state_dict())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return snaps, elapsed_ms / max_step


@torch.no_grad()
def final_elbo(model, log_target, n_samples=8192):
    """Estimator-agnostic ELBO of the trained model: E_q[log p(z) - log q(z)]."""
    z, _ = model.sample(n_samples)
    elbo = log_target(z) - model.log_prob(z)
    return elbo.mean().item(), (elbo.std() / n_samples ** 0.5).item()


# ── Run all four estimators ─────────────────────────────────────────────────
# Display label is "Straight-Through", not "Gumbel-Softmax": make_estimator's
# registry key is still "gumbel_softmax" (gmvi core, unchanged), but the
# estimator it builds does z = z_soft + (z_hard - z_soft).detach() -- the
# straight-through trick (hard sample forward, soft/Gumbel-Softmax gradient
# backward) -- which is what's actually plotted here, not the plain soft
# relaxation. See GumbelSoftmaxEstimator's docstring in gradient_estimators.py.
estimator_specs = {
    "Straight-Through":          make_estimator("gumbel_softmax", MC_samples=MC_SAMPLES),
    "Score Function":            make_estimator("score_function", MC_samples=MC_SAMPLES),
    "ODE Transport (linear)":    make_estimator("ode_transport", MC_samples=MC_SAMPLES,
                                                 ode_steps=ODE_STEPS, path="linear"),
    "ODE Transport (geometric)": make_estimator("ode_transport", MC_samples=MC_SAMPLES,
                                                 ode_steps=ODE_STEPS, path="geometric"),
}
est_colors = {
    "Straight-Through":          "#457B9D",
    "Score Function":            "#E63946",
    "ODE Transport (linear)":    "#2A9D8F",
    "ODE Transport (geometric)": "#1B5E56",
}
# Two-line titles: at \textwidth/4 per panel, "ODE Transport (linear)" on one
# line collides with its neighbor at 11pt.
title_labels = {
    "Straight-Through":          "Straight-\nThrough",
    "Score Function":            "Score\nFunction",
    "ODE Transport (linear)":    "ODE Transp.\n(linear)",
    "ODE Transport (geometric)": "ODE Transp.\n(geometric)",
}

# Training is ~3min; cache results so cosmetic plot tweaks don't require a
# full retrain. Delete the cache file (or set FORCE_RETRAIN=1) to retrain.
CACHE_PATH = os.path.join(OUTDIR, ".banana_level_sets_cache.pt")
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"

snap_steps_set = set(SNAPSHOT_STEPS)

if os.path.exists(CACHE_PATH) and not FORCE_RETRAIN:
    print(f"Loading cached training results from {CACHE_PATH} "
          f"(set FORCE_RETRAIN=1 to retrain)")
    cache = torch.load(CACHE_PATH, weights_only=False)
    all_snaps, ms_per_step, elbo_final = (
        cache["all_snaps"], cache["ms_per_step"], cache["elbo_final"])
else:
    all_snaps = {}
    ms_per_step = {}
    elbo_final = {}

    for label, est in estimator_specs.items():
        torch.manual_seed(SEED)
        ref   = make_reference("normal", dim=2)
        model = GeneralizedMixture(n_components=COMPONENTS, dim=2, reference=ref,
                                   param_type=PARAM_TYPE, init_scale=2.5)
        print(f"Training {label}...")
        all_snaps[label], ms_per_step[label] = train_with_snapshots(
            model, est, target_banana.log_prob,
            lr=LR, max_step=max(SNAPSHOT_STEPS), snapshot_steps=snap_steps_set,
        )
        model.eval()
        elbo_final[label] = final_elbo(model, target_banana.log_prob)

    torch.save({"all_snaps": all_snaps, "ms_per_step": ms_per_step,
                "elbo_final": elbo_final}, CACHE_PATH)

print("Done training.")
print()
print(f"{'estimator':<28} {'final ELBO':>16} {'ms/step':>10}")
for label in estimator_specs:
    mean, stderr = elbo_final[label]
    print(f"{label:<28} {mean:>9.3f} +/- {stderr:<5.3f} {ms_per_step[label]:>9.2f}")

# ── Pre-compute target grid (used only for the dashed reference contours) ──────
xx, yy, grid = _make_grid(XLIM, YLIM, RESOLUTION)
with torch.no_grad():
    log_p = target_banana.log_prob(grid).numpy().reshape(xx.shape)
log_p -= log_p.max()
p_density = np.exp(log_p)

# ── Plot: rows = iterations, cols = estimators (no separate Target column).
#    Figure width pinned to \textwidth so the rcParams point sizes above read
#    correctly at final print size; constrained_layout + near-zero wspace/
#    hspace keep it compact instead of padded. ──────────────────────────────────
n_iters = len(SNAPSHOT_STEPS)
n_cols  = len(estimator_specs)
panel_aspect = (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])   # height/width per panel

fig, axes = plt.subplots(
    n_iters, n_cols,
    figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / n_cols * panel_aspect * n_iters),
    sharex=True, sharey=True,
    constrained_layout=True,
    gridspec_kw={"wspace": 0.04, "hspace": 0.04},
)

for row, step in enumerate(SNAPSHOT_STEPS):
    for col, (label, color) in enumerate(zip(estimator_specs, est_colors.values())):
        ax = axes[row, col]

        ref_tmp = make_reference("normal", dim=2)
        m = GeneralizedMixture(n_components=COMPONENTS, dim=2, reference=ref_tmp,
                               param_type=PARAM_TYPE, init_scale=2.5)
        m.load_state_dict(all_snaps[label][step])
        m.eval()

        with torch.no_grad():
            log_q = m.log_prob(grid).numpy().reshape(xx.shape)
        log_q -= log_q.max()
        q_density = np.exp(log_q)

        ax.contourf(xx, yy, q_density, levels=8, cmap="Reds", alpha=0.75)
        ax.contour( xx, yy, q_density, levels=8, colors=color, linewidths=0.5, alpha=0.8)
        ax.contour( xx, yy, p_density, levels=5, colors="steelblue", linewidths=0.7,
                    linestyles="--", alpha=0.4)
        ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if row == 0:
            ax.set_title(title_labels[label], fontweight="bold", color=color, pad=3)
        if col == 0:
            ax.set_ylabel(f"iter {step}", labelpad=4)

for ext in ("png", "pdf"):
    out_path = os.path.join(OUTDIR, f"banana_level_sets.{ext}")
    fig.savefig(out_path, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {out_path}")
