"""
theory_visualization_geometric.py

Companion to theory_visualization.py: shows the intermediate distributions rho_t
under the GEOMETRIC path (cf. eq:geometric_interpolation in 03_ODEMethod.tex)
instead of the linear path:

    A_{j,t} = A_j^t = exp(t log A_j)      (geometric, this script)
    A_{j,t} = t A_j + (1-t) I             (linear, theory_visualization.py)

Same mixture / reference / grid as the linear script so the two are directly
comparable. Produces:
  - rho_t_all_geometric.png            5-panel geometric-path analogue of rho_t_all.png
  - linear_vs_geometric_rho_t.png      2x5 grid: linear (top) vs. geometric (bottom)

No training. No gmvi internals. Only numpy / scipy / matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
from scipy.special import logsumexp

# ── Hand-crafted target mixture (identical to theory_visualization.py) ─────────
K, D = 3, 2

log_weights = np.log([0.40, 0.35, 0.25])

means = np.array([
    [ 3.0,  1.0],
    [-2.2,  2.6],
    [ 0.4, -2.8],
], dtype=float)

A_mats = np.array([
    [[ 1.2,  0.4],
     [ 0.0,  0.6]],

    [[ 0.9, -0.3],
     [ 0.3,  0.7]],

    [[ 0.8,  0.2],
     [-0.2,  1.1]],
], dtype=float)

log_A_mats = np.stack([logm(A_mats[j]).real for j in range(K)])   # log A_j, (K, D, D)
logdet_A   = np.array([np.linalg.slogdet(A_mats[j])[1] for j in range(K)])  # log|det A_j|


# ── Density helpers ────────────────────────────────────────────────────────────

def log_ref(z: np.ndarray) -> np.ndarray:
    """N(0, I) log-density.  z: (N, D) -> (N,)"""
    return -0.5 * (np.sum(z ** 2, axis=-1) + D * np.log(2 * np.pi))


def log_rho_t(x: np.ndarray, t: float, path: str) -> np.ndarray:
    """
    Log density of the intermediate distribution rho_t under the linear or
    geometric path.

    path='linear':     A_{j,t} = t A_j + (1-t) I
    path='geometric':  A_{j,t} = A_j^t = exp(t log A_j)

    x: (N, D) grid points
    t: scalar in [0, 1]
    returns: (N,) log densities
    """
    N  = x.shape[0]
    lw = log_weights - logsumexp(log_weights)
    log_comp = np.zeros((N, K))

    if path == "linear":
        I = np.eye(D)
        for j in range(K):
            A_jt     = t * A_mats[j] + (1.0 - t) * I
            A_jt_inv = np.linalg.inv(A_jt)
            _, logdet = np.linalg.slogdet(A_jt)
            z = (x - t * means[j]) @ A_jt_inv.T
            log_comp[:, j] = log_ref(z) - logdet

    elif path == "geometric":
        for j in range(K):
            A_jt_inv = expm(-t * log_A_mats[j])          # A_j^{-t}
            logdet   = t * logdet_A[j]                    # log|det A_j^t| = t log|det A_j|
            z = (x - t * means[j]) @ A_jt_inv.T
            log_comp[:, j] = log_ref(z) - logdet

    else:
        raise ValueError(f"unknown path: {path!r}")

    return logsumexp(lw[None, :] + log_comp, axis=1)


# ── Evaluation grid (identical to theory_visualization.py) ─────────────────────
XLIM, YLIM = (-5.5, 6.0), (-5.5, 5.5)
RES = 350

xs = np.linspace(*XLIM, RES)
ys = np.linspace(*YLIM, RES)
xx, yy = np.meshgrid(xs, ys)
grid   = np.stack([xx.ravel(), yy.ravel()], axis=1)

T_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
COLORS = ["#2166AC", "#4393C3", "#878787", "#D6604D", "#B2182B"]
LEVELS_FRAC = [0.05, 0.20, 0.45, 0.75, 0.95]

# Pre-compute densities for both paths (peak-normalised for fair contour comparison)
densities = {"linear": {}, "geometric": {}}
for path in densities:
    for t in T_VALS:
        log_d = log_rho_t(grid, t, path).reshape(xx.shape)
        log_d -= log_d.max()
        densities[path][t] = np.exp(log_d)


def _draw_panel(ax, t, color, d):
    ax.set_facecolor(BG)
    levels = [f * d.max() for f in LEVELS_FRAC]
    ax.contourf(xx, yy, d, levels=20, cmap="Greys", alpha=0.12)
    ax.contour(xx, yy, d, levels=levels, colors=color, linewidths=1.8, alpha=0.95)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: five panels, geometric path only (analogue of rho_t_all.png)
# ══════════════════════════════════════════════════════════════════════════════
BG = "#ffffff"

TITLES = {
    0.00: r"$q_0$ — reference $\mathcal{N}(0,I)$",
    0.25: r"$q_{0.25}$",
    0.50: r"$q_{0.50}$",
    0.75: r"$q_{0.75}$",
    1.00: r"$q_1$ — target $q_\theta$",
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG,
                          gridspec_kw={"wspace": 0.06})

for ax, t, color in zip(axes, T_VALS, COLORS):
    _draw_panel(ax, t, color, densities["geometric"][t])
    ax.set_title(TITLES[t], color=color, fontsize=11, fontweight="bold", pad=8)

plt.savefig("rho_t_all_geometric.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: rho_t_all_geometric.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: linear (top row) vs. geometric (bottom row), same t / color per column
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(18, 7.6), facecolor=BG,
                          gridspec_kw={"wspace": 0.06, "hspace": 0.16})

for row, path in zip(axes, ["linear", "geometric"]):
    for ax, t, color in zip(row, T_VALS, COLORS):
        _draw_panel(ax, t, color, densities[path][t])
        ax.set_title(TITLES[t], color=color, fontsize=11, fontweight="bold", pad=8)

axes[0, 0].set_ylabel("linear path\n" + r"$A_{j,t}=tA_j+(1-t)I$",
                       fontsize=10, labelpad=10)
axes[1, 0].set_ylabel("geometric path\n" + r"$A_{j,t}=\exp(t\log A_j)$",
                       fontsize=10, labelpad=10)

plt.savefig("linear_vs_geometric_rho_t.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: linear_vs_geometric_rho_t.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: rho_t^linear - rho_t^geometric difference heatmap
# (The two paths agree exactly at t=0 and t=1 by construction, so only the
#  interior t values carry any signal.)
# ══════════════════════════════════════════════════════════════════════════════
DIFF_T_VALS = [0.25, 0.5, 0.75]

diffs = [densities["linear"][t] - densities["geometric"][t] for t in DIFF_T_VALS]
vmax  = max(np.abs(d).max() for d in diffs)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), facecolor=BG,
                          gridspec_kw={"wspace": 0.08})

for ax, t, d, color in zip(axes, DIFF_T_VALS, diffs, COLORS[1:4]):
    ax.set_facecolor(BG)
    im = ax.contourf(xx, yy, d, levels=21, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(xx, yy, densities["linear"][t], levels=LEVELS_FRAC,
               colors="k", linewidths=0.6, alpha=0.35)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb")
    ax.set_title(rf"$t={t}$: $q_t^{{\mathrm{{lin}}}} - q_t^{{\mathrm{{geo}}}}$",
                 color=color, fontsize=11, fontweight="bold", pad=8)

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("density difference (peak-normalised)", fontsize=9)

plt.savefig("linear_vs_geometric_rho_t_diff.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: linear_vs_geometric_rho_t_diff.png")
