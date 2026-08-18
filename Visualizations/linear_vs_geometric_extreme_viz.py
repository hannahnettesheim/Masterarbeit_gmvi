"""
linear_vs_geometric_extreme_viz.py

Two strongly anisotropic Gaussian components (A_1: eigenvalues 10, 1, tilted
+30deg; A_2: eigenvalues 6, 1, tilted -30deg), offset so the mixture bends
into a banana-like curve, make the gap between the linear path
A_t = tA + (1-t)I and the geometric path A_t = exp(t log A) = A^t clearly
visible -- unlike the mild mixture used in rho_t_all.png where the two paths
look almost identical. Top row: linear path. Middle row: geometric path.
Bottom row: their density difference. No caption text is rendered in the
figure -- write it directly in LaTeX; the ingredients are summarized below.

Why the gap is large here: along a component's eigenvector with eigenvalue
lambda, both paths interpolate the *scalar* eigenvalue between 1 (reference)
and lambda (target) -- linear via the arithmetic mean t*lambda + (1-t),
geometric via the geometric mean lambda^t. By AM-GM,
t*lambda + (1-t)*1 >= lambda^t * 1^(1-t) always, with the gap growing with
|log(lambda)|. High-condition-number components (as here, lambda=10 and 6)
are exactly the regime where linear and geometric paths should be expected to
diverge; near-isotropic components (as in the original rho_t_all example)
are exactly the regime where they should look alike.

No training. No gmvi internals. Only numpy / scipy / matplotlib.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import logm, expm
from scipy.special import logsumexp

OUTDIR = os.path.dirname(__file__)

# ── Thesis color palette (Visualizations/Instructions.txt) ─────────────────────
PALETTE = {
    "score_function": "#E63946",
    "gumbel_softmax": "#457B9D",
    "ode_transport":  "#2A9D8F",
    "target":         "#264653",
    "samples":        "#A8DADC",
}
LINEAR_COLOR    = PALETTE["gumbel_softmax"]
GEOMETRIC_COLOR = PALETTE["ode_transport"]

# ── LaTeX-matched fonts. No working latex/dvipng on this machine (checked), so
#    fall back to matplotlib's built-in Computer-Modern mathtext, which reads
#    the same at these sizes without needing a LaTeX install. ─────────────────
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# ── Two strongly anisotropic SPD components, offset to bend like a banana ──────
D = 2
COMPONENTS = [
    dict(weight=0.5, mean=np.array([-3.0, -1.2]), lam1=10.0, lam2=1.0, theta_deg=30.0),
    dict(weight=0.5, mean=np.array([ 3.0,  1.2]), lam1=6.0,  lam2=1.0, theta_deg=-30.0),
]
K = len(COMPONENTS)


def _spd_from_eigs(lam1, lam2, theta_deg):
    theta = np.deg2rad(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R @ np.diag([lam1, lam2]) @ R.T


A_list       = [_spd_from_eigs(c["lam1"], c["lam2"], c["theta_deg"]) for c in COMPONENTS]
means        = [c["mean"] for c in COMPONENTS]
log_weights  = np.log([c["weight"] for c in COMPONENTS])
log_A_list   = [logm(A).real for A in A_list]
logdet_list  = [np.linalg.slogdet(A)[1] for A in A_list]


def log_ref(z: np.ndarray) -> np.ndarray:
    return -0.5 * (np.sum(z ** 2, axis=-1) + D * np.log(2 * np.pi))


def log_rho_t(x: np.ndarray, t: float, path: str) -> np.ndarray:
    """Mixture log-density under the linear or geometric path."""
    N = x.shape[0]
    lw = log_weights - logsumexp(log_weights)
    log_comp = np.zeros((N, K))

    for j in range(K):
        if path == "linear":
            A_t = t * A_list[j] + (1.0 - t) * np.eye(D)
            A_t_inv = np.linalg.inv(A_t)
            logdet = np.linalg.slogdet(A_t)[1]
        elif path == "geometric":
            A_t_inv = expm(-t * log_A_list[j])
            logdet = t * logdet_list[j]
        else:
            raise ValueError(f"unknown path: {path!r}")

        z = (x - t * means[j]) @ A_t_inv.T
        log_comp[:, j] = log_ref(z) - logdet

    return logsumexp(lw[None, :] + log_comp, axis=1)


# ── Evaluation grid, sized to the t=1 mixture extent (5% of peak, principal axes) ─
XLIM, YLIM = (-27.0, 20.0), (-16.0, 13.0)
RES = 450
xs = np.linspace(*XLIM, RES)
ys = np.linspace(*YLIM, RES)
xx, yy = np.meshgrid(xs, ys)
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

T_VALS = [0.25, 0.5, 0.75]   # t=0,1 are identical for both paths by construction

densities = {"linear": {}, "geometric": {}}
for path in densities:
    for t in T_VALS:
        log_d = log_rho_t(grid, t, path).reshape(xx.shape)
        log_d -= log_d.max()
        densities[path][t] = np.exp(log_d)

# Reference: t=1 target shape (identical for both paths) so the banana bend is visible
log_target = log_rho_t(grid, 1.0, "linear").reshape(xx.shape)
log_target -= log_target.max()
target_density = np.exp(log_target)

print("stretching-axis eigenvalue A_t per component:")
for j, c in enumerate(COMPONENTS):
    for t in T_VALS:
        lin = t * c["lam1"] + (1 - t)
        geo = c["lam1"] ** t
        print(f"  comp {j} (lambda1={c['lam1']:.0f}), t={t}:  "
              f"linear={lin:.3f}  geometric={geo:.3f}  relative gap={100*(lin-geo)/geo:.1f}%")


def _heatmap_panel(ax, d, color):
    cmap = LinearSegmentedColormap.from_list("density", ["#ffffff", color])
    ax.contourf(xx, yy, d, levels=25, cmap=cmap, zorder=1)
    ax.contour(xx, yy, d, levels=6, colors=color, linewidths=0.5, alpha=0.4, zorder=2)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# Figure: linear (top) vs geometric (middle) vs difference (bottom), t=.25/.5/.75
# No in-figure caption/title -- sized to \textwidth (418pt) for the thesis.
# ══════════════════════════════════════════════════════════════════════════════
TEXTWIDTH_IN = 418 / 72.27

diffs = {t: densities["linear"][t] - densities["geometric"][t] for t in T_VALS}
vmax = max(np.abs(d).max() for d in diffs.values())
diff_cmap = LinearSegmentedColormap.from_list(
    "rdbu5", ["#2166AC", "#4393C3", "#F7F7F7", "#D6604D", "#B2182B"])

fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85))

for ax, t in zip(axes[0], T_VALS):
    _heatmap_panel(ax, densities["linear"][t], LINEAR_COLOR)
    ax.set_title(rf"$t={t}$", fontsize=11)

for ax, t in zip(axes[1], T_VALS):
    _heatmap_panel(ax, densities["geometric"][t], GEOMETRIC_COLOR)

diff_im = None
for ax, t in zip(axes[2], T_VALS):
    diff_im = ax.contourf(xx, yy, diffs[t], levels=21, cmap=diff_cmap,
                           vmin=-vmax, vmax=vmax, zorder=1)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

axes[0, 0].set_ylabel("linear\n" + r"$A_t = tA + (1-t)I$", fontsize=9)
axes[1, 0].set_ylabel("geometric\n" + r"$A_t = \exp(t\log A)$", fontsize=9)
axes[2, 0].set_ylabel(r"$q_t^{\mathrm{lin}} - q_t^{\mathrm{geo}}$", fontsize=9)

cbar = fig.colorbar(diff_im, ax=axes[2], shrink=0.85, pad=0.02, orientation="horizontal",
                     location="bottom")
cbar.ax.tick_params(labelsize=7)

for ext in ("png", "pdf"):
    out_path = os.path.join(OUTDIR, f"linear_vs_geometric_extreme.{ext}")
    fig.savefig(out_path, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ── Supplementary: the t=1 target shape alone, to check the banana bend ────────
fig2, ax2 = plt.subplots(figsize=(TEXTWIDTH_IN * 0.45, TEXTWIDTH_IN * 0.30))
_heatmap_panel(ax2, target_density, PALETTE["target"])
for ext in ("png", "pdf"):
    out_path = os.path.join(OUTDIR, f"linear_vs_geometric_extreme_target.{ext}")
    fig2.savefig(out_path, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {out_path}")
