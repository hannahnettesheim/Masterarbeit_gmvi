"""
theory_visualization.py

Shows the intermediate distributions ρ_t interpolating between a standard-normal
reference (t=0) and a hand-crafted Gaussian mixture target (t=1), under the
linear ODE path:

    A_{j,t} = t A_j + (1-t) I

The intermediate density is computed analytically:

    log ρ_t(x) = logsumexp_j [ log w_j
                                + log ρ_ref( A_{j,t}^{-1}(x - t a_j) )
                                - log|det A_{j,t}| ]

No training. No gmvi internals.  Only numpy / scipy / matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# ── Hand-crafted target mixture ────────────────────────────────────────────────
# T_j(x) = a_j + A_j x,   x ~ N(0, I)
# q(z) = Σ_j w_j · N(z ; a_j, A_j Aⱼᵀ)

K, D = 3, 2

log_weights = np.log([0.40, 0.35, 0.25])   # will be softmax-normalised below

means = np.array([
    [ 3.0,  1.0],
    [-2.2,  2.6],
    [ 0.4, -2.8],
], dtype=float)

# Invertible A_j matrices — give each component a different shape / orientation
A_mats = np.array([
    [[ 1.2,  0.4],          # Component 1: elongated, tilted right
     [ 0.0,  0.6]],

    [[ 0.9, -0.3],          # Component 2: compressed, rotated ~20°
     [ 0.3,  0.7]],

    [[ 0.8,  0.2],          # Component 3: slightly sheared
     [-0.2,  1.1]],
], dtype=float)


# ── Density helpers ────────────────────────────────────────────────────────────

def log_ref(z: np.ndarray) -> np.ndarray:
    """N(0, I) log-density.  z: (N, D) → (N,)"""
    return -0.5 * (np.sum(z ** 2, axis=-1) + D * np.log(2 * np.pi))


def log_rho_t(x: np.ndarray, t: float) -> np.ndarray:
    """
    Log density of the linear-path intermediate distribution ρ_t.

    x: (N, D) grid points
    t: scalar in [0, 1]
    returns: (N,) log densities
    """
    I  = np.eye(D)
    N  = x.shape[0]
    lw = log_weights - logsumexp(log_weights)   # normalise log-weights
    log_comp = np.zeros((N, K))

    for j in range(K):
        A_jt     = t * A_mats[j] + (1.0 - t) * I      # (D, D)
        A_jt_inv = np.linalg.inv(A_jt)
        _, logdet = np.linalg.slogdet(A_jt)

        z = (x - t * means[j]) @ A_jt_inv.T            # (N, D) pre-images
        log_comp[:, j] = log_ref(z) - logdet

    return logsumexp(lw[None, :] + log_comp, axis=1)   # (N,)


# ── Evaluation grid ────────────────────────────────────────────────────────────
XLIM, YLIM = (-5.5, 6.0), (-5.5, 5.5)
RES = 350

xs = np.linspace(*XLIM, RES)
ys = np.linspace(*YLIM, RES)
xx, yy = np.meshgrid(xs, ys)
grid   = np.stack([xx.ravel(), yy.ravel()], axis=1)   # (RES², 2)

# ── t-values, colours, labels ─────────────────────────────────────────────────
T_VALS  = [0.0, 0.25, 0.5, 0.75, 1.0]

# ColorBrewer RdBu-5: dark blue → mid blue → gray → mid red → dark red
COLORS  = ["#2166AC", "#4393C3", "#878787", "#D6604D", "#B2182B"]
LABELS  = {
    0.00: r"$t=0$  (reference  $\rho_0 = \mathcal{N}(0,I)$)",
    0.25: r"$t=0.25$",
    0.50: r"$t=0.50$",
    0.75: r"$t=0.75$",
    1.00: r"$t=1$  (target  $\rho_1 = q$)",
}

# Level-set thresholds as fraction of the peak density
LEVELS_FRAC = [0.05, 0.20, 0.45, 0.75, 0.95]

# Pre-compute densities (normalised so peak = 1 for fair contour comparison)
densities = {}
for t in T_VALS:
    log_d = log_rho_t(grid, t).reshape(xx.shape)
    log_d -= log_d.max()
    densities[t] = np.exp(log_d)


# ══════════════════════════════════════════════════════════════════════════════
# Five individual images — one per t
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
    ax.set_facecolor(BG)

    d = densities[t]
    levels = [f * d.max() for f in LEVELS_FRAC]

    ax.contourf(xx, yy, d, levels=20, cmap="Greys", alpha=0.12)
    ax.contour( xx, yy, d, levels=levels, colors=color, linewidths=1.8, alpha=0.95)

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_title(TITLES[t], color=color, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb")

plt.savefig("rho_t_all.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("Saved: rho_t_all.png")
