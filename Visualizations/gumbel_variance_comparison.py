"""
gumbel_variance_comparison.py

Variance (and bias) comparison of three gradient estimators for a
discrete categorical distribution, as a function of the Gumbel-Softmax
temperature tau.

Problem setup
-------------
Z ~ Categorical(p),  p = [0.5, 0.3, 0.2],  K = 3
f(Z) = [1, 4, 9]  (deterministic reward per category)
We estimate the gradient w.r.t. alpha_0,  where p = softmax(alpha).

True gradient:
    nabla_{alpha_0} E[f(Z)] = p_0 * (f_0 - E[f]) = 0.5 * (1 - 3.5) = -1.25

Three estimators
----------------
1. Score function (REINFORCE, no CV):
       g_score = f(Z) * (1_{Z=0} - p_0)       [unbiased]
2. Score function with control variate:
       g_cv    = g_score + c* (f(Z) - E[f(Z)]) [unbiased, reduced variance]
       c* = -Cov(g_score, f(Z)-E[f]) / Var[f(Z)-E[f]]
3. Gumbel-Softmax pathwise gradient (temperature tau):
       Z_tau = softmax((log p + G) / tau),  G_k ~ Gumbel(0,1)
       g_GS  = d/d_{alpha_0} f_soft(Z_tau)
             = Z_tau_0 * (f_0 - f_soft(Z_tau)) / tau    [BIASED for tau>0]
   Bias -> 0 as tau -> 0;  Variance -> inf as tau -> 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LATEX FIGURE CAPTION — copy into thesis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Variance (top) and absolute bias (bottom) of three gradient estimators for
$Z\sim\mathrm{Cat}(p)$, $p=(0.5,0.3,0.2)$, $f(Z)\in\{1,4,9\}$, plotted as
a function of the Gumbel-Softmax temperature $\tau$.
The score-function estimator $g^{\mathrm{score}}=f(Z)(1_{Z=0}-p_0)$ (red, dashed)
and its control-variate variant
$\hat{g}^{\mathrm{CV}}=g^{\mathrm{score}}+c^*(f(Z)-\mathbb{E}[f])$ (blue, dashed)
are both unbiased with fixed variances.
The Gumbel-Softmax pathwise estimator
$g^{\mathrm{GS}}_\tau = Z_{\tau,0}(f_0 - f_\mathrm{soft}(Z_\tau))/\tau$,
where $Z_\tau=\mathrm{softmax}((\log p + G)/\tau)$ with $G_k\overset{\mathrm{iid}}{\sim}\mathrm{Gumbel}(0,1)$,
trades variance for bias as $\tau$ increases:
at small $\tau$ it is nearly unbiased but has high variance;
at large $\tau$ the variance drops but bias grows.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ── Problem setup ──────────────────────────────────────────────────────────────
p      = np.array([0.5, 0.3, 0.2])
f_vals = np.array([1.0, 4.0, 9.0])
K      = len(p)
alpha  = np.log(p)          # softmax(alpha) = p

f_bar     = p @ f_vals      # E[f(Z)] = 3.5
true_grad = p[0] * (f_vals[0] - f_bar)   # = -1.25

n_reps = 500_000

# ── 1. Score function (REINFORCE) ──────────────────────────────────────────────
Z       = rng.choice(K, size=n_reps, p=p)
g_score = f_vals[Z] * (np.where(Z == 0, 1.0, 0.0) - p[0])

var_score = g_score.var()   # analytic: ~3.81

# ── 2. Score function + optimal CV ────────────────────────────────────────────
b_centered = f_vals[Z] - f_bar                       # b(Z) - E[b]
c_opt      = -np.cov(g_score, b_centered)[0, 1] / np.var(b_centered)
g_cv       = g_score + c_opt * b_centered

var_cv = g_cv.var()   # analytic: ~0.08

# ── 3. Gumbel-Softmax: sweep over tau ────────────────────────────────────────
taus      = np.logspace(-1.5, 1.0, 60)    # tau from ~0.03 to ~10
var_GS    = np.zeros(len(taus))
bias_GS   = np.zeros(len(taus))

# Pre-draw Gumbel noise (same noise for all tau for comparability)
G = rng.gumbel(0, 1, size=(n_reps, K))

for i, tau in enumerate(taus):
    logits = (alpha[None, :] + G) / tau
    logits -= logits.max(axis=1, keepdims=True)   # numerical stability
    z_tau  = np.exp(logits)
    z_tau /= z_tau.sum(axis=1, keepdims=True)

    f_soft = z_tau @ f_vals                        # soft function value
    g_GS   = z_tau[:, 0] * (f_vals[0] - f_soft) / tau   # pathwise gradient

    var_GS[i]  = g_GS.var()
    bias_GS[i] = abs(g_GS.mean() - true_grad)

# ── Colours ────────────────────────────────────────────────────────────────────
BG   = "#ffffff"
C_NO = "#B2182B"    # dark red
C_CV = "#2166AC"    # dark blue
C_GS = "#4DAF4A"    # green for GS

# ── Figure: two stacked panels ────────────────────────────────────────────────
fig, (ax_var, ax_bias) = plt.subplots(
    2, 1, figsize=(7, 6), facecolor=BG, sharex=True,
    gridspec_kw={"hspace": 0.1, "height_ratios": [1.4, 1]}
)

for ax in (ax_var, ax_bias):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.tick_params(labelsize=10)

# ── Top panel: Variance ────────────────────────────────────────────────────────
ax_var.axhline(var_score, color=C_NO, linewidth=2.0, linestyle="--",
               label=rf"$g^{{\mathrm{{score}}}}$"
                     rf"$\quad\mathbb{{V}} \approx {var_score:.2f}$")
ax_var.axhline(var_cv,    color=C_CV, linewidth=2.0, linestyle="--",
               label=rf"$\hat{{g}}^{{\mathrm{{CV}}}}$"
                     rf"$\quad\mathbb{{V}} \approx {var_cv:.2f}$")
ax_var.plot(taus, var_GS, color=C_GS, linewidth=2.4,
            label=r"$g^{\mathrm{GS}}_\tau$  (Gumbel-Softmax)")

ax_var.set_yscale("log")
ax_var.set_ylabel(r"Variance  $\mathbb{V}[\hat{g}]$", fontsize=11)
ax_var.legend(fontsize=10, framealpha=0.95, edgecolor="#cccccc",
              loc="upper right", handlelength=1.6)

# ── Bottom panel: |Bias| ───────────────────────────────────────────────────────
ax_bias.axhline(0.0, color=C_NO, linewidth=2.0, linestyle="--",
                label=r"$g^{\mathrm{score}}$ and $\hat{g}^{\mathrm{CV}}$: bias $= 0$")
ax_bias.plot(taus, bias_GS, color=C_GS, linewidth=2.4,
             label=r"$|$bias$|$ of $g^{\mathrm{GS}}_\tau$")
ax_bias.fill_between(taus, bias_GS, color=C_GS, alpha=0.15)

ax_bias.set_xscale("log")
ax_bias.set_xlabel(r"Temperature $\tau$", fontsize=11)
ax_bias.set_ylabel(r"$|\mathrm{bias}|$", fontsize=11)
ax_bias.legend(fontsize=10, framealpha=0.95, edgecolor="#cccccc",
               loc="upper left", handlelength=1.6)

# Shared x-axis label — add minor annotation
ax_var.set_title(
    r"Gumbel-Softmax vs.\ score-function gradient estimators:"
    "\n"
    r"$Z\sim\mathrm{Cat}(p)$, $p=(0.5,\,0.3,\,0.2)$, "
    r"$f(Z)\in\{1,4,9\}$,  true gradient $= -1.25$",
    fontsize=10.5, pad=8
)

plt.savefig("gumbel_variance_comparison.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"Saved: gumbel_variance_comparison.png")
print(f"var_score = {var_score:.4f}")
print(f"var_cv    = {var_cv:.4f}")
print(f"GS var at tau=0.1: {var_GS[np.argmin(np.abs(taus-0.1))]:.4f}")
print(f"GS var at tau=1.0: {var_GS[np.argmin(np.abs(taus-1.0))]:.4f}")
