"""
control_variate_viz.py

Score-function (REINFORCE) gradient estimator: variance reduction via control variate.

Setup
-----
X ~ N(mu, 1),   f(X) = X^2,   mu = 2
g_score  = f(X) * (X - mu)                   score-function gradient, single sample
Baseline: b(X) = X^2,  E[b] = mu^2 + 1 = 5  (known analytically)
c_opt    = -Cov(g, b - E[b]) / Var(b - E[b]) = -2   (analytic)
g_hat    = g_score + c_opt * (b(X) - E[b(X)])

Each plotted estimate is the average over N = 10 i.i.d. samples.
By CLT both distributions are approximately Gaussian, centred at the true gradient.

Analytic single-sample variances:  Var[g_score] = 87,  Var[g_hat] = 15  (83% reduction)
Batch (N=10) variances:            87/10 = 8.7,         15/10 = 1.5

True gradient:  nabla_mu E[X^2] = 2*mu = 4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LATEX FIGURE CAPTION — copy into thesis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Distribution of Monte Carlo gradient estimates
$\hat g_\theta^{\mathrm{score}} = \frac{1}{N}\sum_{i=1}^N f(X_i)\,\nabla_\theta\log p_\theta(X_i)$
with (blue) and without (red) a control variate, for $X_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(\mu,1)$,
$\mu=2$, $f(X)=X^2$, and $N=10$ samples per estimate.
The baseline $b(X)=X^2$ has known expectation $\mathbb{E}[b]=\mu^2+1$;
the optimal scalar $c^*=-\mathrm{Cov}(g^{\mathrm{score}},\,b-\mathbb{E}[b])\,/\,\mathbb{V}[b-\mathbb{E}[b]]=-2$
minimises the variance of the corrected estimator
$\hat g^{\mathrm{CV}}=\hat g^{\mathrm{score}}+c^*(b(X)-\mathbb{E}[b(X)])$.
Both estimators are unbiased — centred at the true gradient
$\nabla_\mu\,\mathbb{E}[X^2]=2\mu=4$ (dashed) — but the control variate
reduces the variance by $83\%$
(from $\mathbb{V}[\hat g^{\mathrm{score}}]/N\approx 8.7$ to $\mathbb{V}[\hat g^{\mathrm{CV}}]/N\approx 1.5$).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

rng = np.random.default_rng(0)

# ── Parameters ─────────────────────────────────────────────────────────────────
mu, sigma = 2.0, 1.0
N_batch   = 10      # samples per gradient estimate
n_reps    = 300_000  # number of independent gradient estimates

# ── Simulate batch gradient estimates ──────────────────────────────────────────
X          = rng.normal(mu, sigma, (n_reps, N_batch))   # (n_reps, N)
g_single   = X**2 * (X - mu)                            # score-fn gradient
b_centered = X**2 - (mu**2 + sigma**2)                  # b(X) - E[b],  analytic E[b] = mu^2+1

c_opt = -2.0   # analytic optimal scalar

g_batch    = g_single.mean(axis=1)                        # average over N samples
g_cv_batch = (g_single + c_opt * b_centered).mean(axis=1)

true_grad = 2.0 * mu   # = 4.0
var_no    = g_batch.var()
var_cv    = g_cv_batch.var()
reduction = 100.0 * (1.0 - var_cv / var_no)

# ── KDE for smooth curves ──────────────────────────────────────────────────────
std_no = g_batch.std()
xs     = np.linspace(true_grad - 5 * std_no, true_grad + 5 * std_no, 1000)

kde_no = gaussian_kde(g_batch,    bw_method="scott")
kde_cv = gaussian_kde(g_cv_batch, bw_method="scott")

# ── Colours  (same palette as theory_visualization.py) ────────────────────────
BG   = "#ffffff"
C_NO = "#B2182B"   # dark red  — high variance
C_CV = "#2166AC"   # dark blue — low variance

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
ax.set_facecolor(BG)

# Filled areas (subtle)
ax.fill_between(xs, kde_no(xs), color=C_NO, alpha=0.13, zorder=2)
ax.fill_between(xs, kde_cv(xs), color=C_CV, alpha=0.22, zorder=2)

# KDE curves
ax.plot(xs, kde_no(xs), color=C_NO, linewidth=2.4, zorder=3,
        label=(
            r"$g_\theta^{\mathrm{score}}$  (no CV)"
            "\n"
            rf"$\mathbb{{V}}[g] / N \approx {var_no:.1f}$"
        ))
ax.plot(xs, kde_cv(xs), color=C_CV, linewidth=2.4, zorder=3,
        label=(
            r"$\hat{g}_\theta^{\mathrm{score}}$  (with CV, $c^*\!=\!-2$)"
            "\n"
            rf"$\mathbb{{V}}[\hat{{g}}] / N \approx {var_cv:.1f}$"
            rf"$\;\;({reduction:.0f}\%$ red.$)$"
        ))

# True gradient: dashed vertical line + label above
ymax_cv = float(kde_cv(np.array([true_grad]))[0])
ax.axvline(true_grad, color="#555555", linewidth=1.5, linestyle="--", zorder=4,
           label=rf"True gradient $= 2\mu = {true_grad:.0f}$")

# Axes labels
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel(r"Gradient estimate  ($N = 10$ samples per estimate)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)

ax.legend(fontsize=9.5, framealpha=0.95, edgecolor="#cccccc",
          loc="upper left", handlelength=1.5, labelspacing=0.6)

for sp in ax.spines.values():
    sp.set_edgecolor("#cccccc")
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("control_variate_variance.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.show()
print("Saved: control_variate_variance.png")
