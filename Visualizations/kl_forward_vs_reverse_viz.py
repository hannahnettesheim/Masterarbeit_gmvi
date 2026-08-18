"""
kl_forward_vs_reverse_viz.py

Forward vs. reverse KL when fitting a single Gaussian q to a bimodal target p.

Setup
-----
p(x) = 0.5 * N(x; -2, 0.6^2) + 0.5 * N(x; 2, 0.6^2)

Forward KL   min_q KL(p || q):  closed form (moment matching) —
             mu_q = E_p[x],  sigma_q^2 = Var_p[x]
Reverse KL   min_q KL(q || p):  no closed form — numerical optimisation
             (Nelder-Mead on a discretised integral), initialised at the
             right-hand mode so it locks onto that mode (mode-seeking).

Both integrals are evaluated on the same fine grid via the trapezoidal rule.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LATEX FIGURE CAPTION — copy into thesis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fitting a single Gaussian $q$ to a bimodal target
$p(x) = \\tfrac{1}{2}\\mathcal{N}(x;-2,0.6^2) + \\tfrac{1}{2}\\mathcal{N}(x;2,0.6^2)$
under the two directions of the KL divergence. Minimising the forward KL,
$\\arg\\min_q \\mathrm{KL}(p\\,\\Vert\\,q)$, admits a closed form (moment matching)
and is \\emph{mode-covering}: since the expectation is taken under $p$, any
region with $p(x)>0$ must also have $q(x)>0$, so $q$ widens
($\\mu=0,\\ \\sigma=2.09$) to straddle both modes at the cost of placing most of
its mass in the low-density valley between them. Minimising the reverse KL,
$\\arg\\min_q \\mathrm{KL}(q\\,\\Vert\\,p)$ (solved numerically here, since no
closed form exists), is \\emph{mode-seeking}: the expectation is taken under
$q$ itself, so $q$ is never penalised for ignoring a mode entirely and instead
collapses onto one of them ($\\mu\\approx 2,\\ \\sigma\\approx 0.60$) at the
optimal local fit. Initialising the reverse-KL optimisation near $x=-2$
converges to the mirror-image solution at the other mode.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ── Target: bimodal Gaussian mixture ───────────────────────────────────────────
w = 0.5
mu1, s1 = -2.0, 0.6
mu2, s2 = 2.0, 0.6


def p_pdf(x):
    return w * norm.pdf(x, mu1, s1) + (1 - w) * norm.pdf(x, mu2, s2)


x_grid = np.linspace(-8, 8, 4001)
p_vals = p_pdf(x_grid)

# ── Forward KL:  min_q KL(p || q)  — closed form (moment matching) ────────────
mean_p = w * mu1 + (1 - w) * mu2
Ex2 = w * (s1 ** 2 + mu1 ** 2) + (1 - w) * (s2 ** 2 + mu2 ** 2)
var_p = Ex2 - mean_p ** 2
mu_fwd, sigma_fwd = mean_p, np.sqrt(var_p)


def kl_p_q(mu, sigma):
    q_vals = np.clip(norm.pdf(x_grid, mu, sigma), 1e-300, None)
    integrand = p_vals * (np.log(np.clip(p_vals, 1e-300, None)) - np.log(q_vals))
    return np.trapezoid(integrand, x_grid)


fwd_kl_value = kl_p_q(mu_fwd, sigma_fwd)

# ── Reverse KL:  min_q KL(q || p)  — numerical, mode-seeking ──────────────────
def kl_q_p(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q_vals = norm.pdf(x_grid, mu, sigma)
    q_safe = np.clip(q_vals, 1e-300, None)
    p_safe = np.clip(p_vals, 1e-300, None)
    integrand = q_vals * (np.log(q_safe) - np.log(p_safe))
    return np.trapezoid(integrand, x_grid)


res_rev = minimize(kl_q_p, x0=[2.0, np.log(0.6)], method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
mu_rev, sigma_rev = res_rev.x[0], np.exp(res_rev.x[1])
rev_kl_value = res_rev.fun

print(f"Forward KL fit (mode-covering): mu={mu_fwd:.3f} sigma={sigma_fwd:.3f}  KL(p||q)={fwd_kl_value:.4f}")
print(f"Reverse KL fit (mode-seeking):  mu={mu_rev:.3f} sigma={sigma_rev:.3f}  KL(q||p)={rev_kl_value:.4f}")

# ── Colours — same palette as gmvi/utils/visualization.py ─────────────────────
COLORS = {
    "score_function": "#E63946",   # red
    "gumbel_softmax": "#457B9D",   # blue
    "ode_transport":  "#2A9D8F",   # teal
    "target":         "#264653",
    "samples":        "#A8DADC",
}
C_TARGET = COLORS["target"]
C_FWD = COLORS["score_function"]
C_REV = COLORS["ode_transport"]
BG = "#ffffff"

# ── Figure ──────────────────────────────────────────────────────────────────────
x_plot = np.linspace(-6, 6, 601)
p_plot = p_pdf(x_plot)
q_fwd_plot = norm.pdf(x_plot, mu_fwd, sigma_fwd)
q_rev_plot = norm.pdf(x_plot, mu_rev, sigma_rev)

fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=BG)
ax.set_facecolor(BG)

ax.fill_between(x_plot, p_plot, color=C_TARGET, alpha=0.12, zorder=1)
ax.plot(x_plot, p_plot, color=C_TARGET, linewidth=2.4, zorder=3,
        label=r"Target $p(x)$")

ax.plot(x_plot, q_fwd_plot, color=C_FWD, linewidth=2.2, zorder=3,
        label=(r"$\arg\min_q\,\mathrm{KL}(p\,\Vert\,q)$"
               "\n"
               rf"mode-covering, $\mu\!=\!{mu_fwd:.2f},\ \sigma\!=\!{sigma_fwd:.2f}$"))

ax.plot(x_plot, q_rev_plot, color=C_REV, linewidth=2.2, zorder=3,
        label=(r"$\arg\min_q\,\mathrm{KL}(q\,\Vert\,p)$"
               "\n"
               rf"mode-seeking, $\mu\!=\!{mu_rev:.2f},\ \sigma\!=\!{sigma_rev:.2f}$"))

ax.set_xlim(x_plot[0], x_plot[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$x$", fontsize=11)
ax.set_ylabel("Density", fontsize=11)

ax.legend(fontsize=9, framealpha=0.95, edgecolor="#cccccc",
          loc="upper left", handlelength=1.5, labelspacing=0.9)

for sp in ax.spines.values():
    sp.set_edgecolor("#cccccc")
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("kl_forward_vs_reverse.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.savefig("kl_forward_vs_reverse.pdf", bbox_inches="tight", facecolor=BG)
print("Saved: kl_forward_vs_reverse.png, kl_forward_vs_reverse.pdf")
