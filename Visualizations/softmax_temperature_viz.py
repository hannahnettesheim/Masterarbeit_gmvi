"""
softmax_temperature_viz.py

Shows how softmax(z / τ) approaches argmax as τ → 0.
Logits z = [3.0, 1.5, 0.5, -0.5] over four classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Setup ───────────────────────────────────────────────────────────────────────
logits      = np.array([3.0, 1.5, 0.5, -0.5])
n_classes   = len(logits)
class_labels = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$"]
temps       = [5.0, 2.0, 1.0, 0.5, 0.1]
argmax_idx  = int(np.argmax(logits))

# ── Style ───────────────────────────────────────────────────────────────────────
BG    = "#ffffff"
C_ARG = "#B2182B"   # red  — argmax class
C_BAR = "#2166AC"   # blue — other classes
GRAY  = "#cccccc"

def softmax(z, tau):
    e = np.exp((z - z.max()) / tau)
    return e / e.sum()

# ── Figure ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(temps), figsize=(10, 3.8), facecolor=BG)
fig.subplots_adjust(wspace=0.35)

x = np.arange(n_classes)

for i, (ax, tau) in enumerate(zip(axes, temps)):
    probs = softmax(logits, tau)
    ax.set_facecolor(BG)

    # bars
    colors = [C_ARG if j == argmax_idx else C_BAR for j in range(n_classes)]
    alphas = [0.85 if j == argmax_idx else 0.65 for j in range(n_classes)]
    for j in range(n_classes):
        ax.bar(x[j], probs[j], color=colors[j], alpha=alphas[j],
               width=0.6, zorder=2)

    # horizontal guides
    for y in [0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y, color=GRAY, linewidth=0.6, linestyle="--", zorder=1)

    ax.set_xlim(-0.6, n_classes - 0.4)
    ax.set_ylim(0, 1.09)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10.5)

    if i == 0:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "", "0.5", "", "1"], fontsize=9)
        ax.set_ylabel("Probability", fontsize=10.5)
    else:
        ax.set_yticks([])

    ax.set_title(rf"$\tau = {tau}$", fontsize=11.5, pad=7)

    # annotate argmax probability
    p_max = probs[argmax_idx]
    label = rf"$p_1={p_max:.3f}$" if p_max < 0.999 else r"$p_1\approx 1$"
    ax.text(0.5, 0.97, label, ha="center", va="top",
            transform=ax.transAxes, fontsize=8.5, color="#333333")

    for sp in ax.spines.values():
        sp.set_edgecolor(GRAY)
    ax.tick_params(length=3, labelsize=9)


# ── Main title ───────────────────────────────────────────────────────────────────

plt.tight_layout()
plt.savefig("softmax_temperature.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.show()
print("Saved: softmax_temperature.png")