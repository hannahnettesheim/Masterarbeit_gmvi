"""
gumbel_softmax_placement_viz.py

Where do Gumbel-Softmax samples land as tau -> 0?

Draws 100 relaxed-categorical samples z = sum_k w_k(tau) * x_k from a 1D,
3-component Gaussian mixture with well-separated means (distinct valleys of
low density between components), for tau in {1, 0.5, 0.1, 0}. Samples are
colored by blending the 3 component colors (gmvi.utils.visualization.COLORS)
according to their soft assignment weights: muddy colors mean the sample
landed in a valley between components; pure colors mean it collapsed onto a
single component. tau = 0 draws hard (one-hot / argmax) samples.

Uses the existing color palette and plotting infrastructure in
gmvi/utils/visualization.py (plot_gumbel_softmax_placement).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gmvi.utils.visualization import plot_gumbel_softmax_placement

fig = plot_gumbel_softmax_placement(
    means=(-4.0, 0.0, 4.0),
    scale=0.6,
    temperatures=(1.0, 0.5, 0.1, 0.0),
    n_samples=100,
    seed=0,
)

fig.savefig("gumbel_softmax_placement.png", dpi=180, bbox_inches="tight")
print("Saved: gumbel_softmax_placement.png")
