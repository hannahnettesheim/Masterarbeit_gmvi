"""
gumbel_softmax_placement_2d_viz.py

Where do Gumbel-Softmax samples land as tau -> 0? (2D version)

Draws 100 relaxed-categorical samples z = sum_k w_k(tau) * x_k from a 2D,
3-component Gaussian mixture with well-separated means (distinct valleys of
low density between components), one rectangular panel per tau in
{1, 0.5, 0.1, 0}, laid out side by side. Background color encodes the
mixture density; sample colors blend the 3 component colors
(gmvi.utils.visualization.COLORS) according to their soft assignment
weights: muddy colors mean the sample landed in a valley between
components, pure colors mean it collapsed onto a single component.
tau = 0 draws hard (one-hot / argmax) samples.

Uses the existing color palette and plotting infrastructure in
gmvi/utils/visualization.py (plot_gumbel_softmax_placement_2d).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gmvi.utils.visualization import plot_gumbel_softmax_placement_2d

fig = plot_gumbel_softmax_placement_2d(
    temperatures=(1.0, 0.5, 0.1, 0.0),
    n_samples=100,
    seed=0,
)

out_path = os.path.join(os.path.dirname(__file__), "gumbel_softmax_placement_2d.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
