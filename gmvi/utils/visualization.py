"""
Visualization utilities for GMM-VI experiments.

Functions:
  - plot_2d_approximation:   contour of target + GMM samples side-by-side
  - plot_training_curves:    ELBO over training steps, all estimators
  - plot_comparison_summary: bar chart of final ELBOs across runs
  - plot_component_evolution: component means/weights over training (2D only)
  - make_training_gif:       animated GIF of density evolution during training
"""

import copy
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from typing import Dict, Optional
from typing import Dict, List, Optional, Tuple

from gmvi.models.generalized_mixture import GaussianMixture
from gmvi.targets.distributions import Target, GaussianMixtureTarget
from gmvi.experiments.trainer import TrainResult


# ─── Color palette ─────────────────────────────────────────────────────────────

COLORS = {
    "score_function":   "#E63946",   # red
    "gumbel_softmax":   "#457B9D",   # blue
    "ode_transport":    "#2A9D8F",   # teal (new method)
    "target":           "#264653",
    "samples":          "#A8DADC",
}

LABELS = {
    "score_function":   "Score Function (REINFORCE)",
    "gumbel_softmax":   "Gumbel-Softmax",
    "ode_transport": "ODE Transport",
}


# ─── 2D density helpers ─────────────────────────────────────────────────────────

def _make_grid(
    xlim: Tuple[float, float] = (-5, 5),
    ylim: Tuple[float, float] = (-5, 5),
    resolution: int = 200,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    xx, yy = np.meshgrid(
        np.linspace(*xlim, resolution),
        np.linspace(*ylim, resolution),
    )
    grid = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
    )
    return xx, yy, grid


def plot_2d_approximation(
    model: GaussianMixture,
    target: Target,
    n_samples: int = 2000,
    xlim: Tuple = (-5, 5),
    ylim: Tuple = (-5, 5),
    resolution: int = 200,
    title: str = "",
    ax_target=None,
    ax_approx=None,
    show_components: bool = True,
) -> plt.Figure:
    """
    Side-by-side: target density (left) vs GMM approximation (right).
    """
    assert model.D == 2, "plot_2d_approximation requires 2D model"

    standalone = ax_target is None
    if standalone:
        fig, (ax_target, ax_approx) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax_target.figure

    xx, yy, grid = _make_grid(xlim, ylim, resolution)

    with torch.no_grad():
        # Target density
        log_p = target.log_prob(grid).numpy().reshape(xx.shape)
        log_p -= log_p.max()
        p = np.exp(log_p)

        # GMM density
        log_q = model.log_prob(grid).numpy().reshape(xx.shape)
        log_q -= log_q.max()
        q = np.exp(log_q)

        # Samples from q
        z_samples, _ = model.sample(n_samples)
        z_samples = z_samples.numpy()

    ax_target.contourf(xx, yy, p, levels=30, cmap="Blues")
    ax_target.set_title("Target p(z)")
    ax_target.set_xlim(xlim)
    ax_target.set_ylim(ylim)
    ax_target.set_aspect("equal")

    ax_approx.contourf(xx, yy, q, levels=30, cmap="Reds", alpha=0.7)
    ax_approx.scatter(z_samples[:500, 0], z_samples[:500, 1],
                      s=2, alpha=0.3, color=COLORS["samples"], zorder=3)

    if show_components:
        with torch.no_grad():
            weights = model.weights.numpy()
            for k in range(model.K):
                w = weights[k]
                if w < 0.01:
                    continue
                comp = model.components[k]
                center = comp.a.numpy()
                A = comp.get_A().numpy()  # (D, D)

                ax_approx.scatter(*center, s=80 * w * model.K, marker="x",
                                   color="white", zorder=5, linewidths=2)

                # Derive ellipse from A via SVD: A = U S V^T
                # The 2-sigma ellipse axes are 2*singular values, rotated by U
                U, S, _ = np.linalg.svd(A)
                angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                ell = Ellipse(
                    xy=center,
                    width=4 * S[0],
                    height=4 * S[1],
                    angle=angle,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=max(0.5, w * 3),
                    alpha=0.6,
                )
                ax_approx.add_patch(ell)

    ax_approx.set_title("GMM Approximation q(z)")
    ax_approx.set_xlim(xlim)
    ax_approx.set_ylim(ylim)
    ax_approx.set_aspect("equal")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    if standalone:
        plt.tight_layout()
    return fig


def plot_training_curves(
    results: Dict[str, List[TrainResult]],
    smooth_window: int = 20,
    figsize: Tuple = (10, 4),
    show_std: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot ELBO training curves for all estimators across runs.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax_elbo, ax_time = axes

    for est_name, run_results in results.items():
        color = COLORS.get(est_name, "gray")
        label = LABELS.get(est_name, est_name)
        n_steps = len(run_results[0].elbo_history)
        steps = np.arange(n_steps)

        # Stack runs
        matrix = np.array([r.elbo_history for r in run_results])  # (runs, steps)

        # Smooth
        def smooth(x, w):
            return np.convolve(x, np.ones(w) / w, mode="valid")

        smoothed = np.array([smooth(row, smooth_window) for row in matrix])
        s_steps = steps[smooth_window - 1:]

        mean = smoothed.mean(0)
        std = smoothed.std(0)

        ax_elbo.plot(s_steps, mean, color=color, label=label, linewidth=2)
        if show_std and len(run_results) > 1:
            ax_elbo.fill_between(s_steps, mean - std, mean + std,
                                  color=color, alpha=0.15)

    if log_scale:
        ax_elbo.set_yscale("symlog", linthresh=1.0)

    ax_elbo.set_xlabel("Training Step")
    ax_elbo.set_ylabel("ELBO")
    ax_elbo.set_title("ELBO During Training")
    ax_elbo.legend(fontsize=9)
    ax_elbo.grid(True, alpha=0.3)

    # Per-step time comparison (box plots)
    times_data = []
    time_labels = []
    time_colors = []
    for est_name, run_results in results.items():
        step_times = np.concatenate([r.step_times for r in run_results]) * 1000  # ms
        times_data.append(step_times)
        time_labels.append(LABELS.get(est_name, est_name).split("(")[0].strip())
        time_colors.append(COLORS.get(est_name, "gray"))

    bp = ax_time.boxplot(times_data, labels=time_labels, patch_artist=True,
                          showfliers=False, medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], time_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax_time.set_ylabel("Time per step (ms)")
    ax_time.set_title("Computational Cost")
    ax_time.grid(True, alpha=0.3, axis="y")
    plt.setp(ax_time.get_xticklabels(), rotation=15, ha="right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_comparison_summary(
    summary: Dict,
    figsize: Tuple = (8, 4),
) -> plt.Figure:
    """
    Bar chart: final ELBO per estimator with error bars (across runs).
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(summary.keys())
    means = [summary[n]["mean_elbo"] for n in names]
    stds = [summary[n]["std_elbo"] for n in names]
    colors = [COLORS.get(n, "gray") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    bars = ax.bar(labels, means, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=1.5)
    ax.errorbar(labels, means, yerr=stds, fmt="none",
                 color="black", capsize=5, linewidth=2)

    # Annotate values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(std) + 0.05 * abs(min(means)),
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_ylabel("Final ELBO (mean ± std)")
    ax.set_title("Estimator Comparison: Final ELBO")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_all_2d(
    models: Dict[str, GaussianMixture],
    target: Target,
    xlim: Tuple = (-5, 5),
    ylim: Tuple = (-5, 5),
    resolution: int = 150,
    n_samples: int = 1000,
) -> plt.Figure:
    """
    3-column figure: each estimator's final approximation vs target.
    """
    n_est = len(models)
    fig = plt.figure(figsize=(5 * (n_est + 1), 4))
    gs = gridspec.GridSpec(1, n_est + 1, figure=fig)

    # Target in first column
    xx, yy, grid = _make_grid(xlim, ylim, resolution)
    with torch.no_grad():
        log_p = target.log_prob(grid).numpy().reshape(xx.shape)

    ax0 = fig.add_subplot(gs[0])
    ax0.contourf(xx, yy, np.exp(log_p - log_p.max()), levels=30, cmap="Blues")
    ax0.set_title("Target p(z)", fontweight="bold")
    ax0.set_aspect("equal")
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)

    for i, (est_name, model) in enumerate(models.items()):
        ax = fig.add_subplot(gs[i + 1])
        with torch.no_grad():
            log_q = model.log_prob(grid).numpy().reshape(xx.shape)
            z_samp, _ = model.sample(n_samples)
            z_samp = z_samp.numpy()
            weights = model.weights.numpy()

        color = COLORS.get(est_name, "gray")
        ax.contourf(xx, yy, np.exp(log_q - log_q.max()), levels=30,
                     cmap="Reds", alpha=0.7)
        ax.scatter(z_samp[:300, 0], z_samp[:300, 1], s=3, alpha=0.4,
                    color=color, zorder=3)

        for k in range(model.K):
            if weights[k] < 0.01:
                continue
            comp = model.components[k]
            center = comp.a.detach().numpy()
            A = comp.get_A().detach().numpy()
            U, S, _ = np.linalg.svd(A)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            ax.scatter(*center, s=60, marker="x", color="white", zorder=5)
            ell = Ellipse(xy=center, width=4 * S[0], height=4 * S[1],
                           angle=angle, edgecolor="white", facecolor="none",
                           linewidth=1, alpha=0.7)
            ax.add_patch(ell)

        ax.set_title(LABELS.get(est_name, est_name), fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.suptitle(f"Target: {target.name}", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_gumbel_softmax_placement(
    means: Tuple[float, float, float] = (-4.0, 0.0, 4.0),
    scale: float = 0.6,
    temperatures: Tuple[float, ...] = (1.0, 0.5, 0.1, 0.0),
    n_samples: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
    seed: int = 0,
    figsize: Tuple = (9, 7.5),
) -> plt.Figure:
    """
    Where do Gumbel-Softmax samples land as τ → 0?

    Draws `n_samples` relaxed-categorical samples z = Σ_k w̃_k(τ) x_k from a
    1-D, K=3 Gaussian mixture with well-separated means (i.e. clear valleys
    of low density between components), one row per τ in `temperatures`.
    Each point is colored by blending the K component colors according to
    its soft assignment weights w̃(τ): muddy/blended colors mean the sample
    landed between components (in a valley); pure colors mean it collapsed
    onto one component. τ = 0 draws hard (one-hot / argmax) samples.
    """
    K = 3
    assert len(means) == K, "uses the 3 method colors as the 3 component colors"
    torch.manual_seed(seed)

    means_t  = torch.tensor(means, dtype=torch.float32)
    scales_t = torch.full((K,), scale)
    log_weights = torch.zeros(K)  # uniform mixture weights

    target = GaussianMixtureTarget(means=means_t.unsqueeze(-1), scales=scales_t.unsqueeze(-1))

    if xlim is None:
        pad = 3 * scale + 0.5
        xlim = (min(means) - pad, max(means) + pad)

    xs = torch.linspace(*xlim, 400).unsqueeze(-1)
    with torch.no_grad():
        density = target.log_prob(xs).exp().numpy()
    xs_np = xs.squeeze(-1).numpy()

    comp_colors = [COLORS["score_function"], COLORS["gumbel_softmax"], COLORS["ode_transport"]]
    comp_rgb = np.array([to_rgb(c) for c in comp_colors])  # (K, 3)

    fig, axes = plt.subplots(len(temperatures), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, tau in zip(axes, temperatures):
        ax.fill_between(xs_np, density, color=COLORS["target"], alpha=0.12, zorder=1)
        ax.plot(xs_np, density, color=COLORS["target"], linewidth=1.2, alpha=0.55, zorder=1)
        ax.axhline(0, color=COLORS["target"], linewidth=0.8, alpha=0.3, zorder=1)

        gumbel = -torch.log(-torch.log(torch.rand(n_samples, K)))
        if tau > 0:
            soft_w = torch.softmax((log_weights + gumbel) / tau, dim=-1)  # (N, K)
        else:
            k_hard = (log_weights + gumbel).argmax(dim=-1)
            soft_w = torch.eye(K)[k_hard]

        comp_draws = means_t[None, :] + scales_t[None, :] * torch.randn(n_samples, K)
        z = (soft_w * comp_draws).sum(-1).numpy()
        blended = soft_w.numpy() @ comp_rgb

        y = np.random.uniform(-0.22, -0.05, size=n_samples) * density.max()
        ax.scatter(z, y, c=blended, s=26, alpha=0.9,
                    edgecolor="white", linewidth=0.4, zorder=3)

        for m, c in zip(means, comp_colors):
            ax.axvline(m, color=c, linestyle="--", linewidth=1, alpha=0.45, zorder=2)

        label = r"$\tau \to 0$ (hard)" if tau == 0 else rf"$\tau = {tau}$"
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11, fontweight="bold")
        ax.set_yticks([])
        ax.set_ylim(-0.30 * density.max(), density.max() * 1.15)
        ax.set_xlim(xlim)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("z")
    fig.suptitle(
        f"Gumbel-Softmax sample placement vs. temperature "
        f"({n_samples} samples, K={K} components)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    return fig


def plot_gumbel_softmax_placement_2d(
    means: Tuple[Tuple[float, float], ...] = (
        (0.0, 4.5), (-3.897, -2.25), (3.897, -2.25)
    ),
    scale: float = 0.9,
    temperatures: Tuple[float, ...] = (1.0, 0.5, 0.1, 0.0),
    n_samples: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    resolution: int = 200,
    seed: int = 0,
    figsize: Tuple = (16, 4.2),
) -> plt.Figure:
    """
    2D companion to plot_gumbel_softmax_placement — one rectangular panel per
    τ, laid out side by side (easier to place as a single figure in the
    thesis). Background color encodes the target density of a 2D, K=3
    Gaussian mixture with well-separated means (distinct valleys between
    components); scattered points are relaxed Gumbel-Softmax samples
    z = Σ_k w̃_k(τ) x_k, colored by blending the K component colors according
    to their soft assignment weights w̃(τ). τ = 0 draws hard samples.
    """
    K = 3
    assert len(means) == K, "uses the 3 method colors as the 3 component colors"
    torch.manual_seed(seed)

    means_t = torch.tensor(means, dtype=torch.float32)   # (K, 2)
    scales_t = torch.full((K, 2), scale)
    log_weights = torch.zeros(K)  # uniform mixture weights

    target = GaussianMixtureTarget(means=means_t, scales=scales_t)

    if xlim is None or ylim is None:
        pad = 3 * scale + 0.5
        xlim = xlim or (float(means_t[:, 0].min()) - pad, float(means_t[:, 0].max()) + pad)
        ylim = ylim or (float(means_t[:, 1].min()) - pad, float(means_t[:, 1].max()) + pad)

    xx, yy, grid = _make_grid(xlim, ylim, resolution)
    with torch.no_grad():
        density = target.log_prob(grid).exp().numpy().reshape(xx.shape)

    density_cmap = LinearSegmentedColormap.from_list("density", ["#ffffff", COLORS["target"]])
    comp_colors = [COLORS["score_function"], COLORS["gumbel_softmax"], COLORS["ode_transport"]]
    comp_rgb = np.array([to_rgb(c) for c in comp_colors])  # (K, 3)

    fig, axes = plt.subplots(1, len(temperatures), figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, tau in zip(axes, temperatures):
        ax.contourf(xx, yy, density, levels=25, cmap=density_cmap, zorder=1)
        ax.contour(xx, yy, density, levels=6, colors=COLORS["target"],
                    linewidths=0.4, alpha=0.35, zorder=2)

        gumbel = -torch.log(-torch.log(torch.rand(n_samples, K)))
        if tau > 0:
            soft_w = torch.softmax((log_weights + gumbel) / tau, dim=-1)  # (N, K)
        else:
            k_hard = (log_weights + gumbel).argmax(dim=-1)
            soft_w = torch.eye(K)[k_hard]

        comp_draws = means_t[None, :, :] + scales_t[None, :, :] * torch.randn(n_samples, K, 2)
        z = torch.einsum("nk,nkd->nd", soft_w, comp_draws).numpy()
        blended = soft_w.numpy() @ comp_rgb

        ax.scatter(z[:, 0], z[:, 1], c=blended, s=22, alpha=0.9,
                    edgecolor="white", linewidth=0.4, zorder=3)

        for (mx, my), c in zip(means, comp_colors):
            ax.scatter([mx], [my], marker="x", color=c, s=70, linewidths=2, zorder=4)

        label = r"$\tau \to 0$ (hard)" if tau == 0 else rf"$\tau = {tau}$"
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"Gumbel-Softmax sample placement vs. temperature "
        f"({n_samples} samples, K={K} components)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    return fig


def make_training_gif(
    target,
    gif_path: str = "training.gif",
    n_steps: int = 2000,
    snap_every: int = 10,
    n_components: int = 5,
    param_type: str = "eigenvaluedecomp",
    mc_samples: int = 256,
    lr: float = 5e-3,
    ode_steps: int = 5,
    xlim: Tuple = (-5, 5),
    ylim: Tuple = (-5, 5),
    resolution: int = 150,
    fps: int = 10,
    seed: int = 3,
    estimator_specs: Optional[Dict] = None,
) -> str:
    """
    Train three estimators on a 2-D target and save an animated GIF of the
    approximation density evolving over training.

    Args:
        target:          a Target with .log_prob and .name
        gif_path:        output file path (e.g. "banana.gif")
        n_steps:         total gradient steps per estimator
        snap_every:      save a snapshot every this many steps
        n_components:    number of mixture components
        param_type:      matrix parametrisation passed to GeneralizedMixture
        mc_samples:      Monte Carlo samples per ELBO estimate
        lr:              Adam learning rate
        ode_steps:       number of ODE integration steps (ODE estimator only)
        xlim / ylim:     axis limits for the density grid
        resolution:      grid resolution per axis
        fps:             frames per second in the output GIF
        seed:            torch manual seed (same for all estimators)
        estimator_specs: optional dict {label: estimator} to override defaults

    Returns:
        gif_path (str)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install pillow")

    from gmvi.models.generalized_mixture import GeneralizedMixture
    from gmvi.models.reference_distributions import make_reference
    from gmvi.estimators.gradient_estimators import make_estimator

    # ── default estimators ────────────────────────────────────────────────────
    if estimator_specs is None:
        estimator_specs = {
            "Gumbel-Softmax": make_estimator("gumbel_softmax", MC_samples=mc_samples),
            "Score Function":  make_estimator("score_function",  MC_samples=mc_samples),
            "ODE Transport":   make_estimator("ode_transport",   MC_samples=mc_samples,
                                              ode_steps=ode_steps),
        }

    # ── training helper ───────────────────────────────────────────────────────
    def _train_with_snapshots(model, estimator, log_target):
        opt       = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        snaps = {}
        for step in range(n_steps + 1):
            if step % snap_every == 0:
                snaps[step] = copy.deepcopy(model.state_dict())
            if step == n_steps:
                break
            opt.zero_grad()
            loss, _ = estimator.loss(model, log_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        return snaps

    # ── train all estimators ──────────────────────────────────────────────────
    all_snaps = {}
    for label, est in estimator_specs.items():
        torch.manual_seed(seed)
        ref   = make_reference("normal", dim=2)
        model = GeneralizedMixture(n_components=n_components, dim=2, reference=ref,
                                   param_type=param_type, init_scale=2.5)
        print(f"Training {label}  ({n_steps} steps, snapshot every {snap_every})…")
        all_snaps[label] = _train_with_snapshots(model, est, target.log_prob)
        print(f"  → {len(all_snaps[label])} snapshots")

    print("Training complete.\n")

    # ── target density grid ───────────────────────────────────────────────────
    xx, yy, grid = _make_grid(xlim, ylim, resolution)
    with torch.no_grad():
        log_p = target.log_prob(grid).numpy().reshape(xx.shape)
    log_p -= log_p.max()
    p_density = np.exp(log_p)

    # ── render frames ─────────────────────────────────────────────────────────
    est_labels  = list(estimator_specs.keys())
    est_colors  = ["#457B9D", "#E63946", "#2A9D8F", "#F4A261", "#8338EC"]
    snap_steps  = sorted(all_snaps[est_labels[0]].keys())
    pil_frames  = []

    print(f"Rendering {len(snap_steps)} frames…")
    for i, step in enumerate(snap_steps):
        fig, axes = plt.subplots(1, len(est_labels) + 1,
                                  figsize=(4 * (len(est_labels) + 1), 4))

        axes[0].contourf(xx, yy, p_density, levels=12, cmap="Blues", alpha=0.85)
        axes[0].contour( xx, yy, p_density, levels=12, colors="steelblue",
                         linewidths=0.5, alpha=0.6)
        axes[0].set_title("Target p(z)", fontsize=11, fontweight="bold")
        axes[0].set_xlim(xlim); axes[0].set_ylim(ylim); axes[0].set_aspect("equal")

        for ax, label, color in zip(axes[1:], est_labels, est_colors):
            ref_tmp = make_reference("normal", dim=2)
            m = GeneralizedMixture(n_components=n_components, dim=2, reference=ref_tmp,
                                   param_type=param_type, init_scale=2.5)
            m.load_state_dict(all_snaps[label][step])
            m.eval()
            with torch.no_grad():
                log_q = m.log_prob(grid).numpy().reshape(xx.shape)
            log_q -= log_q.max()
            q_density = np.exp(log_q)

            ax.contourf(xx, yy, q_density, levels=12, cmap="Reds", alpha=0.75)
            ax.contour( xx, yy, q_density, levels=12, colors=color, linewidths=0.6, alpha=0.7)
            ax.contour( xx, yy, p_density, levels=6,  colors="steelblue", linewidths=0.8,
                        linestyles="--", alpha=0.35)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")

        fig.suptitle(f"{target.name}  —  Step {step:>4} / {n_steps}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        plt.close(fig)

        if i % 20 == 0:
            print(f"  frame {i+1:>3}/{len(snap_steps)}  (step {step})")

    # ── save GIF ──────────────────────────────────────────────────────────────
    print(f"\nSaving GIF ({len(pil_frames)} frames @ {fps} fps) → {gif_path}")
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Saved: {gif_path}")
    return gif_path
