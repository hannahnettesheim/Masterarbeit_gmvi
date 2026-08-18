"""
RK4 discretization error of the transport ODE, linear vs. geometric path.

Question: does W2-optimality of the interpolation path transfer to the
discretized map?

  - 'linear':    A_{j,t} = t A_j + (1-t) I  — the displacement interpolation
                 (McCann/W2-geodesic for Gaussians), energy-minimal, but
                 A_{j,t} A_{j,t}^{-1}-derivative is not constant in t.
  - 'geometric': A_{j,t} = exp(t log A_j)   — constant Ȧ A^{-1}, not the
                 W2-geodesic.

For K=1 (a single component) the flow is exactly affine and both paths are
solved exactly by any consistent scheme, up to floating-point error — the
experiment has no content. For K>=2 the softmax mixture responsibilities
gamma_j(x, t) make the velocity field x-dependent and nonlinear, so RK4 with
a finite step incurs genuine discretization error. This script measures that
error against step size h = 1/ode_steps, for a K=2 mixture whose components'
matrices A_j are constructed (via the 'eigenvaluedecomp' parametrization) to
have an exact, prescribed condition number kappa.

Error metric: pathwise RMSE  sqrt(E_x0 ||x1_h(x0) - x1_ref(x0)||^2), computed
against a fine-step (ode_steps=4096) reference solution of the *same*
continuous path, using a shared, paired set of x0 samples across all runs.
Since both the coarse and reference maps are driven by the same coupling
(the shared x0's), this pathwise RMSE upper-bounds the W2 distance between
the two induced pushforward measures, and is the standard quantity for
reporting ODE-integrator discretization error.
"""

import math
import torch
import matplotlib.pyplot as plt

from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.estimators.gradient_estimators import integrate_ode


def build_model(kappa: float, angles=(0.3, -0.6), means=((-1.2, 0.0), (1.2, 0.0))) -> GeneralizedMixture:
    """K=2, dim=2 mixture whose component matrices A_j have condition number kappa."""
    model = GeneralizedMixture(n_components=2, dim=2, param_type="eigenvaluedecomp", init_scale=1.0)
    ln_k = math.log(kappa)
    with torch.no_grad():
        for j, comp in enumerate(model.components):
            comp.skew_raw.data[:] = angles[j]
            comp.log_diag.data[:] = torch.tensor([0.5 * ln_k, -0.5 * ln_k], dtype=comp.log_diag.dtype)
            comp.a.data[:] = torch.tensor(means[j], dtype=comp.a.dtype)
        model.log_weights.data.zero_()
    return model


def run(
    kappas=(10.0, 1e3, 1e5),
    paths=("linear", "geometric"),
    h_steps=(2, 4, 8, 16, 32, 64, 128, 256),
    ref_steps: int = 4096,
    n_samples: int = 500,
    seed: int = 0,
    out_path: str = "rk4_convergence.png",
):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    x0 = torch.randn(n_samples, 2)  # shared, paired across all runs

    results = {}
    for kappa in kappas:
        model = build_model(kappa)
        for path in paths:
            with torch.no_grad():
                x1_ref = integrate_ode(x0, model, ode_steps=ref_steps, path=path)
            errs = []
            for steps in h_steps:
                with torch.no_grad():
                    x1_h = integrate_ode(x0, model, ode_steps=steps, path=path)
                err = (x1_h - x1_ref).pow(2).sum(-1).mean().sqrt().item()
                errs.append(err)
                print(f"kappa={kappa:>8.0f}  path={path:9s}  ode_steps={steps:4d}  "
                      f"h={1/steps:.4f}  RMSE={err:.3e}", flush=True)
            results[(kappa, path)] = errs

    _plot(results, kappas, paths, h_steps, out_path)
    return results


def _plot(results, kappas, paths, h_steps, out_path):
    h = [1.0 / s for s in h_steps]
    colors = {kappas[0]: "#457B9D", kappas[1]: "#E9A94A", kappas[2]: "#E63946"}
    linestyles = {"linear": "-", "geometric": "--"}
    markers = {"linear": "o", "geometric": "s"}

    fig, ax = plt.subplots(figsize=(6, 5))
    for kappa in kappas:
        for path in paths:
            ax.loglog(h, results[(kappa, path)],
                       color=colors[kappa], linestyle=linestyles[path],
                       marker=markers[path], markersize=5,
                       label=f"$\\kappa$={kappa:.0e}, {path}")

    # h^4 reference slope, anchored near the finest-h, best-conditioned point
    h_ref = torch.tensor(h[-3:])
    anchor = results[(kappas[0], "linear")][-3]
    ref = anchor * (h_ref / h_ref[-1]) ** 4
    ax.loglog(h_ref, ref, color="gray", linestyle=":", linewidth=1.5, label="$O(h^4)$")

    ax.set_xlabel("step size $h = 1/\\mathrm{ode\\_steps}$")
    ax.set_ylabel("pathwise RMSE  $\\sqrt{\\mathbb{E}\\|x_1^h - x_1^{\\mathrm{ref}}\\|^2}$")
    ax.set_title("RK4 discretization error: linear vs. geometric path (K=2)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved figure to {out_path}")


if __name__ == "__main__":
    run()
