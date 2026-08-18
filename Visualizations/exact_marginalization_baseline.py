"""
exact_marginalization_baseline.py

A single high-fidelity Exact Marginalization run on the banana target:
MC_samples=4096, n_steps=10000, seed=3, K=5 components, eigenvaluedecomp
parameterization, Adam lr=5e-3 cosine-annealed over the run, gradient
clipped to norm 1 -- otherwise identical protocol to the runs in
ode_solver_comparison_viz.py, just a single seed at much higher MC_samples
and training length as a high-quality baseline reference point.

Reports: final ELBO (both the training-loss value at the last step, and an
independent 8192-sample cross-check using the same estimator-agnostic
E_q[log p - log q] used throughout this project), wall-clock time, and the
final trained parameters (weights, means, A matrices per component).
KL(q||p) = -ELBO exactly since the banana target's log_prob is an exactly
normalized density.

No plot -- numeric report only, also written to
exact_marginalization_baseline.json for reproducibility.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from gmvi.targets.distributions import make_target
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.estimators.gradient_estimators import make_estimator

OUTDIR = os.path.dirname(__file__)

SEED       = 3
N_STEPS    = 10_000
MC_SAMPLES = 4096
COMPONENTS = 5
PARAM_TYPE = "eigenvaluedecomp"
LR         = 5e-3

target = make_target("banana")


@torch.no_grad()
def eval_elbo(model, log_target, n_samples=8192):
    """Estimator-agnostic ELBO: E_q[log p(z) - log q(z)]. KL(q||p) = -ELBO here
    because log_target is an exactly normalized density (log Z = 0)."""
    z, _ = model.sample(n_samples)
    elbo = log_target(z) - model.log_prob(z)
    return elbo.mean().item(), (elbo.std() / n_samples ** 0.5).item()


torch.manual_seed(SEED)
ref = make_reference("normal", dim=2)
model = GeneralizedMixture(n_components=COMPONENTS, dim=2, reference=ref,
                           param_type=PARAM_TYPE, init_scale=2.5)
est = make_estimator("marginal_estimator", MC_samples=MC_SAMPLES)

opt = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

print(f"Training Exact Marginalization: MC_samples={MC_SAMPLES}, n_steps={N_STEPS}, seed={SEED}")
t0 = time.perf_counter()
last_train_elbo = None
for step in range(N_STEPS + 1):
    opt.zero_grad()
    loss, info = est.loss(model, target.log_prob)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()
    last_train_elbo = info["elbo"]
    if step % 2000 == 0:
        print(f"  step {step:>6d}   train ELBO {last_train_elbo:.4f}")
total_s = time.perf_counter() - t0

model.eval()
final_elbo_mean, final_elbo_stderr = eval_elbo(model, target.log_prob, n_samples=8192)

# ── Final parameters ─────────────────────────────────────────────────────────
weights = torch.softmax(model.log_weights, dim=0).detach().tolist()
components = []
for i, comp in enumerate(model.components):
    components.append({
        "weight": weights[i],
        "mean_a": comp.a.detach().tolist(),
        "A": comp.get_A().detach().tolist(),
        "log_diag": comp.log_diag.detach().tolist(),
        "skew_raw": comp.skew_raw.detach().tolist(),
    })

report = {
    "config": {
        "seed": SEED, "n_steps": N_STEPS, "MC_samples": MC_SAMPLES,
        "n_components": COMPONENTS, "param_type": PARAM_TYPE, "lr": LR,
    },
    "timing": {
        "total_train_s": total_s,
        "ms_per_step": total_s * 1000 / N_STEPS,
    },
    "elbo": {
        "final_train_loss_elbo": last_train_elbo,
        "final_eval_elbo_mean_8192samples": final_elbo_mean,
        "final_eval_elbo_stderr": final_elbo_stderr,
        "kl_q_p_mean": -final_elbo_mean,
        "kl_q_p_stderr": final_elbo_stderr,
    },
    "final_parameters": {
        "weights": weights,
        "components": components,
    },
}

out_path = os.path.join(OUTDIR, "exact_marginalization_baseline.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print()
print("=" * 70)
print(f"Total training time:   {total_s:.1f} s  ({report['timing']['ms_per_step']:.2f} ms/step)")
print(f"Final train-loss ELBO: {last_train_elbo:.4f}")
print(f"Final eval ELBO (8192-sample MC): {final_elbo_mean:.4f} +/- {final_elbo_stderr:.4f}")
print(f"KL(q||p):              {-final_elbo_mean:.4f} +/- {final_elbo_stderr:.4f}")
print()
print("Final parameters:")
print(f"  weights: {[f'{w:.4f}' for w in weights]}")
for i, c in enumerate(components):
    print(f"  component {i}: weight={c['weight']:.4f}  a={c['mean_a']}")
    print(f"    A = {c['A']}")
print("=" * 70)
print(f"Saved: {out_path}")
