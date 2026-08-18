"""
baseline_sweep_floor.py

Section 2.1's floor run: Exact Marginalization, MC_samples=4096,
n_steps=20000, 3 seeds. Checkpointed at the same points as the main sweep
plus 20000, so we can report whether it's still falling between 10k and
20k (per the spec: if so, quote as an upper bound on the floor, not the
floor itself). K=5, matrixexponential, Adam lr=5e-3 cosine-annealed over
the full 20k, no gradient clipping -- otherwise identical protocol to
baseline_sweep_main.py.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from gmvi.targets.distributions import make_target
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.estimators.gradient_estimators import make_estimator

OUTDIR = os.path.dirname(__file__)

SEEDS = [1, 2, 3]
CHECKPOINTS = [200, 500, 1000, 2000, 5000, 10000, 20000]
COMPONENTS = 5
PARAM_TYPE = "matrixexponential"
LR = 5e-3
MC_SAMPLES = 4096
N_STEPS = 20000

target = make_target("banana")


@torch.no_grad()
def eval_elbo(model, log_target, n_samples=8192):
    z, _ = model.sample(n_samples)
    elbo = log_target(z) - model.log_prob(z)
    return elbo.mean().item(), (elbo.std() / n_samples ** 0.5).item()


def train_with_checkpoints(seed):
    torch.manual_seed(seed)
    ref = make_reference("normal", dim=2)
    model = GeneralizedMixture(n_components=COMPONENTS, dim=2, reference=ref,
                               param_type=PARAM_TYPE, init_scale=2.5)
    estimator = make_estimator("marginal_estimator", MC_samples=MC_SAMPLES)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)
    checkpoint_set = set(CHECKPOINTS)

    checkpoints = {}
    grad_norms = []
    diverged_at = None
    t0 = time.perf_counter()
    for step in range(N_STEPS + 1):
        opt.zero_grad()
        try:
            loss, info = estimator.loss(model, target.log_prob)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss.item()}")
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(gnorm):
                raise FloatingPointError(f"non-finite grad norm: {gnorm.item()}")
        except Exception as e:
            diverged_at = step
            print(f"  seed={seed}  DIVERGED at step {step}: {type(e).__name__}: {e}")
            break
        grad_norms.append(gnorm.item())
        opt.step()
        scheduler.step()
        if step in checkpoint_set:
            cumulative_s = time.perf_counter() - t0
            model.eval()
            elbo_mean, elbo_stderr = eval_elbo(model, target.log_prob)
            model.train()
            checkpoints[step] = {
                "elbo": elbo_mean, "elbo_mc_stderr": elbo_stderr,
                "cumulative_train_s": cumulative_s,
            }
            print(f"  seed={seed} step={step:>6d}  KL={-elbo_mean:.4f} +/- {elbo_stderr:.4f}")

    for step in CHECKPOINTS:
        if step not in checkpoints:
            checkpoints[step] = {"diverged": True, "diverged_at_step": diverged_at}

    gn = np.array(grad_norms) if grad_norms else np.array([np.nan])
    return {
        "checkpoints": checkpoints,
        "grad_norm_mean": float(gn.mean()) if grad_norms else None,
        "grad_norm_p95": float(np.percentile(gn, 95)) if grad_norms else None,
        "diverged_at_step": diverged_at,
    }


CACHE_PATH = os.path.join(OUTDIR, "baseline_sweep_floor.json")
results = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        results = json.load(f)

for seed in SEEDS:
    skey = str(seed)
    if skey in results:
        continue
    print(f"Training floor run, seed={seed}...")
    results[skey] = train_with_checkpoints(seed)
    with open(CACHE_PATH, "w") as f:
        json.dump(results, f, indent=2)

print("\n=== Floor run summary (Exact Marginalization, MC=4096) ===")
for step in CHECKPOINTS:
    kls = [-results[str(s)]["checkpoints"][str(step)]["elbo"] for s in SEEDS
           if not results[str(s)]["checkpoints"][str(step)].get("diverged")]
    n_div = len(SEEDS) - len(kls)
    if kls:
        print(f"  n_steps={step:>6d}  KL = {np.mean(kls):.4f} +/- {np.std(kls, ddof=1) if len(kls)>1 else 0:.4f}"
              f"  (n={len(kls)}{f', {n_div} diverged' if n_div else ''})")
    else:
        print(f"  n_steps={step:>6d}  all {len(SEEDS)} seeds diverged")

kl10 = [-results[str(s)]["checkpoints"]["10000"]["elbo"] for s in SEEDS
        if not results[str(s)]["checkpoints"]["10000"].get("diverged")]
kl20 = [-results[str(s)]["checkpoints"]["20000"]["elbo"] for s in SEEDS
        if not results[str(s)]["checkpoints"]["20000"].get("diverged")]
if kl10 and kl20:
    delta_10k_20k = np.mean(kl10) - np.mean(kl20)
    print(f"\nKL(10k) - KL(20k) = {delta_10k_20k:.4f}  "
          f"({'still falling -- report 20k as an UPPER BOUND on the floor' if delta_10k_20k > 0.002 else 'converged, this is the floor'})")
print(f"Saved: {CACHE_PATH}")
