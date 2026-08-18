"""
baseline_sweep_timing.py

Section 3's authoritative timing pass: ms_per_step and NFE re-measured
SOLO (one process, no sharding, fixed thread count) for one seed of every
arm. Do not use baseline_sweep_main.py's (sharded, shared-machine)
ms_per_step numbers as final -- this script is the one whose numbers get
reported. Short measurement (5 warmup + 50 measured steps) is sufficient
since we're not tracking accuracy here, just wall-clock/NFE per step.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from gmvi.targets.distributions import make_target
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.estimators.gradient_estimators import make_estimator

from sweep_arms import MAIN_ARMS, ABLATION_ARMS

OUTDIR = os.path.dirname(__file__)
FIXED_THREADS = 4  # solo, single process, fixed thread count (section 3)
torch.set_num_threads(FIXED_THREADS)

SEED = 1
N_WARMUP = 5
N_MEASURED = 50
COMPONENTS = 5
PARAM_TYPE = "matrixexponential"
LR = 5e-3

target = make_target("banana")
results = {}

for label, est_name, kwargs, _ in MAIN_ARMS + ABLATION_ARMS:
    torch.manual_seed(SEED)
    ref = make_reference("normal", dim=2)
    model = GeneralizedMixture(n_components=COMPONENTS, dim=2, reference=ref,
                               param_type=PARAM_TYPE, init_scale=2.5)
    estimator = make_estimator(est_name, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    try:
        for _ in range(N_WARMUP):
            opt.zero_grad()
            loss, info = estimator.loss(model, target.log_prob)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss during warmup")
            loss.backward()
            opt.step()
    except Exception as e:
        print(f"{label:45s} SKIPPED (diverged during warmup: {e})")
        results[label] = {"ms_per_step": None, "nfe_per_step": None, "diverged_during_warmup": True}
        continue

    nfes = []
    t0 = time.perf_counter()
    for _ in range(N_MEASURED):
        opt.zero_grad()
        loss, info = estimator.loss(model, target.log_prob)
        loss.backward()
        opt.step()
        if info.get("nfe") is not None:
            nfes.append(info["nfe"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ms_per_step = elapsed_ms / N_MEASURED
    nfe_per_step = sum(nfes) / len(nfes) if nfes else None
    results[label] = {"ms_per_step": ms_per_step, "nfe_per_step": nfe_per_step,
                       "fixed_threads": FIXED_THREADS, "seed": SEED}
    nfe_str = f"{nfe_per_step:.1f}" if nfe_per_step is not None else "n/a"
    print(f"{label:45s} {ms_per_step:8.2f} ms/step   nfe/step={nfe_str}")

out_path = os.path.join(OUTDIR, "baseline_sweep_timing.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
