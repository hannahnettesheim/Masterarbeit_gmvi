"""
baseline_sweep_main.py

Main accuracy sweep per BASELINE_SWEEP.md section 1-2: 45 arms (33 main +
12 new points from the section 2.4 MC_samples ablation) x 5 seeds, K=5,
matrixexponential parameterization, Adam lr=5e-3 cosine-annealed over
10000 steps, NO gradient clipping, checkpoints at
{200,500,1000,2000,5000,10000} (all checkpoints of a single 10k-step run,
not separate retrains -- "training step" not "training budget"). Every
checkpoint's ELBO is an estimator-agnostic 8192-sample eval
(E_q[log p - log q]); KL(q||p) = -ELBO exactly, banana log_prob is an
exactly normalized density. Also logs pre-clip gradient norm (mean, p95)
over the whole run per arm/seed, per section 1's request -- clipping is
off, this is a diagnostic on whether ODE gradients are systematically
larger.

ms_per_step recorded here is informal (shared-machine, sharded); the
authoritative timing numbers come from the separate solo harness
(baseline_sweep_timing.py) per section 3 -- do not report this run's
timings as final.

Usage:
  Sharded parallel workers (write only their own shard cache):
    python baseline_sweep_main.py --shard 0 --num-shards 5
    ...
  Merge all shards + fill gaps + write long-format CSV/XLSX:
    python baseline_sweep_main.py
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch

from gmvi.targets.distributions import make_target
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.estimators.gradient_estimators import make_estimator

from sweep_arms import MAIN_ARMS, ABLATION_ARMS

OUTDIR = os.path.dirname(__file__)

SEEDS = [1, 2, 3, 4, 5]
CHECKPOINTS = [200, 500, 1000, 2000, 5000, 10000]
COMPONENTS = 5
PARAM_TYPE = "matrixexponential"
LR = 5e-3

ALL_ARMS = MAIN_ARMS + ABLATION_ARMS  # (label, est_name, kwargs, ablation_group)
ARM_BY_LABEL = {label: (est_name, kwargs) for label, est_name, kwargs, _ in ALL_ARMS}
JOBS = [(label, seed) for label, _, _, _ in ALL_ARMS for seed in SEEDS]

target = make_target("banana")


@torch.no_grad()
def eval_elbo(model, log_target, n_samples=8192):
    z, _ = model.sample(n_samples)
    elbo = log_target(z) - model.log_prob(z)
    return elbo.mean().item(), (elbo.std() / n_samples ** 0.5).item()


def train_with_checkpoints(est_name, kwargs, seed):
    torch.manual_seed(seed)
    ref = make_reference("normal", dim=2)
    model = GeneralizedMixture(n_components=COMPONENTS, dim=2, reference=ref,
                               param_type=PARAM_TYPE, init_scale=2.5)
    estimator = make_estimator(est_name, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    max_steps = max(CHECKPOINTS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    checkpoint_set = set(CHECKPOINTS)

    checkpoints = {}
    grad_norms = []
    last_info = {}
    diverged_at = None
    t0 = time.perf_counter()
    for step in range(max_steps + 1):
        opt.zero_grad()
        try:
            loss, info = estimator.loss(model, target.log_prob)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss.item()}")
            loss.backward()
            # gradient clipping OFF (section 1) -- max_norm=inf never rescales,
            # just returns the pre-clip total norm for logging.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(gnorm):
                raise FloatingPointError(f"non-finite grad norm: {gnorm.item()}")
        except Exception as e:
            # Divergence (e.g. plain Gumbel-Softmax's z_soft landing in a
            # low-density valley -> NaN loss/params) is a real, reportable
            # result under no-clipping, not a crash to lose the whole shard
            # over. Record it and stop this (arm, seed) run.
            diverged_at = step
            print(f"    DIVERGED at step {step}: {type(e).__name__}: {e}")
            break
        grad_norms.append(gnorm.item())
        opt.step()
        scheduler.step()
        last_info = info
        if step in checkpoint_set:
            cumulative_s = time.perf_counter() - t0
            model.eval()
            elbo_mean, elbo_stderr = eval_elbo(model, target.log_prob)
            model.train()
            checkpoints[step] = {
                "elbo": elbo_mean,
                "elbo_mc_stderr": elbo_stderr,
                "cumulative_train_s": cumulative_s,
                "ms_per_step": cumulative_s * 1000 / step,
                "nfe": last_info.get("nfe"),
            }

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


def run_job(label, seed):
    est_name, kwargs = ARM_BY_LABEL[label]
    return train_with_checkpoints(est_name, kwargs, seed)


# ── CLI: sharded worker mode vs. merge/report mode ──────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=None)
parser.add_argument("--num-shards", type=int, default=1)
args = parser.parse_args()

MASTER_CACHE = os.path.join(OUTDIR, ".baseline_sweep_main_cache.json")

if args.shard is not None:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // args.num_shards))
    shard_path = os.path.join(
        OUTDIR, f".baseline_sweep_main_shard{args.shard}_of_{args.num_shards}.json")
    shard_results = {}
    if os.path.exists(shard_path):
        with open(shard_path) as f:
            shard_results = json.load(f)

    my_jobs = JOBS[args.shard::args.num_shards]
    print(f"Shard {args.shard}/{args.num_shards}: {len(my_jobs)} jobs")
    for label, seed in my_jobs:
        skey = str(seed)
        if label in shard_results and skey in shard_results[label]:
            continue
        print(f"[shard {args.shard}] Training {label} (seed={seed})...")
        shard_results.setdefault(label, {})[skey] = run_job(label, seed)
        with open(shard_path, "w") as f:
            json.dump(shard_results, f, indent=2)

    print(f"Shard {args.shard}/{args.num_shards} done.")
    sys.exit(0)

# ── Merge/report mode ────────────────────────────────────────────────────────
results = {}
if os.path.exists(MASTER_CACHE):
    with open(MASTER_CACHE) as f:
        results = json.load(f)

for shard_file in glob.glob(os.path.join(OUTDIR, ".baseline_sweep_main_shard*.json")):
    with open(shard_file) as f:
        shard_results = json.load(f)
    for label, per_seed in shard_results.items():
        results.setdefault(label, {}).update(per_seed)

for label, seed in JOBS:
    skey = str(seed)
    if label in results and skey in results[label]:
        continue
    print(f"Training {label} (seed={seed})... (not found in any shard cache)")
    results.setdefault(label, {})[skey] = run_job(label, seed)
    with open(MASTER_CACHE, "w") as f:
        json.dump(results, f, indent=2)

with open(MASTER_CACHE, "w") as f:
    json.dump(results, f, indent=2)
print(f"Merged cache written to {MASTER_CACHE}")

# ── Long-format DataFrame ────────────────────────────────────────────────────
arm_meta = {label: kwargs for label, _, kwargs, _ in ALL_ARMS}
arm_group = {label: group for label, _, _, group in ALL_ARMS}
arm_est = {label: est_name for label, est_name, _, _ in ALL_ARMS}

rows = []
for label, per_seed in results.items():
    for seed_str, run in per_seed.items():
        for n_steps_str, ck in run["checkpoints"].items():
            diverged = ck.get("diverged", False)
            rows.append({
                "config": label,
                "estimator": arm_est.get(label),
                "ablation_group": arm_group.get(label),
                "seed": int(seed_str),
                "n_steps": int(n_steps_str),
                "diverged": diverged,
                "diverged_at_step": ck.get("diverged_at_step") if diverged else None,
                "elbo": ck.get("elbo"),
                "kl": -ck["elbo"] if not diverged else float("nan"),
                "elbo_mc_stderr": ck.get("elbo_mc_stderr"),
                "ms_per_step": ck.get("ms_per_step"),
                "nfe_per_step": ck.get("nfe"),
                "cumulative_train_s": ck.get("cumulative_train_s"),
                "grad_norm_mean": run.get("grad_norm_mean"),
                "grad_norm_p95": run.get("grad_norm_p95"),
                **{f"kwarg_{k}": v for k, v in arm_meta.get(label, {}).items()},
            })
# Post-hoc instability flag: the in-loop isfinite() check only catches
# literal NaN/Inf. Gradients can also explode to enormous-but-finite values
# (e.g. euler-4-linear's known instability, worse without clipping) and
# "complete" all n_steps while reporting a meaningless KL. Healthy runs in
# this sweep sit at KL < ~2 and grad_norm_mean < ~2; anything past these
# generous thresholds is numerically unstable, not a real accuracy number.
UNSTABLE_KL_THRESHOLD = 5.0
UNSTABLE_GRAD_NORM_THRESHOLD = 50.0

df = pd.DataFrame(rows)
df["unstable"] = (
    df["diverged"]
    | (df["kl"].abs() > UNSTABLE_KL_THRESHOLD)
    | (df["grad_norm_mean"].fillna(0) > UNSTABLE_GRAD_NORM_THRESHOLD)
)

csv_path = os.path.join(OUTDIR, "baseline_sweep_main.csv")
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

xlsx_path = os.path.join(OUTDIR, "baseline_sweep_main.xlsx")
with pd.ExcelWriter(xlsx_path) as writer:
    df.to_excel(writer, sheet_name="raw_per_seed", index=False)
    agg = (df[~df.unstable].groupby(["config", "n_steps"], dropna=False)
             .agg(elbo_mean=("elbo", "mean"), elbo_std=("elbo", "std"),
                  kl_mean=("kl", "mean"), kl_std=("kl", "std"),
                  n_seeds=("seed", "count"))
             .reset_index())
    agg.to_excel(writer, sheet_name="aggregated", index=False)
print(f"Saved: {xlsx_path}")

final_n = max(CHECKPOINTS)
final = df[df.n_steps == final_n]

final_tbl = final[~final.unstable].groupby("config").agg(
    kl_mean=("kl", "mean"), kl_std=("kl", "std"), n_seeds=("seed", "count"),
    grad_norm_mean=("grad_norm_mean", "mean")).reset_index().sort_values("kl_mean")
print(f"\n=== Final checkpoint (n_steps={final_n}), numerically stable runs only, sorted by KL ===")
print(final_tbl.to_string(index=False))

unstable_summary = (final[final.unstable].groupby("config")
                     .agg(n_unstable=("seed", "count"),
                          n_hard_diverged=("diverged", "sum"))
                     .reset_index())
unstable_summary["n_unstable_of_5"] = unstable_summary["n_unstable"]
if len(unstable_summary):
    print(f"\n=== Unstable configs ({unstable_summary.n_unstable.sum()} of {len(JOBS)} "
          f"(config,seed) pairs: hard NaN/Inf divergence OR |KL|>{UNSTABLE_KL_THRESHOLD:g} "
          f"OR grad_norm_mean>{UNSTABLE_GRAD_NORM_THRESHOLD:g}) ===")
    print(unstable_summary.sort_values("n_unstable", ascending=False).to_string(index=False))
