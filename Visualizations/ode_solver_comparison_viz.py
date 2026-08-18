"""
ode_solver_comparison_viz.py

Caption (put in LaTeX, not rendered in the figure): Final approximation
quality vs. compute and vs. training budget for the ODE Transport
estimator under both the linear and geometric path, for three fixed-step
solvers (Euler order 1, midpoint order 2, RK4-3/8-rule order 4, each
swept over ode_steps in {4,8,16,32}) and the adaptive Dormand-Prince
solver (dopri5, rtol=1e-5, atol=1e-7); against Straight-Through at five
fixed temperatures tau in {2.0,1.0,0.5,0.3,0.1} plus its usual annealed
schedule, Score Function, and Exact Marginalization as baselines. Every
config is trained to n_steps in {200,400,800,1600,3200} (checkpointed
within a single run, not five separate retrains) with >=3 seeds.
KL(q||p) = -ELBO exactly because the banana target's log_prob is an
exactly normalized joint density (Normal(x) times Normal(y|x)), so
log Z = 0. Hyperparameters: K=5 components, eigenvaluedecomp
parameterization, MC_samples=256, Adam lr=5e-3 cosine-annealed over the
run, gradient clipped to norm 1, seeds={1,2,3}.

Trains real gmvi models -- this is not a closed-form/no-training theory
sketch like the other Visualizations/*.py scripts.

Usage:
  Sharded (parallel) training workers -- each writes its own cache file,
  does NOT plot/export:
    python ode_solver_comparison_viz.py --shard 0 --num-shards 5
    python ode_solver_comparison_viz.py --shard 1 --num-shards 5
    ...
  Merge all shard caches + fill any gaps + produce table/plots/CSV/XLSX:
    python ode_solver_comparison_viz.py
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from gmvi.targets.distributions import make_target
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.estimators.gradient_estimators import make_estimator

OUTDIR = os.path.dirname(__file__)

TEXTWIDTH_IN = 418 / 72.27
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# ── Config ───────────────────────────────────────────────────────────────────
SEEDS           = [1, 2, 3]
N_STEPS_SWEEP   = [200, 400, 800, 1600, 3200]
COMPONENTS      = 5
PARAM_TYPE      = "eigenvaluedecomp"
MC_SAMPLES      = 256
LR              = 5e-3
ODE_STEPS_SWEEP = [4, 8, 16, 32]
SOLVERS         = ["euler", "midpoint", "rk4"]
# Euler stays in SOLVERS (already trained/cached, kept in the table/CSV/XLSX);
# it's just excluded from the plots below since its steps=4 outlier squashes
# the rest of the comparison on a linear y-axis.
PLOT_SOLVERS    = ["midpoint", "rk4"]
PATHS           = ["linear", "geometric"]
TAUS            = [2.0, 1.0, 0.5, 0.3, 0.1]

target_banana = make_target("banana")

path_linestyle = {"linear": "-", "geometric": "--"}
solver_colors  = {"euler": "#E9C46A", "midpoint": "#F4A261", "rk4": "#2A9D8F"}
solver_markers = {"euler": "o", "midpoint": "s", "rk4": "^"}
baseline_colors = {
    "Score Function":         "#E63946",
    "Exact Marginalization":  "#6A4C93",
    "Straight-Through (annealed)": "#457B9D",
}
dopri5_color = "#264653"
rk4_light_color = "#9BD4C9"  # steps=8 in the training-budget plot (steps=32 uses solver_colors["rk4"])


# ── Build the full job list: 34 configs x len(SEEDS) seeds ─────────────────────
def _make_runs():
    runs = []
    runs.append(("Score Function",
                 lambda: make_estimator("score_function", MC_samples=MC_SAMPLES),
                 {}))
    runs.append(("Exact Marginalization",
                 lambda: make_estimator("marginal_estimator", MC_samples=MC_SAMPLES),
                 {}))
    runs.append(("Straight-Through (annealed)",
                 lambda: make_estimator("gumbel_softmax", MC_samples=MC_SAMPLES),
                 {}))
    for tau in TAUS:
        runs.append((
            f"Straight-Through (tau={tau})",
            lambda tau=tau: make_estimator("gumbel_softmax", MC_samples=MC_SAMPLES,
                                            temperature=tau, anneal_rate=1.0,
                                            min_temperature=1e-6),
            {"tau": tau},
        ))
    for path in PATHS:
        for solver in SOLVERS:
            for steps in ODE_STEPS_SWEEP:
                runs.append((
                    f"ODE {path} {solver} (steps={steps})",
                    lambda path=path, solver=solver, steps=steps: make_estimator(
                        "ode_transport", MC_samples=MC_SAMPLES, ode_steps=steps,
                        path=path, ode_solver=solver),
                    {"path": path, "solver": solver, "ode_steps": steps},
                ))
        runs.append((
            f"ODE {path} dopri5",
            lambda path=path: make_estimator("ode_transport", MC_samples=MC_SAMPLES,
                                              path=path, ode_solver="dopri5",
                                              rtol=1e-5, atol=1e-7),
            {"path": path, "solver": "dopri5"},
        ))
    return runs


RUNS = _make_runs()
JOBS = [(label, seed) for label, _, _ in RUNS for seed in SEEDS]
MAKE_EST = {label: make_est for label, make_est, _ in RUNS}
META = {label: meta for label, _, meta in RUNS}


@torch.no_grad()
def eval_elbo(model, log_target, n_samples=8192):
    """Estimator-agnostic ELBO: E_q[log p(z) - log q(z)]. KL(q||p) = -ELBO here
    because log_target is an exactly normalized density (log Z = 0)."""
    z, _ = model.sample(n_samples)
    elbo = log_target(z) - model.log_prob(z)
    return elbo.mean().item(), (elbo.std() / n_samples ** 0.5).item()


def train_with_checkpoints(estimator, log_target, lr, n_steps_sweep, seed):
    """Trains one model up to max(n_steps_sweep), evaluating ELBO at each
    checkpoint along the way (no separate retrain per n_steps value)."""
    torch.manual_seed(seed)
    ref = make_reference("normal", dim=2)
    model = GeneralizedMixture(n_components=COMPONENTS, dim=2, reference=ref,
                               param_type=PARAM_TYPE, init_scale=2.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    max_steps = max(n_steps_sweep)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    checkpoint_set = set(n_steps_sweep)

    checkpoints = {}
    t0 = time.perf_counter()
    for step in range(max_steps + 1):
        opt.zero_grad()
        loss, _ = estimator.loss(model, log_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if step in checkpoint_set:
            cumulative_s = time.perf_counter() - t0
            model.eval()
            elbo_mean, elbo_stderr = eval_elbo(model, log_target)
            checkpoints[step] = {
                "elbo": elbo_mean,
                "elbo_mc_stderr": elbo_stderr,
                "cumulative_train_s": cumulative_s,
                "ms_per_step": cumulative_s * 1000 / step,
            }
    return checkpoints


def run_job(label, seed):
    est = MAKE_EST[label]()
    checkpoints = train_with_checkpoints(est, target_banana.log_prob, LR, N_STEPS_SWEEP, seed)
    return {"meta": META[label], "checkpoints": checkpoints}


# ── CLI: sharded worker mode vs. merge/report mode ──────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=None)
parser.add_argument("--num-shards", type=int, default=1)
args = parser.parse_args()

MASTER_CACHE = os.path.join(OUTDIR, ".ode_solver_comparison_cache.json")

if args.shard is not None:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // args.num_shards))
    shard_path = os.path.join(
        OUTDIR, f".ode_solver_comparison_shard{args.shard}_of_{args.num_shards}.json")
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

for shard_file in glob.glob(os.path.join(OUTDIR, ".ode_solver_comparison_shard*.json")):
    with open(shard_file) as f:
        shard_results = json.load(f)
    for label, per_seed in shard_results.items():
        results.setdefault(label, {}).update(per_seed)

# Fallback: run any jobs missing from every shard (e.g. standalone, unsharded use)
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

# ── Long-format DataFrame: one row per (config, seed, n_steps checkpoint) ──────
rows = []
for label, per_seed in results.items():
    meta = META.get(label, next(iter(per_seed.values()))["meta"])
    for seed_str, run in per_seed.items():
        for n_steps_str, ck in run["checkpoints"].items():
            rows.append({
                "config": label,
                "path": meta.get("path"),
                "solver": meta.get("solver"),
                "ode_steps": meta.get("ode_steps"),
                "tau": meta.get("tau"),
                "seed": int(seed_str),
                "n_steps": int(n_steps_str),
                "elbo": ck["elbo"],
                "kl": -ck["elbo"],
                "elbo_mc_stderr": ck["elbo_mc_stderr"],
                "ms_per_step": ck["ms_per_step"],
                "cumulative_train_s": ck["cumulative_train_s"],
            })
df = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR, "ode_solver_comparison.csv")
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

agg_df = (df.groupby(["config", "path", "solver", "ode_steps", "tau", "n_steps"],
                      dropna=False)
            .agg(elbo_mean=("elbo", "mean"), elbo_std=("elbo", "std"),
                 kl_mean=("kl", "mean"), kl_std=("kl", "std"),
                 ms_per_step_mean=("ms_per_step", "mean"), n_seeds=("seed", "count"))
            .reset_index())
xlsx_path = os.path.join(OUTDIR, "ode_solver_comparison.xlsx")
with pd.ExcelWriter(xlsx_path) as writer:
    df.to_excel(writer, sheet_name="raw_per_seed", index=False)
    agg_df.to_excel(writer, sheet_name="aggregated", index=False)
print(f"Saved: {xlsx_path}")

# ── Print final-checkpoint (n_steps=max) summary table ──────────────────────────
final_n = max(N_STEPS_SWEEP)
final_tbl = agg_df[agg_df.n_steps == final_n].sort_values(
    ["path", "solver", "ode_steps"], na_position="first")
print(f"\n=== Final checkpoint (n_steps={final_n}) ===")
print(final_tbl.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: accuracy vs ode_steps and vs compute, at the final checkpoint,
#         one panel row per path.
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85),
                          constrained_layout=True)

for row, path in enumerate(PATHS):
    ax_steps, ax_time = axes[row]
    sub = final_tbl[final_tbl.path == path]
    for solver in PLOT_SOLVERS:
        s = sub[sub.solver == solver].sort_values("ode_steps")
        ax_steps.errorbar(s.ode_steps, s.kl_mean, yerr=s.kl_std,
                           color=solver_colors[solver], marker=solver_markers[solver],
                           label=solver, capsize=2)
        ax_time.errorbar(s.ms_per_step_mean, s.kl_mean, yerr=s.kl_std,
                          color=solver_colors[solver], marker=solver_markers[solver],
                          label=solver, capsize=2)
    dop = sub[sub.solver == "dopri5"]
    if len(dop):
        ax_time.errorbar(dop.ms_per_step_mean, dop.kl_mean, yerr=dop.kl_std,
                          color=dopri5_color, marker="D", markersize=6, capsize=2,
                          label="dopri5")
    for label, color in baseline_colors.items():
        r = final_tbl[final_tbl.config == label]
        if len(r):
            ax_steps.axhline(r.kl_mean.iloc[0], color=color, linestyle=":", linewidth=1)
            ax_time.scatter(r.ms_per_step_mean, r.kl_mean, color=color, marker="*",
                             s=60, zorder=5)

    ax_steps.set_xscale("log", base=2)
    ax_steps.set_xticks(ODE_STEPS_SWEEP)
    ax_steps.set_xticklabels(ODE_STEPS_SWEEP)
    ax_steps.set_ylabel(f"{path}\n" + r"$\mathrm{KL}(q\,\|\,p)$")
    ax_time.set_xscale("log")
    if row == 0:
        ax_steps.set_title("accuracy vs. integration steps")
        ax_time.set_title("accuracy vs. compute")
    if row == 1:
        ax_steps.set_xlabel("ode_steps")
        ax_time.set_xlabel("ms / training step")

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="outside lower center", ncol=6, frameon=False)

for ext in ("png", "pdf"):
    out_path = os.path.join(OUTDIR, f"ode_solver_comparison.{ext}")
    fig.savefig(out_path, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: accuracy vs training budget (n_steps), rk4 (representative solver)
#         for both paths, plus ST-annealed / Score / ExactMarg baselines.
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.5),
                         constrained_layout=True)
for path in PATHS:
    for steps in [8, 32]:
        s = agg_df[(agg_df.config == f"ODE {path} rk4 (steps={steps})")].sort_values("n_steps")
        ax.errorbar(s.n_steps, s.kl_mean, yerr=s.kl_std,
                    linestyle=path_linestyle[path],
                    color=solver_colors["rk4"] if steps == 32 else rk4_light_color,
                    marker="^", label=f"{path} rk4 (steps={steps})", capsize=2)
for label, color in baseline_colors.items():
    s = agg_df[agg_df.config == label].sort_values("n_steps")
    ax.errorbar(s.n_steps, s.kl_mean, yerr=s.kl_std, color=color, marker="*",
                label=label, capsize=2)
ax.set_xscale("log", base=2)
ax.set_xticks(N_STEPS_SWEEP)
ax.set_xticklabels(N_STEPS_SWEEP)
ax.set_xlabel("training steps")
ax.set_ylabel(r"$\mathrm{KL}(q\,\|\,p)$")
ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=7)

for ext in ("png", "pdf"):
    out_path = os.path.join(OUTDIR, f"ode_solver_comparison_vs_training_budget.{ext}")
    fig2.savefig(out_path, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {out_path}")
