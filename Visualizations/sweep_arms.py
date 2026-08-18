"""
sweep_arms.py

Shared arm-list definition for BASELINE_SWEEP.md, imported by the timing
harness and the main sweep script so the two never drift apart.

Assumption (spec section 1's protocol table has no MC_samples row -- it's
its own axis per section 2.4, varied only for a named subset): MC_samples=256
is the default for every main-sweep arm not in that subset. The section 2.4
ablation then only needs to add the MC values NOT already covered by an
arm's default (256) or by the section 2.1 Exact Marginalization reference
(1024) -- see the comments below for exactly which (config, MC) pairs are
new.
"""

MC_DEFAULT = 256

TAUS = [2.0, 1.0, 0.5, 0.3, 0.1]
ODE_STEPS_EULER = [4, 8, 16, 32]
ODE_STEPS_MID_RK4 = [4, 8]
PATHS = ["linear", "geometric"]


def make_estimator_kwargs():
    """Returns a list of (label, estimator_name, kwargs, mc_axis_group) tuples.
    mc_axis_group is None, or a string key used by the section 2.4 ablation to
    find which arms need extra MC_samples points."""
    arms = []

    # ── 2.1 Reference (not a competitor): Exact Marginalization, MC=1024 ──────
    arms.append(("Exact Marginalization (MC=1024)", "marginal_estimator",
                 dict(MC_samples=1024), "exact_marg"))

    # ── 2.2 Baselines ──────────────────────────────────────────────────────────
    arms.append(("Score Function", "score_function",
                 dict(MC_samples=MC_DEFAULT, baseline="none"), "score_function"))
    arms.append(("Score Function + RLOO", "score_function",
                 dict(MC_samples=MC_DEFAULT, baseline="rloo"), "score_function_rloo"))

    for tau in TAUS:
        arms.append((f"Gumbel-Softmax (soft, tau={tau})", "gumbel_softmax",
                     dict(MC_samples=MC_DEFAULT, temperature=tau, anneal_rate=1.0,
                          min_temperature=1e-6, mode="soft"),
                     "gs_soft_tau0.5" if tau == 0.5 else None))
    arms.append(("Gumbel-Softmax (soft, annealed)", "gumbel_softmax",
                 dict(MC_samples=MC_DEFAULT, temperature=1.0, anneal_rate=0.9995,
                      min_temperature=0.3, mode="soft"), None))

    for tau in TAUS:
        arms.append((f"Straight-Through (tau={tau})", "gumbel_softmax",
                     dict(MC_samples=MC_DEFAULT, temperature=tau, anneal_rate=1.0,
                          min_temperature=1e-6, mode="straight_through"),
                     "st_tau0.5" if tau == 0.5 else None))
    arms.append(("Straight-Through (annealed)", "gumbel_softmax",
                 dict(MC_samples=MC_DEFAULT, temperature=1.0, anneal_rate=0.9995,
                      min_temperature=0.3, mode="straight_through"), None))

    # ── 2.3 ODE transport: 18 arms ──────────────────────────────────────────────
    for path in PATHS:
        for steps in ODE_STEPS_EULER:
            arms.append((f"ODE {path} euler (steps={steps})", "ode_transport",
                         dict(MC_samples=MC_DEFAULT, ode_steps=steps, path=path,
                              ode_solver="euler"), None))
        for steps in ODE_STEPS_MID_RK4:
            label = f"ODE {path} midpoint (steps={steps})"
            group = "ode_linear_midpoint4" if (path == "linear" and steps == 4) else None
            arms.append((label, "ode_transport",
                         dict(MC_samples=MC_DEFAULT, ode_steps=steps, path=path,
                              ode_solver="midpoint"), group))
        for steps in ODE_STEPS_MID_RK4:
            arms.append((f"ODE {path} rk4 (steps={steps})", "ode_transport",
                         dict(MC_samples=MC_DEFAULT, ode_steps=steps, path=path,
                              ode_solver="rk4"), None))
        arms.append((f"ODE {path} dopri5", "ode_transport",
                     dict(MC_samples=MC_DEFAULT, path=path, ode_solver="dopri5",
                          rtol=1e-5, atol=1e-7), None))

    return arms


MAIN_ARMS = make_estimator_kwargs()

# ── 2.4 MC_samples ablation: only the NEW (group, mc) points, existing
#      MC=256 (or MC=1024 for exact_marg) arms above are reused, not rerun. ────
MC_ABLATION_VALUES = [64, 256, 1024]
MC_ABLATION_GROUPS = {
    "score_function":       ("score_function", dict(baseline="none")),
    "score_function_rloo":  ("score_function", dict(baseline="rloo")),
    "gs_soft_tau0.5":       ("gumbel_softmax", dict(temperature=0.5, anneal_rate=1.0,
                                                     min_temperature=1e-6, mode="soft")),
    "st_tau0.5":            ("gumbel_softmax", dict(temperature=0.5, anneal_rate=1.0,
                                                     min_temperature=1e-6, mode="straight_through")),
    "ode_linear_midpoint4": ("ode_transport", dict(ode_steps=4, path="linear",
                                                    ode_solver="midpoint")),
    "exact_marg":           ("marginal_estimator", dict()),
}
# MC values already covered by MAIN_ARMS for each group (don't rerun):
MC_ALREADY_COVERED = {
    "score_function": {256}, "score_function_rloo": {256}, "gs_soft_tau0.5": {256},
    "st_tau0.5": {256}, "ode_linear_midpoint4": {256}, "exact_marg": {1024},
}


def make_ablation_arms():
    arms = []
    for group, (est_name, kwargs) in MC_ABLATION_GROUPS.items():
        for mc in MC_ABLATION_VALUES:
            if mc in MC_ALREADY_COVERED[group]:
                continue
            label = f"{group} (MC={mc})"
            arms.append((label, est_name, dict(kwargs, MC_samples=mc), group))
    return arms


ABLATION_ARMS = make_ablation_arms()

if __name__ == "__main__":
    print(f"MAIN_ARMS: {len(MAIN_ARMS)}")
    for label, est, kwargs, group in MAIN_ARMS:
        print(f"  {label:45s} {est:20s} {kwargs}")
    print(f"\nABLATION_ARMS (new points only): {len(ABLATION_ARMS)}")
    for label, est, kwargs, group in ABLATION_ARMS:
        print(f"  {label:35s} {est:20s} {kwargs}")
    print(f"\nTotal distinct arms: {len(MAIN_ARMS) + len(ABLATION_ARMS)}")
