"""
Sanity checks for LotkaVolterraPosterior. Run this before wiring it into the sweep.

    python test_lotka_volterra.py
"""

import time
import torch

from gmvi.targets.lotka_volterra import LotkaVolterraPosterior as LV


def main() -> None:
    torch.manual_seed(0)
    t = LV(n_obs=20, t_max=20.0, ode_steps=200)

    print(f"dim               {t.dim}")
    print(f"data shape        {tuple(t.log_y.shape)}")
    y = torch.exp(t.log_y)
    print(f"observed u range  [{y[:,0].min():.2f}, {y[:,0].max():.2f}]")
    print(f"observed v range  [{y[:,1].min():.2f}, {y[:,1].max():.2f}]")

    # 1. The truth should be far more probable than typical prior draws.
    lp_true = t.log_prob(t.x_true)
    z = t.prior.sample((1024,))
    lp = t.log_prob(z)
    print(f"\nlog_prob(x_true)  {lp_true.item():>12.2f}")
    print(f"prior draws       median {lp.median():.1f}   max {lp.max():.1f}")
    print(f"finite fraction   {torch.isfinite(lp).float().mean():.4f}")
    assert torch.isfinite(lp_true), "log_prob at the truth is not finite"
    assert lp_true > lp.max(), "truth is not the most probable point among prior draws"

    # 2. Gradients must be finite at the truth.
    x = t.x_true.clone().requires_grad_(True)
    t.log_prob(x).backward()
    print(f"\ngrad at truth     {[round(g, 2) for g in x.grad.tolist()]}")
    assert torch.isfinite(x.grad).all(), "non-finite gradient at the truth"

    # 3. Extreme inputs -- which the variational family WILL propose early in training --
    #    must give finite values and finite gradients, not NaN. Tested well past anything
    #    training should reach, since one NaN poisons the whole batch gradient.
    #    x4 is already well beyond anything training reaches (init_scale=2.5, prior sds
    #    of 0.5-1.0) and must be perfectly clean. x8 and x16 are stress tests: report the
    #    rate, but do not demand perfection -- explosive parameter regions have genuinely
    #    exponential sensitivity through a 200-step solver, and the durable protection is
    #    the guard in the training loop (see below), not more clamping.
    #    Only x4 is asserted. Beyond that the rate is reported, not required: sigma is no
    #    longer clamped at a model-meaningful bound, so at sigma ~ e^-30 a mismatched
    #    observation gives (delta/sigma)^2 ~ 1e28 and the 200-step solver chain can
    #    genuinely overflow. That is the honest cost of not baking a modelling assumption
    #    into a numerical guard, and it is handled by the training-loop guard below.
    for scale, required in ((4, 1.0), (8, None), (16, None)):
        bad = (torch.randn(512, 8) * scale).requires_grad_(True)
        lpb = t.log_prob(bad)
        lpb.sum().backward()
        fv = torch.isfinite(lpb).float().mean().item()
        fg = torch.isfinite(bad.grad).float().mean().item()
        tag = "asserted" if required is not None else "reported"
        print(f"\nextreme (x{scale:<2})      finite values {fv:.4f}   "
              f"finite grads {fg:.4f}   [{tag}]")
        if required is not None:
            assert fv == 1.0, f"non-finite log_prob at scale {scale}"
            assert fg >= required, f"finite-gradient rate {fg:.4f} at scale {scale}"

    # 3b. The field clamp must not bind anywhere that carries posterior mass, or it would
    #     be distorting the model rather than just guarding the numerics.
    from gmvi.targets.lotka_volterra import _FIELD_CLAMP
    with torch.no_grad():
        rates = torch.exp(t.x_true[None, 0:4])
        s = t.x_true[None, 4:6]
        worst = 0.0
        for _ in range(t.ode_steps):
            f = t._field(s, rates)
            worst = max(worst, f.abs().max().item())
            s = s + t.h * f
    print(f"\nfield at truth    max |ds/dt| = {worst:.3f}   clamp = {_FIELD_CLAMP:.0f}"
          f"   headroom {_FIELD_CLAMP/worst:.0f}x")
    assert worst < 0.01 * _FIELD_CLAMP, "field clamp is close to binding at the truth"


    # 3c. Regression: the target must remain proper outside the numerical guard region.
    #     The Gaussian prior is evaluated at the raw x, so moving farther into a tail must
    #     decrease log_prob rather than hitting a constant plateau.
    x10 = t.prior.loc.clone()
    x20 = t.prior.loc.clone()
    x10[0] += 10.0 * t.prior.scale[0]
    x20[0] += 20.0 * t.prior.scale[0]
    lp10 = t.log_prob(x10)
    lp20 = t.log_prob(x20)
    print(f"\nproper-tail check  logp(10sd)={lp10.item():.1f}  logp(20sd)={lp20.item():.1f}")
    assert lp20 < lp10, "log_prob has a flat tail; raw parameters are probably being clamped"

    # 3d. Regression: custom observation times must actually control where the forward
    #     model is evaluated.  They need not lie on the base RK4 grid.
    custom_times = torch.tensor([0.37, 1.11, 2.05, 3.14, 4.2])
    probe = LV(n_obs=5, t_max=5.0, ode_steps=47, obs_times=custom_times, seed=0)
    uniform = LV(n_obs=5, t_max=5.0, ode_steps=47, seed=0)
    with torch.no_grad():
        traj_custom = probe._solve(probe.x_true[None])
        traj_uniform = uniform._solve(uniform.x_true[None])
    print(f"custom times      {probe.obs_times.tolist()}")
    assert traj_custom.shape == (1, 5, 2)
    assert not torch.allclose(traj_custom, traj_uniform), "obs_times are being ignored"

    # 3d. No guard may bind in the region carrying posterior mass. A guard that never
    #     activates there provably is not part of the model there -- which is the claim
    #     the thesis needs to be able to make about the numerical forward model G-hat.
    from gmvi.targets.lotka_volterra import (
        _LOG_STATE_CLAMP, _LOG_RATE_CLAMP, _LOG_SIGMA_CLAMP,
    )
    with torch.no_grad():
        # Plausible parameters: the truth, jittered by the prior scale.
        xs = t.x_true + torch.randn(2048, 8) * t.prior.scale
        margins = {
            "log rate ": (xs[:, 0:4].abs().max().item(), _LOG_RATE_CLAMP),
            "log u0,v0": (xs[:, 4:6].abs().max().item(), _LOG_STATE_CLAMP),
            "log sigma": (xs[:, 6:8].abs().max().item(), _LOG_SIGMA_CLAMP),
        }
    print()
    for label, (worst, bound) in margins.items():
        binds = worst >= bound
        print(f"guard {label}   max |x| = {worst:6.2f}   bound = {bound:6.1f}"
              f"   headroom {bound/max(worst, 1e-9):5.1f}x"
              f"{'   <-- BINDS' if binds else ''}")
        # The claim to protect is 'this guard never activates where the posterior has
        # mass', so assert exactly that rather than an arbitrary headroom factor. The
        # headroom is printed as information: if it drops near 1, widen the bound, since
        # a binding guard zeroes the gradient in a region the optimizer still needs.
        assert not binds, f"{label} guard activates on plausible parameters"

    # 4. Solver accuracy: the trajectory should be converged at ode_steps=200.
    coarse = LV(n_obs=20, t_max=20.0, ode_steps=200, seed=0)
    fine = LV(n_obs=20, t_max=20.0, ode_steps=2000, seed=0)
    with torch.no_grad():
        d = (coarse._solve(t.x_true[None]) - fine._solve(t.x_true[None])).abs().max()
    print(f"\nsolver check      max |log-traj(200) - log-traj(2000)| = {d:.2e}")
    if d > 1e-4:
        print("  WARNING: increase ode_steps, the forward model is not converged")

    # 5. C_p, the quantity the cost model needs.
    cp = t.time_per_eval(batch=256, repeats=10)
    print(f"\nC_p forward       {cp*1e6:.1f} us per log_prob evaluation")
    cpg = t.time_per_grad_eval(batch=64, repeats=3)
    print(f"C_p + grad        {cpg*1e6:.1f} us per value+gradient evaluation")
    print(f"                  (the latter is closer to the VI training cost)")

    # 6. Eval counter works.
    t.reset_eval_counter()
    t.log_prob(t.prior.sample((37,)))
    assert t.n_evals == 37, f"counter is {t.n_evals}, expected 37"
    print(f"\neval counter      ok")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
