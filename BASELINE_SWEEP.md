# Banana baseline sweep — finalized specification

Supersedes the 34-config sweep of 2026-08-18. That run is still useful as a pilot, but its
headline numbers are not publishable for three reasons, all fixed below: nothing was converged
(everything sat at 2–5× the attainable floor), `clip_grad_norm_(1.0)` may have penalised
estimators unequally, and the `ms_per_step` column was measured under sharded execution with
divided thread counts.

---

## 0. Two things to fix before running

**0.1 — Kill the per-stage linear algebra.** `_velocity_general` calls `get_A()`, `linalg.inv`
and `linalg.slogdet` *per RK stage per component*; `_velocity_geometric` calls `matrix_exp`
per stage per component. With `eigenvaluedecomp` all of these have closed forms on the cached
$Q$, $s$:

```
geometric:  A_it^{-1} = Q diag(exp(-t*s)) Q^T      logdet A_it = t * sum(s)
linear:     A_it^{-1} = Q diag(1/(t*e^s + 1 - t)) Q^T
            logdet A_it = sum(log(t*e^s + 1 - t))
```

Compute `_get_Q()` **once per gradient step**, not once per stage. Until this is done the ODE
wall-clock numbers are measuring the implementation, not the method.

**0.2 — Settle the Straight-Through / Gumbel-Softmax labelling.** Every arm labelled
"Straight-Through" in the pilot was built with `make_estimator("gumbel_softmax", ...)`. Confirm
which estimator actually runs. Both belong in the sweep, and they are different claims.

---

## 1. Shared training protocol

| setting | value | why |
|---|---|---|
| target | banana, `Normal(x) * Normal(y|x)` | exactly normalized, so `KL(q‖p) = -ELBO` with no quadrature |
| dimension | 2 | |
| components | $k = 5$ | |
| parametrization | **`matrixexponential`** | global diffeomorphism $\mathrm{Sym}(n)\to S^{++}$; `eigenvaluedecomp` is non-injective and its Cayley map is not onto $SO(n)$ |
| reference | $\mathcal N(0, I_2)$ | $C^\infty$, so RK4's order conditions are satisfied |
| init | `init_scale=2.5` | unchanged from pilot |
| optimizer | Adam, lr $5\times10^{-3}$, cosine over the full 10k | |
| **gradient clipping** | **off** | the main confound in the pilot |
| steps | **10 000** | the pilot never converged; the floor run needed 10k |
| checkpoints | 200, 500, 1000, 2000, 5000, 10000 | log-spaced |
| seeds | **5** for the sweep, **10** for the final headline arms | 3 could not separate 0.055 from 0.059 |
| eval | `eval_elbo`, 8192 samples, estimator-agnostic | unchanged — this part was right |

**Label the x-axis "training step", not "training budget".** With a cosine schedule over 10k
steps, the checkpoint at 200 is an intermediate state of a 10k run, not a converged 200-step
run. It is a learning curve; call it one.

**Log pre-clip gradient norms** (mean and 95th percentile over the run) for every arm even with
clipping off. If the ODE gradients are systematically larger, that is itself a finding and it
explains the pilot.

---

## 2. Arms

### 2.1 References (not competitors)

- **Exact Marginalization**, `MC_samples=1024`
- **Exact Marginalization**, `MC_samples=4096`, 20 000 steps — the **floor**, run once with 3
  seeds. Report whether it is still falling between 10k and 20k; if it is, quote it as an upper
  bound on the floor, not the floor.

### 2.2 Baseline estimators

- Score Function
- Score Function **+ leave-one-out control variate** — without this the baseline is a straw man
- Gumbel–Softmax, fixed $\tau \in \{2.0, 1.0, 0.5, 0.3, 0.1\}$
- Gumbel–Softmax, annealed (`anneal_rate=0.9995`, `min_temperature=0.3`)
- Straight-Through, same six settings

### 2.3 ODE transport

Pruned from the pilot, which showed midpoint and RK4 are already at the estimator's accuracy
floor at 4 steps, and that euler is the only discretization-limited solver.

| path | solver | steps |
|---|---|---|
| linear, geometric | euler | 4, 8, 16, 32 |
| linear, geometric | midpoint | 4, 8 |
| linear, geometric | rk4 | 4, 8 |
| linear, geometric | dopri5 (rtol 1e-5, atol 1e-7) | — |

18 ODE arms. Dropped: midpoint/rk4 at 16 and 32 steps — the pilot showed them indistinguishable
from 4 steps. `dopri5` is retained as the *exact-transport reference*, not as a competitor.

### 2.4 Sample-count axis

`MC_samples` $\in \{64, 256, 1024\}$ for a **subset**: Score Function, Score Function + CV,
Gumbel–Softmax $\tau=0.5$, ODE linear midpoint-4, Exact Marginalization.

This axis is not optional. The pilot fixed `MC_samples=256`, and at 2-D the cost is
launch-bound — 4096 samples cost only 1.3× the time of 256. Fixing it high hides exactly the
variance differences that separate these estimators.

**Total: roughly 50 arms × 5 seeds.**

---

## 2.5 Measuring the expense ratio $\rho = C_v / C_p$

The central claim of the chapter is that the ODE estimator wins when the target is expensive
relative to the velocity field. That ratio is directly measurable from a timing line.

Sweep $N$ for one ODE arm at fixed $M$ and fit

$$\text{ms per step} \;=\; a + bN$$

Then $b = M\,C_v$ (one stage for the whole batch) and $a \approx M\,C_p$ plus fixed overhead, so

$$\boxed{\;\rho \;=\; C_v / C_p \;\approx\; b / a\;}$$

— the slope-to-intercept ratio of the timing line. No separate instrumentation needed.

From the pilot (29 / 58 / 118 / 227 ms at $N = 4/8/16/32$): $b \approx 7.1$, $a \approx 0.7$, so
$\rho \approx 10$. **On the banana the velocity field is an order of magnitude more expensive
than the target.** Caching improves $b$ but cannot change the sign of the conclusion: the banana
target is two Gaussian log-densities and will never be the expensive one.

Consequence for the write-up:

- **Banana → report on the target-evaluation axis.** It is the controlled setting where
  $\sigma^2_{\mathrm{SF}}$, $\sigma^2_{\mathrm{ODE}}$ and hence the crossover threshold
  $\rho^\star = \frac{1}{N}\left(\sigma^2_{\mathrm{SF}}/\sigma^2_{\mathrm{ODE}} - 1\right)$
  are measured. Wall-clock on the banana is *expected* to look bad and should be reported as
  such, with $\rho \approx 10$ quoted so the reader sees why.
- **Lotka–Volterra → report on wall-clock.** There $C_p$ is a full predator–prey ODE solve, so
  $\rho \ll 1$ genuinely holds and the wall-clock advantage is real rather than counted.

Report $\rho$ for every target used. It is the single number that tells a reader whether the
method applies to their problem.

## 3. Timing

Re-measure `ms_per_step` **solo**, one process, fixed thread count, for one seed of each arm.
Do not reuse timings from sharded runs. Report **velocity-field evaluations per gradient step**
as the primary compute axis and wall-clock as secondary — NFE is implementation-independent and
survives the fix in §0.1.

---

## 4. The separate fixed-$\theta$ experiment

This is not part of the training sweep and costs almost nothing — no training at all.

Pick three parameter values per seed: the initialization, a mid-trajectory point from a
converged run, and the near-optimal point. At each, draw $M = 200$ replicate gradients from
every estimator and compute

- $\widehat{\mathrm{bias}} = \|\bar g - g^\star\|$ against $g^\star$ = Exact Marginalization at
  `MC_samples=16384`
- $\widehat{\mathrm{var}} = \operatorname{tr}\widehat{\operatorname{cov}}(g)$
- **the bias restricted to the weight coordinates $w$ and to $(a, A)$ separately**

The last one is the experiment that tests the thesis's actual claim — that the pathwise failure
is confined to the weights. Nothing in the training sweep tests it.

Report bias with its own Monte Carlo error bar ($\propto M^{-1/2}$); anything below the bar is
"indistinguishable from zero", not "zero".

---

## 5. Figures this produces

1. **Learning curves**, KL vs training step, median over seeds with interquartile band, one
   panel per estimator family, with the floor as a horizontal line.
2. **Bias–variance scatter** at fixed $\theta$, log axes, with the Gumbel–Softmax temperature
   sweep tracing a curve.
3. **Weight-block vs shape-block bias**, the thesis's central claim.
4. **KL vs NFE**, the honest compute comparison.
5. **Solver/step ablation**, KL vs steps per solver and path. The known result to reproduce:
   euler-4 linear diverges (KL 9.3) while euler-4 geometric survives (KL 0.95); everything at
   midpoint-4 and above is indistinguishable.
6. **`MC_samples` ablation**, KL vs sample count per estimator.

---

## 6. What the pilot already established

Keep these; they held up and only need rerunning under the corrected protocol.

- Gumbel–Softmax bias is monotone in $\tau$: 0.109, 0.107, 0.103, 0.084, 0.061 for
  $\tau = 2.0 \ldots 0.1$, with the seed spread widening sharply at $\tau = 0.1$
  ($[0.042, 0.103]$ versus $[0.087, 0.115]$ at $\tau=2.0$). Clean bias–variance trade-off.
- The annealed schedule lands at 0.104 — **no better than fixed $\tau = 0.5$**.
- Midpoint at 4 steps already matches rk4 at 32 steps (0.0586 vs 0.0569) at 1/15 the cost.
- The linear and geometric paths differ decisively **only** at euler-4.
- ODE cost is exactly linear in stage count, confirming per-stage overhead dominates.
