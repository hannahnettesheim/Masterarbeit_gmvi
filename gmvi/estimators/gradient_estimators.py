"""
Gradient estimators for GMM-VI — corrected.

Bugs fixed:
  1. [Gumbel-Softmax] z_soft = Σ_k w̃_k T_k(x_k) lies in low-density
     inter-component regions of the GMM → log q(z_soft) ≪ 0 → ELBO explodes.
     Fix: do NOT evaluate log q at z_soft. Instead use the *mixture density*
     directly from the component weights and per-component log-probs, which
     gives a proper lower bound even for the relaxed sample.
     Alternatively (and more correctly): use the IWAE-style bound or simply
     only use z_soft to drive gradients and evaluate the ELBO at z_hard.
     We take the cleaner route: evaluate ELBO on z_hard in all estimators
     (unbiased objective), and use the respective method only for gradients.

  2. [Score Function] The loss had an incorrect extra `entropy_loss = log_q.mean()`
     term that conflicted with and nearly cancelled the REINFORCE gradient.
     The correct objective is simply: loss = -E[(elbo - b) * log_q].
     Note: the entropy is already implicitly maximised via the -log q in elbo.

  3. [Straight-Through] Was partially affected by bug 1 via z_soft path.
     Now fixed by ensuring gradient proxy z_soft is only used for .backward(),
     not for ELBO evaluation.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, Literal, Tuple, Optional
from torch.quasirandom import SobolEngine
from torchdiffeq import odeint_adjoint, odeint

import torch.distributions as dist

from gmvi.models.generalized_mixture import GeneralizedMixture

SampleMode = Literal["mc", "qmc", "rqmc"]


def _qmc_normal_sample(n: int, d: int, device, scramble: bool) -> Tensor:
    """Low-discrepancy normal samples via Sobol sequence + inverse-normal CDF."""
    engine = SobolEngine(dimension=d, scramble=scramble)
    u = engine.draw(n).to(device).clamp(1e-6, 1 - 1e-6)
    return torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5)


def _ref_sample(n: int, model: "GeneralizedMixture", mode: SampleMode) -> Tensor:
    """Draw n reference samples using the requested sampling mode."""
    if mode == "mc":
        return model.reference.sample(n)
    return _qmc_normal_sample(n, model.D, model.log_weights.device, scramble=(mode == "rqmc"))


# ─── Utility ───────────────────────────────────────────────────────────────────

def _elbo_samples(
    model: GeneralizedMixture,
    log_target: Callable[[Tensor], Tensor],
    z: Tensor,
) -> Tensor:
    """
    Per-sample ELBO = log p(z) - log q(z).
    z must be a genuine sample from q (not a soft convex mixture).
    z: (N, D) -> (N,)
    """
    return log_target(z) - model.log_prob(z)


def _sample_hard(model: GeneralizedMixture, n: int, gumbel: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Hard ancestral sample using pre-drawn Gumbel noise.
    Returns z_hard (N, D) and k_hard (N,).
    Differentiable w.r.t. component parameters (a_k, A_k) but NOT log_weights.
    """
    K, D = model.K, model.D
    device = model.log_weights.device

    with torch.no_grad():
        k_hard = (model.log_weights + gumbel).argmax(dim=-1)  # (N,)

    x_per_comp = torch.stack(
        [model.reference.sample(n) for _ in range(K)], dim=1
    )  # (N, K, D)

    z_hard = torch.zeros(n, D, device=device)
    for i in range(K):
        mask = (k_hard == i)
        if mask.any():
            z_hard[mask] = model.components[i].forward(x_per_comp[mask, i, :])

    return z_hard, k_hard, x_per_comp


# ─── 1. Score Function (REINFORCE) ─────────────────────────────────────────────

class ScoreFunctionEstimator:
    """
    REINFORCE / log-derivative trick.

    ∇_θ ELBO = E_q[(log p(z) - log q(z)) * ∇_θ log q(z)]

    The loss whose .backward() gives this gradient is:
        L = -E[(elbo - b) * log_q(z)]
    where b is a baseline (control variate) and z ~ q is detached.

    Note: the -log q term inside elbo already drives entropy maximisation;
    there is no need for a separate entropy loss term.

    Args:
        n_samples:  MC samples per gradient estimate
        baseline:   control variate ('none' | 'mean' | 'rloo')
    """
    name = "score_function"

    def __init__(self, MC_samples: int = 64, baseline: str = "none", sample_mode: SampleMode = "mc"):
        self.MC_samples = MC_samples
        assert baseline in ("none", "mean", "rloo")
        self.baseline = baseline
        self.sample_mode = sample_mode

    def loss(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        N = self.MC_samples

        # Sample from q — no gradient through sampling
        if self.sample_mode == "mc":
            z, _ = model.sample(N)
            z = z.detach()
        else:
            with torch.no_grad():
                k = dist.Categorical(probs=model.weights).sample((N,))
                x = _ref_sample(N, model, self.sample_mode)
                z = torch.zeros(N, model.D, device=x.device)
                for i, comp in enumerate(model.components):
                    mask = (k == i)
                    if mask.any():
                        z[mask] = comp.forward(x[mask])

        # ELBO values (detached — used as signal, not differentiated)
        with torch.no_grad():
            elbo = _elbo_samples(model, log_target, z)  # (N,)

        # log q(z; θ) — this carries gradients w.r.t. θ
        log_q = model.log_prob(z)  # (N,)

        # Control variate
        if self.baseline == "none":
            b = 0.0
        elif self.baseline == "mean":
            b = elbo.mean()
        elif self.baseline == "rloo":
            # Leave-one-out: b_i = mean over all j ≠ i
            b = (elbo.sum() - elbo) / (N - 1)

        # REINFORCE loss: minimise -E[(elbo - b) * log q(z)]
        # Gradient of this = -E[(elbo - b) * ∇_θ log q(z)] = -∇_θ ELBO
        loss = -((elbo - b) * log_q).mean()

        return loss, {
            "elbo": elbo.mean().item(),
            "elbo_std": elbo.std().item(),
            "log_q_mean": log_q.mean().item(),
        }


# ─── 2. Gumbel-Softmax Estimator ───────────────────────────────────────────────

class GumbelSoftmaxEstimator:
    """
    Gumbel-Softmax (Concrete) relaxation.

    The key insight for correctness: we cannot evaluate log q at z_soft because
    z_soft = Σ_k w̃_k T_k(x_k) is a convex combination — it lies between
    components, in a region where q assigns very low density. This makes
    -log q(z_soft) huge and the ELBO explodes.

    Correct approach: use z_soft ONLY to drive gradients. Evaluate the ELBO
    on z_hard (an exact GMM sample). The gradient flows through z_soft via
    the straight-through identity:

        z = z_soft + stop_grad(z_hard - z_soft)

    This makes GumbelSoftmaxEstimator equivalent to StraightThroughEstimator
    with the ELBO evaluated at the hard sample. The difference between the two
    is in the temperature schedule and intended use — GS is typically run
    warmer and relies more on the soft path for stability.

    An alternative used in some works: evaluate a *surrogate* ELBO using the
    soft density Σ_k w̃_k log q_k(z_soft), which avoids the collapse but
    introduces a different approximation. We prefer the STE-style fix as it
    keeps the ELBO evaluation unbiased.

    mode='straight_through' (default) is the estimator described above.
    mode='soft' is the textbook Gumbel-Softmax gradient estimator as
    formally defined in the thesis (def:gumbel_softmax_estimator): z = z_soft
    directly (no hard/detach mixing), ELBO evaluated at z_soft using the
    *true* mixture density log q_theta(z_soft) = model.log_prob(z_soft).
    This is the biased estimator the correctness note above works around --
    z_soft can land in a low-density valley between components, so
    -log q(z_soft) can be huge and the loss/gradient can spike. That
    instability is itself part of what distinguishes the two estimators'
    claims, not a bug to paper over, so mode='soft' does not apply the
    straight-through fix.

    Args:
        n_samples:       MC samples per estimate
        temperature:     Gumbel-Softmax temperature τ (lower = harder)
        anneal_rate:     multiplicative decay per step
        min_temperature: floor for annealing
        mode:            'straight_through' (hard fwd/soft bwd, ELBO at
                         z_hard) or 'soft' (plain relaxation, ELBO at z_soft
                         via the true mixture density)
    """
    name = "gumbel_softmax"

    def __init__(
        self,
        MC_samples: int = 64,
        temperature: float = 1.0,
        anneal_rate: float = 0.9995,
        min_temperature: float = 0.3,
        sample_mode: SampleMode = "mc",
        mode: Literal["straight_through", "soft"] = "straight_through",
    ):
        self.MC_samples = MC_samples
        self.temperature = temperature
        self.anneal_rate = anneal_rate
        self.min_temperature = min_temperature
        self.sample_mode = sample_mode
        assert mode in ("straight_through", "soft")
        self.mode = mode
        self._step = 0

    def _current_temp(self) -> float:
        return max(self.min_temperature,
                   self.temperature * (self.anneal_rate ** self._step))

    def loss(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        tau = self._current_temp()
        self._step += 1

        N, K = self.MC_samples, model.K

        # Shared Gumbel noise
        U = torch.rand(N, K, device=model.log_weights.device).clamp(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(U))

        # ── Soft path (carries gradients) ──
        soft_w = F.softmax((model.log_weights + gumbel) / tau, dim=-1)  # (N, K)

        x_per_comp = torch.stack(
            [_ref_sample(N, model, self.sample_mode) for _ in range(K)], dim=1
        )  # (N, K, D)

        z_per_comp = torch.stack(
            [model.components[k].forward(x_per_comp[:, k, :]) for k in range(K)],
            dim=1,
        )  # (N, K, D)

        z_soft = (soft_w.unsqueeze(-1) * z_per_comp).sum(dim=1)  # (N, D)

        if self.mode == "straight_through":
            hard_w = F.one_hot(soft_w.argmax(dim=-1), K).float()  # (N, K)
            z_hard = (hard_w.unsqueeze(-1) * z_per_comp).sum(dim=1)  # (N, D)
            z = z_soft + (z_hard - z_soft).detach()
            # ELBO evaluated at z_hard (value-wise); gradients flow through z_soft
            elbo = _elbo_samples(model, log_target, z)
        else:
            # mode='soft': textbook Gumbel-Softmax estimator (def:gumbel_softmax_estimator).
            # ELBO at z_soft using the TRUE mixture density -- biased, and can spike
            # when z_soft lands between components; that's the point of this arm.
            log_p = log_target(z_soft)
            log_q = model.log_prob(z_soft)
            elbo = log_p - log_q

        loss = -elbo.mean()

        return loss, {
            "elbo": elbo.mean().item(),
            "elbo_std": elbo.std().item(),
            "temperature": tau,
        }
    

class ExactMarginalizationEstimator:
    """
    Exact marginalization over the discrete mixture component.

    Instead of sampling k ~ Categorical(pi), we enumerate all components:

        ELBO = sum_k pi_k E_{x ~ p0}[
            log p(T_k(x)) - log q(T_k(x))
        ]

    This gives an unbiased Monte Carlo estimate of the original mixture ELBO,
    with the discrete expectation computed exactly. The remaining Monte Carlo
    noise comes only from the continuous base samples x.

    Cost: O(K) component forward passes per estimate, and potentially O(K^2)
    if evaluating log q(z) itself loops over all mixture components.
    """

    name = "exact_marginalization"

    def __init__(
        self,
        MC_samples: int = 64,
        sample_mode: SampleMode = "mc",
    ):
        self.MC_samples = MC_samples
        self.sample_mode = sample_mode

    def loss(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:

        N, K = self.MC_samples, model.K

        weights = torch.softmax(model.log_weights, dim=0)  # (K,)

        elbo_per_component = []

        for k in range(K):
            # Sample from the base distribution
            x_k = _ref_sample(N, model, self.sample_mode)  # (N, D)

            # Push through component k
            z_k = model.components[k].forward(x_k)  # (N, D)

            # Evaluate the true mixture ELBO at z_k:
            # log p(z_k) - log q_theta(z_k)
            elbo_k = _elbo_samples(model, log_target, z_k)  # (N,)

            elbo_per_component.append(elbo_k.mean())

        elbo_per_component = torch.stack(elbo_per_component)  # (K,)

        # Exact sum over mixture components
        elbo = torch.sum(weights * elbo_per_component)

        loss = -elbo

        return loss, {
            "elbo": elbo.item(),
            "component_elbos": elbo_per_component.detach().cpu(),
            "weights": weights.detach().cpu(),
        }

# ─── ODE Transport ────────────────────────────────────────────────────────────


# this is the simpler case: we are looking for diagonal ODES
def _velocity_diagonal(
    x: Tensor,              # (N, D)
    t: float,
    log_weights: Tensor,    # (K,) unnormalized
    means: Tensor,          # (K, D)
    diagonal_entries: Tensor,         # (K, D)  diagonal entries of A_j = diag(scales_j)
    ref_log_prob: Callable,
) -> Tensor:
    N, D = x.shape

    # A_{j,t} diagonal: t * sigma_j + (1 - t)
    Ajt_diag = t * diagonal_entries + (1.0 - t)                        # (K, D)

    # Pre-image: z_j = A_{j,t}^{-1} (x - t a_j)
    x_shift = x[:, None, :] - t * means[None, :, :]          # (N, K, D)
    z       = x_shift / Ajt_diag[None, :, :]                  # (N, K, D)

    # log rho_{j,t}(x) = log rho_ref(z_j) - log|det A_{j,t}|
    N, K, D = z.shape
    log_ref    = ref_log_prob(z.reshape(N*K, D)).reshape(N, K)  # (N, K)
    logdet_Ajt = torch.log(Ajt_diag).sum(dim=-1)             # (K,)
    log_rho_jt = log_ref - logdet_Ajt[None, :]               # (N, K)

    # Responsibilities: gamma_j = softmax_j(log w_j + log rho_{j,t})
    log_w  = torch.log_softmax(log_weights, dim=0)            # (K,)
    gamma  = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    # v_{j,t}(x) = a_j + (sigma_j - 1) * z_j
    v_jt = means[None, :, :] + (diagonal_entries - 1.0)[None, :, :] * z  # (N, K, D)

    return (gamma[:, :, None] * v_jt).sum(dim=1)             # (N, D)


def _velocity_general(
    x: Tensor,              # (N, D)
    t: float,
    log_weights: Tensor,    # (K,)
    components,             # list of AffineComponent
    ref_log_prob: Callable,
) -> Tensor:
    """
    General velocity field supporting tril and full A_j.
    Loops over components, uses matrix ops.
    """
    N, D   = x.shape
    K      = len(components)
    I      = torch.eye(D, device=x.device, dtype=x.dtype)

    log_rho_jt = torch.zeros(N, K, device=x.device, dtype=x.dtype)
    v_jt       = torch.zeros(N, K, D, device=x.device, dtype=x.dtype)

    for j, comp in enumerate(components):
        a_j  = comp.a                                         # (D,)
        A_j  = comp.get_A()                                   # (D, D)

        A_jt     = t * A_j + (1.0 - t) * I                   # (D, D)
        A_jt_inv = torch.linalg.inv(A_jt)                    # (D, D)
        _, logdet = torch.linalg.slogdet(A_jt)               # scalar

        z = (x - t * a_j) @ A_jt_inv.T                       # (N, D)

        log_rho_jt[:, j] = ref_log_prob(z) - logdet          # (N,)
        v_jt[:, j, :]    = a_j + z @ (A_j - I).T             # (N, D)

    log_w = torch.log_softmax(log_weights, dim=0)             # (K,)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    return (gamma[:, :, None] * v_jt).sum(dim=1)             # (N, D)


def _velocity_geometric_diagonal(
    x: Tensor,              # (N, D)
    t: float,
    log_weights: Tensor,    # (K,) unnormalized
    means: Tensor,          # (K, D)
    log_diag_stack: Tensor, # (K, D) — log_diag of each component
    ref_log_prob: Callable,
) -> Tensor:
    """
    Fast geometric velocity for diagonal A_j = diag(exp(s_j)).

    A_{j,t} = diag(exp(t s_j)),  v_{j,t}(x) = a_j + s_j ⊙ (x - t a_j).
    Fully differentiable w.r.t. log_diag parameters.
    """
    N, D = x.shape

    x_shift = x[:, None, :] - t * means[None, :, :]                        # (N, K, D)
    z       = x_shift * torch.exp(-t * log_diag_stack)[None, :, :]         # (N, K, D)

    N, K, D = z.shape
    log_ref    = ref_log_prob(z.reshape(N * K, D)).reshape(N, K)            # (N, K)
    logdet     = t * log_diag_stack.sum(dim=-1)                             # (K,)
    log_rho_jt = log_ref - logdet[None, :]                                  # (N, K)

    log_w  = torch.log_softmax(log_weights, dim=0)                          # (K,)
    gamma  = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)             # (N, K)

    v_jt = means[None, :, :] + log_diag_stack[None, :, :] * x_shift        # (N, K, D)

    return (gamma[:, :, None] * v_jt).sum(dim=1)                           # (N, D)


def _velocity_geometric(
    x: Tensor,              # (N, D)
    t: float,
    log_weights: Tensor,    # (K,)
    components,             # list of AffineComponent
    ref_log_prob: Callable,
) -> Tensor:
    """
    Geometric velocity field: A_{j,t} = A_j^t = exp(t log A_j).

    v_{j,t}(x) = a_j + log(A_j)(x - t a_j)
    rho_{j,t}  = |det A_j|^{-t} rho_ref(A_j^{-t}(x - t a_j))

    log(A_j) is exact and differentiable for all supported param_types
    (diagonal, eigenvaluedecomp, matrixexponential).
    """
    N, D = x.shape
    K    = len(components)

    log_rho_jt = torch.zeros(N, K, device=x.device, dtype=x.dtype)
    v_jt       = torch.zeros(N, K, D, device=x.device, dtype=x.dtype)

    for j, comp in enumerate(components):
        a_j     = comp.a                                              # (D,)
        log_A_j = comp.get_log_A()                                   # (D, D)

        A_jt_inv = torch.linalg.matrix_exp(-t * log_A_j)            # (D, D)
        logdet   = t * comp.log_abs_det()                            # scalar

        z = (x - t * a_j) @ A_jt_inv.T                              # (N, D)

        log_rho_jt[:, j] = ref_log_prob(z) - logdet
        v_jt[:, j, :]    = a_j + (x - t * a_j) @ log_A_j.T         # (N, D)

    log_w = torch.log_softmax(log_weights, dim=0)                    # (K,)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)       # (N, K)

    return (gamma[:, :, None] * v_jt).sum(dim=1)                    # (N, D)


def _velocity_general_cached(
    x: Tensor,              # (N, D)
    t: float,
    log_weights: Tensor,    # (K,)
    a_stack: Tensor,        # (K, D)   -- static, precomputed once per grad step
    A_stack: Tensor,        # (K, D, D) -- static, precomputed once per grad step
    ref_log_prob: Callable,
) -> Tensor:
    """
    Same math as _velocity_general, but takes the per-component a_j/A_j as
    precomputed (K, ...) stacks instead of re-deriving them from `components`
    (i.e. re-running get_A(), which for matrixexponential/eigenvaluedecomp is
    a matrix_exp / Cayley-map call) on every invocation. The caller is
    expected to compute a_stack/A_stack ONCE per gradient step -- A_j is
    t-independent, so recomputing it at every one of the ode_steps x
    stages-per-step calls to the velocity field is pure waste. Vectorized
    over K via batched matmul instead of a Python loop.
    """
    N, D = x.shape
    K = a_stack.shape[0]
    I = torch.eye(D, device=x.device, dtype=x.dtype)

    A_jt = t * A_stack + (1.0 - t) * I                     # (K, D, D)
    A_jt_inv = torch.linalg.inv(A_jt)                      # (K, D, D)
    logdet = torch.linalg.slogdet(A_jt)[1]                 # (K,)

    x_shift = x[:, None, :] - t * a_stack[None, :, :]       # (N, K, D)
    z = torch.einsum("nkd,ked->nke", x_shift, A_jt_inv)     # (N, K, D)

    log_ref = ref_log_prob(z.reshape(N * K, D)).reshape(N, K)
    log_rho_jt = log_ref - logdet[None, :]

    log_w = torch.log_softmax(log_weights, dim=0)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    v_jt = a_stack[None, :, :] + torch.einsum("nkd,ked->nke", z, A_stack - I)

    return (gamma[:, :, None] * v_jt).sum(dim=1)


def _velocity_geometric_cached(
    x: Tensor,              # (N, D)
    t: float,
    log_weights: Tensor,    # (K,)
    a_stack: Tensor,        # (K, D)    -- static
    log_A_stack: Tensor,    # (K, D, D) -- static, = log(A_j), precomputed once
    logdet_A_stack: Tensor, # (K,)      -- static, = log|det A_j|, precomputed once
    ref_log_prob: Callable,
) -> Tensor:
    """Cached/vectorized analogue of _velocity_geometric -- see
    _velocity_general_cached's docstring for the rationale. The
    t-dependent matrix_exp(-t * log_A_j) itself is irreducibly per-stage
    (it genuinely depends on t), but log_A_j and log|det A_j| no longer are."""
    N, D = x.shape
    K = a_stack.shape[0]

    A_jt_inv = torch.linalg.matrix_exp(-t * log_A_stack)   # (K, D, D)
    logdet = t * logdet_A_stack                              # (K,)

    x_shift = x[:, None, :] - t * a_stack[None, :, :]        # (N, K, D)
    z = torch.einsum("nkd,ked->nke", x_shift, A_jt_inv)      # (N, K, D)

    log_ref = ref_log_prob(z.reshape(N * K, D)).reshape(N, K)
    log_rho_jt = log_ref - logdet[None, :]

    log_w = torch.log_softmax(log_weights, dim=0)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)

    v_jt = a_stack[None, :, :] + torch.einsum("nkd,ked->nke", x_shift, log_A_stack)

    return (gamma[:, :, None] * v_jt).sum(dim=1)


def velocity(
    x: Tensor,
    t: float,
    model: GeneralizedMixture,
    path: Literal["linear", "geometric"] = "linear",
) -> Tensor:
    """
    Velocity field v_t(x). Dispatches to fast diagonal path when possible.

    path='linear':    A_{j,t} = t A_j + (1-t) I
    path='geometric': A_{j,t} = A_j^t = exp(t log A_j)
    """
    if path == "geometric":
        if model.param_type == "diagonal":
            log_diags = torch.stack([c.log_diag for c in model.components])
            return _velocity_geometric_diagonal(
                x, t, model.log_weights, model.means, log_diags,
                model.reference.log_prob,
            )
        else:
            return _velocity_geometric(
                x, t, model.log_weights, model.components,
                model.reference.log_prob,
            )
    else:
        if model.param_type == "diagonal":
            scales = torch.stack([torch.exp(c.log_diag) for c in model.components])
            means  = model.means
            return _velocity_diagonal(
                x, t, model.log_weights, means, scales,
                model.reference.log_prob,
            )
        else:
            return _velocity_general(
                x, t, model.log_weights, model.components,
                model.reference.log_prob,
            )


# ─── Integrators ───────────────────────────────────────────────────────────────

class _VelocityFunc(nn.Module):
    """Wraps velocity() in the (t, x) -> dx signature torchdiffeq expects.

    Precomputes the per-component static matrices (A_j for the linear path;
    log_A_j and log|det A_j| for the geometric path) ONCE here, since
    integrate_ode() constructs a fresh _VelocityFunc per gradient step and
    this instance's forward() is then called once per RK stage per
    ode_step (e.g. 4x per step for rk4, more for dopri5). A_j is
    t-independent, so recomputing get_A()/get_log_A() -- a matrix_exp or
    Cayley-map call per component -- at every stage was pure waste; see
    BASELINE_SWEEP.md section 0.1. The diagonal param_type path is already
    cheap (no matrix ops) and is left as before.
    """
    def __init__(self, model: GeneralizedMixture, path: Literal["linear", "geometric"] = "linear"):
        super().__init__()
        self.model = model
        self.path  = path
        self.diagonal = model.param_type == "diagonal"
        self.nfe = 0  # velocity-field evaluations so far; implementation-independent compute axis

        if not self.diagonal:
            self.a_stack = torch.stack([c.a for c in model.components])              # (K, D)
            if path == "geometric":
                self.log_A_stack = torch.stack([c.get_log_A() for c in model.components])      # (K, D, D)
                self.logdet_A_stack = torch.stack([c.log_abs_det() for c in model.components])  # (K,)
            else:
                self.A_stack = torch.stack([c.get_A() for c in model.components])    # (K, D, D)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        self.nfe += 1
        t_val = t.item()
        model = self.model

        if self.diagonal:
            if self.path == "geometric":
                log_diags = torch.stack([c.log_diag for c in model.components])
                return _velocity_geometric_diagonal(
                    x, t_val, model.log_weights, model.means, log_diags,
                    model.reference.log_prob,
                )
            else:
                scales = torch.stack([torch.exp(c.log_diag) for c in model.components])
                return _velocity_diagonal(
                    x, t_val, model.log_weights, model.means, scales,
                    model.reference.log_prob,
                )
        else:
            if self.path == "geometric":
                return _velocity_geometric_cached(
                    x, t_val, model.log_weights, self.a_stack, self.log_A_stack,
                    self.logdet_A_stack, model.reference.log_prob,
                )
            else:
                return _velocity_general_cached(
                    x, t_val, model.log_weights, self.a_stack, self.A_stack,
                    model.reference.log_prob,
                )


ADAPTIVE_SOLVERS = {"dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"}


def integrate_ode(
    x0: Tensor,
    model: GeneralizedMixture,
    ode_steps: int = 20,
    path: Literal["linear", "geometric"] = "linear",
    ode_solver: Literal["euler", "midpoint", "rk4", "dopri5"] = "rk4",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    return_nfe: bool = False,
):
    func = _VelocityFunc(model, path=path)
    if ode_solver in ADAPTIVE_SOLVERS:
        # Adaptive solvers pick their own internal step size; ode_steps doesn't
        # apply. Only the two endpoints are requested, tolerance controls cost.
        t_span = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
        x1 = odeint(func, x0, t_span, method=ode_solver, rtol=rtol, atol=atol)
    else:
        t_span = torch.linspace(0.0, 1.0, ode_steps + 1, device=x0.device, dtype=x0.dtype)
        x1 = odeint(func, x0, t_span, method=ode_solver)
    if return_nfe:
        return x1[-1], func.nfe
    return x1[-1]


# ─── Estimator ─────────────────────────────────────────────────────────────────

class ODETransportEstimator:
    """
    ODE transport estimator based on Theorem 5.1.

    Samples x1 = T(x0) by integrating the theorem's velocity field,
    then computes the ELBO directly:

        ELBO = E_{x0 ~ rho_ref}[ log p(x1) - log q(x1) ]

    This works for any A_j parameterization (diagonal, tril, full).
    Gradients flow through x1 back to (a_j, A_j, w_j).

    Args:
        n_samples:   MC samples per training step
        ode_steps:   number of RK4 steps for the ODE integrator (20 default, 40 for accuracy).
                     Distinct from the number of training steps in TrainConfig.n_steps.
    """
    name = "ode_transport"

    def __init__(
        self,
        MC_samples: int = 128,
        ode_steps: int = 20,
        use_adjoint: bool = True,
        sample_mode: SampleMode = "mc",
        path: Literal["linear", "geometric"] = "linear",
        ode_solver: Literal["euler", "midpoint", "rk4", "dopri5"] = "rk4",
        rtol: float = 1e-5,
        atol: float = 1e-7,
    ):
        self.MC_samples   = MC_samples
        self.ode_steps    = ode_steps
        self.use_adjoint  = use_adjoint
        self.sample_mode  = sample_mode
        self.path         = path
        self.ode_solver   = ode_solver
        self.rtol         = rtol
        self.atol         = atol

    def loss(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        N = self.MC_samples

        # Sample from reference and integrate ODE
        with torch.no_grad():
            x0 = _ref_sample(N, model, self.sample_mode)          # (N, D), no grad needed

        x1, nfe = integrate_ode(x0, model, ode_steps=self.ode_steps, path=self.path,
                                 ode_solver=self.ode_solver,
                                 rtol=self.rtol, atol=self.atol, return_nfe=True)  # (N, D)

        # ELBO: both terms evaluated at x1
        log_p = log_target(x1)                                    # (N,)
        log_q = model.log_prob(x1)                                # (N,), exact GMM density

        elbo = log_p - log_q                                      # (N,)
        loss = -elbo.mean()

        return loss, {
            "elbo":     elbo.mean().item(),
            "elbo_std": elbo.std().item(),
            "log_p":    log_p.mean().item(),
            "log_q":    log_q.mean().item(),
            "nfe":      nfe,
        }

    def transport(
        self,
        model: GeneralizedMixture,
        n_samples: int = 1000,
        ode_steps: Optional[int] = None,
        # deprecated — use ode_steps:
        n_steps: Optional[int] = None,
    ) -> Tensor:
        """Draw samples by integrating the ODE (no grad)."""
        if n_steps is not None:
            import warnings
            warnings.warn(
                f"transport(): n_steps={n_steps} is deprecated, use ode_steps=.",
                DeprecationWarning, stacklevel=2,
            )
            ode_steps = n_steps
        steps = ode_steps or self.ode_steps
        with torch.no_grad():
            x0 = model.reference.sample(n_samples)
            x1 = integrate_ode(x0, model, ode_steps=steps, path=self.path, ode_solver= self.ode_solver)
        return x1

# ─── Registry ──────────────────────────────────────────────────────────────────


def make_estimator(name: str, **kwargs):
    registry = {
        "score_function":   ScoreFunctionEstimator,
        "gumbel_softmax":   GumbelSoftmaxEstimator,
        "ode_transport":    ODETransportEstimator,
        "marginal_estimator": ExactMarginalizationEstimator,
    }
    if name not in registry:
        raise ValueError(f"Unknown estimator '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)