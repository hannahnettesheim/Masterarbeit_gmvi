"""
GeneralizedMixture: variational distribution q(x) = Σ_i w_i * q_i(x)

where q_i(x) = |det A_i|^{-1} * q_ref(A_i^{-1}(x - a_i))
and T_i(x) = a_i + A_i * x pushes Q_ref to the i-th component.

This strictly generalizes GaussianMixture:
  GaussianMixture = GeneralizedMixture(ref=N(0,I), param_type='diagonal')

Key properties:
  - log_prob(z):      exact mixture log density (logsumexp over components)
  - sample(n):        sample by (1) drawing component k ~ Cat(w), (2) z = T_k(x), x ~ Q_ref
  - rsample(n):       reparameterized: component selection is hard (non-differentiable),
                      Gaussian part is differentiable (as in original GaussianMixture)

The GaussianMixture class is preserved as a thin wrapper for backward compatibility.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional
import torch.distributions as dist

from gmvi.models.affine_component import AffineComponent, AffineParamType
from gmvi.models.reference_distributions import ReferenceDistribution, StandardNormal


class GeneralizedMixture(nn.Module):
    """
    Generalized k-component mixture with affine-pushed reference distribution.

    q(x) = Σ_{i=1}^k  w_i * |det A_i|^{-1} * q_ref(A_i^{-1}(x - a_i))

    Args:
        n_components:   k, number of mixture components
        dim:            n, dimensionality of the latent space
        reference:      ReferenceDistribution (Q_ref)
        param_type:     how to parameterize A_i ('diagonal' | 'eigenvaluedecomp' | 'matrixexponential')
        init_scale:     initial spread of component means/matrices
    """

    def __init__(
        self,
        n_components: int,
        dim: int,
        reference: Optional[ReferenceDistribution] = None,
        param_type: AffineParamType = "diagonal",
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.K = n_components
        self.D = dim
        self.param_type = param_type

        # Reference distribution
        self.reference = reference if reference is not None else StandardNormal(dim)
        assert self.reference.dim == dim, \
            f"Reference dim {self.reference.dim} != latent dim {dim}"

        # Mixture weights (unnormalized log-weights)
        self.log_weights = nn.Parameter(torch.zeros(n_components))

        # One learnable affine component per mixture element
        self.components = nn.ModuleList([
            AffineComponent(
                dim=dim,
                param_type=param_type,
                init_scale=init_scale,
            )
            for _ in range(n_components)
        ])

        # Spread initial shifts so components start apart
        with torch.no_grad():
            for i, comp in enumerate(self.components):
                comp.a.data = torch.randn(dim) * init_scale

    # ── Weight access ────────────────────────────────────────────────────────

    @property
    def weights(self) -> Tensor:
        """Normalized mixture weights. Shape: (K,)"""
        return torch.softmax(self.log_weights, dim=0)

    # ── Density ──────────────────────────────────────────────────────────────

    def component_log_probs(self, z: Tensor) -> Tensor:
        """
        Compute log q_i(z) for all i simultaneously.
        """
        N = z.shape[0]
        log_probs = torch.zeros(N, self.K, device=z.device)
        for i, comp in enumerate(self.components):
            log_probs[:, i] = comp.component_log_prob(z, self.reference.log_prob)
        return log_probs  # (N, K)

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Log mixture density: log q(z) = logsumexp_i [log w_i + log q_i(z)]
        """
        log_comp = self.component_log_probs(z)                  # (N, K)
        log_w = torch.log_softmax(self.log_weights, dim=0)      # (K,)
        return torch.logsumexp(log_w + log_comp, dim=1)         # (N,)

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(self, n: int) -> Tuple[Tensor, Tensor]:
        """
        Samples from the mixture
        """
        with torch.no_grad():
            k = dist.Categorical(probs=self.weights).sample((n,))  # (N,)
            x = self.reference.sample(n)                           # (N, D)

            z = torch.zeros(n, self.D, device=x.device)
            for i, comp in enumerate(self.components):
                mask = (k == i)
                if mask.any():
                    z[mask] = comp.forward(x[mask])

        return z, k

    def rsample(self, n: int) -> Tuple[Tensor, Tensor]:
        """
        Reparameterized sample. Differentiable w.r.t. a_i and A_i.
        Component selection k is hard (straight-through or score function
        handles the gradient through log_weights).

        Returns:
            z: (N, D) — differentiable w.r.t. component parameters
            k: (N,)   — hard component assignments (detached)
        """
        k = dist.Categorical(probs=self.weights.detach()).sample((n,))
        x = self.reference.sample(n)  # (N, D) — from ref, not differentiable here

        z = torch.zeros(n, self.D, device=self.log_weights.device)
        for i, comp in enumerate(self.components):
            mask = (k == i)
            if mask.any():
                z[mask] = comp.forward(x[mask])  # differentiable w.r.t. a_i, A_i

        return z, k

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def means(self) -> Tensor:
        """Component shifts a_i stacked. Shape: (K, D). For visualization."""
        return torch.stack([c.a for c in self.components])

    @property
    def log_abs_dets(self) -> Tensor:
        """log|det A_i| for each component. Shape: (K,)"""
        return torch.stack([c.log_abs_det() for c in self.components])

    def get_matrices(self) -> List[Tensor]:
        """Return list of A_i matrices. Each (D, D)."""
        return [c.get_A() for c in self.components]

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, D={self.D}, "
            f"ref={self.reference.__class__.__name__}, "
            f"param_type={self.param_type}"
        )


# ── Backward-compatible GaussianMixture wrapper ──────────────────────────────

class GaussianMixture(GeneralizedMixture):
    """
    Classical Gaussian mixture: GeneralizedMixture with N(0,I) reference
    and diagonal A_i = diag(sigma_i).

    Preserved for full backward compatibility with existing estimator code.
    Additional properties expose means/scales in the original format.
    """

    def __init__(self, n_components: int, latent_dim: int, init_scale: float = 1.0):
        super().__init__(
            n_components=n_components,
            dim=latent_dim,
            reference=StandardNormal(latent_dim),
            param_type="diagonal",
            init_scale=init_scale,
        )

    @property
    def scales(self) -> Tensor:
        """Diagonal std devs for each component. Shape: (K, D)."""
        return torch.stack([
            torch.exp(c.log_diag) for c in self.components
        ])

    @property
    def log_scales(self) -> Tensor:
        """Log std devs. Shape: (K, D)."""
        return torch.stack([c.log_diag for c in self.components])