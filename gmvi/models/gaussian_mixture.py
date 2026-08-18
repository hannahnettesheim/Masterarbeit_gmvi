"""
Gaussian Mixture Model as variational distribution q(z|x).

Parameters are:
  - log_weights: (K,)         unnormalized log mixture weights
  - means:       (K, D)       component means
  - log_scales:  (K, D)       log of component std devs (diagonal covariance)
"""

import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor
from typing import Tuple


class GaussianMixture(nn.Module):
    """
    Trainable Gaussian Mixture Model used as the variational approximation q(z).

    Args:
        n_components (int): Number of mixture components K.
        latent_dim (int): Dimensionality of the latent space D.
        init_scale (float): Initial std dev of each component.
    """

    def __init__(self, n_components: int, latent_dim: int, init_scale: float = 1.0):
        super().__init__()
        self.K = n_components
        self.D = latent_dim

        # Learnable parameters
        self.log_weights = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, latent_dim) * init_scale)
        self.log_scales = nn.Parameter(
            torch.full((n_components, latent_dim), fill_value=torch.log(torch.tensor(init_scale)))
        )

    @property
    def weights(self) -> Tensor:
        """Normalized mixture weights via softmax. Shape: (K,)"""
        return torch.softmax(self.log_weights, dim=0)

    @property
    def scales(self) -> Tensor:
        """Positive scales via exp. Shape: (K, D)"""
        return torch.exp(self.log_scales)

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Log probability of z under the mixture: log q(z).

        Args:
            z: Tensor of shape (N, D)
        Returns:
            log_prob: Tensor of shape (N,)
        """
        # z: (N, D) -> (N, 1, D) for broadcasting with (K, D)
        z_exp = z.unsqueeze(1)                          # (N, 1, D)
        means = self.means.unsqueeze(0)                 # (1, K, D)
        scales = self.scales.unsqueeze(0)               # (1, K, D)

        # Component log probs: (N, K)
        component_log_prob = dist.Normal(means, scales).log_prob(z_exp).sum(-1)

        # log pi_k + log N(z | mu_k, sigma_k)
        log_w = torch.log_softmax(self.log_weights, dim=0)  # (K,)
        log_mix = log_w.unsqueeze(0) + component_log_prob   # (N, K)

        return torch.logsumexp(log_mix, dim=1)              # (N,)

    def sample(self, n: int) -> Tuple[Tensor, Tensor]:
        """
        Sample from the mixture and return (samples, component_indices).

        Args:
            n: Number of samples
        Returns:
            z:   (N, D) samples
            k:   (N,)   component assignments
        """
        # Sample component indices
        k = dist.Categorical(probs=self.weights).sample((n,))  # (N,)

        # Sample from selected components
        selected_means = self.means[k]       # (N, D)
        selected_scales = self.scales[k]     # (N, D)
        eps = torch.randn_like(selected_means)
        z = selected_means + selected_scales * eps
        return z, k

    def rsample(self, n: int) -> Tuple[Tensor, Tensor]:
        """
        Reparameterized sample (differentiable w.r.t. means and scales).
        Component selection is still non-differentiable (hard).

        Args:
            n: Number of samples
        Returns:
            z:   (N, D) reparameterized samples
            k:   (N,)   component assignments (detached)
        """
        k = dist.Categorical(probs=self.weights.detach()).sample((n,))
        selected_means = self.means[k]
        selected_scales = self.scales[k]
        eps = torch.randn_like(selected_means)
        z = selected_means + selected_scales * eps
        return z, k

    def entropy_lb(self) -> Tensor:
        """
        Lower bound on entropy via Jensen: H(q) >= -E_q[log q(z)].
        Estimated via MC with 1000 samples.
        """
        with torch.no_grad():
            z, _ = self.sample(1000)
        return -self.log_prob(z).mean()

    def extra_repr(self) -> str:
        return f"K={self.K}, D={self.D}"