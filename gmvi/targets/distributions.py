"""
Target posterior distributions p(z) for variational inference experiments.

Each target implements:
  - log_prob(z): unnormalized log probability
  - sample(n):   exact samples (for evaluation)
"""

import torch
import torch.distributions as dist
from torch import Tensor
import math


class Target:
    """Base class for target distributions."""
    name: str = "base"

    def log_prob(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, n: int) -> Tensor:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


# ─── 2D Toy Targets ────────────────────────────────────────────────────────────

class TwoMoonsTarget(Target):
    """
    2D two-moons distribution — a classic non-convex posterior.
    """
    name = "two_moons"

    def __init__(self, noise: float = 0.1):
        self.noise = noise

    @property
    def dim(self) -> int:
        return 2

    def log_prob(self, z: Tensor) -> Tensor:
        x, y = z[..., 0], z[..., 1]

        # Upper moon: center (-1, 0.5), radius 1, upper half (sin θ ≥ 0)
        dx1, dy1 = x + 1.0, y - 0.5
        r1   = torch.sqrt(dx1 ** 2 + dy1 ** 2 + 1e-8)
        sin1 = dy1 / r1                                     # ∈ [0,1] on upper arc
        lp1  = (dist.Normal(1.0, self.noise).log_prob(r1)
                + dist.Normal(0.5, 0.4).log_prob(sin1))

        # Lower moon: center (1, -0.5), radius 1, lower half (−sin θ ≥ 0)
        dx2, dy2 = x - 1.0, y + 0.5
        r2   = torch.sqrt(dx2 ** 2 + dy2 ** 2 + 1e-8)
        sin2 = -dy2 / r2                                    # ∈ [0,1] on lower arc
        lp2  = (dist.Normal(1.0, self.noise).log_prob(r2)
                + dist.Normal(0.5, 0.4).log_prob(sin2))

        return torch.logaddexp(lp1, lp2) - math.log(2)

    def sample(self, n: int) -> Tensor:
        half = n // 2
        # Upper moon
        theta1 = torch.rand(half) * math.pi
        x1 = torch.stack([torch.cos(theta1) - 1.0,
                           torch.sin(theta1) + 0.5], dim=-1)
        # Lower moon
        theta2 = torch.rand(n - half) * math.pi
        x2 = torch.stack([-torch.cos(theta2) + 1.0,
                            -torch.sin(theta2) - 0.5], dim=-1)
        samples = torch.cat([x1, x2], dim=0)
        samples += torch.randn_like(samples) * self.noise
        return samples[torch.randperm(n)]


class GaussianMixtureTarget(Target):
    """
    Mixture of K Gaussians in D dimensions.
    """
    name = "gmm_target"

    def __init__(self, means: Tensor, scales: Tensor, weights: Tensor = None):
        """
        Args:
            means:   (K, D)
            scales:  (K, D) — diagonal std devs
            weights: (K,) optional, defaults to uniform
        """
        self.means_ = means
        self.scales_ = scales
        K = means.shape[0]
        self.weights_ = weights if weights is not None else torch.ones(K) / K

    @property
    def dim(self) -> int:
        return self.means_.shape[1]

    def log_prob(self, z: Tensor) -> Tensor:
        z_exp = z.unsqueeze(1)                               # (N, 1, D)
        m = self.means_.unsqueeze(0)                         # (1, K, D)
        s = self.scales_.unsqueeze(0)                        # (1, K, D)
        comp_lp = dist.Normal(m, s).log_prob(z_exp).sum(-1) # (N, K)
        log_w = torch.log(self.weights_)                     # (K,)
        return torch.logsumexp(log_w + comp_lp, dim=1)

    def sample(self, n: int) -> Tensor:
        k = dist.Categorical(probs=self.weights_).sample((n,))
        means = self.means_[k]
        scales = self.scales_[k]
        return means + scales * torch.randn_like(means)


class BananaTarget(Target):
    """
    2D banana-shaped (funnel-like) distribution.
    """
    name = "banana"

    def __init__(self, b: float = 0.5, sigma: float = 2.0):
        self.b = b
        self.sigma = sigma

    @property
    def dim(self) -> int:
        return 2

    def log_prob(self, z: Tensor) -> Tensor:
        x, y = z[..., 0], z[..., 1]
        lp_x = dist.Normal(0.0, self.sigma).log_prob(x)
        lp_y = dist.Normal(self.b * (x**2 - self.sigma**2), 1.0).log_prob(y)
        return lp_x + lp_y

    def sample(self, n: int) -> Tensor:
        x = torch.randn(n) * self.sigma
        y = self.b * (x**2 - self.sigma**2) + torch.randn(n)
        return torch.stack([x, y], dim=-1)


class RingTarget(Target):
    """
    2D ring (annulus) distribution.
    """
    name = "ring"

    def __init__(self, radius: float = 2.0, width: float = 0.3):
        self.radius = radius
        self.width = width

    @property
    def dim(self) -> int:
        return 2

    def log_prob(self, z: Tensor) -> Tensor:
        r = torch.norm(z, dim=-1)
        return dist.Normal(self.radius, self.width).log_prob(r)

    def sample(self, n: int) -> Tensor:
        r = dist.Normal(self.radius, self.width).sample((n,))
        theta = torch.rand(n) * 2 * math.pi
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=-1)


# ─── Higher-Dimensional Targets ─────────────────────────────────────────────────

class NealFunnelTarget(Target):
    """
    Neal's funnel: a hierarchical distribution that is notoriously hard
    for variational inference. D-dimensional.
    """
    name = "neal_funnel"

    def __init__(self, dim: int = 10):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def log_prob(self, z: Tensor) -> Tensor:
        v = z[..., 0]           # log-scale coordinate
        x = z[..., 1:]          # (N, D-1)
        lp_v = dist.Normal(0.0, 3.0).log_prob(v)
        lp_x = dist.Normal(0.0, torch.exp(v / 2).unsqueeze(-1)).log_prob(x).sum(-1)
        return lp_v + lp_x

    def sample(self, n: int) -> Tensor:
        v = torch.randn(n) * 3.0
        x = torch.randn(n, self._dim - 1) * torch.exp(v / 2).unsqueeze(-1)
        return torch.cat([v.unsqueeze(-1), x], dim=-1)


class LogisticRegressionPosterior(Target):
    """
    Bayesian logistic regression posterior p(w | X, y).
    p(w) = N(0, prior_scale^2 I), likelihood = Bernoulli(sigma(Xw)).
    """
    name = "logistic_regression"

    def __init__(self, X: Tensor, y: Tensor, prior_scale: float = 1.0):
        """
        Args:
            X: (N, D) design matrix
            y: (N,) binary labels in {0, 1}
            prior_scale: prior std dev
        """
        self.X = X
        self.y = y
        self.prior_scale = prior_scale
        self._dim = X.shape[1]

    @property
    def dim(self) -> int:
        return self._dim

    def log_prob(self, w: Tensor) -> Tensor:
        # w: (S, D)
        # Prior
        lp_prior = dist.Normal(0.0, self.prior_scale).log_prob(w).sum(-1)  # (S,)
        # Likelihood
        logits = w @ self.X.T   # (S, N)
        lp_lik = dist.Bernoulli(logits=logits).log_prob(self.y).sum(-1)    # (S,)
        return lp_prior + lp_lik

    def sample(self, n: int) -> Tensor:
        raise NotImplementedError("Exact sampling not available; use MCMC.")


class RandomGMTarget(GaussianMixtureTarget):
    """
    Random 3-component Gaussian mixture in D dimensions.

    Means are drawn from N(0, spread), scales from Uniform(scale_lo, scale_hi),
    and weights from a symmetric Dirichlet(alpha). A seed makes it reproducible.
    """
    name = "random_gm"

    def __init__(
        self,
        dim: int = 2,
        n_components: int = 3,
        spread: float = 3.0,
        scale_lo: float = 0.3,
        scale_hi: float = 1.0,
        seed: int = 0,
    ):
        rng = torch.Generator()
        rng.manual_seed(seed)
        means   = torch.randn(n_components, dim, generator=rng) * spread
        scales  = torch.rand(n_components, dim, generator=rng) * (scale_hi - scale_lo) + scale_lo
        weights = torch.distributions.Dirichlet(torch.ones(n_components)).sample()
        super().__init__(means=means, scales=scales, weights=weights)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim


# ─── Registry ──────────────────────────────────────────────────────────────────

def make_target(name: str, **kwargs) -> Target:
    registry = {
        "two_moons": TwoMoonsTarget,
        "gmm": GaussianMixtureTarget,
        "random_gm": RandomGMTarget,
        "banana": BananaTarget,
        "ring": RingTarget,
        "funnel": NealFunnelTarget,
        "logreg": LogisticRegressionPosterior,
    }
    if name not in registry:
        raise ValueError(f"Unknown target '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)