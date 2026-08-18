"""
Reference distributions Q_ref for generalized mixture models.

Each reference distribution implements:
  - log_prob(x):  log q_ref(x), shape (N,)
  - sample(n):    samples from Q_ref, shape (N, D)
  - dim:          dimensionality n

The generalized mixture pushes Q_ref through affine maps T_i(x) = a_i + A_i x.
"""

import torch
import torch.distributions as dist
from torch import Tensor
import math
from abc import ABC, abstractmethod


class ReferenceDistribution(ABC):
    """Abstract base class for reference distributions."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        """Log probability density. x: (N, D) -> (N,)"""
        ...

    @abstractmethod
    def sample(self, n: int) -> Tensor:
        """Draw n samples. Returns (N, D)."""
        ...


class StandardNormal(ReferenceDistribution):
    """
    Q_ref = N(0, I_n).
    Yields the classical Gaussian mixture when used with diagonal A_i.
    """
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def log_prob(self, x: Tensor) -> Tensor:
        return dist.Normal(0., 1.).log_prob(x).sum(-1)

    def sample(self, n: int) -> Tensor:
        return torch.randn(n, self._dim)


class StudentT(ReferenceDistribution):
    """
    Q_ref = multivariate Student-t with df degrees of freedom (isotropic).
    Heavy-tailed reference — useful when the target has heavy tails.
    """
    def __init__(self, dim: int, df: float = 3.0):
        self._dim = dim
        self.df = df

    @property
    def dim(self) -> int:
        return self._dim

    def log_prob(self, x: Tensor) -> Tensor:
        # Product of univariate t (isotropic)
        return dist.StudentT(df=self.df).log_prob(x).sum(-1)

    def sample(self, n: int) -> Tensor:
        return dist.StudentT(df=self.df).sample((n, self._dim))


class Laplace(ReferenceDistribution):
    """
    Q_ref = Laplace(0, 1)^n (isotropic Laplace).
    """
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def log_prob(self, x: Tensor) -> Tensor:
        return dist.Laplace(0., 1.).log_prob(x).sum(-1)

    def sample(self, n: int) -> Tensor:
        return dist.Laplace(torch.zeros(self._dim), torch.ones(self._dim)).sample((n,))


class LogisticRef(ReferenceDistribution):
    """
    Q_ref = Logistic(0, 1)^n (isotropic logistic).
    """
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def log_prob(self, x: Tensor) -> Tensor:
        return dist.Logistic(0., 1.).log_prob(x).sum(-1)

    def sample(self, n: int) -> Tensor:
        return dist.Logistic(torch.zeros(self._dim), torch.ones(self._dim)).sample((n,))


class UniformRef(ReferenceDistribution):
    """
    Q_ref = Uniform[-1, 1]^n.
    log_prob = -n * log(2) inside the cube, -inf outside.
    """
    def __init__(self, dim: int):
        self._dim = dim
        self._log_vol = -dim * math.log(2.0)

    @property
    def dim(self) -> int:
        return self._dim

    def log_prob(self, x: Tensor) -> Tensor:
        inside = (x.abs() <= 1.0).all(dim=-1)
        return torch.where(inside,
                           torch.full(inside.shape, self._log_vol),
                           torch.full(inside.shape, float("-inf")))

    def sample(self, n: int) -> Tensor:
        return torch.rand(n, self._dim) * 2 - 1


def make_reference(name: str, dim: int, **kwargs) -> ReferenceDistribution:
    registry = {
        "normal":   StandardNormal,
        "student_t": StudentT,
        "laplace":  Laplace,
        "logistic": LogisticRef,
        "uniform":  UniformRef,
    }
    if name not in registry:
        raise ValueError(f"Unknown reference '{name}'. Available: {list(registry)}")
    return registry[name](dim=dim, **kwargs)