"""
Learnable affine transformations T_i(x) = a_i + A_i x for generalized mixture components.

Each component maintains:
  - a_i:  shift vector  (n,)
  - A_i:  symmetric positive-definite matrix (n, n)

The log-abs-det-Jacobian of T_i is log|det A_i|, so the component density is:
    log q_i(x) = log q_ref(A_i^{-1}(x - a_i)) - log|det A_i|

Three parameterizations of A_i are supported (chosen at construction), each
guaranteeing A_i ∈ S++^n (symmetric positive-definite):

  'diagonal'         – A_i = diag(exp(s_i)), s_i ∈ R^n. Trivially symmetric
                        positive-definite. Recovers classical Gaussian mixture
                        when ref = N(0,I).

  'eigenvaluedecomp'  – A_i = Q_i diag(exp(s_i)) Q_i^T, where Q_i is orthogonal
                        (the eigenvectors) and exp(s_i) are positive eigenvalue
                        scales. Q_i is parametrized via a skew-symmetric
                        generator (Cayley map, or closed-form rotation in 2-D).
                        log|det A_i| = sum(s_i); A_i^{-1} = Q_i diag(exp(-s_i)) Q_i^T;
                        log(A_i) = Q_i diag(s_i) Q_i^T — all exact and cheap.

  'matrixexponential' – A_i = exp(S_i), where S_i is an unconstrained symmetric
                        matrix (its n(n+1)/2 upper-triangular entries are free
                        parameters). The exponential of a symmetric matrix is
                        always SPD, so this maps freely onto S++^n with no
                        orthogonality constraint to enforce.
                        log(A_i) = S_i exactly; log|det A_i| = trace(S_i);
                        A_i^{-1} = exp(-S_i).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal
import math


AffineParamType = Literal["diagonal", "eigenvaluedecomp", "matrixexponential"]


class AffineComponent(nn.Module):
    """
    Learnable affine map T_i(x) = a_i + A_i x, with A_i symmetric positive-definite.

    Args:
        dim:        dimensionality n
        param_type: how to parameterize A_i
        init_scale: initial scale of A_i (roughly std of the component)
    """

    def __init__(
        self,
        dim: int,
        param_type: AffineParamType = "diagonal",
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.param_type = param_type

        # Shift a_i
        self.a = nn.Parameter(torch.randn(dim) * init_scale)

        # Matrix A_i parameterization
        if param_type == "diagonal":
            # A_i = diag(exp(s_i)), log scale parameters
            self.log_diag = nn.Parameter(
                torch.full((dim,), math.log(init_scale))
            )

        elif param_type == "eigenvaluedecomp":
            # A = Q diag(exp(s)) Q^T, Q orthogonal via a skew-symmetric generator.
            self.log_diag = nn.Parameter(
                torch.full((dim,), math.log(init_scale))
            )
            n_skew = dim * (dim - 1) // 2
            self.skew_raw = nn.Parameter(torch.zeros(n_skew))

        elif param_type == "matrixexponential":
            # A = exp(S), S symmetric and otherwise unconstrained.
            # Stored as the n(n+1)/2 upper-triangular (incl. diagonal) entries.
            # Initialized so S ≈ log(init_scale) * I, i.e. A ≈ init_scale * I.
            self.sym_raw = nn.Parameter(torch.zeros(dim * (dim + 1) // 2))
            with torch.no_grad():
                diag_idx = torch.arange(dim)
                triu_idx = torch.triu_indices(dim, dim, offset=0)
                is_diag = triu_idx[0] == triu_idx[1]
                self.sym_raw[is_diag] = math.log(init_scale)

        else:
            raise ValueError(f"Unknown param_type '{param_type}'")

    # ── Matrix access ────────────────────────────────────────────────────────

    def _get_Q(self) -> Tensor:
        """Orthogonal Q from skew_raw.

        dim=2: closed-form rotation via cos/sin (fast path).
        dim>2: Cayley map Q=(I+S)^{-1}(I-S) via linalg.solve — avoids matrix_exp.
        """
        if self.dim == 2:
            θ = self.skew_raw[0]
            c, s = torch.cos(θ), torch.sin(θ)
            return torch.stack([c, -s, s, c]).reshape(2, 2)

        idx = torch.triu_indices(self.dim, self.dim, offset=1,
                                 device=self.skew_raw.device)
        S = torch.zeros(self.dim, self.dim, dtype=self.skew_raw.dtype,
                        device=self.skew_raw.device)
        S = S.index_put((idx[0], idx[1]), self.skew_raw)
        S = S - S.T
        I = torch.eye(self.dim, dtype=S.dtype, device=S.device)
        # For skew-symmetric S: (I+S) and (I-S) commute, so Cayley = (I+S)^{-1}(I-S)
        return torch.linalg.solve(I + S, I - S)

    def _get_S(self) -> Tensor:
        """Symmetric generator S from sym_raw, for param_type='matrixexponential'."""
        idx = torch.triu_indices(self.dim, self.dim, offset=0,
                                 device=self.sym_raw.device)
        S = torch.zeros(self.dim, self.dim, dtype=self.sym_raw.dtype,
                        device=self.sym_raw.device)
        S = S.index_put((idx[0], idx[1]), self.sym_raw)
        diag = torch.diag(torch.diagonal(S))
        return S + S.T - diag  # symmetrize without double-counting the diagonal

    def get_A(self) -> Tensor:
        """Return the (n, n) SPD matrix A_i."""
        if self.param_type == "diagonal":
            return torch.diag(torch.exp(self.log_diag))

        elif self.param_type == "eigenvaluedecomp":
            Q = self._get_Q()
            return Q @ torch.diag(torch.exp(self.log_diag)) @ Q.T

        elif self.param_type == "matrixexponential":
            return torch.matrix_exp(self._get_S())

    def get_A_inv(self) -> Tensor:
        """Return A_i^{-1}. Closed-form for all supported param_types."""
        if self.param_type == "diagonal":
            return torch.diag(torch.exp(-self.log_diag))
        elif self.param_type == "eigenvaluedecomp":
            Q = self._get_Q()
            return Q @ torch.diag(torch.exp(-self.log_diag)) @ Q.T
        elif self.param_type == "matrixexponential":
            return torch.matrix_exp(-self._get_S())

    def get_log_A(self) -> Tensor:
        """
        Return log(A_i), the matrix logarithm of A_i. Exact and differentiable
        for all supported param_types.

        diagonal:           torch.diag(log_diag)
        eigenvaluedecomp:   Q diag(log_diag) Q^T
        matrixexponential:  S  (by construction, A = exp(S))
        """
        if self.param_type == "diagonal":
            return torch.diag(self.log_diag)

        elif self.param_type == "eigenvaluedecomp":
            Q = self._get_Q()
            return Q @ torch.diag(self.log_diag) @ Q.T

        elif self.param_type == "matrixexponential":
            return self._get_S()

    def log_abs_det(self) -> Tensor:
        """
        log|det A_i|  — scalar.

        diagonal/eigenvaluedecomp: sum of log-eigenvalues.
        matrixexponential:        trace(S), since det(exp(S)) = exp(trace(S)).
        """
        if self.param_type in ("diagonal", "eigenvaluedecomp"):
            return self.log_diag.sum()
        elif self.param_type == "matrixexponential":
            return torch.diagonal(self._get_S()).sum()

    # ── Forward map T_i(x) = a_i + A_i x ────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Push x through T_i.
        Args:
            x: (N, n) samples from Q_ref
        Returns:
            z: (N, n) transformed samples
        """
        A = self.get_A()
        return x @ A.T + self.a  # (N, n)

    def inverse(self, z: Tensor) -> Tensor:
        """
        Apply T_i^{-1}(z) = A_i^{-1}(z - a_i).
        Args:
            z: (N, n) points in the output space
        Returns:
            x: (N, n) pre-images
        """
        A_inv = self.get_A_inv()
        return (z - self.a) @ A_inv.T  # (N, n)

    def component_log_prob(self, z: Tensor, ref_log_prob_fn) -> Tensor:
        """
        log q_i(z) = log q_ref(A_i^{-1}(z - a_i)) - log|det A_i|

        Args:
            z:                (N, n) evaluation points
            ref_log_prob_fn:  callable x:(N,n) -> (N,) log q_ref(x)
        Returns:
            log_prob: (N,)
        """
        x = self.inverse(z)               # (N, n) pre-image
        log_ref = ref_log_prob_fn(x)      # (N,)
        log_jac = self.log_abs_det()      # scalar (positive = expansion)
        return log_ref - log_jac

    def rsample(self, n: int, ref_sample_fn) -> Tensor:
        """
        Reparameterized sample: z = a_i + A_i x,  x ~ Q_ref.
        Differentiable w.r.t. a_i and A_i.
        """
        x = ref_sample_fn(n)              # (N, n)
        return self.forward(x)

    def extra_repr(self) -> str:
        lad = self.log_abs_det().item()
        return f"dim={self.dim}, param_type={self.param_type}, log|det A|={lad:.3f}"
