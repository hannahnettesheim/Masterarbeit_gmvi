"""
Training loop for GMM variational inference.
"""

import torch
import torch.optim as optim
from torch import Tensor
from typing import Callable, Dict, List, Optional
import time
from dataclasses import dataclass, field
from gmvi.models.generalized_mixture import GaussianMixture, GeneralizedMixture
from gmvi.estimators.gradient_estimators import ScoreFunctionEstimator, GumbelSoftmaxEstimator, ODETransportEstimator, ExactMarginalizationEstimator


@dataclass
class TrainConfig:
    n_steps: int = 2000
    lr: float = 1e-2
    minibatch_samples: int = 64
    optimizer: str = "adam"          # 'adam' | 'sgd'
    clip_grad_norm: Optional[float] = 1.0
    log_every: int = 100
    eval_samples: int = 2000          # samples for evaluation metrics


@dataclass
class TrainResult:
    estimator_name: str
    elbo_history: List[float] = field(default_factory=list)
    metrics_history: List[Dict] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)
    final_model_state: Optional[Dict] = None
    total_time: float = 0.0

    def best_elbo(self) -> float:
        return max(self.elbo_history) if self.elbo_history else float("-inf")


def train(
    model: GeneralizedMixture,
    estimator,
    log_target: Callable[[Tensor], Tensor],
    config: TrainConfig = None,
    verbose: bool = True,
) -> TrainResult:
    """
    Train a GaussianMixture model with a given gradient estimator.

    Args:
        model:       GaussianMixture to optimize (modified in-place)
        estimator:   one of ScoreFunctionEstimator / GumbelSoftmaxEstimator / ODEEstimator
        log_target:  callable (N, D) -> (N,) unnormalized log posterior
        config:      TrainConfig
        verbose:     print progress

    Returns:
        TrainResult with full training history
    """
    if config is None:
        config = TrainConfig()

    result = TrainResult(estimator_name=estimator.name)

    # Optimizer
    if config.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=config.lr)
    else:
        opt = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    # the annealer ensures that the learning rate decays (this uses cosine-anneling)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_steps)

    t_start = time.time()

    for step in range(config.n_steps):
        t0 = time.time()
        opt.zero_grad()

        loss, metrics = estimator.loss(model, log_target)
        loss.backward()

        if config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        opt.step()
        scheduler.step()

        result.elbo_history.append(metrics["elbo"])
        result.metrics_history.append(metrics)
        result.step_times.append(time.time() - t0)

        if verbose and (step % config.log_every == 0 or step == config.n_steps - 1):
            temp_str = f"  τ={metrics['temperature']:.3f}" if "temperature" in metrics else ""
            print(
                f"[{estimator.name:20s}] step {step:4d}/{config.n_steps} | "
                f"ELBO={metrics['elbo']:8.3f} ± {metrics.get('elbo_std', 0):.3f}{temp_str}"
            )

    result.total_time = time.time() - t_start
    result.final_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    return result


def evaluate_model(
    model: GeneralizedMixture,
    log_target: Callable[[Tensor], Tensor],
    n_samples: int = 5000,
) -> Dict:
    """
    Compute evaluation metrics for a trained model.

    Returns dict with:
      - elbo:           Monte Carlo ELBO estimate
      - elbo_std:       std dev across samples
      - entropy_lb:     lower bound on q's entropy
      - mean_log_p:     mean log p(z) under q
      - mean_log_q:     mean log q(z) under q (= -entropy estimate)
      - weights:        learned mixture weights
    """
    model.eval()
    with torch.no_grad():
        z, _ = model.sample(n_samples)
        log_p = log_target(z)
        log_q = model.log_prob(z)
        elbo_samples = log_p - log_q

    model.train()
    return {
        "elbo": elbo_samples.mean().item(),
        "elbo_std": elbo_samples.std().item(),
        "elbo_se": (elbo_samples.std() / (n_samples ** 0.5)).item(),
        "entropy_lb": (-log_q.mean()).item(),
        "mean_log_p": log_p.mean().item(),
        "mean_log_q": log_q.mean().item(),
        "weights": model.weights.detach().tolist(),
    }