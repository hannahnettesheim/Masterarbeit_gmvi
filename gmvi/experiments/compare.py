"""
Run comparison experiments between all three gradient estimators.

Usage:
    from gmvi.experiments.compare import run_comparison
    results = run_comparison(target, n_components=6, n_runs=5)
"""

import torch
import copy
from typing import Dict, List, Optional

from gmvi.models.generalized_mixture import GaussianMixture
from gmvi.estimators.gradient_estimators import (
    ScoreFunctionEstimator,
    GumbelSoftmaxEstimator,
    ODETransportEstimator,
    make_estimator,
)
from gmvi.experiments.trainer import train, evaluate_model, TrainConfig, TrainResult
from gmvi.targets.distributions import Target


def run_comparison(
    target: Target,
    n_components: int = 6,
    init_scale: float = 2.0,
    config: TrainConfig = None,
    n_runs: int = 3,
    estimator_kwargs: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, List[TrainResult]]:
    """
    Train all three estimators on the same target, multiple runs each.

    Args:
        target:           Target distribution to approximate
        n_components:     GMM components K
        init_scale:       initial spread of component means
        config:           TrainConfig (shared across estimators)
        n_runs:           number of random restarts per estimator
        estimator_kwargs: dict of per-estimator kwargs overrides
        seed:             base random seed
        verbose:          print progress

    Returns:
        Dict mapping estimator name -> list of TrainResult (one per run)
    """
    if config is None:
        config = TrainConfig()

    if estimator_kwargs is None:
        estimator_kwargs = {}

    estimator_cfgs = {
        "score_function": {"n_samples": config.minibatch_samples, "baseline": "rloo",
                           **estimator_kwargs.get("score_function", {})},
        "gumbel_softmax": {"n_samples": config.minibatch_samples, "temperature": 1.0,
                           **estimator_kwargs.get("gumbel_softmax", {})},
        "ode_transport": {"n_samples": config.minibatch_samples,
                              **estimator_kwargs.get("ode_transport", {})},
    }

    all_results: Dict[str, List[TrainResult]] = {name: [] for name in estimator_cfgs}

    for run in range(n_runs):
        run_seed = seed + run
        torch.manual_seed(run_seed)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Run {run+1}/{n_runs}  (seed={run_seed})")
            print(f"{'='*60}")

        for est_name, est_kwargs in estimator_cfgs.items():
            # Fresh model and estimator for each run
            torch.manual_seed(run_seed)
            model = GaussianMixture(
                n_components=n_components,
                latent_dim=target.dim,
                init_scale=init_scale,
            )

            estimator = make_estimator(est_name, **est_kwargs)

            result = train(
                model=model,
                estimator=estimator,
                log_target=target.log_prob,
                config=config,
                verbose=verbose,
            )

            # Final evaluation
            result.eval_metrics = evaluate_model(model, target.log_prob, n_samples=5000)

            if verbose:
                m = result.eval_metrics
                print(
                    f"  → Final ELBO: {m['elbo']:.3f} ± {m['elbo_se']:.3f} | "
                    f"Entropy: {m['entropy_lb']:.3f} | "
                    f"Time: {result.total_time:.1f}s"
                )

            all_results[est_name].append(result)

    return all_results


def summarize_comparison(results: Dict[str, List[TrainResult]]) -> Dict:
    """
    Aggregate statistics across runs for each estimator.

    Returns dict: estimator_name -> {mean_elbo, std_elbo, mean_time, ...}
    """
    import numpy as np

    summary = {}
    for est_name, run_results in results.items():
        elbos = [r.eval_metrics["elbo"] for r in run_results]
        times = [r.total_time for r in run_results]
        summary[est_name] = {
            "mean_elbo": float(np.mean(elbos)),
            "std_elbo": float(np.std(elbos)),
            "median_elbo": float(np.median(elbos)),
            "mean_time": float(np.mean(times)),
            "n_runs": len(run_results),
            "elbo_per_run": elbos,
        }
    return summary