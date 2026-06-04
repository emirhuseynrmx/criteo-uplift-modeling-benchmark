from __future__ import annotations

import time

import numpy as np

from criteo_uplift_benchmark.config import BenchmarkConfig
from criteo_uplift_benchmark.learners import CausalForestModel, NaiveResponseRanker
from criteo_uplift_benchmark.metrics import evaluate_predictions, paired_auuc_diff_ci
from criteo_uplift_benchmark.resources import PeakRSSMonitor
from criteo_uplift_benchmark.schemas import FitResult, ResourceMetrics, SplitData


def fit_and_score_models(
    *,
    models: dict[str, object],
    split: SplitData,
    config: BenchmarkConfig,
    outcome_name: str,
) -> FitResult:
    """Fit models, score validation/test splits, and track runtime/RSS."""
    validation_predictions = {}
    test_predictions = {}
    validation_metrics = []
    test_metrics = []
    resource_metrics = []

    for name, model in models.items():
        start = time.perf_counter()
        with PeakRSSMonitor() as monitor:
            _fit_model(model, split, config)
            pred_val = model.predict(split.X_val)
            pred_test = model.predict(split.X_test)
        runtime = time.perf_counter() - start

        validation_predictions[name] = pred_val
        test_predictions[name] = pred_test
        validation_metrics.append(
            evaluate_predictions(
                name=name,
                uplift=pred_val.uplift,
                y=split.y_val,
                t=split.t_val,
                config=config,
                split="validation",
            )
        )
        test_metrics.append(
            evaluate_predictions(
                name=name,
                uplift=pred_test.uplift,
                y=split.y_test,
                t=split.t_test,
                config=config,
                split="test",
            )
        )
        resource_metrics.append(
            ResourceMetrics(
                model=name,
                outcome=outcome_name,
                runtime_s=runtime,
                peak_rss_mb=monitor.peak_rss_mb,
                peak_rss_delta_mb=monitor.peak_delta_mb,
            )
        )

    return FitResult(
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        resource_metrics=resource_metrics,
    )


def select_validation_winner(result: FitResult) -> tuple[str, str | None]:
    """Select the best non-response model on validation AUUC."""
    frame = result.validation_frame
    candidates = frame[frame["model"] != "Naive Response Ranker"]
    if candidates.empty:
        winner = frame.iloc[0]["model"]
        return str(winner), None
    winner = str(candidates.iloc[0]["model"])
    runner_up = str(candidates.iloc[1]["model"]) if len(candidates) > 1 else None
    return winner, runner_up


def winner_diff_ci(
    *,
    result: FitResult,
    split: SplitData,
    winner: str,
    runner_up: str,
    config: BenchmarkConfig,
) -> tuple[float, float]:
    low, high, _ = paired_auuc_diff_ci(
        result.validation_predictions[winner].uplift,
        result.validation_predictions[runner_up].uplift,
        split.y_val,
        split.t_val,
        n_boot=config.runtime.bootstrap_n,
        seed=config.split.seed,
    )
    return low, high


def _fit_model(model: object, split: SplitData, config: BenchmarkConfig) -> None:
    if isinstance(model, NaiveResponseRanker):
        model.fit(split.X_train, split.y_train)
        return

    too_large_for_full_causal_forest = (
        len(split.X_train) > config.runtime.causal_forest_train_sample
    )
    if isinstance(model, CausalForestModel) and too_large_for_full_causal_forest:
        rng = np.random.default_rng(config.split.seed)
        idx = rng.choice(
            len(split.X_train),
            size=config.runtime.causal_forest_train_sample,
            replace=False,
        )
        model.fit(split.X_train[idx], split.y_train[idx], split.t_train[idx])
        return

    model.fit(split.X_train, split.y_train, split.t_train)
