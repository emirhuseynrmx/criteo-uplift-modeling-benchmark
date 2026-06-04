from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from criteo_uplift_benchmark.config import BenchmarkConfig
from criteo_uplift_benchmark.data import (
    load_data,
    make_index_split,
    make_split_from_arrays,
    materialize_arrays,
)
from criteo_uplift_benchmark.evaluation import (
    fit_and_score_models,
    select_validation_winner,
    winner_diff_ci,
)
from criteo_uplift_benchmark.learners import build_models
from criteo_uplift_benchmark.metrics import policy_curve, segment_policy
from criteo_uplift_benchmark.schemas import FitResult, PolicyRow, SegmentSummary


class BenchmarkRun(BaseModel):
    """Full benchmark output returned by the Python pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: BenchmarkConfig
    outcome: str
    result: FitResult
    winner: str
    runner_up: str | None
    paired_diff_ci: tuple[float, float] | None
    winner_policy: list[PolicyRow]
    winner_segments: SegmentSummary


def run_benchmark(config: BenchmarkConfig | None = None) -> BenchmarkRun:
    """Run the complete benchmark as Python modules instead of a notebook."""
    config = config or BenchmarkConfig()
    df = load_data(config)
    idx_split = make_index_split(df, config)
    arrays = materialize_arrays(df, config)
    outcome = config.runtime.outcome
    split = make_split_from_arrays(arrays, outcome, idx_split)

    result = fit_and_score_models(
        models=build_models(config),
        split=split,
        config=config,
        outcome_name=outcome,
    )
    winner, runner_up = select_validation_winner(result)
    paired_ci = (
        winner_diff_ci(
            result=result,
            split=split,
            winner=winner,
            runner_up=runner_up,
            config=config,
        )
        if runner_up is not None
        else None
    )

    winner_pred = result.test_predictions[winner]
    baseline = result.test_predictions.get("Naive Response Ranker")
    segments = segment_policy(
        uplift=winner_pred.uplift,
        prob_t0=winner_pred.prob_t0,
        baseline_score=baseline.uplift if baseline is not None else None,
        config=config,
    )
    return BenchmarkRun(
        config=config,
        outcome=outcome,
        result=result,
        winner=winner,
        runner_up=runner_up,
        paired_diff_ci=paired_ci,
        winner_policy=policy_curve(winner_pred.uplift, split.y_test, split.t_test),
        winner_segments=segments,
    )
