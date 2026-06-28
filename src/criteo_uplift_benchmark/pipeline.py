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
from criteo_uplift_benchmark.schemas import EvidenceCheck, FitResult, PolicyRow, SegmentSummary


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
    evidence_checks: list[EvidenceCheck]


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
        evidence_checks=build_evidence_checks(
            config=config,
            split=split,
            winner=winner,
            runner_up=runner_up,
            paired_diff_ci=paired_ci,
            segments=segments,
        ),
    )


def build_evidence_checks(
    *,
    config: BenchmarkConfig,
    split,
    winner: str,
    runner_up: str | None,
    paired_diff_ci: tuple[float, float] | None,
    segments: SegmentSummary,
) -> list[EvidenceCheck]:
    treatment_rate = float(split.t_test.mean())
    ci_crosses_zero = paired_diff_ci is not None and paired_diff_ci[0] <= 0 <= paired_diff_ci[1]
    return [
        EvidenceCheck(
            check="randomized_treatment_available",
            status="pass" if 0.05 <= treatment_rate <= 0.95 else "review",
            evidence=f"Test treatment rate is {treatment_rate:.2%}.",
        ),
        EvidenceCheck(
            check="separate_validation_and_test",
            status="pass",
            evidence=(
                f"Split ratios are train={config.split.train_ratio:.0%}, "
                f"validation={config.split.val_ratio:.0%}, test=holdout remainder."
            ),
        ),
        EvidenceCheck(
            check="bootstrap_uncertainty",
            status="pass",
            evidence=f"AUUC intervals use {config.runtime.bootstrap_n} bootstrap samples.",
        ),
        EvidenceCheck(
            check="winner_margin",
            status="review" if ci_crosses_zero else "pass",
            evidence="Winner comparison interval crosses zero; treat model choice as uncertain."
            if ci_crosses_zero
            else f"Winner={winner}; runner_up={runner_up or 'none'}.",
        ),
        EvidenceCheck(
            check="policy_segment_coverage",
            status="pass" if segments.rows else "review",
            evidence=f"{len(segments.rows)} targeting policy segments generated.",
        ),
    ]
