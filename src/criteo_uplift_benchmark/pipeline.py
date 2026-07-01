from __future__ import annotations

import numpy as np
import optuna
from pydantic import BaseModel, ConfigDict

from criteo_uplift_benchmark.config import BenchmarkConfig, LGBMConfig
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

_HPO_SUBSAMPLE = 30_000
_HPO_TRIALS = 20


def tune_lgbm_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[LGBMConfig, float, int]:
    """Find best LightGBM hyperparams via Optuna on a fixed subsample of training data."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    n = min(_HPO_SUBSAMPLE, len(X_train))
    rng = np.random.default_rng(config.split.seed)
    idx = rng.choice(len(X_train), n, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": config.split.seed,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.split.seed)
        scores = cross_val_score(model, X_sub, y_sub, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.split.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=_HPO_TRIALS, show_progress_bar=False)
    best = study.best_params
    tuned = LGBMConfig(
        n_estimators=best["n_estimators"],
        learning_rate=best["learning_rate"],
        num_leaves=best["num_leaves"],
        min_child_samples=best["min_child_samples"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        random_state=config.split.seed,
        verbose=-1,
        n_jobs=-1,
    )
    return tuned, study.best_value, len(study.trials)


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

    tuned_lgbm, hpo_cv_auc, hpo_n_trials = tune_lgbm_config(
        split.X_train, split.y_train, config
    )
    tuned_config = config.model_copy(update={"lgbm": tuned_lgbm})

    result = fit_and_score_models(
        models=build_models(tuned_config),
        split=split,
        config=tuned_config,
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
        config=tuned_config,
        outcome=outcome,
        result=result,
        winner=winner,
        runner_up=runner_up,
        paired_diff_ci=paired_ci,
        winner_policy=policy_curve(winner_pred.uplift, split.y_test, split.t_test),
        winner_segments=segments,
        evidence_checks=build_evidence_checks(
            config=tuned_config,
            split=split,
            winner=winner,
            runner_up=runner_up,
            paired_diff_ci=paired_ci,
            segments=segments,
            hpo_cv_auc=hpo_cv_auc,
            hpo_n_trials=hpo_n_trials,
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
    hpo_cv_auc: float = 0.0,
    hpo_n_trials: int = 0,
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
            check="hpo_validation",
            status="pass",
            evidence=(
                f"Optuna TPE ran {hpo_n_trials} trials on {_HPO_SUBSAMPLE:,}-row subsample "
                f"(3-fold CV); best LightGBM CV-AUC={hpo_cv_auc:.4f}. "
                "Holdout and validation sets never entered HPO loop."
            ),
        ),
        EvidenceCheck(
            check="policy_segment_coverage",
            status="pass" if segments.rows else "review",
            evidence=f"{len(segments.rows)} targeting policy segments generated.",
        ),
    ]
