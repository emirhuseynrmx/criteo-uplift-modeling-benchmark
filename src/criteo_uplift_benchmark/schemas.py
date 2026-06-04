from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field


class IndexSplit(BaseModel):
    """Index arrays for a deterministic split."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class SplitData(BaseModel):
    """Materialized arrays for one outcome."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    t_train: np.ndarray
    t_val: np.ndarray
    t_test: np.ndarray


class ArrayBundle(BaseModel):
    """Feature, treatment, and outcome arrays materialized from a dataframe."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: np.ndarray
    t: np.ndarray
    outcomes: dict[str, np.ndarray]


class UpliftPrediction(BaseModel):
    """Predictions from an uplift learner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    uplift: np.ndarray
    prob_t1: np.ndarray
    prob_t0: np.ndarray


class UpliftMetrics(BaseModel):
    """Evaluation metrics for one model on one split."""

    split: Literal["validation", "test", "train"]
    model: str
    auuc: float
    normalized_auuc: float
    auuc_ci_low: float
    auuc_ci_high: float
    top_decile_uplift: float
    top_decile_relative_uplift: float | None
    population_uplift: float


class ResourceMetrics(BaseModel):
    """Runtime and memory summary for one fit/predict cycle."""

    model: str
    outcome: str
    runtime_s: float = Field(ge=0)
    peak_rss_mb: float = Field(ge=0)
    peak_rss_delta_mb: float = Field(ge=0)


class FitResult(BaseModel):
    """Predictions and metrics from a benchmark pass."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    validation_predictions: dict[str, UpliftPrediction]
    test_predictions: dict[str, UpliftPrediction]
    validation_metrics: list[UpliftMetrics]
    test_metrics: list[UpliftMetrics]
    resource_metrics: list[ResourceMetrics]

    @computed_field
    @property
    def validation_frame(self) -> pd.DataFrame:
        rows = [metric.model_dump() for metric in self.validation_metrics]
        return pd.DataFrame(rows).sort_values("auuc", ascending=False).reset_index(drop=True)

    @computed_field
    @property
    def test_frame(self) -> pd.DataFrame:
        rows = [metric.model_dump() for metric in self.test_metrics]
        return pd.DataFrame(rows).sort_values("auuc", ascending=False).reset_index(drop=True)


class PolicyRow(BaseModel):
    """One point on a targeting policy curve."""

    target_pct: float
    target_n: int
    incremental_rate: float
    incremental_events: float
    events_per_targeted_user: float


class SegmentRow(BaseModel):
    """One model-based policy segment."""

    policy_segment: str
    rule: str
    action: str
    count: int
    share: float


class SegmentSummary(BaseModel):
    """Policy segments and avoidable campaign spend."""

    rows: list[SegmentRow]
    avoidable_campaign_spend: float
