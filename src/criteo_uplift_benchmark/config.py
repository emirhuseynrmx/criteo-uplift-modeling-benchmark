from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetConfig(BaseModel):
    """Dataset location and column layout."""

    model_config = ConfigDict(extra="forbid")

    data_path: Path | None = None
    data_dir: Path = Path("data")
    kaggle_dataset: str = "arashnic/uplift-modeling"
    treatment_col: str = "treatment"
    primary_outcome_col: str = "visit"
    conversion_col: str = "conversion"
    feature_cols: tuple[str, ...] = tuple(f"f{i}" for i in range(12))

    @field_validator("data_path", "data_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path | None) -> Path | None:
        return None if value is None else Path(value)


class SplitConfig(BaseModel):
    """Train/validation/test split settings."""

    model_config = ConfigDict(extra="forbid")

    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _check_ratios(self) -> SplitConfig:
        if self.train_ratio + self.val_ratio >= 1.0:
            msg = "train_ratio + val_ratio must be less than 1.0"
            raise ValueError(msg)
        return self


class LGBMConfig(BaseModel):
    """LightGBM parameters shared by the meta-learners."""

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = Field(default=300, gt=0)
    learning_rate: float = Field(default=0.05, gt=0.0)
    num_leaves: int = Field(default=63, gt=1)
    min_child_samples: int = Field(default=50, gt=0)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    random_state: int = 42
    verbose: int = -1
    n_jobs: int = -1

    def classifier_params(self) -> dict[str, int | float]:
        return self.model_dump()

    def regressor_params(self, n_estimators: int | None = 200) -> dict[str, int | float]:
        params = self.model_dump()
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        return params


class RuntimeConfig(BaseModel):
    """Runtime switches for expensive learners and output generation."""

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(default=7_000_000, gt=0)
    calibration_cv: int = Field(default=3, ge=2)
    bootstrap_n: int = Field(default=200, ge=1)
    run_dr_learner: bool = True
    run_causal_forest: bool = False
    causal_forest_train_sample: int = Field(default=300_000, gt=0)
    run_conversion: bool = True
    run_conversion_comparison: bool = False
    shap_sample_size: int = Field(default=3_000, gt=0)
    outcome: Literal["visit", "conversion"] = "visit"


class PolicyConfig(BaseModel):
    """Policy thresholds and campaign economics."""

    model_config = ConfigDict(extra="forbid")

    positive_uplift_threshold: float = 0.02
    negative_uplift_threshold: float = -0.02
    campaign_cost_per_contact: float = Field(default=0.05, ge=0.0)


class BenchmarkConfig(BaseModel):
    """Top-level validated benchmark configuration."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    lgbm: LGBMConfig = Field(default_factory=LGBMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @classmethod
    def quick(
        cls,
        *,
        sample_size: int = 500_000,
        run_dr_learner: bool = False,
        run_causal_forest: bool = False,
    ) -> BenchmarkConfig:
        return cls(
            runtime=RuntimeConfig(
                sample_size=sample_size,
                run_dr_learner=run_dr_learner,
                run_causal_forest=run_causal_forest,
            )
        )


class ProjectConfig(BaseModel):
    """Small project-level defaults used by downloader and CLI."""

    model_config = ConfigDict(extra="forbid")

    default_dataset: str = "arashnic/uplift-modeling"
    default_data_dir: Path = Path("data")
    default_assets_dir: Path = Path("assets")
    expected_csv_name: str = "criteo-uplift-v2.1.csv"
