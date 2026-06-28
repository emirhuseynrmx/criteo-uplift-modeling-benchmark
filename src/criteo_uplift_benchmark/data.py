from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from criteo_uplift_benchmark.config import BenchmarkConfig
from criteo_uplift_benchmark.schemas import ArrayBundle, IndexSplit, SplitData


def resolve_data_path(config: BenchmarkConfig) -> Path | None:
    """Resolve the Criteo CSV path from explicit config, local data, or Kaggle mount."""
    dataset = config.dataset
    if dataset.data_path and dataset.data_path.exists():
        return dataset.data_path

    local_matches = list(dataset.data_dir.rglob("criteo*.csv")) if dataset.data_dir.exists() else []
    if local_matches:
        return local_matches[0]

    kaggle_matches = glob.glob("/kaggle/input/**/criteo*.csv", recursive=True)
    return Path(kaggle_matches[0]) if kaggle_matches else None


def load_data(config: BenchmarkConfig) -> pl.DataFrame:
    """Load Criteo data from a local file, data directory, or Kaggle mount."""
    data_path = resolve_data_path(config)
    if data_path is None:
        raise FileNotFoundError(
            "Criteo uplift data was not found. Run `python start.py download --data-dir data` "
            "or pass --data-path to a real Kaggle CSV."
        )

    schema_names = pl.scan_csv(data_path).collect_schema().names()
    outcome_cols = [
        col
        for col in [config.dataset.primary_outcome_col, config.dataset.conversion_col]
        if col in schema_names
    ]
    select_cols = list(config.dataset.feature_cols) + [
        config.dataset.treatment_col,
        *outcome_cols,
    ]

    df = (
        pl.scan_csv(data_path)
        .select(select_cols)
        .with_columns([pl.col(col).cast(pl.Float32) for col in config.dataset.feature_cols])
        .collect()
    )

    if config.runtime.sample_size < len(df):
        return df.sample(
            n=config.runtime.sample_size,
            seed=config.split.seed,
            shuffle=True,
        )
    return df.sample(fraction=1.0, seed=config.split.seed, shuffle=True)


def make_stratification_labels(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    labels = np.zeros(len(df), dtype=np.int64)
    for col in cols:
        labels = labels * 2 + df[col].to_numpy().astype(np.int64)
    return labels


def _safe_train_test_split(
    indices: np.ndarray,
    *,
    test_size: float,
    seed: int,
    stratify: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def make_index_split(df: pl.DataFrame, config: BenchmarkConfig) -> IndexSplit:
    """Create one stratified index split reused across outcomes."""
    indices = np.arange(len(df))
    strat_cols = [
        config.dataset.treatment_col,
        config.dataset.primary_outcome_col,
    ]
    if config.dataset.conversion_col in df.columns:
        strat_cols.append(config.dataset.conversion_col)
    strat = make_stratification_labels(df, strat_cols)

    test_val_size = 1.0 - config.split.train_ratio
    val_share = config.split.val_ratio / test_val_size
    train_idx, temp_idx = _safe_train_test_split(
        indices,
        test_size=test_val_size,
        seed=config.split.seed,
        stratify=strat,
    )
    val_idx, test_idx = _safe_train_test_split(
        temp_idx,
        test_size=1.0 - val_share,
        seed=config.split.seed,
        stratify=strat[temp_idx],
    )
    return IndexSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def materialize_arrays(df: pl.DataFrame, config: BenchmarkConfig) -> ArrayBundle:
    """Convert a Polars dataframe into numpy arrays."""
    outcomes = {
        col: df[col].to_numpy().astype(np.int8, copy=False)
        for col in [config.dataset.primary_outcome_col, config.dataset.conversion_col]
        if col in df.columns
    }
    return ArrayBundle(
        X=df.select(list(config.dataset.feature_cols)).to_numpy().astype(np.float32, copy=False),
        t=df[config.dataset.treatment_col].to_numpy().astype(np.int8, copy=False),
        outcomes=outcomes,
    )


def make_split_from_arrays(bundle: ArrayBundle, outcome: str, split: IndexSplit) -> SplitData:
    """Materialize train/validation/test arrays for one outcome."""
    y = bundle.outcomes[outcome]
    return SplitData(
        X_train=bundle.X[split.train_idx],
        X_val=bundle.X[split.val_idx],
        X_test=bundle.X[split.test_idx],
        y_train=y[split.train_idx],
        y_val=y[split.val_idx],
        y_test=y[split.test_idx],
        t_train=bundle.t[split.train_idx],
        t_val=bundle.t[split.val_idx],
        t_test=bundle.t[split.test_idx],
    )
