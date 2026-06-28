from pathlib import Path

import pytest

from criteo_uplift_benchmark.config import BenchmarkConfig, DatasetConfig, RuntimeConfig
from criteo_uplift_benchmark.data import load_data


def test_load_data_requires_real_dataset(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        dataset=DatasetConfig(data_dir=tmp_path),
        runtime=RuntimeConfig(sample_size=1000),
    )

    with pytest.raises(FileNotFoundError, match="Criteo uplift data was not found"):
        load_data(config)
