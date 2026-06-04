from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from criteo_uplift_benchmark.config import ProjectConfig

PROJECT = ProjectConfig()

def has_kaggle_credentials(home: Path | None = None) -> bool:
    """Check whether Kaggle credentials are available."""
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    home_path = home or Path.home()
    has_file = (home_path / ".kaggle" / "kaggle.json").exists()
    return has_env or has_file


def ensure_kaggle_ready() -> None:
    if shutil.which("kaggle") is None:
        msg = "Kaggle CLI is not installed. Run: pip install kaggle"
        raise RuntimeError(msg)
    if not has_kaggle_credentials():
        msg = (
            "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or create "
            "~/.kaggle/kaggle.json."
        )
        raise RuntimeError(msg)


def download_dataset(
    data_dir: Path | str = PROJECT.default_data_dir,
    dataset: str = PROJECT.default_dataset,
    unzip: bool = True,
) -> Path:
    """Download a Kaggle dataset into the local data directory."""
    ensure_kaggle_ready()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(data_path)]
    if unzip:
        command.append("--unzip")

    subprocess.run(command, check=True)
    return data_path


def find_criteo_csv(data_dir: Path | str = PROJECT.default_data_dir) -> Path | None:
    """Find the Criteo uplift CSV after download or Kaggle notebook mounting."""
    root = Path(data_dir)
    candidates = [
        root / PROJECT.expected_csv_name,
        Path("/kaggle/input/uplift-modeling") / PROJECT.expected_csv_name,
    ]
    candidates.extend(root.rglob("criteo*.csv") if root.exists() else [])
    for path in candidates:
        if path.exists():
            return path
    return None
