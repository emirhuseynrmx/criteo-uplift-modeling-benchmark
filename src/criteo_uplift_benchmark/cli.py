from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from criteo_uplift_benchmark.assets import generate_assets
from criteo_uplift_benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    ProjectConfig,
    RuntimeConfig,
)
from criteo_uplift_benchmark.downloader import download_dataset, find_criteo_csv

PROJECT = ProjectConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="criteo-uplift",
        description="Utilities for the Criteo uplift modeling benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command")

    assets = subparsers.add_parser("assets", help="Generate README visual assets.")
    assets.add_argument("--output-dir", default=str(PROJECT.default_assets_dir))

    download = subparsers.add_parser("download", help="Download the Criteo dataset from Kaggle.")
    download.add_argument("--dataset", default=PROJECT.default_dataset)
    download.add_argument("--data-dir", default=str(PROJECT.default_data_dir))
    download.add_argument("--no-unzip", action="store_true")

    locate = subparsers.add_parser("locate-data", help="Locate the Criteo CSV file.")
    locate.add_argument("--data-dir", default=str(PROJECT.default_data_dir))

    run = subparsers.add_parser("run", help="Run the benchmark from Python modules.")
    run.add_argument("--data-path", default=None)
    run.add_argument("--data-dir", default=str(PROJECT.default_data_dir))
    run.add_argument("--sample-size", type=int, default=500_000)
    run.add_argument("--full", action="store_true", help="Use the default 7M sample.")
    run.add_argument("--with-dr", action="store_true")
    run.add_argument("--with-causal-forest", action="store_true")
    run.add_argument("--outcome", choices=["visit", "conversion"], default="visit")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "assets":
        output = generate_assets(Path(args.output_dir))
        print(f"Generated assets in: {output}")
        return 0

    if args.command == "download":
        output = download_dataset(
            data_dir=Path(args.data_dir),
            dataset=args.dataset,
            unzip=not args.no_unzip,
        )
        print(f"Downloaded dataset into: {output}")
        return 0

    if args.command == "locate-data":
        found = find_criteo_csv(Path(args.data_dir))
        if found is None:
            print("Criteo CSV not found.")
            return 1
        print(found)
        return 0

    if args.command == "run":
        from criteo_uplift_benchmark.pipeline import run_benchmark

        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        sample_size = 7_000_000 if args.full else args.sample_size
        config = BenchmarkConfig(
            dataset=DatasetConfig(
                data_path=args.data_path,
                data_dir=Path(args.data_dir),
            ),
            runtime=RuntimeConfig(
                sample_size=sample_size,
                run_dr_learner=args.with_dr,
                run_causal_forest=args.with_causal_forest,
                outcome=args.outcome,
            ),
        )
        run_result = run_benchmark(config)
        print(f"Winner: {run_result.winner}")
        if run_result.paired_diff_ci is not None:
            low, high = run_result.paired_diff_ci
            print(f"Paired AUUC diff CI vs runner-up: [{low:.2f}, {high:.2f}]")
        print(run_result.result.validation_frame.to_string(index=False))
        return 0

    parser.print_help()
    return 0
