"""
End-to-end demo: generate synthetic data, run the uplift benchmark, print results.

Usage:
    python examples/run_demo.py
    python examples/run_demo.py --rows 30000 --no-dr-learner
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from examples.generate_synthetic import generate_criteo_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uplift benchmark on synthetic data.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--no-dr-learner", action="store_true")
    parser.add_argument("--causal-forest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Generate and save synthetic data
    data_path = Path("data/criteo-demo.csv")
    data_path.parent.mkdir(exist_ok=True)
    df = generate_criteo_synthetic(n_rows=args.rows, seed=args.seed)
    df.to_csv(data_path, index=False)
    print(f"Generated {len(df):,} synthetic rows → {data_path}")

    # Import here so the synthetic CSV is ready before data loading
    from criteo_uplift_benchmark.config import BenchmarkConfig, DatasetConfig, RuntimeConfig
    from criteo_uplift_benchmark.pipeline import run_benchmark

    config = BenchmarkConfig(
        dataset=DatasetConfig(data_path=data_path),
        runtime=RuntimeConfig(
            sample_size=args.rows,
            run_dr_learner=not args.no_dr_learner,
            run_causal_forest=args.causal_forest,
            bootstrap_n=50,  # fewer for demo speed
        ),
    )

    print("\nRunning benchmark (this may take ~1-2 min for 20k rows)…\n")
    run = run_benchmark(config)

    # Print results
    print("=" * 55)
    print("Criteo Uplift Benchmark — Results")
    print("=" * 55)
    print(f"  Outcome          : {run.outcome}")
    print(f"  Winner           : {run.winner}")
    if run.runner_up:
        print(f"  Runner-up        : {run.runner_up}")
    if run.paired_diff_ci:
        lo, hi = run.paired_diff_ci
        print(f"  AUUC diff CI     : [{lo:+.4f}, {hi:+.4f}]")

    print("\nModel Scores (AUUC on test set)")
    print("-" * 45)
    for name, _score in sorted(
        run.result.test_auuc.items(), key=lambda x: -x[1].mean
    ):
        ci = run.result.test_auuc[name]
        print(f"  {name:<30} {ci.mean:+.5f}  [{ci.lower:+.5f}, {ci.upper:+.5f}]")

    if run.winner_segments.rows:
        print("\nWinner Targeting Policy Segments")
        print("-" * 45)
        print(f"  {'Segment':<12} {'Users':>8}  {'Avg Uplift':>12}  {'Action'}")
        print("  " + "-" * 50)
        for row in run.winner_segments.rows:
            print(
                f"  {row.segment:<12} {row.users:>8,}  "
                f"{row.avg_uplift:>+12.5f}  {row.action}"
            )

    print("\nEvidence Contract")
    print("-" * 45)
    for check in run.evidence_checks:
        icon = "✓" if check.status == "pass" else "⚠" if check.status == "review" else "✗"
        print(f"  [{icon}] {check.check}: {check.evidence}")
    print()


if __name__ == "__main__":
    main()
