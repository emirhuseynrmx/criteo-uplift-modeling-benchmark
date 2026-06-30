"""
Generate synthetic Criteo-like uplift data for local testing without downloading from Kaggle.

Usage:
    python examples/generate_synthetic.py
    python examples/generate_synthetic.py --rows 50000 --out data/criteo-synthetic.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_criteo_synthetic(n_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 12 feature columns matching Criteo schema (f0..f11), mix of sparse and dense
    features = {}
    for i in range(12):
        if i < 5:
            # dense continuous features
            features[f"f{i}"] = rng.standard_normal(n_rows).astype(np.float32)
        elif i < 9:
            # sparse binary-ish features
            features[f"f{i}"] = rng.binomial(1, 0.3, n_rows).astype(np.float32)
        else:
            # count-like features
            features[f"f{i}"] = rng.poisson(2.0, n_rows).astype(np.float32)

    frame = pd.DataFrame(features)

    # Treatment assignment: ~85% treated (matching real Criteo distribution)
    frame["treatment"] = rng.binomial(1, 0.85, n_rows).astype(np.int8)

    # Build latent conversion propensity
    propensity = (
        0.3 * frame["f0"]
        + 0.2 * frame["f1"]
        - 0.15 * frame["f2"]
        + 0.1 * frame["f3"]
        + 0.05 * frame["f5"]
    )
    uplift_effect = 0.04  # ~4% average treatment effect

    visit_prob_control = 1 / (1 + np.exp(-0.5 - propensity * 0.3))
    visit_prob_treated = np.clip(visit_prob_control + uplift_effect, 0, 1)

    visit_prob = np.where(frame["treatment"] == 1, visit_prob_treated, visit_prob_control)
    frame["visit"] = rng.binomial(1, visit_prob, n_rows).astype(np.int8)

    # Conversion: subset of visits, smaller effect
    conv_prob_base = visit_prob * 0.4
    conv_effect = 0.01
    conv_prob_control = conv_prob_base
    conv_prob_treated = np.clip(conv_prob_base + conv_effect, 0, 1)
    conv_prob = np.where(frame["treatment"] == 1, conv_prob_treated, conv_prob_control)
    frame["conversion"] = rng.binomial(1, conv_prob, n_rows).astype(np.int8)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Criteo-like uplift data.")
    parser.add_argument("--rows", type=int, default=20_000, help="Number of rows (default 20000)")
    parser.add_argument(
        "--out", type=Path, default=Path("data/criteo-synthetic.csv"), help="Output CSV path"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_criteo_synthetic(n_rows=args.rows, seed=args.seed)
    df.to_csv(args.out, index=False)

    treatment_rate = df["treatment"].mean()
    visit_rate_treated = df[df["treatment"] == 1]["visit"].mean()
    visit_rate_control = df[df["treatment"] == 0]["visit"].mean()
    uplift = visit_rate_treated - visit_rate_control

    print(f"Generated {len(df):,} rows → {args.out}")
    print(f"  Treatment rate   : {treatment_rate:.1%}")
    print(f"  Visit rate (T=1) : {visit_rate_treated:.3%}")
    print(f"  Visit rate (T=0) : {visit_rate_control:.3%}")
    print(f"  ATE estimate     : {uplift:+.4f}")
    print(f"  Conversion rate  : {df['conversion'].mean():.3%}")


if __name__ == "__main__":
    main()
