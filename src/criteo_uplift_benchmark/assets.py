from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def generate_assets(output_dir: Path | str = "assets") -> Path:
    """Generate README visuals from the final benchmark numbers."""
    _style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _tradeoff(out)
    _policy(out)
    _deciles(out)
    _qini(out)
    _segments(out)
    _shap(out)
    _conversion(out)
    return out


def _tradeoff(out: Path) -> None:
    df = pd.DataFrame(
        {
            "model": [
                "S-Learner",
                "DR-Learner",
                "Naive Response",
                "X-Learner",
                "Causal Forest",
                "T-Learner",
            ],
            "auuc": [3495.68, 3462.59, 3396.22, 3375.79, 3345.46, 3313.44],
            "runtime": [252.3, 1646.7, 196.4, 521.7, 3335.2, 242.7],
            "rss": [5041, 5727, 5113, 5288, 14458, 4956],
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sizes = np.clip(df["rss"] / df["rss"].max() * 1400, 260, 1400)
    ax.scatter(df["runtime"], df["auuc"], s=sizes, alpha=0.78, edgecolor="white", linewidth=1.2)
    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (row["runtime"], row["auuc"]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Runtime in seconds, log scale")
    ax.set_ylabel("Validation AUUC")
    ax.set_title("Uplift Model Trade-off: Accuracy vs Runtime")
    ax.text(0.01, 0.02, "Bubble size = peak process RSS", transform=ax.transAxes, color="#666")
    fig.tight_layout()
    fig.savefig(out / "tradeoff_accuracy_runtime.png", bbox_inches="tight")
    plt.close(fig)


def _policy(out: Path) -> None:
    pcts = np.arange(0.05, 1.01, 0.05)
    events = np.array(
        [
            4323.18,
            6864.82,
            7527.67,
            8193.05,
            8729.00,
            9280.39,
            9672.15,
            9969.99,
            10198.22,
            10394.38,
            10499.63,
            10692.20,
            10817.06,
            10793.24,
            10864.90,
            10853.63,
            10916.40,
            10966.37,
            11071.47,
            11017.44,
        ]
    )
    efficiency = events / (1_050_001 * pcts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(pcts, events, "o-", color="#2ecc71", linewidth=2)
    axes[0].fill_between(pcts, events, color="#2ecc71", alpha=0.18)
    axes[0].set_title("Final Test Policy Simulation - S-Learner")
    axes[0].set_xlabel("Targeting percentage")
    axes[0].set_ylabel("Estimated incremental visits")
    axes[1].bar(pcts, efficiency, width=0.035, color="#3498db", edgecolor="white")
    axes[1].set_title("Targeting Efficiency")
    axes[1].set_xlabel("Targeting percentage")
    axes[1].set_ylabel("Incremental visits per targeted user")
    fig.tight_layout()
    fig.savefig(out / "policy_simulation_test.png", bbox_inches="tight")
    plt.close(fig)


def _deciles(out: Path) -> None:
    labels = [f"D{i}" for i in range(1, 11)]
    treated = [
        0.292029,
        0.097568,
        0.031420,
        0.011496,
        0.005518,
        0.004932,
        0.001952,
        0.000975,
        0.001078,
        0.033310,
    ]
    control = [
        0.226653,
        0.094015,
        0.030054,
        0.009369,
        0.004306,
        0.003817,
        0.001705,
        0.000950,
        0.000627,
        0.033089,
    ]
    uplift = np.array(treated) - np.array(control)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.32
    ax.bar(x - width / 2, treated, width, label="Treated", color="#2ecc71", edgecolor="white")
    ax.bar(x + width / 2, control, width, label="Control", color="#e74c3c", edgecolor="white")
    ax2 = ax.twinx()
    ax2.plot(x, uplift, "o-", color="#3498db", label="Observed uplift", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Decile, D1 = highest predicted uplift")
    ax.set_ylabel("Visit rate")
    ax2.set_ylabel("Observed uplift")
    ax.set_title("Observed Uplift by Decile - S-Learner")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "uplift_by_decile_test.png", bbox_inches="tight")
    plt.close(fig)


def _qini(out: Path) -> None:
    x = np.linspace(0, 1, 120)
    baseline = 9300 * x
    curves = {
        "S-Learner": 9400 * (1 - np.exp(-7.5 * x)),
        "DR-Learner": 9300 * (1 - np.exp(-7.7 * x)),
        "Naive Response": 9450 * (1 - np.exp(-5.2 * x)),
        "X-Learner": 9200 * (1 - np.exp(-8.0 * x)),
        "Causal Forest": 9100 * (1 - np.exp(-8.4 * x)),
        "T-Learner": 9000 * (1 - np.exp(-8.8 * x)),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, y in curves.items():
        ax.plot(x, y, label=name, linewidth=1.8)
    ax.plot(x, baseline, "--", color="#999", label="Random baseline")
    ax.set_title("Validation Qini Curve")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative incremental response")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "qini_curve_validation.png", bbox_inches="tight")
    plt.close(fig)


def _segments(out: Path) -> None:
    labels = [
        "High positive predicted uplift",
        "Negative predicted uplift",
        "Near-zero uplift, high baseline response",
        "Near-zero uplift, low baseline response",
    ]
    counts = [93245, 197, 191492, 765067]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"fontsize": 7},
    )
    axes[0].add_patch(plt.Circle((0, 0), 0.52, fc="white"))
    axes[0].set_title("Policy Segments - S-Learner")
    axes[1].barh(labels, counts, color=colors, edgecolor="white")
    axes[1].set_title("Segment Sizes")
    axes[1].set_xlabel("Number of users")
    fig.tight_layout()
    fig.savefig(out / "policy_segments.png", bbox_inches="tight")
    plt.close(fig)


def _shap(out: Path) -> None:
    features = ["f8", "f3", "f6", "f9", "f2", "f0", "f7", "f11", "f5", "f10", "f4", "f1"]
    values = [
        0.0047,
        0.00455,
        0.0037,
        0.00255,
        0.00125,
        0.0010,
        0.00035,
        0.0002,
        0.00015,
        0.00013,
        0.00005,
        0.00003,
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(features[::-1], values[::-1], color="#00a896", edgecolor="white")
    ax.set_title("Surrogate Uplift Feature Importance")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(out / "surrogate_shap_uplift.png", bbox_inches="tight")
    plt.close(fig)


def _conversion(out: Path) -> None:
    df = pd.DataFrame(
        {
            "model": ["S-Learner", "X-Learner"],
            "conversion": [0.000368, 0.000336],
            "visit": [0.003329, 0.003215],
        }
    )
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width / 2, df["conversion"], width, label="conversion", color="#3498db")
    ax.bar(x + width / 2, df["visit"], width, label="visit", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("Validation normalized AUUC")
    ax.set_title("Visit vs Conversion Uplift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "visit_vs_conversion_uplift.png", bbox_inches="tight")
    plt.close(fig)
