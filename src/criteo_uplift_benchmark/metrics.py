from __future__ import annotations

import numpy as np

from criteo_uplift_benchmark.config import BenchmarkConfig
from criteo_uplift_benchmark.schemas import PolicyRow, SegmentRow, SegmentSummary, UpliftMetrics

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def group_uplift(y: np.ndarray, t: np.ndarray) -> float:
    """Estimate observed uplift as treated rate minus control rate."""
    y = np.asarray(y)
    t = np.asarray(t)
    treated = y[t == 1]
    control = y[t == 0]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def top_decile_uplift(
    uplift: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    pct: float = 0.10,
) -> float:
    """Measure incremental uplift inside the highest predicted-uplift group."""
    uplift = np.asarray(uplift)
    n_top = max(1, int(len(uplift) * pct))
    idx = np.argsort(-uplift)[:n_top]
    return group_uplift(np.asarray(y)[idx], np.asarray(t)[idx])


def qini_curve(
    uplift: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return qini values, random baseline, and targeted population fractions."""
    uplift = np.asarray(uplift)
    y = np.asarray(y)
    t = np.asarray(t)
    order = np.argsort(-uplift)
    y_s = y[order]
    t_s = t[order]

    n1 = (t == 1).sum()
    n0 = (t == 0).sum()
    cum_t1 = np.cumsum(y_s * (t_s == 1))
    cum_t0 = np.cumsum(y_s * (t_s == 0))
    cum_n1 = np.cumsum(t_s == 1)
    cum_n0 = np.cumsum(t_s == 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        qini = np.where(
            cum_n1 > 0,
            cum_t1 - cum_t0 * (cum_n1 / np.maximum(cum_n0, 1)),
            0.0,
        )

    rand_end = float(y_s[t_s == 1].sum()) - float(y_s[t_s == 0].sum()) * (n1 / max(n0, 1))
    random_line = np.linspace(0, rand_end, len(uplift))
    x = np.arange(1, len(uplift) + 1) / len(uplift)
    return qini, random_line, x


def auuc(uplift: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Compute area between Qini curve and random baseline."""
    qini, random_line, x = qini_curve(uplift, y, t)
    return float(_trapz(qini - random_line, x))


def top_decile_relative_uplift(
    uplift: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    pct: float = 0.10,
) -> float | None:
    top = top_decile_uplift(uplift, y, t, pct=pct)
    base = group_uplift(y, t)
    if np.isnan(top) or np.isnan(base) or abs(base) < 1e-12:
        return None
    return float(top / base)


def bootstrap_auuc(
    uplift: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(uplift)
    scores = np.empty(n_boot, dtype=float)
    for boot_idx in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[boot_idx] = auuc(uplift[idx], y[idx], t[idx])
    return scores


def paired_auuc_diff_ci(
    uplift_a: np.ndarray,
    uplift_b: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot, dtype=float)
    for boot_idx in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[boot_idx] = auuc(uplift_a[idx], y[idx], t[idx]) - auuc(
            uplift_b[idx],
            y[idx],
            t[idx],
        )
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), diffs


def evaluate_predictions(
    *,
    name: str,
    uplift: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    config: BenchmarkConfig,
    split: str,
) -> UpliftMetrics:
    auuc_value = auuc(uplift, y, t)
    boot = bootstrap_auuc(
        uplift,
        y,
        t,
        n_boot=config.runtime.bootstrap_n,
        seed=config.split.seed,
    )
    return UpliftMetrics(
        split=split,
        model=name,
        auuc=auuc_value,
        normalized_auuc=auuc_value / len(y),
        auuc_ci_low=float(np.percentile(boot, 2.5)),
        auuc_ci_high=float(np.percentile(boot, 97.5)),
        top_decile_uplift=top_decile_uplift(uplift, y, t),
        top_decile_relative_uplift=top_decile_relative_uplift(uplift, y, t),
        population_uplift=group_uplift(y, t),
    )


def policy_curve(
    uplift: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    pcts: np.ndarray | None = None,
) -> list[PolicyRow]:
    pcts = np.arange(0.05, 1.01, 0.05) if pcts is None else pcts
    order = np.argsort(-uplift)
    y_sorted = y[order]
    t_sorted = t[order]
    rows: list[PolicyRow] = []
    for pct in pcts:
        target_n = max(1, int(len(uplift) * pct))
        inc_rate = group_uplift(y_sorted[:target_n], t_sorted[:target_n])
        incremental_events = inc_rate * target_n
        rows.append(
            PolicyRow(
                target_pct=float(pct),
                target_n=target_n,
                incremental_rate=inc_rate,
                incremental_events=incremental_events,
                events_per_targeted_user=incremental_events / target_n,
            )
        )
    return rows


def segment_policy(
    *,
    uplift: np.ndarray,
    prob_t0: np.ndarray,
    config: BenchmarkConfig,
    baseline_score: np.ndarray | None = None,
) -> SegmentSummary:
    if np.all(np.isnan(prob_t0)):
        prob_t0 = baseline_score if baseline_score is not None else np.zeros_like(uplift)

    positive = uplift > config.policy.positive_uplift_threshold
    negative = uplift < config.policy.negative_uplift_threshold
    neutral = ~(positive | negative)
    neutral_p0 = prob_t0[neutral]
    baseline_cut = np.quantile(neutral_p0, 0.80) if len(neutral_p0) else np.inf
    neutral_high_baseline = neutral & (prob_t0 >= baseline_cut)
    neutral_low_baseline = neutral & ~neutral_high_baseline

    raw_rows = [
        (
            "High positive predicted uplift",
            f"tau > {config.policy.positive_uplift_threshold}",
            "Target first",
            int(positive.sum()),
        ),
        (
            "Negative predicted uplift",
            f"tau < {config.policy.negative_uplift_threshold}",
            "Avoid contact",
            int(negative.sum()),
        ),
        (
            "Near-zero uplift, high baseline response",
            "neutral tau and top 20% baseline score",
            "Do not prioritize paid contact",
            int(neutral_high_baseline.sum()),
        ),
        (
            "Near-zero uplift, low baseline response",
            "neutral tau and lower baseline score",
            "Low priority",
            int(neutral_low_baseline.sum()),
        ),
    ]
    rows = [
        SegmentRow(
            policy_segment=segment,
            rule=rule,
            action=action,
            count=count,
            share=count / len(uplift),
        )
        for segment, rule, action, count in raw_rows
    ]
    return SegmentSummary(
        rows=rows,
        avoidable_campaign_spend=int(negative.sum()) * config.policy.campaign_cost_per_contact,
    )
