import numpy as np

from criteo_uplift_benchmark.config import BenchmarkConfig
from criteo_uplift_benchmark.metrics import auuc, group_uplift, qini_curve, top_decile_uplift
from criteo_uplift_benchmark.pipeline import build_evidence_checks
from criteo_uplift_benchmark.schemas import SegmentSummary


def test_group_uplift_uses_treatment_assignment() -> None:
    y = np.array([1, 0, 0, 0])
    t = np.array([1, 1, 0, 0])

    assert group_uplift(y, t) == 0.5


def test_top_decile_uplift_uses_treatment_not_response_lift() -> None:
    uplift = np.array([0.9, 0.8, 0.1, 0.0])
    y = np.array([1, 0, 0, 0])
    t = np.array([1, 0, 1, 0])

    assert top_decile_uplift(uplift, y, t, pct=0.50) == 1.0


def test_qini_curve_shapes() -> None:
    uplift = np.array([0.4, 0.3, 0.2, 0.1])
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 0, 1, 0])

    qini, random_line, x = qini_curve(uplift, y, t)

    assert len(qini) == len(uplift)
    assert len(random_line) == len(uplift)
    assert len(x) == len(uplift)


def test_auuc_returns_float() -> None:
    uplift = np.array([0.4, 0.3, 0.2, 0.1])
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 0, 1, 0])

    assert isinstance(auuc(uplift, y, t), float)


def test_evidence_checks_flag_uncertain_winner() -> None:
    class Split:
        t_test = np.array([1, 0, 1, 0])

    checks = build_evidence_checks(
        config=BenchmarkConfig.quick(sample_size=100),
        split=Split(),
        winner="S-Learner",
        runner_up="DR-Learner",
        paired_diff_ci=(-1.0, 1.0),
        segments=SegmentSummary(rows=[], avoidable_campaign_spend=0.0),
    )

    by_name = {check.check: check for check in checks}
    assert by_name["winner_margin"].status == "review"
    assert by_name["policy_segment_coverage"].status == "review"

