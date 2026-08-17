from pathlib import Path

from autograder.checks.hw4 import grade_hw4


SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "CSC8851_S2026_HW4_Sample.ipynb"


def test_hw4_sample_scores_full_points():
    report = grade_hw4(SAMPLE)
    assert report.score == report.max_score
    assert report.score == 90  # 5 + 15 + 10 + 15 + 15 + 30


def test_hw4_sample_all_checks_pass():
    report = grade_hw4(SAMPLE)
    assert all(r.passed for r in report.results)
