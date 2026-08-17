"""Tier-1 checks for HW2 (Conditional WGAN-GP)."""

from __future__ import annotations

from pathlib import Path

from autograder.checks.common import grade_tier1


def grade_hw2(notebook: Path):
    return grade_tier1(
        notebook,
        hw_number=2,
        assignment="HW2",
        extra_keywords=["WGAN", "gradient_penalty", "EMA"],
        label="Conditional WGAN-GP",
    )
