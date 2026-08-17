"""Tier-1 checks for HW1 (Conditional beta-VAE)."""

from __future__ import annotations

from pathlib import Path

from autograder.checks.common import grade_tier1


def grade_hw1(notebook: Path):
    return grade_tier1(
        notebook,
        hw_number=1,
        assignment="HW1",
        extra_keywords=["VAE", "KL", "beta"],
        label="Conditional beta-VAE",
    )
