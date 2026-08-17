"""Tier-1 checks for HW3 (Diffusion models)."""

from __future__ import annotations

from pathlib import Path

from autograder.checks.common import grade_tier1


def grade_hw3(notebook: Path):
    return grade_tier1(
        notebook,
        hw_number=3,
        assignment="HW3",
        extra_keywords=["DDPM", "DDIM", "UNet"],
        label="Diffusion models",
    )
