"""Dispatch grading to per-assignment modules."""

from __future__ import annotations

from pathlib import Path

from autograder.checks.base import GradeReport
from autograder.checks.hw1 import grade_hw1
from autograder.checks.hw2 import grade_hw2
from autograder.checks.hw3 import grade_hw3
from autograder.checks.hw4 import grade_hw4

GRADERS = {
    1: grade_hw1,
    2: grade_hw2,
    3: grade_hw3,
    4: grade_hw4,
}


def grade(notebook: Path, hw: int) -> GradeReport:
    if hw not in GRADERS:
        raise ValueError(f"Unsupported homework number: {hw}")
    path = Path(notebook)
    if not path.exists():
        raise FileNotFoundError(path)
    return GRADERS[hw](path)
