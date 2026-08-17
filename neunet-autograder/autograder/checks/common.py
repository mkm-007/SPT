"""Checks shared across all homework assignments."""

from __future__ import annotations

import re
from pathlib import Path

from autograder.checks.base import CheckResult, GradeReport
from autograder.runner import NotebookRunResult, run_notebook


_HW_PATTERN = re.compile(r"CSC8851_S2026_HW(\d)_.+\.ipynb$", re.IGNORECASE)


def check_filename(notebook: Path, hw_number: int) -> CheckResult:
    match = _HW_PATTERN.match(notebook.name)
    if not match:
        return CheckResult(
            name="filename",
            passed=False,
            points=0,
            max_points=5,
            message=(
                f"Expected CSC8851_S2026_HW{hw_number}_YourName.ipynb, "
                f"got {notebook.name}"
            ),
        )
    if int(match.group(1)) != hw_number:
        return CheckResult(
            name="filename",
            passed=False,
            points=0,
            max_points=5,
            message=f"Filename says HW{match.group(1)} but grading HW{hw_number}.",
        )
    return CheckResult(
        name="filename",
        passed=True,
        points=5,
        max_points=5,
        message="Filename follows course convention.",
    )


def check_notebook_runs(notebook: Path, timeout: int = 180) -> tuple[CheckResult, NotebookRunResult]:
    run = run_notebook(notebook, timeout=timeout)
    if run.ok:
        return (
            CheckResult(
                name="executes",
                passed=True,
                points=15,
                max_points=15,
                message="Notebook executed without errors.",
            ),
            run,
        )
    return (
        CheckResult(
            name="executes",
            passed=False,
            points=0,
            max_points=15,
            message=f"Execution failed: {run.error}",
        ),
        run,
    )


def check_code_contains(
    source: str,
    keywords: list[str],
    name: str,
    points: int,
    label: str,
) -> CheckResult:
    missing = [kw for kw in keywords if kw not in source]
    if missing:
        return CheckResult(
            name=name,
            passed=False,
            points=0,
            max_points=points,
            message=f"Missing expected content for {label}: {', '.join(missing)}",
        )
    return CheckResult(
        name=name,
        passed=True,
        points=points,
        max_points=points,
        message=f"{label} core components detected in notebook.",
    )


def grade_tier1(
    notebook: Path,
    hw_number: int,
    assignment: str,
    extra_keywords: list[str],
    label: str,
) -> GradeReport:
    report = GradeReport(assignment=assignment, notebook=str(notebook))
    report.add(check_filename(notebook, hw_number))

    run_check, run = check_notebook_runs(notebook)
    report.add(run_check)
    if run.ok:
        report.add(
            check_code_contains(
                run.source,
                extra_keywords,
                name="components",
                points=10,
                label=label,
            )
        )
    else:
        report.add(
            CheckResult(
                name="components",
                passed=False,
                points=0,
                max_points=10,
                message="Skipped — notebook did not execute.",
            )
        )
    return report
