"""Turn check results into readable reports."""

from __future__ import annotations

import json

from autograder.checks.base import GradeReport


def format_report(report: GradeReport) -> str:
    lines = [
        f"Assignment: {report.assignment}",
        f"Notebook:   {report.notebook}",
        f"Score:      {report.score}/{report.max_score}",
        "",
        "Checks:",
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"  [{status}] {result.name} ({result.earned}/{result.max_points}) — {result.message}"
        )
    return "\n".join(lines)


def to_json(report: GradeReport) -> str:
    payload = {
        "assignment": report.assignment,
        "notebook": report.notebook,
        "score": report.score,
        "max_score": report.max_score,
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "points": r.earned,
                "max_points": r.max_points,
                "message": r.message,
            }
            for r in report.results
        ],
    }
    return json.dumps(payload, indent=2)
