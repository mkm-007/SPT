"""Shared types for autograder checks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    passed: bool
    points: int
    max_points: int
    message: str = ""

    @property
    def earned(self) -> int:
        return self.points if self.passed else 0


@dataclass
class GradeReport:
    assignment: str
    notebook: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(r.earned for r in self.results)

    @property
    def max_score(self) -> int:
        return sum(r.max_points for r in self.results)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)
