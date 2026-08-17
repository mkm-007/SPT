"""Command-line interface for NeuNet Autograder."""

from __future__ import annotations

import argparse
from pathlib import Path

from autograder.grade import grade
from autograder.reporter import format_report, to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade CSC 8851 notebook submissions.")
    parser.add_argument("notebook", type=Path, help="Path to student .ipynb file")
    parser.add_argument("--hw", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    report = grade(args.notebook, args.hw)
    if args.json:
        print(to_json(report))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
