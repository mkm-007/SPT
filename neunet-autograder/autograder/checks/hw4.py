"""HW4 checks — Graph Neural Networks (pure NumPy)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from autograder.checks.base import CheckResult, GradeReport
from autograder.checks.common import check_filename, check_notebook_runs
from autograder.fixtures import hw4_reference as ref


def _call_student_functions(notebook: Path, timeout: int = 180) -> dict:
    script = textwrap.dedent(
        f"""
        import json
        import numpy as np
        from pathlib import Path
        import nbformat

        nb = nbformat.read({str(notebook)!r}, as_version=4)
        code = "\\n\\n".join(c.source for c in nb.cells if c.cell_type == "code")
        ns = {{"__name__": "__main__"}}
        exec(compile(code, {str(notebook)!r}, "exec"), ns)

        edges = {ref.EDGES!r}
        n_nodes = {ref.N_NODES}
        x = np.array({ref.X.tolist()!r}, dtype=np.float64)
        adj_fixture = np.array({ref.ADJ.tolist()!r}, dtype=np.float64)
        weights = [
            np.array({ref.W1.tolist()!r}, dtype=np.float64),
            np.array({ref.W2.tolist()!r}, dtype=np.float64),
            np.array({ref.W3.tolist()!r}, dtype=np.float64),
        ]
        biases = [
            np.array({ref.B1.tolist()!r}, dtype=np.float64),
            np.array({ref.B2.tolist()!r}, dtype=np.float64),
            np.array({ref.B3.tolist()!r}, dtype=np.float64),
        ]
        gat_w = np.array({ref.GAT_W.tolist()!r}, dtype=np.float64)
        gat_a = np.array({ref.GAT_A.tolist()!r}, dtype=np.float64)

        out = {{}}
        if "build_adjacency_matrix" not in ns:
            raise KeyError("Missing function build_adjacency_matrix")
        if "gcn_forward" not in ns:
            raise KeyError("Missing function gcn_forward")
        if "neighborhood_sample" not in ns:
            raise KeyError("Missing function neighborhood_sample")
        if "gat_forward" not in ns:
            raise KeyError("Missing function gat_forward")

        out["adj"] = np.asarray(ns["build_adjacency_matrix"](edges, n_nodes)).tolist()
        out["gcn"] = np.asarray(
            ns["gcn_forward"](x, adj_fixture, weights, biases)
        ).tolist()
        out["sample"] = ns["neighborhood_sample"](adj_fixture, 1, 2, 42)
        out["gat"] = np.asarray(
            ns["gat_forward"](x, adj_fixture, gat_w, gat_a)
        ).tolist()
        print(json.dumps({{"ok": True, "outputs": out}}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())

    import json

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    if not payload.get("ok"):
        raise RuntimeError("Failed to evaluate student functions.")
    return payload["outputs"]


def _layers_disjoint(layers: dict) -> bool:
    seen: set[int] = set()
    for name in ("output", "hidden2", "hidden1", "input"):
        for node in layers.get(name, []):
            if node in seen:
                return False
            seen.add(node)
    return True


def grade_hw4(notebook: Path) -> GradeReport:
    report = GradeReport(assignment="HW4", notebook=str(notebook))
    report.add(check_filename(notebook, 4))

    run_check, _ = check_notebook_runs(notebook)
    report.add(run_check)
    if not run_check.passed:
        for name, pts in [
            ("adjacency", 10),
            ("gcn", 15),
            ("sampling", 15),
            ("gat", 30),
        ]:
            report.add(
                CheckResult(
                    name=name,
                    passed=False,
                    points=0,
                    max_points=pts,
                    message="Skipped — notebook did not execute.",
                )
            )
        return report

    try:
        outputs = _call_student_functions(notebook)
    except Exception as exc:  # noqa: BLE001 — surface student errors to report
        for name, pts in [
            ("adjacency", 10),
            ("gcn", 15),
            ("sampling", 15),
            ("gat", 30),
        ]:
            report.add(
                CheckResult(
                    name=name,
                    passed=False,
                    points=0,
                    max_points=pts,
                    message=str(exc),
                )
            )
        return report

    adj = np.asarray(outputs["adj"], dtype=np.float64)
    report.add(
        CheckResult(
            name="adjacency",
            passed=np.allclose(adj, ref.ADJ),
            points=10 if np.allclose(adj, ref.ADJ) else 0,
            max_points=10,
            message="Adjacency matrix matches reference."
            if np.allclose(adj, ref.ADJ)
            else "Adjacency matrix does not match reference.",
        )
    )

    gcn = np.asarray(outputs["gcn"], dtype=np.float64)
    gcn_ok = np.allclose(gcn, ref.REF_GCN, atol=1e-6)
    report.add(
        CheckResult(
            name="gcn",
            passed=gcn_ok,
            points=15 if gcn_ok else 0,
            max_points=15,
            message="GCN forward pass matches reference."
            if gcn_ok
            else "GCN output differs from reference.",
        )
    )

    sample = outputs["sample"]
    sample_ok = (
        sample.get("output") == ref.REF_SAMPLE["output"]
        and _layers_disjoint(sample)
        and all(
            len(set(sample.get(layer, []))) == len(sample.get(layer, []))
            for layer in ("hidden2", "hidden1", "input")
        )
    )
    report.add(
        CheckResult(
            name="sampling",
            passed=sample_ok,
            points=15 if sample_ok else 0,
            max_points=15,
            message="Neighborhood sampling layers are valid."
            if sample_ok
            else "Sampling layers overlap or have unexpected structure.",
        )
    )

    gat = np.asarray(outputs["gat"], dtype=np.float64)
    gat_ok = np.allclose(gat, ref.REF_GAT, atol=1e-6)
    report.add(
        CheckResult(
            name="gat",
            passed=gat_ok,
            points=30 if gat_ok else 0,
            max_points=30,
            message="GAT forward pass matches reference."
            if gat_ok
            else "GAT output differs from reference.",
        )
    )

    return report
