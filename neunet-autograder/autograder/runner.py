"""Execute student notebooks."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import nbformat


@dataclass
class NotebookRunResult:
    ok: bool
    source: str = ""
    error: str = ""


def _collect_source(notebook: Path) -> str:
    nb = nbformat.read(notebook, as_version=4)
    return "\n\n".join(
        cell.source for cell in nb.cells if cell.cell_type == "code"
    )


def run_notebook(notebook: Path, timeout: int = 180) -> NotebookRunResult:
    """Run notebook code in a subprocess."""
    source = _collect_source(notebook)
    if not source.strip():
        return NotebookRunResult(ok=False, source=source, error="Notebook has no code cells.")

    runner = textwrap.dedent(
        f"""
        from pathlib import Path
        import nbformat

        source = Path({str(notebook)!r}).read_text(encoding="utf-8")
        nb = nbformat.reads(source, as_version=4)
        code = "\\n\\n".join(
            cell.source for cell in nb.cells if cell.cell_type == "code"
        )
        namespace = {{"__name__": "__main__"}}
        exec(compile(code, {str(notebook)!r}, "exec"), namespace)
        print("OK")
        """
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(runner)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return NotebookRunResult(
            ok=False,
            source=source,
            error=f"Timed out after {timeout}s.",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "Unknown execution error."
        return NotebookRunResult(ok=False, source=source, error=err)

    return NotebookRunResult(ok=True, source=source)
