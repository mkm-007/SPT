"""Optional FastAPI layer — install with pip install -e '.[api]'."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from autograder.grade import grade
from autograder.reporter import to_json

app = FastAPI(title="NeuNet Autograder", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/grade")
async def grade_submission(hw: int = Form(...), notebook: UploadFile = File(...)):
    if hw not in (1, 2, 3, 4):
        return JSONResponse({"error": "hw must be 1-4"}, status_code=400)

    suffix = Path(notebook.filename or "submission.ipynb").suffix or ".ipynb"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await notebook.read())
        path = Path(tmp.name)

    try:
        report = grade(path, hw)
        return JSONResponse(content=__import__("json").loads(to_json(report)))
    finally:
        path.unlink(missing_ok=True)
