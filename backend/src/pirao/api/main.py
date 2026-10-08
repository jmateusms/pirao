"""HTTP API.

A thin layer over :mod:`pirao.core`; it contains no statistics of its own.

The one design choice worth calling out is that the **registries are served,
not duplicated**.  ``GET /api/meta`` returns every likelihood, every prior
family, each family's hyperparameter names and constraints, and which families
apply to which parameter roles.  The frontend renders its dropdowns and its
dynamic hyperparameter fields from that payload, so adding a prior is a change
to one Python dict and nothing else.
"""

from __future__ import annotations

import io
import json
import math
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .. import __version__
from ..core import bundle as bundle_mod
from ..core.compile import ToolchainError, toolchain_fingerprint
from ..core.data import (
    ALL_COLUMNS,
    OPTIONAL_COLUMNS,
    RESERVED_COLUMNS,
    DataError,
    template_rows,
    validate_table,
)
from ..core.examples import ExampleNotFound, get_example, list_examples
from ..core.jobs import JobManager
from ..core.preview import prior_preview
from ..core.registry_likelihoods import LIKELIHOODS
from ..core.registry_priors import PRIORS
from ..core.resolve import SpecError, resolve
from ..core.spec import SAMPLER_HELP, ModelSpec, RunRequest
from ..locales import translate

RUNS_ROOT = Path(
    __import__("os").environ.get("PIRAO_RUNS_DIR")
    or __import__("os").environ.get("RELIMCMC_RUNS_DIR", "/tmp/pirao-runs")
)

app = FastAPI(
    title="pirao",
    version=__version__,
    description="Modular Bayesian reliability MCMC builder",
)
# No CORS middleware, deliberately.  Every way of running pirão serves the web
# app and the API from one origin (`pirao gui`, nginx in Docker, the Vite proxy
# in development), so nothing needs cross-origin access -- and allowing it
# would let any page open in the same browser read the runs and download their
# bundles, data included.

jobs = JobManager(RUNS_ROOT)


def _json_number(value: float | None) -> float | None:
    """JSON has no infinity; the UI shows an empty bound instead."""
    if value is None or not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# metadata: everything the UI needs to build its forms


@app.get("/api/meta")
def get_meta(lang: str = "en") -> dict[str, Any]:
    """The registries, with their labels and help in ``lang`` (en or pt)."""

    def tr(text: str) -> str:
        return translate(text, lang)

    likelihoods = [
        {
            "id": lik.id,
            "label": tr(lik.label),
            "description": tr(lik.description),
            "mission": lik.mission,
            "required_columns": list(lik.required_columns),
            "optional_columns": list(OPTIONAL_COLUMNS),
            "parameters": [
                {
                    "name": p.name,
                    "role": p.role,
                    "label": tr(p.display_label),
                    "description": tr(p.description),
                    "natural_lower": _json_number(p.nat_lo),
                    "natural_upper": _json_number(p.nat_hi),
                    "presets": [
                        {"label": tr(label), "lower": lo, "upper": hi}
                        for label, lo, hi in p.presets
                    ],
                }
                for p in lik.params
            ],
        }
        for lik in LIKELIHOODS.values()
    ]

    priors = [
        {
            "id": f.id,
            "label": tr(f.label),
            "notes": tr(f.notes),
            "is_proper": f.is_proper,
            "support_lower": _json_number(f.supp_lo),
            "support_upper": _json_number(f.supp_hi),
            "roles": list(f.roles),
            "hyperparameters": [
                {
                    "name": h.name,
                    "label": tr(h.label),
                    "default": h.default,
                    "minimum": _json_number(h.lo),
                    "maximum": _json_number(h.hi),
                    "exclusive_minimum": h.exclusive_lo,
                }
                for h in f.hypers
            ],
        }
        for f in PRIORS.values()
    ]

    return {
        "likelihoods": likelihoods,
        "priors": priors,
        "sampler_help": {key: tr(text) for key, text in SAMPLER_HELP.items()},
        "columns": {
            "all": list(ALL_COLUMNS),
            "optional": list(OPTIONAL_COLUMNS),
            "reserved": list(RESERVED_COLUMNS),
        },
    }


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    """Report whether the Stan toolchain is usable, before a user waits on it."""
    try:
        return {
            "ok": True,
            "version": __version__,
            "toolchain": toolchain_fingerprint(),
        }
    except ToolchainError as exc:
        return {
            "ok": False,
            "version": __version__,
            "error": str(exc),
            "hint": "Run `pirao install-stan` once (a few minutes; needs a C++ "
            "compiler).",
        }


# ---------------------------------------------------------------------------
# specification: resolve bounds and preview priors before anything is run


class ValidateSpecResponse(BaseModel):
    ok: bool
    errors: list[str] = []
    warnings: list[str] = []
    parameters: list[dict[str, Any]] = []
    stan_source: str | None = None


@app.post("/api/specs/validate", response_model=ValidateSpecResponse)
def validate_spec(spec: ModelSpec) -> ValidateSpecResponse:
    """Resolve a spec and describe the prior the user will actually get.

    The effective bounds are an intersection, applied silently by the engine, so
    the UI has to show them; and the prior preview is the *truncated* density,
    because a bound that quietly discards half the prior mass is the failure
    mode this endpoint exists to make visible.
    """
    try:
        resolved = resolve(spec)
    except SpecError as exc:
        return ValidateSpecResponse(ok=False, errors=exc.messages)

    parameters = []
    for param in resolved.params:
        hyper = {h.stan_name.rsplit("_prior_", 1)[1]: h.value for h in param.hypers}
        if param.family == "uniform":
            hyper = dict(spec.params[param.name].prior.hyper)
        preview = prior_preview(param.family, hyper, param.lo, param.hi)
        parameters.append(
            {
                "name": param.name,
                "role": param.role,
                "family": param.family,
                "family_label": param.family_label,
                "effective_lower": _json_number(param.lo),
                "effective_upper": _json_number(param.hi),
                "truncated_below": param.truncated_below,
                "truncated_above": param.truncated_above,
                "declaration": param.declaration,
                "preview": {
                    "x": preview.x,
                    "pdf": preview.pdf,
                    "retained_mass": preview.retained_mass,
                    "median": preview.median,
                    "mean": preview.mean,
                    "q05": preview.q05,
                    "q95": preview.q95,
                    "warning": preview.warning,
                },
            }
        )

    from ..core.render import render_resolved

    return ValidateSpecResponse(
        ok=True,
        warnings=list(resolved.warnings),
        parameters=parameters,
        stan_source=render_resolved(resolved).source,
    )


# ---------------------------------------------------------------------------
# data: templates, upload parsing, validation


class ValidateDataRequest(BaseModel):
    spec: ModelSpec
    rows: list[dict[str, Any]] = []


@app.post("/api/data/validate")
def validate_data(request: ValidateDataRequest) -> dict[str, Any]:
    try:
        table = validate_table(request.spec, request.rows)
    except DataError as exc:
        return {
            "ok": False,
            "errors": [
                {"row": e.row, "column": e.column, "message": e.message}
                for e in exc.errors
            ],
        }
    except SpecError as exc:
        return {
            "ok": False,
            "errors": [
                {"row": None, "column": None, "message": m} for m in exc.messages
            ],
        }
    return {
        "ok": True,
        "errors": [],
        "warnings": table.warnings,
        "rows": table.rows,
        "ignored_columns": table.ignored_columns,
    }


@app.get("/api/templates/{likelihood}.csv")
def get_csv_template(likelihood: str) -> PlainTextResponse:
    header, rows = _template_or_404(likelihood)
    lines = [",".join(header)]
    lines += [",".join(str(row.get(c, "")) for c in header) for row in rows]
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        headers={
            "Content-Disposition": f'attachment; filename="{likelihood}_template.csv"'
        },
    )


@app.get("/api/templates/{likelihood}.xlsx")
def get_xlsx_template(likelihood: str) -> StreamingResponse:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.worksheet.datavalidation import DataValidation

    header, rows = _template_or_404(likelihood)
    lik = LIKELIHOODS[likelihood]

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "data"
    sheet.append(header)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"
    for row in rows:
        sheet.append([row.get(c, "") for c in header])

    last = 1000
    if "failure" in header:
        column = chr(ord("A") + header.index("failure"))
        rule = DataValidation(type="list", formula1='"0,1"', allow_blank=False)
        sheet.add_data_validation(rule)
        rule.add(f"{column}2:{column}{last}")
    if "relevance" in header:
        column = chr(ord("A") + header.index("relevance"))
        rule = DataValidation(
            type="decimal", operator="between", formula1=0, formula2=1
        )
        sheet.add_data_validation(rule)
        rule.add(f"{column}2:{column}{last}")

    notes = workbook.create_sheet("instructions")
    notes.append(["column", "meaning"])
    notes["A1"].font = Font(bold=True)
    notes["B1"].font = Font(bold=True)
    guide = {
        "device": "Label for the unit.  Optional; defaults to the row number.",
        "time": "Failure time when failure = 1, otherwise the censoring time.",
        "n": "Number of demands or actuations.",
        "failure": "1 = this unit failed.  0 = it survived (censored).",
        "failures": "Number of failures in this group of n trials.",
        "relevance": (
            "Weight between 0 and 1.  Use less than 1 for data that is only "
            "partly relevant to this device.  Optional; defaults to 1."
        ),
    }
    for column in header:
        notes.append([column, guide.get(column, "")])
    notes.append([])
    notes.append(["model", lik.label])
    notes.append(["", lik.description])

    stream = io.BytesIO()
    workbook.save(stream)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{likelihood}_template.xlsx"'
        },
    )


@app.post("/api/data/parse")
async def parse_upload(file: UploadFile) -> dict[str, Any]:
    """Read an uploaded CSV or XLSX into tidy rows, preserving file order."""
    import pandas as pd

    raw = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xlsm", ".xltx")):
            frame = pd.read_excel(io.BytesIO(raw), sheet_name=0)
        else:
            frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"That file could not be read as a table ({exc}).  Save it as "
                f"CSV or XLSX and try again."
            ),
        ) from exc

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    known = [c for c in frame.columns if c in ALL_COLUMNS]
    if not known:
        raise HTTPException(
            status_code=400,
            detail=(
                "No recognised columns were found.  The first row should name "
                f"the columns, using any of: {', '.join(ALL_COLUMNS)}."
            ),
        )

    frame = frame.where(frame.notna(), None)
    return {
        "rows": frame.to_dict(orient="records"),
        "columns": list(frame.columns),
        "ignored_columns": [c for c in frame.columns if c not in ALL_COLUMNS],
    }


def _template_or_404(likelihood: str):
    if likelihood not in LIKELIHOODS:
        raise HTTPException(status_code=404, detail=f"Unknown model {likelihood!r}.")
    return template_rows(likelihood)


# ---------------------------------------------------------------------------
# examples: ready-made analyses on public or made-up data


@app.get("/api/examples")
def get_examples() -> list[dict[str, Any]]:
    return list_examples()


@app.get("/api/examples/{example_id}")
def get_one_example(example_id: str) -> dict[str, Any]:
    try:
        return get_example(example_id)
    except ExampleNotFound:
        raise HTTPException(
            status_code=404, detail=f"Unknown example {example_id!r}."
        ) from None


# ---------------------------------------------------------------------------
# runs


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
    """Start a run.  Returns immediately with an id to poll or stream."""
    try:
        resolve(request.spec)
        validate_table(request.spec, request.rows)
    except SpecError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.messages}) from exc
    except DataError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "errors": [
                    {"row": e.row, "column": e.column, "message": e.message}
                    for e in exc.errors
                ]
            },
        ) from exc

    run_id = uuid.uuid4().hex[:12]
    state = jobs.submit(run_id, request.spec, request.rows, request.sampler)
    return state.public()


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return jobs.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    state = _job_or_404(run_id)
    payload = state.public()
    manifest = state.root / bundle_mod.MANIFEST if state.root else None
    if manifest and manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        payload["summary"] = data.get("summary")
        payload["elapsed"] = data.get("elapsed")
        payload["seed"] = data.get("sampler", {}).get("seed")
    return payload


@app.get("/api/runs/{run_id}/events")
async def stream_run(run_id: str) -> StreamingResponse:
    """Server-sent events for the run's stages.

    Compilation is the stage that actually takes time, so it is reported like
    any other rather than hidden behind a spinner labelled "sampling".
    """
    import asyncio

    _job_or_404(run_id)

    async def generate():
        index = 0
        while True:
            for message in jobs.events_since(run_id, index):
                index += 1
                yield f"data: {json.dumps(message)}\n\n"
            state = jobs.get(run_id)
            if state is None or state.status in ("done", "failed", "cancelled"):
                status = str(state.status) if state else "unknown"
                yield f"data: {json.dumps({'kind': 'end', 'status': status})}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/runs/{run_id}")
def cancel_run(run_id: str) -> dict[str, Any]:
    _job_or_404(run_id)
    return {"cancelled": jobs.cancel(run_id)}


@app.get("/api/runs/{run_id}/posterior")
def get_posterior(run_id: str) -> dict[str, Any]:
    """Thinned draws for plotting.  Downloads serve the full file instead."""
    path = _artifact_or_404(run_id, "posterior.json")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/source")
def get_source(run_id: str) -> PlainTextResponse:
    return PlainTextResponse(_artifact_or_404(run_id, "model.stan").read_text())


@app.get("/api/runs/{run_id}/draws")
def download_draws(run_id: str, format: str = "csv") -> FileResponse:
    state = _job_or_404(run_id)
    assert state.root is not None

    if format == "parquet":
        path = state.root / "draws.parquet"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="Parquet output is unavailable; install pyarrow, or use CSV.",
            )
        return FileResponse(
            path,
            filename=f"draws_{run_id}.parquet",
            media_type="application/octet-stream",
        )

    csv_path = state.root / "draws.csv"
    if not csv_path.exists():
        parquet = state.root / "draws.parquet"
        if not parquet.exists():
            raise HTTPException(status_code=404, detail="This run has no draws yet.")
        import pandas as pd

        pd.read_parquet(parquet).to_csv(csv_path, index=False)
    return FileResponse(csv_path, filename=f"draws_{run_id}.csv", media_type="text/csv")


@app.get("/api/runs/{run_id}/bundle")
def download_bundle(run_id: str) -> FileResponse:
    """The whole analysis: spec, data, seed, generated source, toolchain, draws."""
    state = _job_or_404(run_id)
    assert state.root is not None
    if not (state.root / bundle_mod.MANIFEST).exists():
        raise HTTPException(status_code=404, detail="This run has not finished yet.")
    archive = state.root.parent / f"{run_id}.zip"
    bundle_mod.to_zip(state.root, archive)
    return FileResponse(
        archive, filename=f"analysis_{run_id}.zip", media_type="application/zip"
    )


def _job_or_404(run_id: str):
    state = jobs.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No run with id {run_id!r}.")
    return state


def _artifact_or_404(run_id: str, name: str) -> Path:
    state = _job_or_404(run_id)
    assert state.root is not None
    path = state.root / name
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not available: the run may still be going.",
        )
    return path
