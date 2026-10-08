"""Background run management.

Runs execute in a **child process**, not a thread.  That is not incidental:
cmdstanpy offers no cooperative cancel, and each chain is a separate CmdStan
subprocess, so the only way to stop a run on demand is to terminate the process
group.  A thread could not do it.

The child writes its outputs to the run directory and sends back small JSON
payloads, so nothing has to pickle a ``CmdStanMCMC`` across the boundary.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import bundle as bundle_mod
from .data import DataError
from .resolve import SpecError
from .sample import Progress, Stage
from .spec import ModelSpec, SamplerConfig

#: Cap the draws sent to the browser for plotting.  Downloads are unaffected --
#: they stream the full file from disk.
PLOT_DRAW_LIMIT = 4000


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobState:
    id: str
    status: JobStatus = JobStatus.QUEUED
    stage: str = Stage.RESOLVE
    message: str = "Queued"
    fraction: float | None = None
    error: str | None = None
    #: User-facing validation problems, located by row and column where known.
    error_details: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_utc: str = ""
    root: Path | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": str(self.status),
            "stage": str(self.stage),
            "message": self.message,
            "fraction": self.fraction,
            "error": self.error,
            "error_details": self.error_details,
            "warnings": self.warnings,
            "created_utc": self.created_utc,
        }


def _child(
    run_id: str,
    root_str: str,
    spec_json: str,
    rows: list[dict[str, Any]],
    sampler_json: str,
    channel: Any,
) -> None:
    """Execute one run.  Runs in its own process; never returns a value."""
    os.setpgrp()  # so the parent can signal this process and its CmdStan children

    from .results import (
        as_dict,
        data_summary,
        draws_frame,
        kaplan_meier,
        prior_vs_posterior,
        reliability_curve,
        reliability_over_time,
        summarize,
    )
    from .sample import run as run_sampling

    root = Path(root_str)
    root.mkdir(parents=True, exist_ok=True)

    def emit(progress: Progress) -> None:
        channel.put(
            {
                "kind": "progress",
                "stage": str(progress.stage),
                "message": progress.message,
                "fraction": progress.fraction,
            }
        )

    try:
        spec = ModelSpec.model_validate_json(spec_json)
        sampler = SamplerConfig.model_validate_json(sampler_json)

        result = run_sampling(
            spec, rows, sampler, progress=emit, output_dir=root / "cmdstan"
        )

        emit(Progress(Stage.SUMMARIZE, "Summarising the draws"))
        summary = summarize(result)
        summary_payload = as_dict(summary)

        curve = reliability_curve(result)
        if curve is not None:
            summary_payload["reliability_curve"] = curve.to_dict(orient="records")

        over_time = reliability_over_time(result)
        if over_time is not None:
            summary_payload["reliability_over_time"] = over_time
        summary_payload["data"] = data_summary(result)
        empirical = kaplan_meier(result)
        if empirical is not None:
            summary_payload["empirical"] = empirical

        frame = draws_frame(result)
        try:
            frame.to_parquet(root / "draws.parquet", index=False)
        except Exception:  # pyarrow is optional
            frame.to_csv(root / "draws.csv", index=False)

        # A compact, downsampled slice for the browser's plots.
        names = [p for p in result.parameter_names if p in frame.columns]
        extras, labels = _derived_columns(frame, result)
        plot_cols = names + extras
        step = max(1, len(frame) // PLOT_DRAW_LIMIT)
        thinned = frame.iloc[::step]
        (root / "posterior.json").write_text(
            json.dumps(
                {
                    "parameters": names,
                    "derived": extras,
                    "labels": labels,
                    "columns": {c: thinned[c].tolist() for c in plot_cols},
                    "chain": thinned["chain__"].astype(int).tolist()
                    if "chain__" in thinned
                    else [],
                    "draw": thinned["draw__"].astype(int).tolist()
                    if "draw__" in thinned
                    else list(range(len(thinned))),
                    "total_draws": int(len(frame)),
                    "thinned_by": step,
                    "prior_posterior": prior_vs_posterior(result),
                }
            ),
            encoding="utf-8",
        )

        bundle_mod.save(
            root=root,
            spec=spec,
            rows=rows,
            sampler=result.sampler,
            stan_source=result.rendered.source,
            summary=summary_payload,
            warnings=result.warnings,
            extra={"run_id": run_id, "elapsed": result.elapsed},
        )
        channel.put(
            {
                "kind": "done",
                "warnings": result.warnings,
                "elapsed": result.elapsed,
            }
        )
    except (SpecError, DataError) as exc:
        channel.put(
            {
                "kind": "failed",
                "error": _friendly(exc),
                "details": _details(exc),
            }
        )
    except Exception as exc:
        channel.put(
            {
                "kind": "failed",
                "error": _friendly(exc),
                "details": [],
                "traceback": traceback.format_exc(),
            }
        )


#: Derived quantities worth plotting beside the parameters themselves.  The
#: reliability *curve* is deliberately absent: it is 121 columns of picture,
#: already summarised into the manifest.
_DERIVED = ("reliability", "mttf", "b10", "mean_demands_to_failure")


def _derived_columns(frame: Any, result: Any) -> tuple[list[str], dict[str, str]]:
    """Plottable derived columns, with display labels for the indexed ones.

    cmdstanpy names vector entries from 1 (``reliability[1]``), so the mission
    labels are matched against that convention rather than ArviZ's.
    """
    from .results import reliability_labels

    labels: dict[str, str] = {}
    mission = reliability_labels(result)
    for index, label in enumerate(mission, start=1):
        labels[f"reliability[{index}]"] = label
    if len(mission) == 1:
        labels["reliability"] = mission[0]

    columns = [c for c in frame.columns if c.split("[")[0] in _DERIVED]
    return columns, {c: labels[c] for c in columns if c in labels}


def _friendly(exc: Exception) -> str:
    """Turn an exception into something a reliability engineer can act on."""
    if isinstance(exc, SpecError):
        return "The model settings are not usable yet: " + "  ".join(exc.messages)
    if isinstance(exc, DataError):
        return "The data table has problems that must be fixed first."

    text = str(exc)
    if "Initialization" in text and "failed" in text:
        return (
            "The sampler could not find a valid starting point.  This usually "
            "means the priors and bounds leave no region the data can support "
            "-- widen the bounds, or move the prior toward where the data lies."
        )
    if "not-a-number" in text or "NaN" in text:
        return (
            "The model produced an undefined value.  This normally comes from a "
            "row whose density is zero under the current bounds; check for "
            "zero times or zero-demand failures."
        )
    if "CmdStan" in text or "cmdstan" in text:
        return (
            "The Stan toolchain reported a problem.  The generated program is "
            "available for inspection under 'Show generated source'.\n\n" + text
        )
    return text


def _details(exc: Exception) -> list[dict[str, Any]]:
    if isinstance(exc, DataError):
        return [
            {"row": e.row, "column": e.column, "message": e.message} for e in exc.errors
        ]
    if isinstance(exc, SpecError):
        return [{"row": None, "column": None, "message": m} for m in exc.messages]
    return []


class JobManager:
    """Tracks runs, one child process each."""

    def __init__(self, root: Path, max_concurrent: int = 2):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = {}
        self._procs: dict[str, mp.Process] = {}
        self._queues: dict[str, Any] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")
        self._semaphore = threading.BoundedSemaphore(max_concurrent)

    def submit(
        self,
        run_id: str,
        spec: ModelSpec,
        rows: list[dict[str, Any]],
        sampler: SamplerConfig,
    ) -> JobState:
        root = self.root / run_id
        state = JobState(
            id=run_id,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            root=root,
        )
        channel = self._ctx.Queue()
        proc = self._ctx.Process(
            target=_child,
            args=(
                run_id,
                str(root),
                spec.model_dump_json(),
                rows,
                sampler.model_dump_json(),
                channel,
            ),
            daemon=True,
        )
        with self._lock:
            self._jobs[run_id] = state
            self._queues[run_id] = channel
            self._events[run_id] = []
        proc.start()
        with self._lock:
            self._procs[run_id] = proc
            state.status = JobStatus.RUNNING
            state.message = "Starting"
        threading.Thread(target=self._drain, args=(run_id,), daemon=True).start()
        return state

    def _drain(self, run_id: str) -> None:
        channel = self._queues[run_id]
        state = self._jobs[run_id]
        while True:
            try:
                message = channel.get(timeout=0.5)
            except queue.Empty:
                proc = self._procs.get(run_id)
                if proc is not None and not proc.is_alive():
                    if state.status is JobStatus.RUNNING:
                        state.status = JobStatus.FAILED
                        state.error = (
                            "The run stopped unexpectedly.  This is usually the "
                            "operating system reclaiming memory."
                        )
                        self._record(run_id, {"kind": "failed", "error": state.error})
                    return
                continue

            kind = message.get("kind")
            if kind == "progress":
                state.stage = message["stage"]
                state.message = message["message"]
                state.fraction = message.get("fraction")
            elif kind == "done":
                state.status = JobStatus.DONE
                state.stage = str(Stage.DONE)
                state.message = "Finished"
                state.fraction = 1.0
                state.warnings = message.get("warnings", [])
            elif kind == "failed":
                state.status = JobStatus.FAILED
                state.error = message.get("error")
                state.error_details = message.get("details", [])
            self._record(run_id, message)
            if kind in ("done", "failed"):
                return

    def _record(self, run_id: str, message: dict[str, Any]) -> None:
        with self._lock:
            self._events.setdefault(run_id, []).append(message)

    def get(self, run_id: str) -> JobState | None:
        return self._jobs.get(run_id)

    def events_since(self, run_id: str, index: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(run_id, [])[index:])

    def cancel(self, run_id: str) -> bool:
        proc = self._procs.get(run_id)
        state = self._jobs.get(run_id)
        if proc is None or state is None or not proc.is_alive():
            return False
        try:
            # Signal the whole group: CmdStan chains are grandchildren.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        state.status = JobStatus.CANCELLED
        state.message = "Cancelled"
        self._record(run_id, {"kind": "cancelled"})
        return True

    def list(self) -> list[dict[str, Any]]:
        return [s.public() for s in self._jobs.values()]
