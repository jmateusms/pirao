"""Run a specification end to end: resolve, render, compile, sample.

The stage sequence is exposed rather than hidden because of a measurement worth
remembering: on realistic reliability datasets, sampling one of these models
takes milliseconds, while a cold compile takes ten to forty seconds.  A progress
indicator built around "the long MCMC run" would sit at zero through the only
wait that exists.  So ``compile`` is a first-class stage with its own progress
callback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .compile import CompiledModel, compile_model, is_cached
from .data import ValidatedTable, to_stan_data, validate_table
from .render import RenderedModel, render_resolved
from .resolve import ResolvedModel, resolve
from .spec import ModelSpec, SamplerConfig


class Stage(StrEnum):
    RESOLVE = "resolve"
    RENDER = "render"
    COMPILE = "compile"
    SAMPLE = "sample"
    SUMMARIZE = "summarize"
    DONE = "done"


@dataclass
class Progress:
    stage: Stage
    message: str
    #: ``None`` when a stage cannot report a meaningful fraction -- which is the
    #: honest answer for compilation, since the toolchain emits nothing
    #: parseable.
    fraction: float | None = None


ProgressCallback = Callable[[Progress], None]


@dataclass
class RunResult:
    spec: ModelSpec
    resolved: ResolvedModel
    rendered: RenderedModel
    compiled: CompiledModel
    table: ValidatedTable
    stan_data: dict[str, Any]
    sampler: SamplerConfig
    fit: Any  # cmdstanpy.CmdStanMCMC
    seed: int
    elapsed: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.resolved.params]


def _noop(_: Progress) -> None:
    return None


def run(
    spec: ModelSpec,
    rows: list[dict[str, Any]],
    sampler: SamplerConfig | None = None,
    progress: ProgressCallback | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
) -> RunResult:
    """Fit ``spec`` to ``rows`` and return everything needed to report on it."""
    import numpy as np

    sampler = sampler or SamplerConfig()
    emit = progress or _noop
    elapsed: dict[str, float] = {}
    warnings: list[str] = []

    t0 = time.perf_counter()
    emit(Progress(Stage.RESOLVE, "Checking the model specification"))
    resolved = resolve(spec)
    warnings.extend(resolved.warnings)
    elapsed["resolve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    emit(Progress(Stage.RENDER, "Assembling the Stan program"))
    rendered = render_resolved(resolved)
    elapsed["render"] = time.perf_counter() - t0

    table = validate_table(spec, rows, resolved=resolved)
    warnings.extend(table.warnings)
    stan_data = to_stan_data(spec, table, resolved=resolved)

    t0 = time.perf_counter()
    if is_cached(rendered, cache_dir):
        emit(Progress(Stage.COMPILE, "Reusing the compiled model", 1.0))
    else:
        emit(
            Progress(
                Stage.COMPILE,
                "Compiling the Stan program (this happens once per model "
                "structure and takes a few seconds)",
            )
        )
    compiled = compile_model(rendered, cache_dir=cache_dir)
    elapsed["compile"] = time.perf_counter() - t0

    seed = sampler.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0] % (2**31 - 1))

    t0 = time.perf_counter()
    emit(
        Progress(
            Stage.SAMPLE,
            f"Sampling {sampler.chains} chains "
            f"({sampler.iter_warmup} warmup + {sampler.iter_sampling} draws each)",
        )
    )
    fit = compiled.model.sample(
        data=stan_data,
        chains=sampler.chains,
        parallel_chains=sampler.parallel_chains or sampler.chains,
        iter_warmup=sampler.iter_warmup,
        iter_sampling=sampler.iter_sampling,
        seed=seed,
        adapt_delta=sampler.adapt_delta,
        max_treedepth=sampler.max_treedepth,
        output_dir=str(output_dir) if output_dir else None,
        # CmdStan writes 6 significant figures by default, which is coarse for
        # draws a user is going to download and compute with.
        sig_figs=12,
        show_progress=False,
        show_console=False,
    )
    elapsed["sample"] = time.perf_counter() - t0

    emit(Progress(Stage.DONE, "Finished", 1.0))
    return RunResult(
        spec=spec,
        resolved=resolved,
        rendered=rendered,
        compiled=compiled,
        table=table,
        stan_data=stan_data,
        sampler=sampler.model_copy(update={"seed": seed}),
        fit=fit,
        seed=seed,
        elapsed=elapsed,
        warnings=warnings,
    )
