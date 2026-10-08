"""The analysis specification.

``ModelSpec`` does three jobs at once, and it is worth naming them because they
constrain how it may change:

1. it is the **API contract** between the frontend and the engine;
2. it is the **cache input** -- its rendered form decides which compiled binary
   is reused;
3. it is the **persistence format** -- a saved run bundle stores one verbatim.

That is why ``schema_version`` exists from the first commit.  Without it the
first breaking change orphans every saved analysis.

Note what is *not* here: the observation table.  Data lives outside the spec so
that editing a data cell can never invalidate a compiled binary.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

#: Sampler defaults.  Deliberately modest: these models are small and fast, so
#: the wait a user actually experiences is compilation, not sampling.
DEFAULT_CHAINS = 4
DEFAULT_ITER_WARMUP = 1000
DEFAULT_ITER_SAMPLING = 1000


class PriorSpec(BaseModel):
    """A prior family plus its hyperparameter values, keyed by name."""

    model_config = ConfigDict(extra="forbid")

    family: str
    hyper: dict[str, float] = Field(default_factory=dict)


class ParamSpec(BaseModel):
    """User choices for one parameter: its bounds and its prior."""

    model_config = ConfigDict(extra="forbid")

    #: ``None`` means "no user bound"; the natural support and the prior's own
    #: support still apply.
    lower: float | None = None
    upper: float | None = None
    prior: PriorSpec

    @model_validator(mode="after")
    def _bounds_ordered(self) -> ParamSpec:
        for field_name in ("lower", "upper"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"The {field_name} bound must be a finite number, or left "
                    f"empty for no bound (got {value!r})."
                )
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError(
                f"The lower bound ({self.lower:g}) must be below the upper "
                f"bound ({self.upper:g})."
            )
        return self


class SamplerConfig(BaseModel):
    """MCMC settings, with the plain-language help the GUI renders as tooltips."""

    model_config = ConfigDict(extra="forbid")

    chains: Annotated[int, Field(ge=1, le=64)] = DEFAULT_CHAINS
    iter_warmup: Annotated[int, Field(ge=1, le=1_000_000)] = DEFAULT_ITER_WARMUP
    iter_sampling: Annotated[int, Field(ge=1, le=1_000_000)] = DEFAULT_ITER_SAMPLING
    seed: int | None = None
    adapt_delta: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.8
    max_treedepth: Annotated[int, Field(ge=1, le=20)] = 10
    parallel_chains: int | None = None


SAMPLER_HELP: dict[str, str] = {
    "chains": (
        "How many independent runs to start from different random points.  "
        "Comparing them is what makes convergence detectable, so use at least "
        "2; 4 is the usual choice."
    ),
    "iter_warmup": (
        "Tuning draws, discarded before results are computed.  The sampler "
        "uses them to learn the shape of the posterior.  Raise this if the "
        "run reports that the sampler did not adapt well."
    ),
    "iter_sampling": (
        "Draws kept per chain.  More draws mean less Monte Carlo noise in the "
        "reported means and intervals; they do not make the model itself more "
        "accurate."
    ),
    "seed": (
        "Fixes the random numbers so the same inputs reproduce the same "
        "draws.  Leave empty to draw a fresh seed, which is recorded in the "
        "results either way."
    ),
    "adapt_delta": (
        "Target acceptance rate during warmup.  Raising it toward 0.99 makes "
        "the sampler take smaller, more careful steps -- the standard remedy "
        "for divergence warnings, at the cost of speed."
    ),
    "max_treedepth": (
        "Ceiling on how far the sampler may explore in a single draw.  If the "
        "run warns that this limit was hit, the sampler was cut off early and "
        "was slow rather than wrong; raise it to 12-15."
    ),
    "parallel_chains": (
        "How many chains to run at once.  Defaults to one per available CPU "
        "core, up to the number of chains."
    ),
}


class ModelSpec(BaseModel):
    """A complete, self-contained description of a model to fit."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    likelihood: str
    params: dict[str, ParamSpec]

    #: Mission times at which reliability is reported.  A list rather than a
    #: scalar so the results can show a reliability *curve* rather than a
    #: single number.  Used by time-based likelihoods.
    mission_times: list[float] = Field(default_factory=list)
    #: Mission length in demands, used by demand-based likelihoods.
    mission_demands: int | None = None

    #: Where the reliability curve stops: the largest time (or demand count)
    #: it is evaluated at.  ``None`` means "derive it", which is what the UI
    #: leaves it as -- see :func:`pirao.core.data.curve_points`.  It is not
    #: part of the model: the curve's grid travels as Stan data, so changing
    #: this never recompiles anything.
    curve_horizon: float | None = None

    #: Free-text unit label ("h", "cycles", "days").  Locked for the run and
    #: printed on every plot and export.  It is not cosmetic: weighting a
    #: *density* by a relevance factor is not invariant to a change of unit,
    #: so the same data in hours and in days weights failures against
    #: censorings differently.
    time_unit: str = "h"

    #: Skip the likelihood entirely and sample the prior.  Distinct from
    #: submitting an empty table: this keeps the observed design in place, so
    #: prior *predictive* quantities remain meaningful.
    prior_only: bool = False
    emit_log_lik: bool = True

    @model_validator(mode="after")
    def _check_against_registry(self) -> ModelSpec:
        from .registry_likelihoods import get_likelihood

        lik = get_likelihood(self.likelihood)

        expected = set(lik.param_names)
        got = set(self.params)
        if missing := sorted(expected - got):
            raise ValueError(
                f"The {lik.label} model needs a prior for: {', '.join(missing)}."
            )
        if extra := sorted(got - expected):
            raise ValueError(
                f"The {lik.label} model has no parameter called "
                f"{', '.join(extra)}.  Its parameters are: "
                f"{', '.join(lik.param_names)}."
            )

        if lik.mission == "time":
            if self.mission_demands is not None:
                raise ValueError(
                    f"'Mission demands' does not apply to the {lik.label} "
                    f"model, which is measured in time.  Set a mission time "
                    f"instead."
                )
            for value in self.mission_times:
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Every mission time must be a finite, non-negative "
                        f"number (got {value!r})."
                    )
        else:
            if self.mission_times:
                raise ValueError(
                    f"'Mission time' does not apply to the {lik.label} model, "
                    f"which is measured in demands.  Set a mission demand "
                    f"count instead."
                )
            if self.mission_demands is not None and self.mission_demands < 0:
                raise ValueError(
                    "'Mission demands' must be zero or more "
                    f"(got {self.mission_demands})."
                )

        if self.curve_horizon is not None and (
            not math.isfinite(self.curve_horizon) or self.curve_horizon <= 0
        ):
            raise ValueError(
                f"The reliability curve's horizon must be a positive, finite "
                f"number, or left empty to derive it from the mission "
                f"(got {self.curve_horizon!r})."
            )
        return self

    @property
    def effective_mission_times(self) -> list[float]:
        return list(self.mission_times)

    @property
    def effective_mission_demands(self) -> int:
        return 1 if self.mission_demands is None else self.mission_demands


class RunRequest(BaseModel):
    """A spec, its data, and the sampler settings -- everything one run needs."""

    model_config = ConfigDict(extra="forbid")

    spec: ModelSpec
    #: Tidy observation rows.  Validated against the likelihood by
    #: :mod:`pirao.core.data`.
    rows: list[dict[str, object]] = Field(default_factory=list)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)


ExportFormat = Literal["csv", "parquet", "netcdf"]
