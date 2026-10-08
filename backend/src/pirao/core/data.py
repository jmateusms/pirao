"""The tidy observation table: schema, validation, and the transform to Stan.

One row per observation, for every likelihood.  ``failure = 1`` means *failed*,
everywhere -- no per-model inversion, no separate failure-time and censoring-time
columns of different lengths.  For a time-to-failure model a row is a failure
time when ``failure = 1`` and a right-censoring time when ``failure = 0``.

Retiring the split arrays costs nothing and buys a lot.  The usual argument for
splitting is vectorisation, and it does not apply here: Stan's vectorised
``weibull_lpdf(vector y | ...)`` returns a *sum*, whereas a per-row relevance
weight needs the weighted sum of individual terms.  Both layouts therefore need
a scalar loop, so the tidy one wins on everything that is left -- it matches the
spreadsheet the user uploaded, it keeps row order, and ``log_lik`` comes back
indexed by the user's own rows.

The transform's output keys must match the generated Stan data block exactly.
That is the second-easiest place in this project to be silently wrong (after
argument order), so the contract is tested against the generated source rather
than maintained by hand.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .registry_likelihoods import LikelihoodDef, get_likelihood
from .resolve import ResolvedModel, resolve
from .spec import ModelSpec

#: Columns understood anywhere in the schema.
ALL_COLUMNS = ("device", "n", "time", "failure", "failures", "relevance")

#: Recognised but unused: they belong to temperature-accelerated and
#: multi-segment models that do not exist here yet.  Reserved rather than
#: repurposed, so an old spreadsheet is ignored with a note instead of
#: misinterpreted.
RESERVED_COLUMNS = ("temperature", "subpop")

OPTIONAL_COLUMNS = ("device", "relevance")

#: Points on the dense reliability curve.  Enough that the band reads as a
#: smooth shape rather than a polyline, few enough that the extra generated
#: quantities stay a rounding error next to the draws themselves.
CURVE_POINTS = 121

#: How far past the mission the curve runs when no horizon was given.  The
#: mission then sits at about three quarters of the axis, which shows where the
#: estimate is heading without implying the extrapolation is data.
CURVE_HEADROOM = 1.3


@dataclass(frozen=True)
class CellError:
    """A validation failure, located precisely enough to highlight a cell."""

    #: 1-based row number as the user sees it, or ``None`` for whole-table
    #: problems such as a missing column.
    row: int | None
    column: str | None
    message: str

    def __str__(self) -> str:
        if self.row is None:
            return self.message
        return f"Row {self.row}: {self.message}"


class DataError(ValueError):
    def __init__(self, errors: list[CellError]):
        self.errors = errors
        super().__init__("; ".join(str(e) for e in errors))


@dataclass
class ValidatedTable:
    rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)

    def column(self, name: str) -> list[Any]:
        return [row[name] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


def required_columns(likelihood_id: str) -> tuple[str, ...]:
    return get_likelihood(likelihood_id).required_columns


def _as_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def validate_table(
    spec: ModelSpec,
    rows: Iterable[dict[str, Any]],
    resolved: ResolvedModel | None = None,
) -> ValidatedTable:
    """Check a tidy table against a likelihood, returning normalised rows.

    Collects every problem rather than stopping at the first, so a grid can
    highlight all offending cells at once.
    """
    lik = get_likelihood(spec.likelihood)
    raw = [dict(r) for r in rows]
    errors: list[CellError] = []
    warnings: list[str] = []

    present = {c for row in raw for c in row}
    ignored = sorted(
        (present & set(RESERVED_COLUMNS))
        | (present - set(ALL_COLUMNS) - set(RESERVED_COLUMNS))
    )
    if ignored:
        warnings.append(
            "These columns are not used by the "
            f"{lik.label} model and were ignored: {', '.join(ignored)}."
        )

    if raw:
        for column in lik.required_columns:
            if column not in present:
                errors.append(
                    CellError(
                        None,
                        column,
                        f"Column '{column}' is required by the {lik.label} "
                        f"model but was not found.",
                    )
                )

    if "failure" in present and "failures" in present:
        errors.append(
            CellError(
                None,
                None,
                "Both 'failure' and 'failures' are present.  Use 'failure' "
                "(0 or 1 per specimen) for the per-demand model, or "
                "'failures' (a count per group) for the binomial -- not both.",
            )
        )

    if errors:
        raise DataError(errors)

    normalised: list[dict[str, Any]] = []
    for index, row in enumerate(raw, start=1):
        out: dict[str, Any] = {}

        device = row.get("device")
        out["device"] = str(device).strip() if device not in (None, "") else str(index)

        relevance = _as_number(row.get("relevance"))
        if relevance is None:
            relevance = 1.0
        elif not 0.0 <= relevance <= 1.0:
            errors.append(
                CellError(
                    index,
                    "relevance",
                    f"'relevance' must be between 0 and 1 (got "
                    f"{row.get('relevance')!r}).",
                )
            )
            relevance = 1.0
        out["relevance"] = float(relevance)

        if "time" in lik.required_columns:
            time = _as_number(row.get("time"))
            if time is None:
                errors.append(
                    CellError(
                        index, "time", "'time' must be a number and cannot be blank."
                    )
                )
            elif not math.isfinite(time) or time < 0:
                errors.append(
                    CellError(
                        index,
                        "time",
                        f"'time' must be a finite, non-negative number (got "
                        f"{row.get('time')!r}).",
                    )
                )
            else:
                out["time"] = float(time)

        if "n" in lik.required_columns:
            n = _as_int(row.get("n"))
            if n is None:
                errors.append(
                    CellError(
                        index,
                        "n",
                        f"'n' must be a whole number (got {row.get('n')!r}).",
                    )
                )
            elif n < 0:
                errors.append(
                    CellError(index, "n", f"'n' cannot be negative (got {n}).")
                )
            else:
                out["n"] = n

        if "failure" in lik.required_columns:
            failure = _as_int(row.get("failure"))
            if failure not in (0, 1):
                errors.append(
                    CellError(
                        index,
                        "failure",
                        f"'failure' must be exactly 0 or 1 (got "
                        f"{row.get('failure')!r}).",
                    )
                )
            else:
                out["failure"] = failure

        if "failures" in lik.required_columns:
            failures = _as_int(row.get("failures"))
            if failures is None or failures < 0:
                errors.append(
                    CellError(
                        index,
                        "failures",
                        f"'failures' must be a whole number of zero or more "
                        f"(got {row.get('failures')!r}).",
                    )
                )
            else:
                out["failures"] = failures

        normalised.append(out)

    resolved = resolved or resolve(spec)
    errors.extend(_cross_field_checks(lik, resolved, normalised))

    if errors:
        raise DataError(errors)

    warnings.extend(_advisories(lik, normalised))
    return ValidatedTable(rows=normalised, warnings=warnings, ignored_columns=ignored)


def _shape_lower_bound(resolved: ResolvedModel) -> float | None:
    for p in resolved.params:
        if p.role == "shape":
            return p.lo
    return None


def _cross_field_checks(
    lik: LikelihoodDef, resolved: ResolvedModel, rows: list[dict[str, Any]]
) -> list[CellError]:
    """Rules that depend on more than one cell -- or on the spec itself."""
    errors: list[CellError] = []
    shape_lb = _shape_lower_bound(resolved)

    for index, row in enumerate(rows, start=1):
        if "failure" in row and "n" in row and row["failure"] == 1 and row["n"] < 1:
            errors.append(
                CellError(
                    index,
                    "n",
                    "'failure' = 1 needs 'n' of at least 1: a unit cannot fail "
                    "on demand zero.",
                )
            )
        if "failures" in row and "n" in row and row["failures"] > row["n"]:
            errors.append(
                CellError(
                    index,
                    "failures",
                    f"'failures' ({row['failures']}) cannot exceed 'n' ({row['n']}).",
                )
            )
        if row.get("failure") == 1 and row.get("time") == 0:
            if lik.id in ("lognormal", "gamma"):
                errors.append(
                    CellError(
                        index,
                        "time",
                        f"A failure at time 0 has zero probability under the "
                        f"{lik.label} model.  Use a small positive time, or "
                        f"set 'failure' to 0 if this was a censoring at time "
                        f"zero.",
                    )
                )
            elif lik.id == "weibull" and shape_lb is not None and shape_lb >= 1.0:
                errors.append(
                    CellError(
                        index,
                        "time",
                        "A failure at time 0 has zero probability because the "
                        "Weibull shape is bounded at 1 or above.  Use a small "
                        "positive time, set 'failure' to 0, or lower the "
                        "shape's lower bound below 1.",
                    )
                )
    return errors


def _advisories(lik: LikelihoodDef, rows: list[dict[str, Any]]) -> list[str]:
    """Non-blocking notes: valid data that is worth a second look."""
    notes: list[str] = []
    if not rows:
        return notes

    if "failure" in lik.required_columns:
        failures = sum(r.get("failure", 0) for r in rows)
        if failures == 0:
            notes.append(
                "Every row is censored -- no failure was observed.  The model "
                "still runs and the result is valid, but with no observed "
                "failure the answer will lean heavily on your prior."
            )
        elif failures == len(rows):
            notes.append(
                "Every row is a failure, with no censored units.  That is "
                "fine, but it means there is no direct information about units "
                "that survived."
            )

    devices = [r["device"] for r in rows]
    duplicates = {d for d in devices if devices.count(d) > 1}
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:3])
        notes.append(
            f"Some device labels repeat ({sample}).  That is allowed -- each "
            f"row is still treated as an independent observation -- but check "
            f"it is what you meant."
        )

    effective = sum(r["relevance"] for r in rows)
    if effective < len(rows):
        notes.append(
            f"Relevance weighting is in effect: {len(rows)} rows count as "
            f"{effective:.2f} effective observations.  Results are a "
            f"fractional (pseudo-)posterior, so intervals are deliberately "
            f"wider than a standard Bayesian analysis would give."
        )
    return notes


def curve_horizon(spec: ModelSpec, table: ValidatedTable) -> float:
    """How far the reliability curve should run.

    The mission is the anchor: a curve that stops at the mission time answers
    "will it survive the mission?" and nothing else, while one that runs a
    little past it shows how quickly the answer is deteriorating.  Only when no
    mission was given does the observed data set the scale instead.
    """
    if spec.curve_horizon is not None:
        return float(spec.curve_horizon)

    lik = get_likelihood(spec.likelihood)
    if lik.mission == "time":
        mission = max(spec.effective_mission_times, default=0.0)
        observed = max((r.get("time", 0.0) for r in table.rows), default=0.0)
    else:
        mission = float(spec.effective_mission_demands)
        observed = float(max((r.get("n", 0) for r in table.rows), default=0))

    reference = mission or observed
    return CURVE_HEADROOM * float(reference)


def curve_points(spec: ModelSpec, table: ValidatedTable) -> list[float]:
    """The grid the reliability curve is evaluated on.

    Empty when there is nothing to set a scale by -- no mission and no data --
    in which case the run simply reports no curve rather than inventing an
    axis.
    """
    horizon = curve_horizon(spec, table)
    if not math.isfinite(horizon) or horizon <= 0:
        return []

    lik = get_likelihood(spec.likelihood)
    if lik.mission == "demands":
        # Demands are counted, not measured, so the curve is evaluated at whole
        # demands; a point at 3.5 actuations would not mean anything.
        last = int(math.ceil(horizon))
        if last < CURVE_POINTS:
            return [float(i) for i in range(last + 1)]
        step = horizon / (CURVE_POINTS - 1)
        return sorted({float(round(i * step)) for i in range(CURVE_POINTS)})

    step = horizon / (CURVE_POINTS - 1)
    return [i * step for i in range(CURVE_POINTS)]


def to_stan_data(
    spec: ModelSpec,
    table: ValidatedTable,
    resolved: ResolvedModel | None = None,
) -> dict[str, Any]:
    """Build the dict handed to CmdStan.

    Keys here must match the generated data block exactly; see the module
    docstring.
    """
    resolved = resolved or resolve(spec)
    lik = resolved.likelihood

    data: dict[str, Any] = {
        "N": len(table),
        "relevance": [r["relevance"] for r in table.rows],
        "prior_only": 1 if spec.prior_only else 0,
    }

    grid = curve_points(spec, table)
    data["G"] = len(grid)

    if lik.mission == "time":
        mission = spec.effective_mission_times
        data["M"] = len(mission)
        data["mission_time"] = [float(t) for t in mission]
        data["curve_time"] = grid
        data["time"] = [r["time"] for r in table.rows]
        data["failure"] = [r["failure"] for r in table.rows]
    else:
        data["mission_demands"] = spec.effective_mission_demands
        data["curve_demands"] = grid
        data["n"] = [r["n"] for r in table.rows]
        if "failures" in lik.required_columns:
            data["failures"] = [r["failures"] for r in table.rows]
        else:
            data["failure"] = [r["failure"] for r in table.rows]

    data.update(resolved.data_values())
    return data


def template_rows(likelihood_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Header and example rows for the downloadable CSV/XLSX template."""
    lik = get_likelihood(likelihood_id)
    header = ["device", *lik.required_columns, "relevance"]

    examples: dict[str, list[dict[str, Any]]] = {
        "exponential": [
            {"device": "A1", "time": 120.5, "failure": 1, "relevance": 1.0},
            {"device": "A2", "time": 300.0, "failure": 0, "relevance": 1.0},
            {"device": "B1", "time": 45.2, "failure": 1, "relevance": 0.8},
        ],
        "weibull": [
            {"device": "A1", "time": 850, "failure": 1, "relevance": 1.0},
            {"device": "A2", "time": 1200, "failure": 0, "relevance": 1.0},
            {"device": "A3", "time": 430, "failure": 1, "relevance": 1.0},
        ],
        "lognormal": [
            {"device": "A1", "time": 640, "failure": 1, "relevance": 1.0},
            {"device": "A2", "time": 900, "failure": 0, "relevance": 1.0},
        ],
        "gamma": [
            {"device": "A1", "time": 310, "failure": 1, "relevance": 1.0},
            {"device": "A2", "time": 500, "failure": 0, "relevance": 1.0},
        ],
        "bernoulli_geometric": [
            {"device": "V1", "n": 1, "failure": 0, "relevance": 1.0},
            {"device": "V2", "n": 5, "failure": 1, "relevance": 1.0},
            {"device": "V3", "n": 10, "failure": 0, "relevance": 0.8},
        ],
        "binomial": [
            {"device": "Lot-1", "n": 100, "failures": 5, "relevance": 1.0},
            {"device": "Lot-2", "n": 200, "failures": 3, "relevance": 1.0},
        ],
    }
    return header, examples[lik.id]
