"""Summarise a fit, diagnose it, and hand the numbers over for display.

Two things here are less obvious than they look.

**Non-finite generated quantities are a real hazard, and Stan's bounds do not
catch them.**  ``mttf = scale * tgamma(1 + 1/shape)`` overflows to ``inf`` once
the shape drops below roughly 0.006, and a declaration of ``real<lower=0>``
happily accepts it, because ``inf > 0``.  Left alone, that ``inf`` propagates
into the summary table, breaks the histograms, and is shown to the user with no
explanation.  So every derived column is checked and non-finite ones are
reported as a diagnostic rather than plotted.

**Diagnostics are translated into advice.**  A count of divergent transitions
tells a statistician something and a reliability engineer nothing; each one here
carries the action that follows from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .sample import RunResult

Severity = Literal["ok", "note", "warning", "error"]


@dataclass
class Diagnostic:
    code: str
    severity: Severity
    message: str
    advice: str = ""


@dataclass
class Summary:
    #: One row per reported quantity, with mean, sd, quantiles, R-hat and ESS.
    table: pd.DataFrame
    diagnostics: list[Diagnostic] = field(default_factory=list)
    #: Columns dropped from the display because every draw was non-finite.
    dropped_columns: list[str] = field(default_factory=list)
    elapsed: dict[str, float] = field(default_factory=dict)


#: Columns that are per-observation or per-grid-point rather than per-parameter:
#: useful in the draws export, noise in the summary table.  ``reliability_curve``
#: is 121 rows of picture; ``reliability`` (no suffix) is the answer at the
#: mission and stays.
_BULK_PREFIXES = ("log_lik", "log_lik_weighted", "reliability_curve")


def _display_vars(result: RunResult) -> list[str]:
    names: list[str] = []
    for column in result.fit.column_names:
        base = column.split("[")[0]
        if base.endswith("__") or base.startswith(_BULK_PREFIXES):
            continue
        if base not in names:
            names.append(base)
    return names


def draws_frame(result: RunResult, include_bulk: bool = True) -> pd.DataFrame:
    """Every draw, with chain and draw indices, ready for export."""
    frame = result.fit.draws_pd()
    if not include_bulk:
        keep = [
            c for c in frame.columns if not c.split("[")[0].startswith(_BULK_PREFIXES)
        ]
        frame = frame[keep]
    return frame


def summarize(result: RunResult) -> Summary:
    """Build the summary table and the diagnostics that go beside it."""
    import arviz as az

    idata = az.from_cmdstanpy(posterior=result.fit)
    wanted = _display_vars(result)

    dropped: list[str] = []
    diagnostics: list[Diagnostic] = []

    posterior = idata.posterior
    usable: list[str] = []
    for name in wanted:
        if name not in posterior:
            continue
        values = np.asarray(posterior[name].values, dtype=float)
        if values.size == 0:
            # A zero-length quantity, which is what `reliability` is when no
            # mission was given.  There is nothing to summarise and nothing
            # wrong: resolve() has already said the mission is missing.
            continue
        finite = np.isfinite(values)
        if not finite.any():
            dropped.append(name)
            diagnostics.append(
                Diagnostic(
                    code="non_finite_quantity",
                    severity="warning",
                    message=f"'{name}' was infinite or undefined in every draw.",
                    advice=_non_finite_advice(name),
                )
            )
            continue
        if not finite.all():
            share = 1.0 - float(finite.mean())
            diagnostics.append(
                Diagnostic(
                    code="partly_non_finite",
                    severity="warning",
                    message=(
                        f"'{name}' was infinite or undefined in {share:.1%} of "
                        f"draws; those draws are excluded from its summary."
                    ),
                    advice=_non_finite_advice(name),
                )
            )
        usable.append(name)

    table = az.summary(
        idata,
        var_names=usable,
        hdi_prob=0.9,
        stat_focus="mean",
        # ArviZ rounds to two decimals by default, which turns a B10 life of
        # 1e-9 into a column of zeros.  Keep full precision here and let the
        # display layer decide how to show it.
        round_to="none",
    )
    table = table.rename(
        columns={
            "hdi_5%": "q05",
            "hdi_95%": "q95",
            "ess_bulk": "ess_bulk",
            "ess_tail": "ess_tail",
            "r_hat": "r_hat",
        }
    )

    table = _add_quantiles(table, posterior, usable)
    table = _label_mission_quantities(table, result)
    diagnostics.extend(_heavy_tail_diagnostics(table))
    diagnostics.extend(_sampler_diagnostics(result, table))
    return Summary(
        table=table,
        diagnostics=diagnostics,
        dropped_columns=dropped,
        elapsed=dict(result.elapsed),
    )


def _add_quantiles(
    table: pd.DataFrame, posterior: Any, names: list[str]
) -> pd.DataFrame:
    """Report the median and the equal-tailed 5% and 95% quantiles.

    For derived quantities such as mean life the posterior can be very heavily
    skewed, and there the mean is a far worse summary than the median -- see
    :func:`_heavy_tail_diagnostics`.

    The interval is computed here rather than taken from ArviZ, whose summary
    gives the highest-density interval.  For a skewed posterior the two differ
    a lot -- a failure probability of order 1e-4 has an HDI whose lower end is
    half the 5% quantile -- and the reliability curve's band is equal-tailed,
    so the table must be too or the two would disagree at the mission.
    """
    stats: dict[str, tuple[float, float, float]] = {}
    for name in names:
        values = np.asarray(posterior[name].values, dtype=float)
        if values.ndim <= 2:
            flat = values.reshape(-1, 1)
            keys = [name]
        else:
            flat = values.reshape(values.shape[0] * values.shape[1], -1)
            keys = [f"{name}[{index}]" for index in range(flat.shape[1])]
        quantiles = np.nanquantile(flat, [0.05, 0.5, 0.95], axis=0)
        for key, (q05, median, q95) in zip(keys, quantiles.T, strict=True):
            stats[key] = (float(q05), float(median), float(q95))

    nan3 = (float("nan"),) * 3
    rows = [stats.get(str(idx), nan3) for idx in table.index]
    table = table.copy()
    table["q05"] = [r[0] for r in rows]
    table["q95"] = [r[2] for r in rows]
    table.insert(min(2, len(table.columns)), "median", [r[1] for r in rows])
    return table


def _label_mission_quantities(table: pd.DataFrame, result: RunResult) -> pd.DataFrame:
    """Name the reliability rows after the mission they answer for.

    Stan indexes the vector, so the table arrives saying ``reliability[0]``,
    which asks the reader to remember which mission time they typed first.  The
    row is about a mission, so it is labelled with one.
    """
    labels = reliability_labels(result)
    # ArviZ indexes from 0 -- `reliability[0]` is exactly what the user saw.
    by_index = {f"reliability[{i}]": label for i, label in enumerate(labels)}
    if len(labels) == 1:
        by_index["reliability"] = labels[0]

    renamed = [by_index.get(str(index), str(index)) for index in table.index]
    table = table.copy()
    table.index = pd.Index(renamed, name=table.index.name)
    return table


def reliability_labels(result: RunResult) -> list[str]:
    """Display names for the reliability entries, in Stan's own order."""
    spec = result.spec
    if result.resolved.likelihood.mission == "time":
        unit = f" {spec.time_unit}" if spec.time_unit else ""
        return [f"reliability(t={t:g}{unit})" for t in spec.effective_mission_times]
    return [f"reliability(n={spec.effective_mission_demands})"]


def _heavy_tail_diagnostics(table: pd.DataFrame) -> list[Diagnostic]:
    """Warn when the mean is a misleading summary.

    A Weibull mean life is ``scale * tgamma(1 + 1/shape)``, which grows without
    limit as the shape approaches zero.  When the data does not pin the shape
    down, a handful of draws dominate the average and the reported mean can sit
    far outside the credible interval -- a number that looks authoritative and
    is not.
    """
    if not {"mean", "q05", "q95"} <= set(table.columns):
        return []

    out: list[Diagnostic] = []
    for name, row in table.iterrows():
        mean, lo, hi = row["mean"], row["q05"], row["q95"]
        if not all(np.isfinite([mean, lo, hi])) or hi <= lo:
            continue
        if lo <= mean <= hi:
            continue
        out.append(
            Diagnostic(
                code="skewed_quantity",
                severity="note",
                message=(
                    f"The average of '{name}' ({mean:.4g}) lies outside its own "
                    f"90% interval ({lo:.4g} to {hi:.4g})."
                ),
                advice=(
                    "The distribution has a long tail, so a few extreme draws "
                    "pull the average away from where the estimate actually "
                    "sits.  Quote the median and the interval instead of the "
                    "average.  Constraining the shape parameter more tightly "
                    "shortens the tail."
                ),
            )
        )
    return out


def _non_finite_advice(name: str) -> str:
    if name in ("mttf", "mean_demands_to_failure"):
        return (
            "This happens when the fitted distribution allows a shape or rate "
            "so small that the mean is unbounded.  The reliability at your "
            "mission time is still meaningful; the mean lifetime is not.  "
            "Setting a lower bound on the shape parameter usually resolves it."
        )
    if name == "b10":
        return (
            "The 10%-failure life overflowed.  Constrain the shape parameter, "
            "or read the reliability curve instead."
        )
    return (
        "A derived quantity overflowed.  Tightening the parameter bounds "
        "usually resolves it."
    )


def _sampler_diagnostics(result: RunResult, table: pd.DataFrame) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    fit = result.fit

    try:
        method_vars = fit.method_variables()
        divergences = int(np.sum(method_vars["divergent__"]))
        treedepth = np.asarray(method_vars["treedepth__"])
        saturated = int(np.sum(treedepth >= result.sampler.max_treedepth))
    except Exception:
        divergences, saturated = 0, 0

    total = result.sampler.chains * result.sampler.iter_sampling

    if divergences:
        out.append(
            Diagnostic(
                code="divergences",
                severity="warning",
                message=(
                    f"{divergences} of {total} draws diverged "
                    f"({divergences / total:.1%})."
                ),
                advice=(
                    "The sampler hit places it could not follow accurately, so "
                    "the results may be biased.  Raise 'adapt_delta' toward "
                    "0.99, or use a more informative prior."
                ),
            )
        )
    if saturated:
        out.append(
            Diagnostic(
                code="max_treedepth",
                severity="note",
                message=(f"{saturated} of {total} draws hit the maximum tree depth."),
                advice=(
                    "The sampler was cut short rather than led astray -- this "
                    "costs efficiency, not correctness.  Raise 'max_treedepth' "
                    "to 12-15 if the effective sample size is low."
                ),
            )
        )

    if "r_hat" in table:
        bad = table.index[table["r_hat"] > 1.01].tolist()
        if bad:
            out.append(
                Diagnostic(
                    code="r_hat",
                    severity="warning",
                    message=(
                        f"The chains have not agreed on: {', '.join(bad)} "
                        f"(R-hat above 1.01)."
                    ),
                    advice=(
                        "Run longer, or with more warmup.  If it persists, the "
                        "data may not identify these parameters."
                    ),
                )
            )
    if "ess_bulk" in table:
        thin = table.index[table["ess_bulk"] < 400].tolist()
        if thin:
            out.append(
                Diagnostic(
                    code="low_ess",
                    severity="note",
                    message=(
                        f"Few effective draws for: {', '.join(thin)}.  The "
                        f"reported means carry noticeable Monte Carlo noise."
                    ),
                    advice="Increase 'iter_sampling' (or the number of chains).",
                )
            )

    out.extend(_rejected_proposal_diagnostics(fit))

    if not out:
        out.append(
            Diagnostic(
                code="clean",
                severity="ok",
                message="No sampling problems were detected.",
            )
        )
    return out


def _rejected_proposal_diagnostics(fit: Any) -> list[Diagnostic]:
    """Translate CmdStan's non-fatal per-proposal exceptions.

    These scroll past as alarming console output but are usually benign: a
    parameter bounded below at zero can land on exactly zero in floating point
    during warmup, Stan rejects that one proposal, and sampling continues.  It
    is worth reporting as a note -- persistent rejections do indicate a badly
    placed bound -- but not as a failure.
    """
    try:
        messages = fit.runset.get_err_msgs()
    except Exception:
        return []
    if not messages:
        return []

    text = messages if isinstance(messages, str) else "\n".join(messages)
    count = text.count("Exception:")
    if not count:
        return []

    detail = ""
    for line in text.splitlines():
        if "Exception:" in line:
            detail = line.split("Exception:", 1)[1].split("(in ")[0].strip()
            break

    return [
        Diagnostic(
            code="rejected_proposals",
            severity="note",
            message=(
                f"The sampler rejected {count} proposed steps that landed "
                f"outside the model's valid range"
                + (f' ("{detail}")' if detail else "")
                + "."
            ),
            advice=(
                "This is normal during warmup: the chains start at random "
                "points, often far from where the data put the parameter, and "
                "early steps can overshoot a bound (a rate of 1e-3 starts "
                "near 1).  It does not bias the result.  If it happens "
                "constantly, move a bound away from the region the data "
                "favours."
            ),
        )
    ]


def reliability_curve(result: RunResult) -> pd.DataFrame | None:
    """Posterior reliability at each mission time, as a summary per time."""
    mission = result.spec.effective_mission_times
    if not mission or "reliability" not in result.fit.stan_variables():
        return None
    draws = np.asarray(result.fit.stan_variable("reliability"), dtype=float)
    if draws.ndim == 1:
        draws = draws[:, None]
    return pd.DataFrame(
        {
            "mission_time": mission,
            "mean": draws.mean(axis=0),
            "q05": np.quantile(draws, 0.05, axis=0),
            "median": np.quantile(draws, 0.5, axis=0),
            "q95": np.quantile(draws, 0.95, axis=0),
        }
    )


def reliability_over_time(result: RunResult) -> dict[str, Any] | None:
    """The dense reliability curve, summarised point by point.

    Distinct from :func:`reliability_curve`, which reports only the mission
    times the user asked about.  This is the shape between and beyond them: the
    posterior mean, the median and a 90% band at every point of the grid that
    was handed to Stan as data.

    Everything needed to draw and label it travels with it -- the mission, the
    unit, whether the axis counts time or demands -- because the run that
    produced these numbers is not necessarily the model still on screen.
    """
    lik = result.resolved.likelihood
    key = "curve_time" if lik.mission == "time" else "curve_demands"
    grid = [float(x) for x in result.stan_data.get(key, [])]

    if not grid or "reliability_curve" not in result.fit.stan_variables():
        return None

    draws = np.asarray(result.fit.stan_variable("reliability_curve"), dtype=float)
    if draws.ndim == 1:
        draws = draws[:, None]
    if draws.shape[1] != len(grid):
        return None

    points = [
        {
            "x": x,
            "mean": float(mean),
            "q05": float(q05),
            "median": float(median),
            "q95": float(q95),
        }
        for x, mean, q05, median, q95 in zip(
            grid,
            draws.mean(axis=0),
            np.quantile(draws, 0.05, axis=0),
            np.quantile(draws, 0.5, axis=0),
            np.quantile(draws, 0.95, axis=0),
            strict=True,
        )
    ]

    mission = (
        [float(t) for t in result.spec.effective_mission_times]
        if lik.mission == "time"
        else [float(result.spec.effective_mission_demands)]
    )
    return {
        "kind": lik.mission,
        "unit": result.spec.time_unit if lik.mission == "time" else "demands",
        "mission": mission,
        "points": points,
    }


def export_draws(result: RunResult, path: str, fmt: str = "csv") -> str:
    """Write every draw to ``path`` in the requested format."""
    if fmt == "netcdf":
        import arviz as az

        az.from_cmdstanpy(posterior=result.fit).to_netcdf(path)
        return path
    frame = draws_frame(result)
    if fmt == "csv":
        frame.to_csv(path, index=False)
    elif fmt == "parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"Unsupported export format {fmt!r}; use csv, parquet or netcdf."
        )
    return path


def as_dict(summary: Summary) -> dict[str, Any]:
    """JSON-ready form for the HTTP layer."""
    return {
        "table": summary.table.reset_index(names="parameter").to_dict(orient="records"),
        "diagnostics": [
            {
                "code": d.code,
                "severity": d.severity,
                "message": d.message,
                "advice": d.advice,
            }
            for d in summary.diagnostics
        ],
        "dropped_columns": summary.dropped_columns,
        "elapsed": summary.elapsed,
    }


#: Points on the prior-versus-posterior grid of each parameter.
DENSITY_POINTS = 160


def prior_vs_posterior(result: RunResult) -> dict[str, dict[str, Any]]:
    """Each parameter's posterior density beside the prior that was in force.

    The grid covers the posterior (its central 99.6%, padded), clipped to the
    parameter's bounds, so the picture answers "how far did the data move
    me?" -- a prior that is flat across the window says the data dominated.
    """
    from scipy.stats import gaussian_kde

    from .preview import prior_density

    out: dict[str, dict[str, Any]] = {}
    for param in result.resolved.params:
        draws = np.asarray(result.fit.stan_variable(param.name), dtype=float)
        draws = draws[np.isfinite(draws)]
        if draws.size < 10 or np.ptp(draws) <= 0:
            continue
        lo_q, hi_q = np.quantile(draws, [0.002, 0.998])
        pad = 0.15 * (hi_q - lo_q)
        left, right = lo_q - pad, hi_q + pad
        if param.lo is not None:
            left = max(left, param.lo)
        if param.hi is not None:
            right = min(right, param.hi)
        x = np.linspace(left, right, DENSITY_POINTS)
        try:
            posterior = gaussian_kde(draws)(x)
        except (np.linalg.LinAlgError, ValueError):
            continue
        hyper = dict(result.spec.params[param.name].prior.hyper)
        prior = prior_density(param.family, hyper, param.lo, param.hi, x)
        out[param.name] = {
            "x": x.tolist(),
            "posterior": posterior.tolist(),
            # A Jeffreys or Beta(<1) prior is infinite at its bound, and JSON
            # (as browsers parse it) has no infinity: such points become gaps.
            "prior": None
            if prior is None
            else [float(v) if np.isfinite(v) else None for v in prior],
            "prior_label": param.family_label,
        }
    return out


def data_summary(result: RunResult) -> dict[str, Any]:
    """What went in: counts the results page states beside the answer."""
    rows = result.table.rows
    kind = result.resolved.likelihood.mission
    weights = [float(r.get("relevance", 1.0)) for r in rows]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "effective_n": float(sum(weights)),
        "weighted": any(w < 1.0 for w in weights),
        "prior_only": bool(result.spec.prior_only),
    }
    if "failures" in result.resolved.likelihood.required_columns:
        summary["failures"] = int(sum(int(r["failures"]) for r in rows))
        summary["trials"] = int(sum(int(r["n"]) for r in rows))
    else:
        failed = sum(1 for r in rows if int(r["failure"]) == 1)
        summary["failures"] = failed
        summary["censored"] = len(rows) - failed
        if kind == "time":
            summary["exposure"] = float(sum(float(r["time"]) for r in rows))
        else:
            summary["trials"] = int(sum(int(r["n"]) for r in rows))
    return summary


def kaplan_meier(result: RunResult) -> dict[str, Any] | None:
    """The data's own survival curve, to draw beside the fitted one.

    Product-limit estimate with right censoring; relevance weights enter as
    fractional counts, so a half-relevant failure removes half a unit.  Only
    for time-based models: it is the check a reader makes by eye -- does the
    fitted curve run through the observations?
    """
    if result.resolved.likelihood.mission != "time" or result.spec.prior_only:
        return None
    rows = result.table.rows
    if not rows:
        return None
    events: dict[float, list[float]] = {}
    for row in rows:
        w = float(row.get("relevance", 1.0))
        if w <= 0:
            continue
        slot = events.setdefault(float(row["time"]), [0.0, 0.0])
        slot[0 if int(row["failure"]) == 1 else 1] += w
    at_risk = sum(d + c for d, c in events.values())
    survival = 1.0
    steps: list[dict[str, float]] = [{"t": 0.0, "s": 1.0}]
    censored: list[dict[str, float]] = []
    for t in sorted(events):
        died, lost = events[t]
        if died > 0 and at_risk > 0:
            survival *= 1.0 - died / at_risk
            steps.append({"t": t, "s": survival})
        if lost > 0:
            censored.append({"t": t, "s": survival})
        at_risk -= died + lost
    return {"steps": steps, "censored": censored}
