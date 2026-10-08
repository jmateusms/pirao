"""Prior previews: what the user will actually get, not what they typed.

This module exists because of a specific, silent failure mode.  A parameter
declared ``real<lower=0, upper=1> lambda`` with a ``lognormal(0, 1)`` prior does
not have a lognormal prior -- it has a lognormal *restricted to* [0, 1], which
keeps only half the mass and has a median near 0.53 rather than 1.0.  Stan is
perfectly correct here (the missing normaliser is a constant, so the posterior
is right), but the user elicited one distribution and the model used another,
and nothing told them.

So the UI shows the **truncated** density, its summary statistics, and the share
of the original prior that survived the bounds.  Everything here is
scipy-on-the-CPU; none of it touches Stan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .registry_priors import PriorFamily, get_prior

INF = math.inf


def _frozen(family: PriorFamily, hyper: dict[str, float]):
    """Map a family plus hyperparameter values onto a scipy distribution.

    Returns ``None`` for families with no proper density of their own (flat,
    uniform, Jeffreys), which are handled analytically by the callers.
    """
    h = hyper
    try:
        match family.id:
            case "lognormal":
                return stats.lognorm(s=h["sigma"], scale=math.exp(h["mu"]))
            case "normal":
                return stats.norm(loc=h["mu"], scale=h["sigma"])
            case "student_t":
                return stats.t(df=h["nu"], loc=h["mu"], scale=h["sigma"])
            case "gamma":
                return stats.gamma(a=h["shape"], scale=1.0 / h["rate"])
            case "inv_gamma":
                return stats.invgamma(a=h["shape"], scale=h["scale"])
            case "exponential":
                return stats.expon(scale=1.0 / h["rate"])
            case "weibull":
                return stats.weibull_min(c=h["shape"], scale=h["scale"])
            case "beta":
                return stats.beta(a=h["a"], b=h["b"])
            # The half families are emitted as the symmetric density on a
            # parameter bounded below at 0, but the prior the user specified is
            # the half one: measure retained mass against that, or a plain
            # [0, inf) bound reads as "keeps only 50%".
            case "half_normal":
                return stats.halfnorm(scale=h["sigma"])
            case "half_cauchy":
                return stats.halfcauchy(scale=h["sigma"])
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    return None


def retained_mass(
    family: PriorFamily,
    hyper: dict[str, float],
    lo: float | None,
    hi: float | None,
) -> float | None:
    """Share of the family's mass left inside ``[lo, hi]``.

    ``None`` when the family has no proper density to compare against, in which
    case the question is meaningless rather than merely unanswered.
    """
    dist = _frozen(family, hyper)
    if dist is None:
        return None
    a = -INF if lo is None else lo
    b = INF if hi is None else hi
    try:
        mass = float(dist.cdf(b) - dist.cdf(a))
    except (ValueError, TypeError):
        return None
    if not math.isfinite(mass):
        return None
    return min(max(mass, 0.0), 1.0)


@dataclass(frozen=True)
class PriorPreview:
    """A renderable description of the prior actually in force."""

    family: str
    family_label: str
    lo: float | None
    hi: float | None
    #: Share of the specified prior that survives the bounds.  ``None`` for
    #: families with no proper density (flat, uniform, Jeffreys).
    retained_mass: float | None
    #: Grid for a sparkline: x values and the *truncated*, renormalised density.
    x: list[float]
    pdf: list[float]
    #: Summary statistics of the truncated prior, which are what the user is
    #: actually assuming.
    median: float | None
    mean: float | None
    q05: float | None
    q95: float | None
    is_proper: bool
    warning: str | None = None


def _display_range(
    dist, lo: float | None, hi: float | None
) -> tuple[float, float] | None:
    """Pick a plotting window that shows the truncated prior, not its tails."""
    a = -INF if lo is None else lo
    b = INF if hi is None else hi
    try:
        p_lo, p_hi = float(dist.cdf(a)), float(dist.cdf(b))
    except (ValueError, TypeError):
        return None
    if not (math.isfinite(p_lo) and math.isfinite(p_hi)) or p_hi <= p_lo:
        return None
    inner_lo = float(dist.ppf(p_lo + 0.001 * (p_hi - p_lo)))
    inner_hi = float(dist.ppf(p_lo + 0.999 * (p_hi - p_lo)))
    left = max(a, inner_lo) if math.isfinite(a) else inner_lo
    right = min(b, inner_hi) if math.isfinite(b) else inner_hi
    if not (math.isfinite(left) and math.isfinite(right)) or right <= left:
        return None
    return left, right


def prior_preview(
    family_id: str,
    hyper: dict[str, float],
    lo: float | None,
    hi: float | None,
    points: int = 129,
) -> PriorPreview:
    """Describe the prior a user will actually get, bounds included."""
    family = get_prior(family_id)
    dist = _frozen(family, hyper)

    if dist is None:
        # flat / uniform / Jeffreys: analytic, and only defined on finite bounds.
        if lo is None or hi is None or hi <= lo:
            return PriorPreview(
                family=family.id,
                family_label=family.label,
                lo=lo,
                hi=hi,
                retained_mass=None,
                x=[],
                pdf=[],
                median=None,
                mean=None,
                q05=None,
                q95=None,
                is_proper=False,
                warning=(
                    f"The {family.label} prior needs both a lower and an upper "
                    f"bound before it describes a proper distribution."
                ),
            )
        xs = np.linspace(lo, hi, points)
        if family.id == "jeffreys_scale":
            norm = math.log(hi) - math.log(lo)
            pdf = 1.0 / (xs * norm)
            median = math.exp(math.log(lo) + 0.5 * norm)
            mean = (hi - lo) / norm
            q05 = math.exp(math.log(lo) + 0.05 * norm)
            q95 = math.exp(math.log(lo) + 0.95 * norm)
        else:
            pdf = np.full_like(xs, 1.0 / (hi - lo))
            median = 0.5 * (lo + hi)
            mean = median
            q05 = lo + 0.05 * (hi - lo)
            q95 = lo + 0.95 * (hi - lo)
        return PriorPreview(
            family=family.id,
            family_label=family.label,
            lo=lo,
            hi=hi,
            retained_mass=None,
            x=[float(v) for v in xs],
            pdf=[float(v) for v in pdf],
            median=float(median),
            mean=float(mean),
            q05=float(q05),
            q95=float(q95),
            is_proper=True,
        )

    mass = retained_mass(family, hyper, lo, hi)
    window = _display_range(dist, lo, hi)
    if window is None or mass is None or mass <= 0.0:
        return PriorPreview(
            family=family.id,
            family_label=family.label,
            lo=lo,
            hi=hi,
            retained_mass=mass,
            x=[],
            pdf=[],
            median=None,
            mean=None,
            q05=None,
            q95=None,
            is_proper=True,
            warning=(
                "These bounds leave essentially none of the prior you "
                "specified.  Widen the bounds, or move the prior."
            ),
        )

    left, right = window
    xs = np.linspace(left, right, points)
    pdf = np.asarray(dist.pdf(xs), dtype=float) / mass

    a = -INF if lo is None else lo
    p_lo = float(dist.cdf(a))

    def q(p: float) -> float:
        return float(dist.ppf(p_lo + p * mass))

    # Mean of the truncated density, by quadrature on the visible window --
    # closed forms differ per family and this is accurate enough for a readout.
    dense = np.linspace(left, right, 2001)
    dens = np.asarray(dist.pdf(dense), dtype=float) / mass
    mean = float(np.trapezoid(dense * dens, dense)) if mass > 0 else None

    warning = None
    if mass < 0.99:
        warning = (
            f"Your bounds keep only {mass:.1%} of the {family.label} prior you "
            f"entered.  The curve shown is the restricted prior, which is what "
            f"the model will use."
        )

    return PriorPreview(
        family=family.id,
        family_label=family.label,
        lo=lo,
        hi=hi,
        retained_mass=mass,
        x=[float(v) for v in xs],
        pdf=[float(v) for v in pdf],
        median=q(0.5),
        mean=mean,
        q05=q(0.05),
        q95=q(0.95),
        is_proper=family.is_proper,
        warning=warning,
    )


def prior_density(
    family_id: str,
    hyper: dict[str, float],
    lo: float | None,
    hi: float | None,
    x: np.ndarray,
) -> np.ndarray | None:
    """The prior in force -- truncated to ``[lo, hi]`` and renormalised -- at ``x``.

    This is what the results page draws under the posterior, so the comparison
    is with the prior the model used rather than the one typed in.  ``None``
    when there is no proper density to draw.
    """
    family = get_prior(family_id)
    x = np.asarray(x, dtype=float)
    a = -INF if lo is None else lo
    b = INF if hi is None else hi
    inside = (x >= a) & (x <= b)

    dist = _frozen(family, hyper)
    if dist is None:
        if lo is None or hi is None or hi <= lo:
            return None
        if family.id == "jeffreys_scale":
            if lo <= 0:
                return None
            with np.errstate(divide="ignore"):
                pdf = 1.0 / (x * (math.log(hi) - math.log(lo)))
        else:
            pdf = np.full_like(x, 1.0 / (hi - lo))
        return np.where(inside, pdf, 0.0)

    mass = retained_mass(family, hyper, lo, hi)
    if not mass:
        return None
    return np.where(inside, np.asarray(dist.pdf(x), dtype=float) / mass, 0.0)
