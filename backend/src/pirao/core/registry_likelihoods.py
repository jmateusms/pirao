"""Catalogue of likelihood models.

``stan_arg_order`` is the single most important field in this module.  Stan's
``weibull_lpdf(y | alpha, sigma)`` takes **shape first, then scale**, while the
reliability literature calls the shape ``beta`` and the scale ``alpha`` -- the
exact opposite naming.  A model with the two swapped still compiles, still
samples, and still reports R-hat 1.00; only the answers are wrong.  So the order
lives here once, as data, and every template emits ``{{ args }}`` rather than
writing an argument list by hand.

Parameters are named for their *role* (``shape``, ``scale``, ``rate``,
``prob``, ``mu``, ``sigma``) instead of carrying the legacy ``alpha``/``beta``
labels, which removes the trap at its source rather than guarding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INF = math.inf

#: Roles a parameter can play.  A prior family may restrict itself to a subset
#: (a Beta prior is only meaningful on a probability).
ROLES = ("rate", "scale", "shape", "prob", "mu", "sigma")


@dataclass(frozen=True)
class ParamDef:
    """A parameter exposed by a likelihood."""

    name: str
    role: str
    #: Mathematically natural support, before the user narrows anything.
    nat_lo: float
    nat_hi: float
    label: str = ""
    description: str = ""
    #: Bound preset offered in the UI as a one-click option, e.g. "assume
    #: wear-out" pins a Weibull shape at ``lower = 1``.
    presets: tuple[tuple[str, float | None, float | None], ...] = field(
        default_factory=tuple
    )

    @property
    def display_label(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class LikelihoodDef:
    id: str
    label: str
    params: tuple[ParamDef, ...]
    #: Positional order in which parameters are passed to the Stan
    #: distribution.  SINGLE SOURCE OF TRUTH -- see the module docstring.
    stan_arg_order: tuple[str, ...]
    #: ``"time"`` models consume a ``time`` column and a mission time;
    #: ``"demands"`` models consume an ``n`` column and a mission demand count.
    mission: str
    template: str
    #: Tidy-table columns this likelihood requires, beyond the always-optional
    #: ``device`` and ``relevance``.
    required_columns: tuple[str, ...]
    description: str = ""

    def param(self, name: str) -> ParamDef:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{self.id} has no parameter {name!r}")

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)


_SHAPE_PRESETS = (
    ("Assume wear-out (shape >= 1)", 1.0, None),
    ("Assume infant mortality (shape <= 1)", None, 1.0),
)


LIKELIHOODS: dict[str, LikelihoodDef] = {
    "exponential": LikelihoodDef(
        id="exponential",
        label="Exponential (constant hazard)",
        params=(
            ParamDef(
                name="rate",
                role="rate",
                nat_lo=0.0,
                nat_hi=INF,
                label="rate",
                description=(
                    "Failures per unit time.  The mean time to failure is 1 / rate."
                ),
            ),
        ),
        stan_arg_order=("rate",),
        mission="time",
        template="exponential.stan.j2",
        required_columns=("time", "failure"),
        description=(
            "Memoryless lifetime model: the hazard does not change with age.  "
            "The Weibull with shape fixed at 1."
        ),
    ),
    "weibull": LikelihoodDef(
        id="weibull",
        label="Weibull",
        params=(
            ParamDef(
                name="shape",
                role="shape",
                nat_lo=0.0,
                nat_hi=INF,
                label="shape",
                description=(
                    "Below 1 the hazard falls with age (infant mortality); at "
                    "1 it is constant (exponential); above 1 it rises "
                    "(wear-out).  Called beta in much of the reliability "
                    "literature."
                ),
                presets=_SHAPE_PRESETS,
            ),
            ParamDef(
                name="scale",
                role="scale",
                nat_lo=0.0,
                nat_hi=INF,
                label="scale",
                description=(
                    "Characteristic life: the age by which about 63.2% of "
                    "units have failed.  Called alpha or eta in much of the "
                    "reliability literature."
                ),
            ),
        ),
        stan_arg_order=("shape", "scale"),
        mission="time",
        template="weibull.stan.j2",
        required_columns=("time", "failure"),
        description=(
            "The workhorse lifetime model; its shape parameter distinguishes "
            "infant mortality from wear-out."
        ),
    ),
    "lognormal": LikelihoodDef(
        id="lognormal",
        label="Lognormal",
        params=(
            ParamDef(
                name="mu",
                role="mu",
                nat_lo=-INF,
                nat_hi=INF,
                label="mu (log-scale location)",
                description=(
                    "Mean of log(lifetime).  The median lifetime is exp(mu).  "
                    "This parameter may be negative."
                ),
            ),
            ParamDef(
                name="sigma",
                role="sigma",
                nat_lo=0.0,
                nat_hi=INF,
                label="sigma (log-scale spread)",
                description="Standard deviation of log(lifetime).",
            ),
        ),
        stan_arg_order=("mu", "sigma"),
        mission="time",
        template="lognormal.stan.j2",
        required_columns=("time", "failure"),
        description=(
            "Lifetime model for damage that accumulates multiplicatively; "
            "common for fatigue and crack growth."
        ),
    ),
    "gamma": LikelihoodDef(
        id="gamma",
        label="Gamma",
        params=(
            ParamDef(
                name="shape",
                role="shape",
                nat_lo=0.0,
                nat_hi=INF,
                label="shape",
                description=(
                    "At 1 the gamma reduces to the exponential; above 1 the "
                    "hazard rises toward a constant."
                ),
                presets=_SHAPE_PRESETS,
            ),
            ParamDef(
                name="rate",
                role="rate",
                nat_lo=0.0,
                nat_hi=INF,
                label="rate (= 1 / scale)",
                description="Inverse scale.  The mean lifetime is shape / rate.",
            ),
        ),
        stan_arg_order=("shape", "rate"),
        mission="time",
        template="gamma.stan.j2",
        required_columns=("time", "failure"),
        description=(
            "Lifetime as the waiting time for several exponential shocks; a "
            "flexible alternative to the Weibull."
        ),
    ),
    "bernoulli_geometric": LikelihoodDef(
        id="bernoulli_geometric",
        label="Bernoulli per demand (failed ON demand n)",
        params=(
            ParamDef(
                name="prob",
                role="prob",
                nat_lo=0.0,
                nat_hi=1.0,
                label="failure probability per demand",
                description="Probability that any single demand fails.",
            ),
        ),
        stan_arg_order=("prob",),
        mission="demands",
        template="bernoulli_geometric.stan.j2",
        required_columns=("n", "failure"),
        description=(
            "Right-censored geometric.  Each row is one specimen that survived "
            "n demands (failure = 0) or failed ON the n-th demand "
            "(failure = 1).  Choose this when n records WHEN the unit failed.  "
            "If instead you only know that it failed somewhere within n "
            "attempts, use the binomial."
        ),
    ),
    "binomial": LikelihoodDef(
        id="binomial",
        label="Binomial (failures out of n trials)",
        params=(
            ParamDef(
                name="prob",
                role="prob",
                nat_lo=0.0,
                nat_hi=1.0,
                label="failure probability per trial",
                description="Probability that any single trial fails.",
            ),
        ),
        stan_arg_order=("prob",),
        mission="demands",
        template="binomial.stan.j2",
        required_columns=("n", "failures"),
        description=(
            "Aggregated demand data: each row is a group of n trials with a "
            "count of failures among them.  A different likelihood from the "
            "per-demand geometric, not a different data layout for it -- the "
            "posteriors genuinely differ."
        ),
    ),
}


def get_likelihood(likelihood_id: str) -> LikelihoodDef:
    try:
        return LIKELIHOODS[likelihood_id]
    except KeyError:
        known = ", ".join(sorted(LIKELIHOODS))
        raise KeyError(
            f"Unknown likelihood {likelihood_id!r}. Known likelihoods: {known}"
        ) from None
