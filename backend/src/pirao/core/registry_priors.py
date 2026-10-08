"""Catalogue of prior families.

Every family declares its hyperparameters by *name*, never by position.  The
legacy models packed hyperparameters into an unconstrained ``array[K] real h``,
which is exactly how a location and a scale get swapped without anyone noticing,
and how a negative sigma reaches ``lognormal_lpdf`` and dies with an opaque
initialization error.  Names here become Stan identifiers (``scale_prior_rate``)
and GUI labels, both generated from the same source.

Two conventions in this file are easy to misread and are therefore spelled out
on every affected family:

* ``gamma`` takes a **rate** (inverse scale) as its second argument;
  ``inv_gamma`` takes a **scale**.
* ``weibull`` takes **shape** first, then scale -- the same ordering trap the
  likelihood registry guards.

``is_proper`` drives a hard validation block and must be read literally: it is
*not* a synonym for "emits no target statement".  ``jeffreys_scale`` does emit
one and would slip through that weaker test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INF = math.inf


@dataclass(frozen=True)
class HyperDef:
    """One hyperparameter of a prior family."""

    name: str
    label: str
    #: Stan bound emitted for this hyperparameter in the ``data`` block, e.g.
    #: ``"<lower=0>"``.  Declaring it here means a bad value fails at data load
    #: naming the variable, rather than inside an ``_lpdf`` call.
    stan_constraint: str = ""
    lo: float = -INF
    hi: float = INF
    exclusive_lo: bool = False
    default: float = 0.0

    def validate(self, value: float) -> str | None:
        """Return a user-facing error message, or ``None`` when acceptable."""
        if not math.isfinite(value):
            return f"{self.label} must be a finite number (got {value!r})."
        if self.exclusive_lo:
            if value <= self.lo:
                return f"{self.label} must be greater than {self.lo:g} (got {value:g})."
        elif value < self.lo:
            return f"{self.label} must be at least {self.lo:g} (got {value:g})."
        if value > self.hi:
            return f"{self.label} must be at most {self.hi:g} (got {value:g})."
        return None


def _positive(name: str, label: str, default: float = 1.0) -> HyperDef:
    return HyperDef(
        name=name,
        label=label,
        stan_constraint="<lower=0>",
        lo=0.0,
        exclusive_lo=True,
        default=default,
    )


def _real(name: str, label: str, default: float = 0.0) -> HyperDef:
    return HyperDef(name=name, label=label, default=default)


@dataclass(frozen=True)
class PriorFamily:
    id: str
    label: str
    #: Stan distribution name, or ``None`` when the family emits no
    #: distribution call (``flat``, ``uniform``, ``jeffreys_scale``).
    stan_dist: str | None
    hypers: tuple[HyperDef, ...]
    #: Support the family *advertises*.  Used to intersect with the
    #: parameter's bounds.
    supp_lo: float
    supp_hi: float
    #: Support of the underlying Stan distribution actually called.  Usually
    #: identical to the advertised support, but a half-normal advertises
    #: ``[0, inf)`` while calling ``normal``, whose support is the whole real
    #: line -- and it is the *underlying* support that decides whether a
    #: truncation normaliser is needed.  ``None`` means "same as advertised".
    dist_supp_lo: float | None = None
    dist_supp_hi: float | None = None
    #: Positional arguments passed to ``stan_dist``.  A token beginning with
    #: ``@`` names a hyperparameter; anything else is emitted as a Stan
    #: literal.  Defaults to every hyperparameter, in declaration order.
    stan_args: tuple[str, ...] | None = None
    #: ``False`` for families whose density does not integrate to a finite
    #: value on an unbounded interval.  Such a family is only admissible when
    #: *both* effective bounds are finite.
    is_proper: bool = True
    #: Raw ``target +=`` expression, used by families with no Stan
    #: distribution.  ``{x}`` is replaced by the parameter name.
    raw_target: str | None = None
    #: ``True`` when the family's (a, b) hyperparameters are folded into the
    #: parameter's effective bounds instead of reaching Stan.
    bounds_from_hypers: bool = False
    notes: str = ""
    #: Parameter roles this family is offered for.  Empty means "any role
    #: whose effective support it can cover".
    roles: tuple[str, ...] = field(default_factory=tuple)

    @property
    def underlying_lo(self) -> float:
        """Support lower end of the Stan distribution actually called."""
        return self.supp_lo if self.dist_supp_lo is None else self.dist_supp_lo

    @property
    def underlying_hi(self) -> float:
        """Support upper end of the Stan distribution actually called."""
        return self.supp_hi if self.dist_supp_hi is None else self.dist_supp_hi

    @property
    def arg_tokens(self) -> tuple[str, ...]:
        """Positional argument tokens, defaulting to the hyperparameters."""
        if self.stan_args is not None:
            return self.stan_args
        return tuple(f"@{h.name}" for h in self.hypers)


PRIORS: dict[str, PriorFamily] = {
    "lognormal": PriorFamily(
        id="lognormal",
        label="Lognormal",
        stan_dist="lognormal",
        hypers=(
            _real("mu", "mu (location, on the log scale)"),
            _positive("sigma", "sigma (scale, on the log scale)"),
        ),
        supp_lo=0.0,
        supp_hi=INF,
        notes=(
            "Location and scale are on the LOG scale: the prior median is "
            "exp(mu), not mu."
        ),
    ),
    "normal": PriorFamily(
        id="normal",
        label="Normal",
        stan_dist="normal",
        hypers=(_real("mu", "mu (location)"), _positive("sigma", "sigma (scale)")),
        supp_lo=-INF,
        supp_hi=INF,
        notes=(
            "The only family whose support never constrains a parameter, so "
            "the effective bounds are entirely the user's.  On a parameter "
            "bounded below at 0 this IS a half-normal with a location; between "
            "two finite bounds it IS the truncated normal.  No separate "
            "'truncated normal' family is offered, because two competing "
            "intervals would let the sampler reach a region the bounds call "
            "legal and the truncation calls impossible."
        ),
    ),
    "student_t": PriorFamily(
        id="student_t",
        label="Student t",
        stan_dist="student_t",
        hypers=(
            _positive("nu", "nu (degrees of freedom)", default=3.0),
            _real("mu", "mu (location)"),
            _positive("sigma", "sigma (scale)"),
        ),
        supp_lo=-INF,
        supp_hi=INF,
        notes="Argument order is (nu, mu, sigma) -- degrees of freedom FIRST.",
    ),
    "gamma": PriorFamily(
        id="gamma",
        label="Gamma",
        stan_dist="gamma",
        hypers=(
            _positive("shape", "shape", default=2.0),
            _positive("rate", "rate (= 1 / scale)"),
        ),
        supp_lo=0.0,
        supp_hi=INF,
        notes=(
            "The second argument is a RATE, i.e. an inverse scale; the prior "
            "mean is shape / rate.  Contrast inv_gamma, whose second argument "
            "is a scale."
        ),
    ),
    "inv_gamma": PriorFamily(
        id="inv_gamma",
        label="Inverse gamma",
        stan_dist="inv_gamma",
        hypers=(
            _positive("shape", "shape", default=2.0),
            _positive("scale", "scale"),
        ),
        supp_lo=0.0,
        supp_hi=INF,
        notes=("The second argument is a SCALE, deliberately unlike gamma's rate."),
    ),
    "exponential": PriorFamily(
        id="exponential",
        label="Exponential",
        stan_dist="exponential",
        hypers=(_positive("rate", "rate (= 1 / mean)"),),
        supp_lo=0.0,
        supp_hi=INF,
        notes=(
            "Maximum-entropy prior on a positive parameter given a mean.  A "
            "reasonable default when only an order of magnitude is known: set "
            "rate = 1 / guess."
        ),
    ),
    "weibull": PriorFamily(
        id="weibull",
        label="Weibull",
        stan_dist="weibull",
        hypers=(
            _positive("shape", "shape", default=2.0),
            _positive("scale", "scale"),
        ),
        supp_lo=0.0,
        supp_hi=INF,
        notes="Shape first, then scale -- the same ordering as the likelihood.",
    ),
    "beta": PriorFamily(
        id="beta",
        label="Beta",
        stan_dist="beta",
        hypers=(
            _positive("a", "a (prior successes + 1)"),
            _positive("b", "b (prior failures + 1)"),
        ),
        supp_lo=0.0,
        supp_hi=1.0,
        roles=("prob",),
        notes=(
            "Beta(1, 1) is uniform on [0, 1].  The density is -inf at 0 and 1, "
            "so with a < 1 and a zero-failure dataset the posterior piles onto "
            "the boundary; prefer a >= 1, or a lower bound slightly above 0."
        ),
    ),
    "half_normal": PriorFamily(
        id="half_normal",
        label="Half-normal",
        stan_dist="normal",
        hypers=(_positive("sigma", "sigma (scale)"),),
        supp_lo=0.0,
        supp_hi=INF,
        dist_supp_lo=-INF,
        dist_supp_hi=INF,
        stan_args=("0", "@sigma"),
        notes=(
            "Emitted as a normal centred at 0; the half-ness comes from the "
            "parameter's own lower bound.  A good weakly-informative default "
            "for a shape parameter."
        ),
    ),
    "half_cauchy": PriorFamily(
        id="half_cauchy",
        label="Half-Cauchy",
        stan_dist="cauchy",
        hypers=(_positive("sigma", "sigma (scale)"),),
        supp_lo=0.0,
        supp_hi=INF,
        dist_supp_lo=-INF,
        dist_supp_hi=INF,
        stan_args=("0", "@sigma"),
        notes=(
            "Same construction as the half-normal, with a much heavier tail.  "
            "Pair it with a finite upper bound unless slow mixing is "
            "acceptable."
        ),
    ),
    "uniform": PriorFamily(
        id="uniform",
        label="Uniform",
        stan_dist=None,
        hypers=(
            HyperDef("a", "a (lower)", default=0.0),
            HyperDef("b", "b (upper)", default=1.0),
        ),
        supp_lo=-INF,
        supp_hi=INF,
        is_proper=False,
        bounds_from_hypers=True,
        notes=(
            "(a, b) are folded into the parameter's effective bounds rather "
            "than passed to Stan.  Writing 'x ~ uniform(a, b)' would be a "
            "no-op: under '~' Stan drops the constant -log(b - a), so the "
            "statement contributes exactly nothing while looking like it "
            "does something."
        ),
    ),
    "flat": PriorFamily(
        id="flat",
        label="Flat (uniform over the bounds)",
        stan_dist=None,
        hypers=(),
        supp_lo=-INF,
        supp_hi=INF,
        is_proper=False,
        notes=(
            "Requires both bounds to be finite.  With only one bound the "
            "posterior can be improper, and Stan does not warn: R-hat stays "
            "near 1.0 while the mean runs off to infinity."
        ),
    ),
    "jeffreys_scale": PriorFamily(
        id="jeffreys_scale",
        label="Jeffreys (scale-invariant, 1/x)",
        stan_dist=None,
        raw_target="-log({x})",
        hypers=(),
        supp_lo=0.0,
        supp_hi=INF,
        is_proper=False,
        notes=(
            "The improper 1/x scale prior.  Requires both bounds finite for "
            "the same reason as 'flat'.  Note this family DOES emit a target "
            "increment, so an is_proper check must not be inferred from "
            "'emits no statement'."
        ),
    ),
}


#: Families whose support lies in the non-negative half-line.  Used by the
#: compatibility rules to reject a positive-support prior on a parameter whose
#: effective interval would extend below zero.
POSITIVE_SUPPORT_FAMILIES = frozenset(
    fid for fid, f in PRIORS.items() if f.supp_lo >= 0.0 and f.supp_hi == INF
)


def get_prior(family_id: str) -> PriorFamily:
    try:
        return PRIORS[family_id]
    except KeyError:
        known = ", ".join(sorted(PRIORS))
        raise KeyError(
            f"Unknown prior family {family_id!r}. Known families: {known}"
        ) from None
