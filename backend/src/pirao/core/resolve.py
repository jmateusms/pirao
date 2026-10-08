"""Turn a :class:`~pirao.core.spec.ModelSpec` into everything a template needs.

The important idea here is that **effective bounds are computed, not merely
validated**.  A parameter's admissible interval is the intersection of three
things -- its natural support, its prior's support, and whatever the user typed:

    lo = max(natural_lo, prior_support_lo, user_lower)
    hi = min(natural_hi, prior_support_hi, user_upper)

Clamping rather than warning is what makes "the prior's support is narrower
than the parameter's range" *structurally impossible* rather than merely
discouraged: the sampler never sees a region of zero density, so it can never
stall on one.  The cost is that the clamp is silent, which is why
:class:`ResolvedParam` carries the resulting interval for the UI to display.

Prior statements are emitted as **fully normalised** ``target +=`` increments
rather than ``~ dist(...) T[lb, ub]``.  Two reasons, both load-bearing:

* ``target += ... T[...]`` is a *syntax error* -- ``T[]`` attaches only to
  ``~`` -- so choosing ``T[]`` would foreclose ``target +=`` entirely;
* ``~ ... T[]`` still drops the base density's normalising constants, because
  ``~`` uses the unnormalised ``_lupdf`` form.  Since the point of this tool is
  to let users *compare* prior choices, ``lp__`` has to stay comparable across
  families, and only the explicit normalised form achieves that.

When bounds and hyperparameters are data -- an invariant this module enforces --
the normalisation term is a constant, so it shifts ``lp__`` without touching the
gradient, the trajectory, or any posterior summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .registry_likelihoods import LikelihoodDef, ParamDef, get_likelihood
from .registry_priors import PriorFamily, get_prior
from .spec import ModelSpec

INF = math.inf


class SpecError(ValueError):
    """One or more user-facing problems with a specification."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("; ".join(messages))


@dataclass(frozen=True)
class HyperBinding:
    """A prior hyperparameter, as it appears in the generated Stan data block."""

    stan_name: str
    value: float
    stan_constraint: str
    comment: str


@dataclass(frozen=True)
class ResolvedParam:
    """A parameter with its bounds resolved and its prior statement built."""

    name: str
    role: str
    family: str
    family_label: str
    #: Effective bounds.  ``None`` means unbounded on that side, in which case
    #: no bound is declared and no bound variable is emitted.
    lo: float | None
    hi: float | None
    hypers: tuple[HyperBinding, ...]
    #: Complete ``target += ...;`` line, or ``None`` when the family
    #: contributes nothing (which only happens for a fully unbounded flat
    #: prior, itself rejected earlier).
    prior_statement: str | None
    truncated_below: bool
    truncated_above: bool

    @property
    def lb_name(self) -> str:
        return f"{self.name}_lb"

    @property
    def ub_name(self) -> str:
        return f"{self.name}_ub"

    @property
    def has_lower(self) -> bool:
        return self.lo is not None

    @property
    def has_upper(self) -> bool:
        return self.hi is not None

    @property
    def declaration(self) -> str:
        """The Stan ``parameters`` block declaration for this parameter."""
        parts = []
        if self.has_lower:
            parts.append(f"lower={self.lb_name}")
        if self.has_upper:
            parts.append(f"upper={self.ub_name}")
        bounds = f"<{', '.join(parts)}>" if parts else ""
        return f"real{bounds} {self.name};"

    @property
    def bounds_label(self) -> str:
        lo = "-inf" if self.lo is None else f"{self.lo:g}"
        hi = "+inf" if self.hi is None else f"{self.hi:g}"
        return f"[{lo}, {hi}]"


@dataclass(frozen=True)
class ResolvedModel:
    """Everything the Jinja templates consume."""

    spec: ModelSpec
    likelihood: LikelihoodDef
    params: tuple[ResolvedParam, ...]
    #: Parameters joined in ``stan_arg_order`` -- the string every template
    #: interpolates instead of writing an argument list by hand.
    args: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def param(self, name: str) -> ResolvedParam:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def data_values(self) -> dict[str, float]:
        """Bound and hyperparameter values destined for the Stan data dict."""
        out: dict[str, float] = {}
        for p in self.params:
            if p.has_lower:
                out[p.lb_name] = float(p.lo)  # type: ignore[arg-type]
            if p.has_upper:
                out[p.ub_name] = float(p.hi)  # type: ignore[arg-type]
            for h in p.hypers:
                out[h.stan_name] = float(h.value)
        return out


def _finite_or_none(value: float) -> float | None:
    return None if math.isinf(value) else value


def _check_hypers(
    param_name: str,
    family: PriorFamily,
    supplied: dict[str, float],
    errors: list[str],
) -> tuple[HyperBinding, ...]:
    """Validate hyperparameter names and values; build their Stan bindings."""
    expected = {h.name for h in family.hypers}
    got = set(supplied)

    for missing in sorted(expected - got):
        hyper = next(h for h in family.hypers if h.name == missing)
        errors.append(
            f"{param_name}: the {family.label} prior needs a value for {hyper.label}."
        )
    for extra in sorted(got - expected):
        errors.append(
            f"{param_name}: the {family.label} prior has no hyperparameter "
            f"called {extra!r}."
            + (
                f"  It takes: {', '.join(h.name for h in family.hypers)}."
                if family.hypers
                else "  It takes none."
            )
        )

    bindings: list[HyperBinding] = []
    for hyper in family.hypers:
        if hyper.name not in supplied:
            continue
        value = float(supplied[hyper.name])
        if (message := hyper.validate(value)) is not None:
            errors.append(f"{param_name}: {message}")
            continue
        bindings.append(
            HyperBinding(
                stan_name=f"{param_name}_prior_{hyper.name}",
                value=value,
                stan_constraint=hyper.stan_constraint,
                comment=f"{family.id}: {hyper.label}",
            )
        )

    if family.id == "uniform" and {"a", "b"} <= got:
        if supplied["a"] >= supplied["b"]:
            errors.append(
                f"{param_name}: the uniform prior's lower bound a "
                f"({supplied['a']:g}) must be below its upper bound b "
                f"({supplied['b']:g})."
            )
    return tuple(bindings)


def _prior_statement(
    param: ResolvedParam | None,
    name: str,
    family: PriorFamily,
    hypers: tuple[HyperBinding, ...],
    lo: float | None,
    hi: float | None,
    trunc_lo: bool,
    trunc_hi: bool,
) -> str | None:
    """Build the fully normalised ``target +=`` line for one parameter."""
    lb, ub = f"{name}_lb", f"{name}_ub"

    if family.stan_dist is None:
        if family.raw_target is not None:
            # Jeffreys 1/x on [lb, ub]: the normaliser is log(log(ub/lb)).
            body = family.raw_target.format(x=name)
            return f"target += {body} - log(log({ub}) - log({lb}));"
        # flat / uniform: Stan's own bound transform supplies the shape; this
        # term supplies the normalisation, keeping lp__ comparable.
        return f"target += -log({ub} - {lb});"

    by_name = {h.stan_name.rsplit("_prior_", 1)[1]: h.stan_name for h in hypers}
    args = ", ".join(
        by_name[token[1:]] if token.startswith("@") else token
        for token in family.arg_tokens
    )

    dist = family.stan_dist
    expr = f"{dist}_lpdf({name} | {args})"
    if trunc_lo and trunc_hi:
        expr += (
            f"\n            - log_diff_exp({dist}_lcdf({ub} | {args}), "
            f"{dist}_lcdf({lb} | {args}))"
        )
    elif trunc_hi:
        expr += f" - {dist}_lcdf({ub} | {args})"
    elif trunc_lo:
        expr += f" - {dist}_lccdf({lb} | {args})"
    return f"target += {expr};"


def resolve(spec: ModelSpec) -> ResolvedModel:
    """Validate a spec and expand it into template-ready form.

    Raises :class:`SpecError` with every problem found, rather than only the
    first, so a form can highlight all offending fields at once.
    """
    lik = get_likelihood(spec.likelihood)
    errors: list[str] = []
    warnings: list[str] = []
    resolved: list[ResolvedParam] = []

    for pdef in lik.params:
        pspec = spec.params[pdef.name]
        try:
            family = get_prior(pspec.prior.family)
        except KeyError as exc:
            errors.append(str(exc))
            continue

        if family.roles and pdef.role not in family.roles:
            errors.append(
                f"{pdef.name}: a {family.label} prior only applies to a "
                f"probability, but {pdef.name} is a {pdef.role} parameter."
            )
            continue

        hypers = _check_hypers(pdef.name, family, dict(pspec.prior.hyper), errors)
        supplied = dict(pspec.prior.hyper)
        if len(hypers) != len(family.hypers):
            # A hyperparameter was missing or invalid; the message is already
            # recorded.  Carry on to collect problems with other parameters
            # rather than failing here on a half-built prior.
            continue

        # -- R1: effective bounds are the intersection of three intervals ----
        lo = max(
            pdef.nat_lo,
            family.supp_lo,
            pspec.lower if pspec.lower is not None else -INF,
        )
        hi = min(
            pdef.nat_hi,
            family.supp_hi,
            pspec.upper if pspec.upper is not None else INF,
        )
        if family.bounds_from_hypers:
            if "a" in supplied:
                lo = max(lo, float(supplied["a"]))
            if "b" in supplied:
                hi = min(hi, float(supplied["b"]))

        # -- R2: an empty feasible set names all three contributors ----------
        if lo >= hi:
            errors.append(
                f"{pdef.name}: no value is allowed.  Its natural support is "
                f"({_fmt(pdef.nat_lo)}, {_fmt(pdef.nat_hi)}), the "
                f"{family.label} prior's support is "
                f"({_fmt(family.supp_lo)}, {_fmt(family.supp_hi)}), and your "
                f"bounds are ({_fmt(pspec.lower)}, {_fmt(pspec.upper)}); "
                f"together they leave nothing."
            )
            continue

        # -- R4: a positive-support prior cannot cover a negative interval ---
        if family.supp_lo >= 0.0 and lo < 0.0:
            errors.append(
                f"{pdef.name}: a {family.label} prior only supports "
                f"non-negative values, but {pdef.name} may go as low as "
                f"{lo:g}.  Choose a prior on the whole real line (normal or "
                f"Student t), or set a lower bound of 0."
            )
            continue

        # -- R6: improper families demand two finite bounds ------------------
        if not family.is_proper and (math.isinf(lo) or math.isinf(hi)):
            errors.append(
                f"{pdef.name}: the {family.label} prior does not integrate to "
                f"a finite value on an unbounded range, so it needs BOTH a "
                f"lower and an upper bound.  Without them the posterior can be "
                f"improper, and the sampler will not warn you -- R-hat stays "
                f"near 1 while the estimate runs off to infinity."
            )
            continue
        if family.id == "jeffreys_scale" and lo <= 0.0:
            errors.append(
                f"{pdef.name}: the Jeffreys 1/x prior needs a lower bound "
                f"strictly above zero (its density is unbounded at 0)."
            )
            continue

        trunc_lo = lo > family.underlying_lo
        trunc_hi = hi < family.underlying_hi
        lo_out, hi_out = _finite_or_none(lo), _finite_or_none(hi)

        if (trunc_lo or trunc_hi) and family.stan_dist is not None:
            warnings.extend(
                _truncation_warnings(pdef, family, supplied, lo_out, hi_out)
            )

        statement = _prior_statement(
            None, pdef.name, family, hypers, lo_out, hi_out, trunc_lo, trunc_hi
        )
        resolved.append(
            ResolvedParam(
                name=pdef.name,
                role=pdef.role,
                family=family.id,
                family_label=family.label,
                lo=lo_out,
                hi=hi_out,
                hypers=hypers,
                prior_statement=statement,
                truncated_below=trunc_lo,
                truncated_above=trunc_hi,
            )
        )

    if lik.mission == "time" and not spec.mission_times:
        warnings.append(
            "No mission time was given, so no reliability estimate will be "
            "reported.  Add at least one mission time to get one."
        )

    if errors:
        raise SpecError(errors)

    order = lik.stan_arg_order
    return ResolvedModel(
        spec=spec,
        likelihood=lik,
        params=tuple(resolved),
        args=", ".join(order),
        warnings=tuple(warnings),
    )


def _truncation_warnings(
    pdef: ParamDef,
    family: PriorFamily,
    supplied: dict[str, float],
    lo: float | None,
    hi: float | None,
) -> list[str]:
    """Warn when bounds quietly discard a large share of the prior."""
    from .preview import retained_mass

    mass = retained_mass(family, supplied, lo, hi)
    if mass is None or mass >= 0.99:
        return []
    return [
        f"{pdef.name}: your bounds keep only {mass:.1%} of the "
        f"{family.label} prior you specified.  The prior actually in force is "
        f"that distribution restricted to {_fmt(lo)} to {_fmt(hi)}, which has "
        f"a different centre and spread from the one you entered."
    ]


def _fmt(value: float | None) -> str:
    if value is None:
        return "none"
    if value == INF:
        return "+inf"
    if value == -INF:
        return "-inf"
    return f"{value:g}"
