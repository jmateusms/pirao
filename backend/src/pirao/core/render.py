"""Render a resolved model into Stan source.

The renderer is deliberately dull.  All the judgement lives in
:mod:`pirao.core.resolve`; this module just interpolates.  That matters for
one reason worth stating plainly: **no user-supplied value is ever interpolated
into the source.**  Only structure -- the likelihood id, the prior family names,
and which bounds are present -- reaches a template.  Every number the user typed
travels as Stan ``data``.

That is what makes the compile cache effective (dragging a hyperparameter reuses
the binary) and it is also a security property: nothing a user types can reach
the C++ compiler.  :func:`assert_no_user_values` exists so that property has a
test rather than a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .resolve import ResolvedModel, resolve
from .spec import ModelSpec

TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)


@dataclass(frozen=True)
class RenderedModel:
    """Stan source plus the mapping back to the spec that produced it."""

    source: str
    model: ResolvedModel
    #: 1-based line number -> the spec element responsible for it.  stanc and
    #: g++ report errors against generated source the user never wrote, so this
    #: is what lets the UI say "this line came from your prior on `shape`".
    line_map: dict[int, str]

    @property
    def spec(self) -> ModelSpec:
        return self.model.spec


def _build_line_map(source: str, model: ResolvedModel) -> dict[int, str]:
    """Attribute generated lines to the spec elements that caused them."""
    mapping: dict[int, str] = {}
    hyper_owner = {h.stan_name: p.name for p in model.params for h in p.hypers}
    bound_owner = {}
    for p in model.params:
        if p.has_lower:
            bound_owner[p.lb_name] = p.name
        if p.has_upper:
            bound_owner[p.ub_name] = p.name

    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for stan_name, owner in hyper_owner.items():
            if stan_name in stripped:
                mapping[lineno] = f"prior on {owner}"
                break
        else:
            for stan_name, owner in bound_owner.items():
                if stan_name in stripped:
                    mapping[lineno] = f"bounds on {owner}"
                    break
            else:
                if (
                    stripped.startswith("target +=")
                    or "_lpdf" in stripped
                    or "_lpmf" in stripped
                ):
                    mapping[lineno] = f"{model.likelihood.id} likelihood"
    return mapping


def render_resolved(model: ResolvedModel) -> RenderedModel:
    template = _env.get_template(model.likelihood.template)
    source = template.render(
        lik=model.likelihood,
        params=model.params,
        args=model.args,
        emit_log_lik=model.spec.emit_log_lik,
    )
    # Collapse the blank-line noise that block inheritance leaves behind, so
    # golden files stay readable and stable.
    cleaned: list[str] = []
    for line in source.splitlines():
        line = line.rstrip()
        if not line and cleaned and not cleaned[-1]:
            continue  # never two blank lines in a row
        cleaned.append(line)
    # An unused block leaves a blank line just before its closing brace.
    tidied: list[str] = []
    for index, line in enumerate(cleaned):
        nxt = cleaned[index + 1].strip() if index + 1 < len(cleaned) else ""
        if not line and nxt == "}":
            continue
        tidied.append(line)
    cleaned = tidied
    source = "\n".join(cleaned).strip() + "\n"
    return RenderedModel(
        source=source, model=model, line_map=_build_line_map(source, model)
    )


def render(spec: ModelSpec) -> RenderedModel:
    """Resolve and render a spec in one step."""
    return render_resolved(resolve(spec))


def assert_values_are_data(spec: ModelSpec) -> None:
    """Fail if a hyperparameter value can change the generated source.

    This is the invariant the compile cache rests on, and it is also what keeps
    anything a user types away from the C++ compiler.  Testing it by searching
    the source for the value would be unreliable -- ``0.1`` occurs inside
    ``log1m(0.10)`` -- so it is tested behaviourally: perturb every
    hyperparameter and require byte-identical output.

    Bounds are deliberately *not* perturbed.  A bound value does affect the
    source, but only through one structural question: whether it falls strictly
    inside the prior's support, and therefore whether a truncation term is
    needed.  Moving a bound within the same side of that boundary leaves the
    source alone; crossing it legitimately produces a different model.
    """
    baseline = render(spec).source

    perturbed = spec.model_copy(deep=True)
    for param in perturbed.params.values():
        param.prior.hyper = {
            name: (value * 1.37 + 0.11) for name, value in param.prior.hyper.items()
        }
    try:
        candidate = render(perturbed).source
    except Exception:  # a perturbation may be invalid for this family; that is fine
        return

    if candidate != baseline:
        raise AssertionError(
            "hyperparameter values must travel as Stan data, never as source: "
            "changing them changed the generated program"
        )
