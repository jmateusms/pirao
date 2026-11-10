"""Every likelihood x prior pair the registry allows must pass ``stanc``.

The fast tier only checks that a program renders; the slow tier compiles a
handful.  A template change that breaks one pair would reach users unnoticed,
and ``stanc`` takes milliseconds per program, so all of them are checked here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pirao.core.registry_likelihoods import LIKELIHOODS
from pirao.core.registry_priors import PRIORS
from pirao.core.render import render
from pirao.core.resolve import SpecError, resolve
from pirao.core.spec import ModelSpec, ParamSpec, PriorSpec

# Prior for the parameters that are not the one under test.
_COMPANION = {
    "rate": ("gamma", {"shape": 2.0, "rate": 0.01}),
    "scale": ("gamma", {"shape": 2.0, "rate": 0.01}),
    "shape": ("half_normal", {"sigma": 2.0}),
    "prob": ("beta", {"a": 1.0, "b": 1.0}),
    "mu": ("normal", {"mu": 5.0, "sigma": 2.0}),
    "sigma": ("half_normal", {"sigma": 1.0}),
}


def _spec(likelihood_id: str, param: str, family_id: str) -> ModelSpec:
    lik = LIKELIHOODS[likelihood_id]
    params = {}
    for p in lik.params:
        if p.name == param:
            family = PRIORS[family_id]
            prior = PriorSpec(
                family=family_id, hyper={h.name: h.default for h in family.hypers}
            )
        else:
            name, hyper = _COMPANION[p.role]
            prior = PriorSpec(family=name, hyper=hyper)
        params[p.name] = ParamSpec(lower=0.0 if p.nat_lo == 0.0 else None, prior=prior)
    return ModelSpec(
        likelihood=likelihood_id,
        params=params,
        mission_times=[100.0] if lik.mission == "time" else [],
        mission_demands=1 if lik.mission == "demands" else None,
    )


def _valid_pairs() -> list[tuple[str, str, str]]:
    """(likelihood, parameter, prior family) the resolver accepts."""
    pairs = []
    for lik in LIKELIHOODS.values():
        for p in lik.params:
            for family_id in PRIORS:
                try:
                    resolve(_spec(lik.id, p.name, family_id))
                except SpecError:
                    continue
                pairs.append((lik.id, p.name, family_id))
    return pairs


def _find_stanc() -> Path | None:
    candidates = [os.environ.get("STANC"), shutil.which("stanc")]
    cmdstan = os.environ.get("CMDSTAN")
    if cmdstan is None:
        try:
            import cmdstanpy

            cmdstan = str(cmdstanpy.cmdstan_path())
        except Exception:
            pass
    if cmdstan:
        candidates.append(str(Path(cmdstan) / "bin" / "stanc"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


STANC = _find_stanc()
PAIRS = _valid_pairs()


def test_the_registry_yields_pairs():
    assert len(PAIRS) > 50


def test_ci_has_stanc():
    if not os.environ.get("PIRAO_REQUIRE_STANC"):
        pytest.skip("only enforced where PIRAO_REQUIRE_STANC is set (CI)")
    assert STANC is not None, "PIRAO_REQUIRE_STANC is set but no stanc was found"


@pytest.mark.skipif(
    STANC is None,
    reason="no stanc binary: set STANC, put stanc on PATH or install CmdStan",
)
@pytest.mark.parametrize(
    ("likelihood", "param", "family"), PAIRS, ids=["-".join(p) for p in PAIRS]
)
def test_rendered_program_passes_stanc(tmp_path, likelihood, param, family):
    source = render(_spec(likelihood, param, family)).source
    model = tmp_path / "model.stan"
    model.write_text(source)
    run = subprocess.run(
        [str(STANC), f"--o={tmp_path / 'model.hpp'}", str(model)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, f"{run.stderr}\n--- source ---\n{source}"
