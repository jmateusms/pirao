"""End-to-end correctness checks against closed-form posteriors.

Two of the models in this tool have posteriors that can be written down exactly,
*and the relevance weighting does not break that* -- the weights simply enter the
sufficient statistics.  That makes them the strongest test asset in the project:
each one exercises the entire stack (spec -> resolve -> render -> compile ->
data transform -> sample -> summarise) and compares the result against arithmetic
rather than against a previous run.

    Bernoulli (censored geometric) with a Beta(a, b) prior
        posterior = Beta(a + B, b + A),  A = sum w(n - f),  B = sum w f

    Exponential with a Gamma(a, b) prior
        posterior = Gamma(a + W, b + S),  W = sum w*failure,  S = sum w*time

An argument-order swap, a mis-keyed data dict, a dropped relevance weight, or a
wrong truncation term all fail these loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pirao.core.results import reliability_over_time, summarize
from pirao.core.sample import run
from pirao.core.spec import ModelSpec, ParamSpec, PriorSpec, SamplerConfig

SAMPLER = SamplerConfig(chains=4, iter_warmup=1000, iter_sampling=4000, seed=20260729)


def _check_against(draws: np.ndarray, exact, label: str, tol_q: float = 0.02) -> None:
    """Compare posterior draws with an exact distribution."""
    assert np.isfinite(draws).all(), f"{label}: non-finite draws"

    exact_mean, exact_sd = float(exact.mean()), float(exact.std())
    got_mean, got_sd = float(draws.mean()), float(draws.std())

    # Monte Carlo error on the mean, generously allowed for autocorrelation.
    mcse = exact_sd / np.sqrt(len(draws)) * 6
    assert abs(got_mean - exact_mean) < max(mcse, 0.01 * exact_sd), (
        f"{label}: posterior mean {got_mean:.6g} differs from the exact "
        f"{exact_mean:.6g}"
    )
    assert abs(got_sd - exact_sd) < 0.06 * exact_sd, (
        f"{label}: posterior sd {got_sd:.6g} differs from the exact {exact_sd:.6g}"
    )

    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        got, want = float(np.quantile(draws, p)), float(exact.ppf(p))
        assert abs(got - want) < tol_q * exact_sd + 0.03 * abs(want), (
            f"{label}: {p:.0%} quantile {got:.6g} differs from the exact {want:.6g}"
        )


@pytest.mark.slow
def test_bernoulli_geometric_matches_exact_beta_posterior():
    a, b = 2.0, 5.0
    rows = [
        {"device": "A", "n": 4, "failure": 1, "relevance": 1.0},
        {"device": "B", "n": 12, "failure": 0, "relevance": 1.0},
        {"device": "C", "n": 7, "failure": 1, "relevance": 0.5},
        {"device": "D", "n": 30, "failure": 0, "relevance": 0.25},
        {"device": "E", "n": 3, "failure": 1, "relevance": 1.0},
    ]
    spec = ModelSpec(
        likelihood="bernoulli_geometric",
        mission_demands=1,
        params={
            "prob": ParamSpec(prior=PriorSpec(family="beta", hyper={"a": a, "b": b}))
        },
    )

    A = sum(r["relevance"] * (r["n"] - r["failure"]) for r in rows)
    B = sum(r["relevance"] * r["failure"] for r in rows)
    exact = stats.beta(a + B, b + A)

    result = run(spec, rows, SAMPLER)
    draws = np.asarray(result.fit.stan_variable("prob"), dtype=float)
    _check_against(draws, exact, "bernoulli_geometric / beta")

    # The reported reliability must agree with the same closed form.
    reliability = np.asarray(result.fit.stan_variable("reliability"), dtype=float)
    np.testing.assert_allclose(reliability, 1.0 - draws, rtol=1e-8, atol=1e-12)

    # A demand-based curve is evaluated at whole demands -- a point at 3.5
    # actuations would not mean anything -- and each one is (1 - p)^n.
    curve = reliability_over_time(result)
    assert curve is not None and curve["kind"] == "demands"
    grid = np.asarray([point["x"] for point in curve["points"]], dtype=float)
    assert np.all(grid == np.round(grid))
    expected = np.power(1.0 - draws[:, None], grid[None, :]).mean(axis=0)
    np.testing.assert_allclose(
        [point["mean"] for point in curve["points"]], expected, rtol=1e-8, atol=1e-10
    )

    summary = summarize(result)
    assert not summary.dropped_columns


@pytest.mark.slow
def test_summary_interval_is_the_equal_tailed_one():
    """The table's q05/q95 are quantiles, as the curve's band is, not an HDI.

    A rare-failure binomial makes the difference large: for Beta(3.5, 9997.5)
    the 90% HDI starts near 5e-5, the 5% quantile near 1.1e-4.
    """
    rows = [
        {"n": 2000, "failures": 1},
        {"n": 3500, "failures": 1},
        {"n": 1500, "failures": 0},
        {"n": 3000, "failures": 1},
    ]
    spec = ModelSpec(
        likelihood="binomial",
        mission_demands=1,
        params={
            "prob": ParamSpec(
                prior=PriorSpec(family="beta", hyper={"a": 0.5, "b": 0.5})
            )
        },
    )
    exact = stats.beta(0.5 + 3, 0.5 + 9997)
    summary = summarize(run(spec, rows, SAMPLER))
    row = summary.table.loc["prob"]
    for column, p in (("q05", 0.05), ("median", 0.5), ("q95", 0.95)):
        assert row[column] == pytest.approx(exact.ppf(p), rel=0.08), column


@pytest.mark.slow
def test_exponential_matches_exact_gamma_posterior():
    a, b = 3.0, 500.0
    rows = [
        {"device": "A", "time": 120.0, "failure": 1, "relevance": 1.0},
        {"device": "B", "time": 300.0, "failure": 0, "relevance": 1.0},
        {"device": "C", "time": 45.0, "failure": 1, "relevance": 0.8},
        {"device": "D", "time": 900.0, "failure": 0, "relevance": 0.5},
        {"device": "E", "time": 210.0, "failure": 1, "relevance": 1.0},
    ]
    spec = ModelSpec(
        likelihood="exponential",
        mission_times=[100.0],
        params={
            "rate": ParamSpec(
                lower=0.0,
                prior=PriorSpec(family="gamma", hyper={"shape": a, "rate": b}),
            )
        },
    )

    W = sum(r["relevance"] * r["failure"] for r in rows)
    S = sum(r["relevance"] * r["time"] for r in rows)
    exact = stats.gamma(a + W, scale=1.0 / (b + S))

    result = run(spec, rows, SAMPLER)
    draws = np.asarray(result.fit.stan_variable("rate"), dtype=float)
    _check_against(draws, exact, "exponential / gamma")

    reliability = np.asarray(
        result.fit.stan_variable("reliability"), dtype=float
    ).ravel()
    np.testing.assert_allclose(
        reliability, np.exp(-draws * 100.0), rtol=1e-8, atol=1e-12
    )

    # The curve is generated by the same Stan function as the number above, so
    # this is what proves that function is the survival function it claims to
    # be, at every point of the grid rather than only at the mission.
    curve = reliability_over_time(result)
    assert curve is not None and curve["kind"] == "time"
    grid = np.asarray([point["x"] for point in curve["points"]], dtype=float)
    assert grid[-1] == pytest.approx(130.0)  # 1.3 x the mission time
    expected = np.exp(-draws[:, None] * grid[None, :]).mean(axis=0)
    np.testing.assert_allclose(
        [point["mean"] for point in curve["points"]], expected, rtol=1e-8, atol=1e-10
    )

    # And the summary names that mission rather than a vector index.
    assert "reliability(t=100 h)" in summarize(result).table.index


@pytest.mark.slow
def test_prior_only_recovers_the_prior():
    """With the likelihood switched off, the posterior must be the prior.

    This is the check that catches a wrong truncation term: the normalisation
    is a constant and so cannot show up here, but a *misplaced* truncation
    changes the shape and would.
    """
    a, b = 2.0, 8.0
    spec = ModelSpec(
        likelihood="bernoulli_geometric",
        mission_demands=1,
        prior_only=True,
        params={
            "prob": ParamSpec(prior=PriorSpec(family="beta", hyper={"a": a, "b": b}))
        },
    )
    rows = [{"device": "A", "n": 50, "failure": 1, "relevance": 1.0}]
    result = run(spec, rows, SAMPLER)
    draws = np.asarray(result.fit.stan_variable("prob"), dtype=float)
    _check_against(draws, stats.beta(a, b), "prior-only / beta")


@pytest.mark.slow
def test_truncated_prior_is_the_truncated_distribution():
    """Bounds must produce the *truncated* prior, not the unbounded one."""
    lo, hi = 0.2, 0.6
    a, b = 2.0, 3.0
    spec = ModelSpec(
        likelihood="bernoulli_geometric",
        mission_demands=1,
        prior_only=True,
        params={
            "prob": ParamSpec(
                lower=lo,
                upper=hi,
                prior=PriorSpec(family="beta", hyper={"a": a, "b": b}),
            )
        },
    )
    result = run(spec, [], SAMPLER)
    draws = np.asarray(result.fit.stan_variable("prob"), dtype=float)

    base = stats.beta(a, b)
    lo_p, hi_p = base.cdf(lo), base.cdf(hi)

    class Truncated:
        def mean(self):
            xs = np.linspace(lo, hi, 20001)
            w = base.pdf(xs) / (hi_p - lo_p)
            return np.trapezoid(xs * w, xs)

        def std(self):
            xs = np.linspace(lo, hi, 20001)
            w = base.pdf(xs) / (hi_p - lo_p)
            m = np.trapezoid(xs * w, xs)
            return np.sqrt(np.trapezoid((xs - m) ** 2 * w, xs))

        def ppf(self, p):
            return base.ppf(lo_p + p * (hi_p - lo_p))

    assert draws.min() >= lo - 1e-9 and draws.max() <= hi + 1e-9
    _check_against(draws, Truncated(), "truncated beta prior")
