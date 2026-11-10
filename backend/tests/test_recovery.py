"""Parameter recovery on simulated data, where the truth is known."""

from __future__ import annotations

import numpy as np
import pytest

from pirao.core.results import summarize
from pirao.core.sample import run
from pirao.core.spec import ModelSpec, ParamSpec, PriorSpec, SamplerConfig


@pytest.mark.slow
def test_weibull_recovers_shape_and_scale_from_censored_data():
    """The two true values differ by orders of magnitude, so a swap is obvious.

    Stan's weibull_lpdf takes (shape, scale); the reverse compiles and samples,
    and only the estimates are wrong.
    """
    shape, scale, censor_at = 2.5, 1000.0, 900.0
    rng = np.random.default_rng(20261110)
    lifetimes = scale * rng.weibull(shape, size=100)
    rows = [
        {
            "time": float(min(t, censor_at)),
            "failure": int(t <= censor_at),
            "relevance": 1.0,
        }
        for t in lifetimes
    ]
    assert 0 < sum(r["failure"] for r in rows) < len(rows)

    spec = ModelSpec(
        likelihood="weibull",
        mission_times=[500.0],
        params={
            "shape": ParamSpec(
                lower=0.0,
                prior=PriorSpec(family="half_normal", hyper={"sigma": 5.0}),
            ),
            "scale": ParamSpec(
                lower=0.0,
                prior=PriorSpec(family="gamma", hyper={"shape": 2.0, "rate": 0.001}),
            ),
        },
    )
    sampler = SamplerConfig(chains=4, iter_warmup=1000, iter_sampling=2000, seed=1)
    table = summarize(run(spec, rows, sampler)).table

    for name, truth in (("shape", shape), ("scale", scale)):
        row = table.loc[name]
        assert row["q05"] < truth < row["q95"], (
            f"{name}: truth {truth} outside the 90% interval "
            f"[{row['q05']:.4g}, {row['q95']:.4g}]"
        )

    # Not swapped: each median is nearer its own truth than the other's.
    med_shape, med_scale = table.loc["shape", "median"], table.loc["scale", "median"]
    assert abs(med_shape - shape) < abs(med_shape - scale)
    assert abs(med_scale - scale) < abs(med_scale - shape)
