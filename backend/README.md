# pirão

**P**robabilistic **I**nference for **R**eliability **A**nalysis from **O**bserved
data: Bayesian reliability estimation with [Stan](https://mc-stan.org), with a local
graphical interface. A sibling of [farofa](https://pypi.org/project/farofa/)
(simulation of repairable systems) and [faultree](https://pypi.org/project/faultree/)
(exact fault-tree analysis).

Pick a **likelihood** and, separately, a **prior for each parameter**; supply bounds
and data. pirão writes the Stan program, compiles and caches it, samples it, and
returns reliability at your mission times with 90% intervals, the reliability curve
against the data's own Kaplan–Meier steps, and each parameter's posterior over the
prior that was actually in force.

```bash
pip install pirao
pirao install-stan     # once: compiles CmdStan (a few minutes; needs a C++ compiler)
pirao gui              # opens the interface at http://localhost:8765
```

## What it covers

- **Likelihoods:** exponential, Weibull, lognormal, gamma (time to failure, with right
  censoring); per-demand Bernoulli and binomial (demands).
- **Priors:** lognormal, normal, Student t, gamma, inverse gamma, exponential, Weibull,
  beta, half-normal, half-Cauchy, uniform, flat and Jeffreys, with any bounds. The
  interface shows the prior *after* the bounds and warns when they discard much of it.
- **Relevance weights** per row, for data from similar but not identical equipment
  (a fractional pseudo-posterior, labelled as such).
- **Ready-made examples** on public data (Proschan 1963 air conditioning; Lieblein and
  Zelen 1956 ball bearings) and on clearly illustrative numbers.
- **Interface in Portuguese and English.**
- **Reproducible runs:** a `.zip` with the specification, data, seed, Stan program and
  toolchain versions.

## From the shell or Python

```bash
pirao examples                               # list the examples
pirao example relay-binomial --out relay     # spec.json, data.csv, sampler.json
pirao run relay/spec.json --data relay/data.csv --sampler relay/sampler.json --out relay/run
```

```python
from pirao.core.examples import get_example
from pirao.core.results import summarize
from pirao.core.sample import run
from pirao.core.spec import ModelSpec, ParamSpec, PriorSpec

# 23 ball bearings, millions of revolutions to fatigue (Lieblein and Zelen 1956);
# rows are {"time": ..., "failure": 1 or 0}, one per unit.
rows = get_example("bearings-weibull")["rows"]

spec = ModelSpec(
    likelihood="weibull",
    mission_times=[25, 50],
    time_unit="Mrev",
    params={
        "shape": ParamSpec(lower=0.0, prior=PriorSpec(family="half_normal", hyper={"sigma": 2})),
        "scale": ParamSpec(lower=0.0, prior=PriorSpec(family="gamma", hyper={"shape": 2, "rate": 0.02})),
    },
)
print(summarize(run(spec, rows)).table[["mean", "q05", "q95"]])
```

## More

The [repository](https://github.com/jmateusms/pirao) has the full README (data
format, what the results mean, statistical caveats), the
[design notes](https://github.com/jmateusms/pirao/blob/main/docs/design.md), and a
script that carries the posterior draws into farofa and faultree.

BSD 3-Clause licence. CmdStan (BSD 3-Clause) is downloaded by `pirao install-stan`,
not redistributed.
