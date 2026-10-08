# pirão

**P**robabilistic **I**nference for **R**eliability **A**nalysis from **O**bserved data —
a sibling of [farofa](https://github.com/jmateusms/farofa) and
[faultree](https://github.com/jmateusms/faultree). The package and the command are
`pirao`; until 2026-10-08 they were called `relimcmc`, which still works as a command.

Pick a **likelihood** and, independently, a **prior for each parameter**. Supply
hyperparameters, parameter bounds and data. The Stan program is assembled,
compiled, cached and sampled; results come back as a summary, downloadable
draws, and plots.

The point is the decoupling: the three models this grew out of each had one
hard-coded prior. Here the prior is a choice, and the tool takes responsibility
for telling you what that choice actually means once your bounds are applied.
[`docs/design.md`](docs/design.md) records why it is built this way, with the
evidence for each decision.

## Run it

```bash
docker compose up --build      # then open http://localhost:8080
```

The first build compiles CmdStan, which takes a few minutes. After that,
startup is instant.

### Without Docker

```bash
cd backend
pip install -e ".[api,export,dev]"
python scripts/install_cmdstan.py     # once; a few minutes
uvicorn pirao.api.main:app --port 8000

cd ../frontend
npm install && npm run dev            # http://localhost:5173
```

`GET /api/health` reports whether the Stan toolchain is usable, so a missing
compiler shows up as a message rather than a stack trace mid-run.

### One process, like `farofa gui` and `faultree gui`

```bash
cd frontend && npm install && npm run build && cd ..
pirao gui                     # http://localhost:8765, opens the browser
pirao gui --port 8811 --no-browser
```

The API and the built web app are served together on 127.0.0.1. Without
`--port`, 8765 is tried first and any free port is used if it is taken, the
same convention as the two sibling tools.

## Examples

The **Examples…** menu in the top bar (or `?example=<id>` in the URL, handy for
a link on a slide) loads a complete analysis: model, priors, data and sampler
settings with a fixed seed, plus a note on what to look for. Every data set is
either published or made up for teaching, so all of them can be shown in a
class.

| id | model | data |
|---|---|---|
| `aircondit-exponential` | exponential, Gamma prior (conjugate: exact answer in the note) | Boeing 720 air conditioning, Proschan (1963) |
| `aircondit-gamma` | gamma: is the hazard decreasing? | same |
| `bearings-weibull` | Weibull | ball-bearing fatigue, Lieblein and Zelen (1956) |
| `bearings-weibull-censored` | Weibull, 5 survivors censored at 100 Mrev | same, censored for teaching |
| `bearings-lognormal` | lognormal, to compare tails with the Weibull | same |
| `pumps-relevance` | exponential, generic lognormal prior, relevance 0.5 | illustrative |
| `relay-binomial` | binomial, Jeffreys prior (exact answer in the note) | illustrative, relay K2 of faultree's pressure tank |
| `switch-geometric` | per-demand Bernoulli | illustrative |
| `bounds-trap` | lognormal(2, 1) prior cut at 1: the warning, and what it costs | Proschan (1963) |

```bash
pirao examples                              # list them
pirao example relay-binomial --out relay    # spec.json, data.csv, sampler.json
pirao run relay/spec.json --data relay/data.csv --sampler relay/sampler.json --out relay/run
```

[`docs/integration/farofa_faultree_chain.py`](docs/integration/farofa_faultree_chain.py)
takes the draws of two of these runs further: relay K2's posterior into
faultree's pressure tank as uncertainty samples, and the pumps' failure rate
through farofa (unavailability with repair teams) into a 2-of-3 fault tree.

## What it covers

**Likelihoods** — exponential, Weibull, lognormal, gamma (time to failure);
per-demand Bernoulli and binomial (demand based).

**Priors** — lognormal, normal, Student t, gamma, inverse gamma, exponential,
Weibull, beta, half-normal, half-Cauchy, uniform, flat, and Jeffreys.

The two Bernoulli models are separate, user-visible choices rather than one
model with two data layouts, because they answer different questions:

- **per-demand** (`bernoulli_geometric`) — you know *when* the unit failed: it
  survived `n − 1` demands and failed on the `n`-th.
- **binomial** — you only know it failed *somewhere* within `n` trials.

Their posteriors genuinely differ, so which one you mean should not be inferred
from which column you happened to fill in.

## What you get back

The page opens on the answer: reliability at each mission time with its 90%
interval, mean life and B10, and one card per parameter with a plain reading
(for a Weibull shape, the probability of wear-out). Below come the reliability
curve with the data's own Kaplan–Meier steps drawn over it, and each
parameter's posterior over the prior that was in force, so you can see how far
the data moved you. The full table, sampler traces, marginals, the joint
posterior of each pair (a rotatable surface or a contour) and the fitted Stan
program sit in their own tabs.

The reliability curve is **R(t) across a range**, not a number at one instant:
the posterior mean, the median, and a 90% band at every point of a grid that
runs to 1.3 × the mission by default — set *Curve runs to* on the Model tab to
change it. The mission itself is marked in red and can be switched off from the
legend. Reliability is evaluated inside the Stan program by one function per
likelihood, the same one the mission-time figure uses, so the number in the
table and the curve drawn through it cannot disagree. Its grid travels as Stan
`data`, so moving the horizon resamples but never recompiles.

Reliability rows in the summary are named for the mission they answer for —
`reliability(t=500 h)`, not `reliability[0]`.

## Data format

One tidy table, one row per observation, for every model. `failure = 1` means
the unit failed; `0` means it survived and the row is right-censored.

```csv
device,time,failure,relevance
A1,120.5,1,1.0
A2,300.0,0,1.0
A3,45.2,1,0.8
```

| column | used by | notes |
|---|---|---|
| `device` | all | optional label; defaults to the row number |
| `time` | exponential, Weibull, lognormal, gamma | failure time, or censoring time when `failure = 0` |
| `n` | Bernoulli, binomial | demands or trials |
| `failure` | all but binomial | `1` = failed |
| `failures` | binomial | failure count in a group of `n` |
| `relevance` | all | weight in `[0, 1]`; optional, defaults to `1` |

Download a template for the selected model as CSV or Excel, fill it in, and
upload it — or paste a block straight from a spreadsheet into the grid.

`relevance` below 1 down-weights partly-relevant data. This makes the result a
**fractional (pseudo-)posterior** rather than an ordinary Bayesian one:
intervals widen, which is the intent, but they carry no calibration guarantee.
The tool labels results accordingly and reports the effective number of
observations.

## Using it from Python or the shell

The engine is a plain package; the web layer adds nothing statistical.

```bash
pirao list                                    # models and priors
pirao template weibull > data.csv
pirao run spec.json --data data.csv --out results/   # --sampler s.json, --seed
pirao render spec.json                        # just print the Stan program
```

```python
from pirao.core.results import summarize
from pirao.core.sample import run
from pirao.core.spec import ModelSpec, ParamSpec, PriorSpec

spec = ModelSpec(
    likelihood="weibull",
    mission_times=[100, 500, 1000],
    params={
        "shape": ParamSpec(lower=0.1, upper=10,
                           prior=PriorSpec(family="half_normal", hyper={"sigma": 2})),
        "scale": ParamSpec(lower=0.0,
                           prior=PriorSpec(family="gamma", hyper={"shape": 2, "rate": 0.002})),
    },
)
print(summarize(run(spec, rows)).table)
```

## Three things worth knowing

**Your bounds change your prior, and the tool says so.** A `lognormal(2, 1)`
prior has median 7.4 — but bounded to `[0, 1]` it keeps 2.3% of its mass and
has median 0.76. That is a different prior from the one you asked for. The
model panel plots the *truncated* density, states the interval actually in
force, and warns when the bounds discard more than 1% of the prior. This is not
hypothetical: it is what the original exponential model did silently.

**The wait is compilation, not sampling.** These models sample in
milliseconds; compiling one takes ten seconds or so. So bound and
hyperparameter *values* are Stan `data`, never source literals — changing a
number reuses the compiled binary, and only a structural change (a different
prior family, or a bound crossing the prior's support) triggers a rebuild. The
cache key includes the CmdStan and stanc versions, because a binary built by a
different toolchain is a different program.

**Improper priors are refused, not warned about.** A flat or Jeffreys prior
with only one finite bound can give an improper posterior — and Stan will not
tell you: R-hat stays at 1.00 with no divergences while the estimate runs off
to infinity. Both bounds are required.

## Reproducing a run

Every finished run offers a `.zip` holding the specification, the data, the
sampler settings **including the resolved seed**, the generated Stan program,
and the CmdStan/stanc versions that compiled it. Without the toolchain
versions, "reproducible" would not be true.

## Tests

```bash
cd backend && python -m pytest          # add -m "not slow" to skip sampling
cd frontend && npm run typecheck
```

Two of these models have posteriors with closed forms that survive the
relevance weighting — Bernoulli with a Beta prior gives `Beta(a + B, b + A)`,
exponential with a Gamma prior gives `Gamma(a + W, b + S)`. Those are used as
end-to-end oracles: they check the whole chain, from specification through code
generation, compilation and sampling, against arithmetic rather than against a
previous run.

## Citing

If you use pirão in academic work, please cite it: GitHub's **Cite this repository** button (from `CITATION.cff`) gives APA and BibTeX.

## Licence

BSD 3-Clause (see `LICENSE`). Stan and CmdStan are BSD-3-Clause and are not redistributed here;
`scripts/install_cmdstan.py` fetches CmdStan at install time.
