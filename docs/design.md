# Design notes

Why pirão is built the way it is. The README says what the tool does; this file
records the decisions behind it and the evidence for each, so they can be
revisited with that evidence rather than from memory. Stan behaviour cited here
was checked against the `stan-dev/docs` and `stanc3` sources and against live
CmdStan 2.36 runs.

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| **A1** | FastAPI backend, React (Vite) frontend | The only stack with an editable grid that pastes a block from Excel and highlights single cells, for free. AG Grid–based Python options (Dash, NiceGUI) put range paste behind the Enterprise licence; Panel's Tabulator and Gradio's dataframe have paste gaps. The grid is glide-data-grid (MIT), the engine Streamlit wraps, used directly for its paste and cell-renderer hooks. |
| **A2** | Six likelihoods: exponential, Weibull, lognormal, gamma (time to failure); per-demand Bernoulli and binomial (demands) | Lognormal and gamma also serve as priors. |
| **A3** | Bounds default to each parameter's natural support | A bound is an assertion about the data's units, not domain knowledge. `real<lower=0, upper=1> lambda` asserts a mean life of at least one time unit: 0.002 per hour is inside the box, the same rate written per year (17.5) is outside it. When the data sit outside a bound, the chain drifts to the edge with no divergences and R-hat near 1, and reliability comes out too high, the unsafe direction. Bounds stay available as an explicit choice. |

## 2. Generate Stan, do not switch inside it

The pipeline is: specification → resolve (bounds, priors) → Jinja template →
`.stan` source → content-hashed cache → compile → sample → summarise.

**Bound values and hyperparameter values are Stan `data`, never literals in the
source.** Only structure reaches the template: the likelihood, each parameter's
prior family, and which bounds are present (none, lower, upper, both). So:

- changing `gamma(2, 0.02)` to `gamma(5, 0.1)`, or a bound from 0.1 to 0.5,
  reuses the compiled binary, and the most common interaction never recompiles;
- `real<lower=x_lb, upper=x_ub> x;` with data-valued bounds is legal, and Stan
  applies the right Jacobian;
- user input never reaches the C++ compiler (see the invariants below).

Alternatives considered and rejected:

- **One program with an `int prior_id` switch.** Bound *structure* cannot be
  switched: `real<lower=L, upper=U>`, `real<lower=L>` and `real` are three
  transforms with three Jacobians, fixed at compile time, and
  `lower=negative_infinity()` does not help because the logit transform needs
  finite bounds. A padded `array[K] real h` of hyperparameters also loses
  per-hyperparameter names and checks, which is how a location and a scale get
  swapped unnoticed and a negative sigma ends in an opaque initialisation error.
- **Pre-generating every combination.** Six likelihoods by thirteen priors, with
  two-parameter models squaring that and bound patterns multiplying it again, for
  no benefit: values must be data either way.
- **A hand-written sampler, or PyMC.** Reliability lives in the censored tail,
  where Stan Math's `lccdf`, `log1m_exp` and `log_diff_exp` have long-tested
  asymptotic branches that a hand-written version tends to turn into `-inf` or
  NaN.

### The specification

`ModelSpec` is at once the API contract, the cache input and the saved format,
which is why it has carried a `schema_version` from the start. The data table is
deliberately not part of it, so that editing a cell can never invalidate a
compiled binary.

Parameters are named for their role (`shape`, `scale`, `rate`, `prob`, `mu`,
`sigma`), and each likelihood records the positional order Stan expects
(`stan_arg_order`). Stan's `weibull_lpdf(y | alpha, sigma)` is (shape, scale);
swapping the two still compiles, samples and reports R-hat 1.00, with every
reliability number wrong.

### Effective bounds are computed, not validated

```
lo = max(natural lower, prior support lower, user lower)
hi = min(natural upper, prior support upper, user upper)
reject if lo >= hi   (the message names all three sources)
```

Clamping makes "prior support narrower than the parameter range" impossible, so
the sampler never meets a region of zero prior density. The clamp is silent,
which is why the interface shows the interval actually in force.

### Priors are fully normalised `target +=` statements

```stan
target += d_lpdf(x | ...) - log_diff_exp(d_lcdf(ub | ...), d_lcdf(lb | ...));
```

rather than `x ~ d(...) T[lb, ub]`, for two measured reasons:

- `T[]` is only legal on a `~` statement, so it rules out `target +=` entirely;
- `~ ... T[]` drops the density's constants. At x = 1.2, μ = 1, σ = 3 on
  [0.5, 2.0], `~ normal T[,]` gives `lp__ = 0.638354` and the normalised form
  `-1.379200`; the difference, 2.017554, is exactly 0.5·log(2π) + log(3).

Comparing prior choices is the point of the tool, so `lp__` must be comparable
across families. Special cases:

- `flat` on [L, U] emits `target += -log(U - L);`
- `jeffreys_scale` emits `target += -log(x) - log(log(U) - log(L));`
- `uniform(a, b)` emits nothing: (a, b) are folded into the effective bounds.
  Under `~` it would contribute exactly zero while looking as if it did
  something.

**Improper families (`flat`, `jeffreys_scale`) require both bounds finite, as a
hard error.** Flat or Jeffreys on a Weibull scale with only a lower bound gave a
posterior mean of `inf` and a 95th percentile of 1.6e288, with R-hat 1.0 and no
divergences: the usual diagnostics do not see it. The check reads an explicit
`is_proper` flag on the family, not "emits no statement", because
`jeffreys_scale` does emit one.

### Data: one row per observation

| column | used by | notes |
|---|---|---|
| `device` | all | optional label, never sent to Stan |
| `time` | exponential, Weibull, lognormal, gamma | failure time, or censoring time when `failure = 0` |
| `n` | per-demand, binomial | demands or trials |
| `failure` | all but binomial | 1 = failed, everywhere |
| `failures` | binomial | count in a group of `n` |
| `relevance` | all | weight in [0, 1], default 1 |

Separate failure-time and censoring-time arrays buy nothing here: Stan's
vectorised `weibull_lpdf(vector y | ...)` returns a sum, and a per-row weight
needs a sum of weighted terms, so both layouts need a scalar loop. The single
table keeps the user's row order, matches the spreadsheet they uploaded, and
indexes `log_lik` by their rows.

Every likelihood loop is guarded with `if (relevance[i] > 0)`: `0 × (-inf)` is
NaN, which kills a chain, while `-inf` only rejects one proposal. Rules that
depend on the specification are applied when the data are checked: a failure at
time 0 has zero density when the shape is bounded at or above 1, and a failure
on demand 0 is always rejected.

The two Bernoulli models are separate, visible choices. The per-demand model
says a unit failed *on* demand n; the binomial says it failed *somewhere* in n
trials. Their posteriors differ, so which one is meant should not be inferred
from which column was filled in.

### The wait is compilation

Sampling these models takes milliseconds (a six-row fit took 11 ms); compiling a
new structure takes 9 to 40 seconds and reports no progress. So compilation is
its own stage in the progress stream, a run executes in a child process whose
process group can be killed (each chain is a CmdStan subprocess, and cmdstanpy
has no cooperative cancel), and the reliability curve's grid travels as data, so
moving the horizon resamples without recompiling.

The cache key is a hash of the source **and** the toolchain fingerprint (CmdStan
and cmdstanpy versions, compiler flags, generator version). Source alone would
silently reuse binaries built by an older compiler after an upgrade. A file lock
keeps two tabs from building the same structure into one corrupt executable.

### Invariants

1. **User values never reach the compiler.** Only structure enters the template.
   The day a hyperparameter is inlined "for convenience" this becomes code
   injection into a `g++` call.
2. **Priors attach only to declared parameters**, never to a transformed
   parameter, and no transformed parameter carries a bound its own definition
   can violate.
3. **Hyperparameters and bounds are data, never parameters.** This is what makes
   the truncation normaliser a constant. Hierarchical priors would break it, and
   every truncated prior would then have to be re-derived.

The first is tested (`test_hyperparameter_values_never_reach_the_source`); the
other two hold by construction of the templates and have no test of their own
yet.

## 3. Statistics the interface has to say out loud

### Bounds change the prior

A parameter declared on [0, 1] with a `lognormal(μ, σ)` prior does not have that
prior: it has the lognormal restricted to [0, 1]. The share of mass that
survives, P(λ ≤ 1):

| prior | mass kept |
|---|---|
| lognormal(−6, 1) | 1.000 |
| lognormal(−2, 1) | 0.977 |
| lognormal(0, 1) | 0.500 |
| lognormal(2, 1) | 0.023 |
| lognormal(5, 2) | 0.0062 |

Someone who elicits lognormal(0, 1) gets a prior with median about 0.53, not 1.
So every prior preview and summary in the interface is computed from the
truncated density over the effective bounds, and the interface warns when less
than 99% of the mass survives. The `bounds-trap` example shows it, and shows
that the effect survives twelve observed failures.

### Relevance weighting is a fractional likelihood

`target += w · lpdf` is a power (fractional, tempered) likelihood, a form of
general Bayesian updating (Bhattacharya, Pati and Yang, *Ann. Statist.* 2019,
arXiv:1611.01125; Grünwald and van Ommen's SafeBayes; Lyddon, Holmes and
Walker, arXiv:1709.07616). It is
coherent, with caveats the interface carries:

- it is not a Bayes update for any generative model, so its intervals have no
  automatic calibration: results are labelled a fractional (pseudo-)posterior;
- weights below 1 widen intervals, the conservative direction, which is why
  `relevance` is bounded to [0, 1] in every generated model;
- for the exponential it has an exact reading (a reduced experiment with
  W = Σw·failure failures and S = Σw·time exposure); for the Weibull it is a
  pseudo-likelihood and is described as such;
- weighting a density is not unit-invariant (`w · log f(t)` picks up
  `w · log(unit)`), which is why the time unit is part of the specification;
- Σ relevance is reported as the effective number of observations;
- as weights go to 0 the posterior goes to the prior, which is safe only
  because improper priors are refused.

### Two exact answers to test against

Conjugacy survives the weighting, which gives end-to-end oracles that check the
whole chain (specification, generation, compilation, data, sampling, summary)
against arithmetic:

- per-demand Bernoulli with a Beta(a, b) prior: posterior Beta(a + B, b + A),
  with A = Σ w(n − f) and B = Σ w·f;
- exponential with a Gamma(a, b) prior: posterior Gamma(a + W, b + S), with
  W = Σ w·failure and S = Σ w·time.

An argument-order swap, a mis-keyed data entry, a dropped weight or a wrong
truncation term fails them. The `aircondit-exponential` and `relay-binomial`
examples quote their exact answers for the same reason.

### Intervals are equal-tailed

The summary's 5% and 95% columns are quantiles, the same as the reliability
curve's band. They were briefly the 90% highest-density interval under the same
names, and for a skewed posterior (a failure probability near 10⁻⁴) the two
differ by a factor of two at the lower end.

## 4. Settled questions

| # | Question | Decision |
|---|---|---|
| D1 | Weibull shape prior on `shape` or on `shape − 1`? | On `shape`, with "assume wear-out" (shape ≥ 1) as a labelled preset. |
| D2 | Binomial as its own model, or chosen by which columns are filled? | Its own, visible model; the posteriors differ. |
| D3 | Weighted or unweighted `log_lik`? | Both: `log_lik` unweighted (standard LOO), `log_lik_weighted` to reconcile with `target`. |
| D4 | A row with relevance 0: drop it or keep it? | Keep and guard, so rows stay aligned with `log_lik` and with error messages. |
| D5 | Prior-only runs | A `prior_only` flag that keeps the observed design, so prior-predictive quantities stay meaningful. |
| D6 | Model comparison (LOO) in the interface? | Not yet: `log_lik` is emitted, but LOO on a weighted pseudo-likelihood is contested. |
| D7 | Repeated `device` labels | Allowed; rows are independent. |
| D8 | Licence | BSD 3-Clause, as farofa and faultree. CmdStan is fetched at install time, not redistributed. |
| D9 | Blank cells | An error on required columns; the default on optional ones. |

## 5. Risks

| | Risk | Mitigation |
|---|---|---|
| R1 | CmdStan needs a C++ toolchain, which a locked-down Windows laptop may not have. | Docker as the "one command" path; the native path checks the toolchain at start and says what is missing. |
| R2 | Silent wrong answers from argument order or data keys: both compile, sample and report R-hat 1.00. | `stan_arg_order` in the registry with a test, a contract test between the data block and the data dictionary, and the conjugate oracles. Not yet: golden source files and a parameter-recovery test on simulated Weibull data. |
| R3 | Improper posteriors that look converged. | The `is_proper` hard block and prior-only runs. Not yet: a warning when Σ relevance is small next to the number of parameters. |
| R4 | Stale compiled binaries after a toolchain upgrade. | The toolchain is part of the cache key. |
| R5 | Cache growth: each binary is 10 to 40 MB. | A size-capped cache (not yet implemented). |
| R6 | Users over-trusting a fractional pseudo-posterior. | Labelled as such in the interface and in exports. |

## 6. Repository layout

```
backend/src/pirao/
  core/        headless engine, no web imports
    spec.py  registry_likelihoods.py  registry_priors.py  resolve.py
    render.py  templates/  data.py  compile.py  sample.py  results.py
    preview.py  bundle.py  jobs.py  examples.py
  api/main.py  HTTP layer over core, no statistics of its own
  cli.py       the same engine from the shell
  gui.py       `pirao gui`: API and built frontend from one local process
  examples/    ready-made analyses on public or made-up data
frontend/src/  React app: api/, components/, features/, styles/
docs/          these notes; integration/ for farofa and faultree
```

The registries are served, not duplicated: `GET /api/meta` returns every
likelihood, prior family, hyperparameter and constraint, and the frontend builds
its forms from that payload. Adding a prior family is one entry in
`registry_priors.py` and no frontend change.
