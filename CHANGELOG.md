# Changelog

## Unreleased

- The compiled-model cache has a size limit: past `PIRAO_CACHE_MAX_MB`
  (2048 by default, `0` for no limit) the least recently used binaries are
  removed. Binaries used in the last 15 minutes or being built are kept.
- Packaging: the wheel and the sdist now carry `LICENSE` and `NOTICE`.
- Docker: the image builds again (it lacked the README the package metadata
  reads) and runs as an unprivileged user (uid 10001). A `pirao-data` volume
  created by an older image is owned by root; hand it over once with
  `docker compose run --rm -u root backend chown -R 10001:10001 /data`.
- Packaging: the license is the SPDX expression `BSD-3-Clause`; the matching
  trove classifier is gone, as PEP 639 asks.

## 0.1.1 — 2026-10-08

- Security: the API no longer sends `Access-Control-Allow-Origin: *`. Every way
  of running pirão serves the app and the API from one origin, so nothing
  needed it, and it let any web page open in the same browser read your runs
  and download their bundles, data included.
- Docker: the app listens on 127.0.0.1:8080 only, not on every network
  interface, since it has no login.

## 0.1.0 — 2026-10-08

First release on PyPI (`pip install pirao`).

- Choose a likelihood (exponential, Weibull, lognormal, gamma, per-demand
  Bernoulli, binomial) and, separately, a prior for each parameter from
  thirteen families; the Stan program is generated, compiled, cached and sampled.
- Right censoring, relevance weights (fractional pseudo-posterior), prior-only runs.
- Results open on reliability at the mission times with 90% intervals, the
  reliability curve with the data's Kaplan–Meier steps, and each parameter's
  posterior over the prior in force; full table, sampler diagnostics and the
  fitted Stan program in their own tabs.
- `pirao gui`: the interface from one local process, as `farofa gui` and
  `faultree gui`, in Portuguese and English. `pirao install-stan` installs the
  pinned CmdStan.
- Nine ready-made examples on public or illustrative data; `pirao example`
  writes one out for the shell.
- Reproducible runs: a `.zip` with the specification, data, seed, Stan program
  and toolchain versions.
