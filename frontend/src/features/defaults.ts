/**
 * Sensible starting points, derived from the served registry.
 *
 * Nothing here hardcodes a list of priors or likelihoods; it picks from what
 * the backend reported.  The one piece of judgement is which family to open
 * with for each kind of parameter, which is a UX choice rather than a
 * statistical rule.
 */

import type { LikelihoodMeta, Meta, ModelSpec, ParamMeta, PriorMeta } from '../api/types'

/** Opening prior per parameter role, in order of preference. */
const PREFERRED: Record<string, string[]> = {
  prob: ['beta', 'uniform'],
  rate: ['gamma', 'exponential', 'lognormal'],
  scale: ['gamma', 'lognormal', 'exponential'],
  shape: ['half_normal', 'gamma', 'lognormal'],
  mu: ['normal', 'student_t'],
  sigma: ['half_normal', 'gamma'],
}

export function compatibleFamilies(
  priors: PriorMeta[],
  param: ParamMeta,
): PriorMeta[] {
  return priors.filter((prior) => {
    // A family that names roles applies only to those (a Beta is meaningful
    // only on a probability).
    if (prior.roles.length > 0) return prior.roles.includes(param.role)
    // A positive-support family cannot cover a parameter that ranges below 0.
    const paramCanBeNegative =
      param.natural_lower === null || param.natural_lower < 0
    const priorIsPositive = prior.support_lower !== null && prior.support_lower >= 0
    if (paramCanBeNegative && priorIsPositive) return false
    return true
  })
}

export function defaultFamily(priors: PriorMeta[], param: ParamMeta): PriorMeta {
  const usable = compatibleFamilies(priors, param)
  for (const id of PREFERRED[param.role] ?? []) {
    const found = usable.find((p) => p.id === id)
    if (found) return found
  }
  return usable[0] ?? priors[0]
}

export function defaultHyper(prior: PriorMeta): Record<string, number> {
  const out: Record<string, number> = {}
  for (const hyper of prior.hyperparameters) out[hyper.name] = hyper.default
  return out
}

export function defaultSpecFor(meta: Meta, likelihoodId: string): ModelSpec {
  const likelihood =
    meta.likelihoods.find((l) => l.id === likelihoodId) ?? meta.likelihoods[0]

  const params: ModelSpec['params'] = {}
  for (const param of likelihood.parameters) {
    const prior = defaultFamily(meta.priors, param)
    params[param.name] = {
      // Start at the natural support, per the project's decision to prefer
      // mathematically natural defaults over inherited ad hoc bounds.
      lower: param.natural_lower,
      upper: param.natural_upper,
      prior: { family: prior.id, hyper: defaultHyper(prior) },
    }
  }

  return {
    likelihood: likelihood.id,
    params,
    mission_times: likelihood.mission === 'time' ? [100] : [],
    mission_demands: likelihood.mission === 'demands' ? 1 : null,
    time_unit: 'h',
    prior_only: false,
  }
}

export function missionLabel(likelihood: LikelihoodMeta): string {
  return likelihood.mission === 'time' ? 'Mission time' : 'Mission demands'
}

/** Format a bound for display, where an absent bound means unbounded. */
export function formatBound(value: number | null, fallback: string): string {
  return value === null ? fallback : String(Number(value.toPrecision(6)))
}
