/**
 * Types mirroring the backend's payloads.
 *
 * Note what is *not* here: any list of prior families or likelihoods.  Those
 * arrive from `GET /api/meta` at startup, so adding a prior on the backend
 * needs no change in this codebase.  Hardcoding them here would recreate
 * exactly the duplication the served registry is meant to avoid.
 */

/** `null` stands for an infinite bound: JSON has no infinity. */
export type Bound = number | null

export interface HyperMeta {
  name: string
  label: string
  default: number
  minimum: Bound
  maximum: Bound
  exclusive_minimum: boolean
}

export interface PriorMeta {
  id: string
  label: string
  notes: string
  is_proper: boolean
  support_lower: Bound
  support_upper: Bound
  /** Empty means "any parameter whose support it can cover". */
  roles: string[]
  hyperparameters: HyperMeta[]
}

export interface ParamMeta {
  name: string
  role: string
  label: string
  description: string
  natural_lower: Bound
  natural_upper: Bound
  presets: { label: string; lower: Bound; upper: Bound }[]
}

export interface LikelihoodMeta {
  id: string
  label: string
  description: string
  mission: 'time' | 'demands'
  required_columns: string[]
  optional_columns: string[]
  parameters: ParamMeta[]
}

export interface Meta {
  likelihoods: LikelihoodMeta[]
  priors: PriorMeta[]
  sampler_help: Record<string, string>
  columns: { all: string[]; optional: string[]; reserved: string[] }
}

export interface PriorSpec {
  family: string
  hyper: Record<string, number>
}

export interface ParamSpec {
  lower: number | null
  upper: number | null
  prior: PriorSpec
}

export interface ModelSpec {
  schema_version?: number
  likelihood: string
  params: Record<string, ParamSpec>
  mission_times?: number[]
  mission_demands?: number | null
  /** Where the reliability curve stops; null means "derive it from the mission". */
  curve_horizon?: number | null
  time_unit?: string
  prior_only?: boolean
  emit_log_lik?: boolean
}

export interface SamplerConfig {
  chains: number
  iter_warmup: number
  iter_sampling: number
  seed: number | null
  adapt_delta: number
  max_treedepth: number
}

export interface PriorPreview {
  x: number[]
  pdf: number[]
  /** Share of the chosen prior surviving the bounds; null when meaningless. */
  retained_mass: number | null
  median: number | null
  mean: number | null
  q05: number | null
  q95: number | null
  warning: string | null
}

export interface ResolvedParam {
  name: string
  role: string
  family: string
  family_label: string
  effective_lower: Bound
  effective_upper: Bound
  truncated_below: boolean
  truncated_above: boolean
  declaration: string
  preview: PriorPreview
}

export interface SpecValidation {
  ok: boolean
  errors: string[]
  warnings: string[]
  parameters: ResolvedParam[]
  stan_source: string | null
}

export interface CellError {
  row: number | null
  column: string | null
  message: string
}

export interface DataValidation {
  ok: boolean
  errors: CellError[]
  warnings?: string[]
  rows?: Row[]
  ignored_columns?: string[]
}

export type Row = Record<string, string | number | null>

export interface Diagnostic {
  code: string
  severity: 'ok' | 'note' | 'warning' | 'error'
  message: string
  advice: string
}

export interface SummaryRow {
  parameter: string
  mean: number
  /** Reported beside the mean: for skewed quantities it is the better summary. */
  median: number
  sd: number
  q05: number
  q95: number
  ess_bulk: number
  ess_tail: number
  r_hat: number
  [key: string]: string | number
}

export interface CurvePoint {
  /** Time, or demands, depending on the curve's `kind`. */
  x: number
  mean: number
  q05: number
  median: number
  q95: number
}

/**
 * The dense reliability curve.
 *
 * It carries its own mission and unit rather than reading them off the current
 * spec: the model on screen may have been edited since the run that produced
 * these numbers.
 */
export interface ReliabilityOverTime {
  kind: 'time' | 'demands'
  unit: string
  mission: number[]
  points: CurvePoint[]
}

export interface RunSummary {
  table: SummaryRow[]
  diagnostics: Diagnostic[]
  dropped_columns: string[]
  elapsed: Record<string, number>
  /** Reliability at the mission times the user asked about, exactly. */
  reliability_curve?: {
    mission_time: number
    mean: number
    q05: number
    median: number
    q95: number
  }[]
  reliability_over_time?: ReliabilityOverTime
  /** What went into the fit, stated beside the answer. */
  data?: DataSummary
  /** The data's own Kaplan-Meier curve, for time-based models. */
  empirical?: {
    steps: { t: number; s: number }[]
    censored: { t: number; s: number }[]
  }
}

export interface DataSummary {
  rows: number
  effective_n: number
  weighted: boolean
  prior_only: boolean
  failures: number
  censored?: number
  /** Total time on test, for time-based models. */
  exposure?: number
  /** Total demands or trials, for demand-based models. */
  trials?: number
}

/** One parameter's posterior density beside the prior that was in force. */
export interface PriorPosterior {
  x: number[]
  posterior: number[]
  /** Truncated, renormalised prior on the same grid; null where infinite. */
  prior: (number | null)[] | null
  prior_label: string
}

export type RunStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

export interface RunState {
  id: string
  status: RunStatus
  stage: string
  message: string
  fraction: number | null
  error: string | null
  error_details: CellError[]
  warnings: string[]
  created_utc: string
  summary?: RunSummary
  elapsed?: Record<string, number>
  seed?: number
}

export interface Posterior {
  parameters: string[]
  /** Derived quantities (mttf, b10, reliability at the mission, …). */
  derived?: string[]
  /** Display names for columns Stan could only index, e.g. `reliability[1]`. */
  labels?: Record<string, string>
  columns: Record<string, number[]>
  chain: number[]
  draw: number[]
  total_draws: number
  thinned_by: number
  prior_posterior?: Record<string, PriorPosterior>
}

/** A ready-made analysis, as listed by `GET /api/examples`. */
export interface ExampleSummary {
  id: string
  title: string
  title_pt: string
  note: string
  note_pt: string
  source: string
  /** "public": a published data set; "illustrative": numbers made up for teaching. */
  data_kind: 'public' | 'illustrative'
  likelihood: string
  n_rows: number
}

export interface Example extends ExampleSummary {
  spec: ModelSpec
  rows: Row[]
  sampler: SamplerConfig
}
