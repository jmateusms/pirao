import { useEffect, useMemo, useState } from 'react'

import { api } from '../api/client'
import type {
  DataSummary,
  Diagnostic,
  ModelSpec,
  Posterior,
  PriorPosterior,
  ReliabilityOverTime,
  RunState,
  SpecValidation,
  SummaryRow,
} from '../api/types'
import { Plot } from '../components/Plot'
import { density2d, quantile, thin } from '../components/density'
import {
  BLUE,
  BLUE_WASH,
  CRITICAL,
  DENSITY_SCALE,
  DENSITY_SCALE_CLEAR,
  SCENE_AXIS,
  SERIES,
  VIOLET,
  fmt,
} from '../components/viz'

interface Props {
  run: RunState
  /** The spec the run was fitted with, not whatever is on the left now. */
  spec: ModelSpec
  /** The prior previews as they stood when the run started. */
  validation: SpecValidation | null
}

const VIEWS = ['Overview', 'Table', 'Diagnostics', 'Stan program'] as const
type View = (typeof VIEWS)[number]

/**
 * What a run produced, most important first.
 *
 * The question a reliability study asks is "will it survive the mission, and
 * how sure are we?", so the page opens on reliability at the mission and the
 * reliability curve.  Next comes what the data changed -- each parameter's
 * posterior over the prior that was in force -- because in a Bayesian fit that
 * is what tells the reader whether the answer came from the data or from the
 * prior.  Numbers in full, sampler diagnostics and the Stan program follow.
 */
export function ResultsPanel({ run, spec, validation }: Props) {
  const [view, setView] = useState<View>('Overview')
  const [posterior, setPosterior] = useState<Posterior | null>(null)
  const [source, setSource] = useState<string | null>(null)

  useEffect(() => {
    setPosterior(null)
    api.posterior(run.id).then(setPosterior).catch(() => setPosterior(null))
  }, [run.id])

  useEffect(() => {
    if (view === 'Stan program' && source === null) {
      api.source(run.id).then(setSource).catch(() => setSource(''))
    }
  }, [view, source, run.id])

  const summary = run.summary!
  const unit = summary.reliability_over_time?.unit ?? spec.time_unit ?? ''
  const kind = summary.reliability_over_time?.kind ?? (spec.mission_times?.length ? 'time' : 'demands')
  const groups = useMemo(() => groupRows(summary.table, Object.keys(spec.params)), [summary.table, spec.params])

  return (
    <div className="stack results">
      <section className="results__head">
        <div>
          <h2>Results</h2>
          <p className="muted small">
            Run {run.id} · seed {run.seed} · {fmtElapsed(summary.elapsed)}
            {summary.data && <> · {describeData(summary.data, kind, unit)}</>}
          </p>
        </div>
        <div className="row">
          <a className="chip" href={api.drawsUrl(run.id, 'csv')} download>
            Draws (CSV)
          </a>
          <a className="chip" href={api.drawsUrl(run.id, 'parquet')} download>
            Parquet
          </a>
          <a
            className="chip"
            href={api.bundleUrl(run.id)}
            download
            title="Specification, data, seed, the Stan program and the toolchain versions: everything needed to reproduce the run"
          >
            Full analysis (.zip)
          </a>
        </div>
      </section>

      <SamplerStatus diagnostics={summary.diagnostics} rows={groups.parameters} onMore={() => setView('Diagnostics')} />

      {(summary.data?.weighted || summary.data?.prior_only) && (
        <div className="banner banner--note small">
          {summary.data.prior_only
            ? 'Prior only: the data were ignored, so these numbers show what the prior alone implies.'
            : `Relevance weighting: ${summary.data.rows} rows count as ${fmt(summary.data.effective_n)} observations. This is a fractional (pseudo-)posterior, so the intervals are deliberately wider than a standard Bayesian analysis would give.`}
        </div>
      )}

      <Headline groups={groups} posterior={posterior} spec={spec} kind={kind} unit={unit} />

      <nav className="tabs tabs--sub" role="tablist">
        {VIEWS.map((name) => (
          <button
            key={name}
            role="tab"
            aria-selected={view === name}
            className={`tab${view === name ? ' tab--active' : ''}`}
            onClick={() => setView(name)}
          >
            {name}
          </button>
        ))}
      </nav>

      {view === 'Overview' && (
        <>
          <ReliabilityView
            curve={summary.reliability_over_time}
            marks={groups.reliability}
            empirical={summary.empirical}
          />
          <PriorPosteriorGrid posterior={posterior} validation={validation} rows={groups.parameters} />
        </>
      )}

      {view === 'Table' && (
        <SummaryTable groups={groups} unit={kind === 'time' ? unit : ''} dropped={summary.dropped_columns} />
      )}

      {view === 'Diagnostics' && <DiagnosticsView diagnostics={summary.diagnostics} posterior={posterior} />}

      {view === 'Stan program' && (
        <section className="card">
          <h3>The Stan program that was fitted</h3>
          {source ? <pre className="source">{source}</pre> : <p className="muted">Loading…</p>}
        </section>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// grouping and formatting

interface Groups {
  reliability: SummaryRow[]
  parameters: SummaryRow[]
  life: SummaryRow[]
  other: SummaryRow[]
}

const LIFE = ['mttf', 'mean_demands_to_failure', 'b10']

function groupRows(table: SummaryRow[], parameters: string[]): Groups {
  const groups: Groups = { reliability: [], parameters: [], life: [], other: [] }
  for (const row of table) {
    if (row.parameter.startsWith('reliability')) groups.reliability.push(row)
    else if (parameters.includes(row.parameter)) groups.parameters.push(row)
    else if (LIFE.includes(row.parameter)) groups.life.push(row)
    else groups.other.push(row)
  }
  groups.life.sort((a, b) => LIFE.indexOf(a.parameter) - LIFE.indexOf(b.parameter))
  return groups
}

/** Three significant figures: enough for a headline, few enough to read. */
function sig(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(digits - 1)
  return String(Number(value.toPrecision(digits)))
}

function interval(row: SummaryRow): string {
  return `${sig(row.q05)} – ${sig(row.q95)}`
}

/** `reliability(t=24 h)` -> `24 h`; `reliability(n=100)` -> `100 demands`. */
function missionOf(row: SummaryRow): { label: string; at: number | null } {
  const match = /\((t|n)=([^)]*)\)/.exec(row.parameter)
  if (!match) return { label: 'the mission', at: null }
  const at = Number.parseFloat(match[2])
  const label = match[1] === 'n' ? `${match[2]} demands` : match[2]
  return { label, at: Number.isFinite(at) ? at : null }
}

const LIFE_LABELS: Record<string, { title: string; hint: string }> = {
  mttf: { title: 'Mean time to failure', hint: 'MTTF' },
  b10: { title: 'B10 life', hint: 'age by which 10% have failed' },
  mean_demands_to_failure: { title: 'Mean demands to failure', hint: '1 / prob' },
}

function describeData(data: DataSummary, kind: string, unit: string): string {
  if (data.trials !== undefined && data.censored === undefined) {
    return `${data.failures} failures in ${data.trials.toLocaleString('en')} trials`
  }
  const parts = [`${data.failures} failures`]
  if (data.censored) parts.push(`${data.censored} censored`)
  if (kind === 'time' && data.exposure !== undefined) {
    parts.push(`${sig(data.exposure, 4)} ${unit} on test`)
  } else if (data.trials !== undefined) {
    parts.push(`${data.trials.toLocaleString('en')} demands`)
  }
  return parts.join(', ')
}

// ---------------------------------------------------------------------------
// sampler status: one line when all is well, the details a click away

function SamplerStatus({
  diagnostics,
  rows,
  onMore,
}: {
  diagnostics: Diagnostic[]
  rows: SummaryRow[]
  onMore: () => void
}) {
  const problems = diagnostics.filter((d) => d.severity === 'warning' || d.severity === 'error')
  const notes = diagnostics.filter((d) => d.severity === 'note')
  const rhat = Math.max(...rows.map((r) => r.r_hat).filter(Number.isFinite))
  const ess = Math.min(...rows.map((r) => r.ess_bulk).filter(Number.isFinite))
  const numbers =
    rows.length > 0 ? `max R-hat ${rhat.toFixed(3)} · min ESS ${Math.round(ess)}` : ''

  if (problems.length > 0) {
    return (
      <div className="status status--warn">
        <strong>Check the sampler before trusting these numbers.</strong>
        <ul>
          {problems.map((d, i) => (
            <li key={i}>
              {d.message} {d.advice && <span className="muted">{d.advice}</span>}
            </li>
          ))}
        </ul>
        <span className="muted small">{numbers}</span>
      </div>
    )
  }
  return (
    <div className="status status--ok">
      <span className="status__mark" aria-hidden="true">✓</span>
      <span>
        <strong>The sampler converged.</strong>{' '}
        <span className="muted">{numbers}</span>
        {notes.length > 0 && (
          <>
            {' '}
            ·{' '}
            <button className="link" onClick={onMore}>
              {notes.length === 1 ? '1 note' : `${notes.length} notes`}
            </button>
          </>
        )}
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// headline cards

function Headline({
  groups,
  posterior,
  spec,
  kind,
  unit,
}: {
  groups: Groups
  posterior: Posterior | null
  spec: ModelSpec
  kind: string
  unit: string
}) {
  const shown = groups.reliability.slice(0, 3)
  const lifeUnit = kind === 'time' ? ` ${unit}` : ''

  return (
    <div className="kpis">
      {shown.map((row, index) => {
        const mission = missionOf(row)
        return (
          <div key={row.parameter} className={`kpi${index === 0 ? ' kpi--lead' : ''}`}>
            <div className="kpi__label">Reliability at {mission.label}</div>
            <div className="kpi__value">{sig(row.mean)}</div>
            <div className="kpi__sub">90% interval {interval(row)}</div>
          </div>
        )
      })}
      {groups.life.map((row) => {
          const label = LIFE_LABELS[row.parameter]
          const isDemands = row.parameter === 'mean_demands_to_failure'
          return (
            <div key={row.parameter} className="kpi">
              <div className="kpi__label" title={label?.hint}>
                {label?.title ?? row.parameter}
              </div>
              <div className="kpi__value">
                {sig(row.median)}
                <span className="kpi__unit">{isDemands ? ' demands' : lifeUnit}</span>
              </div>
              <div className="kpi__sub">median · 90% {interval(row)}</div>
            </div>
          )
        })}
      {groups.parameters.map((row) => (
        <div key={row.parameter} className="kpi kpi--param">
          <div className="kpi__label">{row.parameter}</div>
          <div className="kpi__value">{sig(row.median)}</div>
          <div className="kpi__sub">median · 90% {interval(row)}</div>
          <ParamReading name={row.parameter} row={row} posterior={posterior} spec={spec} />
        </div>
      ))}
    </div>
  )
}

/** One plain-language reading of a parameter, where there is a standard one. */
function ParamReading({
  name,
  row,
  posterior,
  spec,
}: {
  name: string
  row: SummaryRow
  posterior: Posterior | null
  spec: ModelSpec
}) {
  if (name === 'shape' && (spec.likelihood === 'weibull' || spec.likelihood === 'gamma')) {
    const draws = posterior?.columns[name]
    if (!draws || draws.length === 0) return null
    const above = draws.filter((v) => v > 1).length / draws.length
    const p = above > 0.99 ? '> 0.99' : above < 0.01 ? '< 0.01' : `= ${sig(above, 2)}`
    const reading =
      above > 0.95
        ? 'wear-out: failures become more frequent with age'
        : above < 0.05
          ? 'early failures: the hazard falls with age'
          : 'the data cannot tell whether the hazard rises or falls'
    return (
      <div className="kpi__note">
        P(shape &gt; 1) {p}: {reading}
      </div>
    )
  }
  if (name === 'prob' && row.median > 0) {
    return <div className="kpi__note">about 1 failure in {sig(1 / row.median, 2)} demands</div>
  }
  if (name === 'rate' && spec.likelihood === 'exponential' && row.median > 0) {
    return <div className="kpi__note">one failure every {sig(1 / row.median)} {spec.time_unit}, typically</div>
  }
  return null
}

// ---------------------------------------------------------------------------
// prior versus posterior

function PriorPosteriorGrid({
  posterior,
  validation,
  rows,
}: {
  posterior: Posterior | null
  validation: SpecValidation | null
  rows: SummaryRow[]
}) {
  const curves = posterior?.prior_posterior
  if (!posterior) return <section className="card"><p className="muted">Loading the draws…</p></section>
  if (!curves || Object.keys(curves).length === 0) return null

  return (
    <section className="stack">
      <div className="section-title">
        <h3>What the data changed</h3>
        <p className="muted small">
          Each parameter&apos;s posterior (filled) over the prior that was in force
          (dashed), which is the prior after your bounds. A posterior much
          narrower than the prior means the data decided; one that copies the
          prior means they could not.
        </p>
      </div>
      <div className="grid2">
        {rows.map((row) =>
          curves[row.parameter] ? (
            <PriorPosteriorCard
              key={row.parameter}
              name={row.parameter}
              curve={curves[row.parameter]}
              row={row}
              prior={validation?.parameters.find((p) => p.name === row.parameter)?.preview ?? null}
            />
          ) : null,
        )}
      </div>
    </section>
  )
}

function PriorPosteriorCard({
  name,
  curve,
  row,
  prior,
}: {
  name: string
  curve: PriorPosterior
  row: SummaryRow
  prior: { median: number | null; q05: number | null; q95: number | null } | null
}) {
  const { data, scale } = useMemo(() => {
    const top = Math.max(...curve.posterior)
    const priorValues = curve.prior ?? []
    const finitePrior = priorValues.filter((v): v is number => v !== null && Number.isFinite(v))
    const priorTop = finitePrior.length ? Math.max(...finitePrior) : 0
    // A prior that is tiny across the posterior's window draws as a flat zero
    // line, which hides its shape -- and the shape (rising toward a bound, say)
    // is what explains a pull.  Rescale it then, and say by how much.
    const scale = priorTop > 0 && priorTop < 0.02 * top ? (0.5 * top) / priorTop : 1
    const traces: unknown[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: curve.x,
        y: curve.posterior,
        fill: 'tozeroy',
        fillcolor: BLUE_WASH,
        line: { color: BLUE, width: 2 },
        name: 'posterior',
        hovertemplate: '%{x:.4g}<br>posterior %{y:.3g}<extra></extra>',
      },
    ]
    if (curve.prior) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: curve.x,
        y: priorValues.map((v) => (v === null ? null : v * scale)),
        line: { color: '#5e5e5a', width: 1.8, dash: 'dash' },
        name: scale === 1 ? `prior (${curve.prior_label})` : `prior × ${sig(scale, 2)}`,
        hovertemplate: '%{x:.4g}<extra>prior</extra>',
        connectgaps: false,
      })
    }
    return { data: traces, scale }
  }, [curve])

  const layout = useMemo(
    () => ({
      margin: { l: 44, r: 12, t: 8, b: 40 },
      xaxis: { title: name },
      yaxis: {
        title: 'density',
        range: [0, 1.12 * Math.max(...curve.posterior)],
        showticklabels: false,
      },
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.12 },
      hovermode: 'x',
    }),
    [name, curve],
  )

  const narrowed =
    prior?.q05 != null && prior?.q95 != null && row.q95 > row.q05
      ? (prior.q95 - prior.q05) / (row.q95 - row.q05)
      : null

  return (
    <section className="card">
      <h3>{name}</h3>
      <Plot data={data} layout={layout} height={230} description={`Prior and posterior of ${name}`} />
      <p className="muted small">
        {prior?.median != null && (
          <>
            prior median {sig(prior.median)} (90% {sig(prior.q05)} – {sig(prior.q95)}) →{' '}
          </>
        )}
        posterior median {sig(row.median)} (90% {interval(row)})
        {narrowed !== null && narrowed > 1.05 && <> · interval {sig(narrowed, 2)}× narrower</>}
        {scale !== 1 && <> · prior drawn ×{sig(scale, 2)} to show its shape</>}
      </p>
    </section>
  )
}

// ---------------------------------------------------------------------------
// the numbers in full

function SummaryTable({
  groups,
  unit,
  dropped,
}: {
  groups: Groups
  unit: string
  dropped: string[]
}) {
  const sections: { title: string; rows: SummaryRow[]; label: (r: SummaryRow) => string }[] = [
    { title: 'Reliability', rows: groups.reliability, label: (r) => `R(${missionOf(r).label})` },
    { title: 'Parameters', rows: groups.parameters, label: (r) => r.parameter },
    {
      title: unit ? `Life (${unit})` : 'Life',
      rows: groups.life,
      label: (r) => LIFE_LABELS[r.parameter]?.title ?? r.parameter,
    },
    { title: 'Other', rows: groups.other, label: (r) => r.parameter },
  ]
  return (
    <section className="card">
      <table className="table table--summary">
        <thead>
          <tr>
            <th>quantity</th>
            <th>mean</th>
            <th>median</th>
            <th>90% interval</th>
            <th>sd</th>
            <th title="Potential scale reduction: should be below 1.01">R-hat</th>
            <th title="Effective sample size, bulk / tail: 400 or more is comfortable">ESS</th>
          </tr>
        </thead>
        {sections
          .filter((section) => section.rows.length > 0)
          .map((section) => (
            <tbody key={section.title}>
              <tr className="table__group">
                <th colSpan={7}>{section.title}</th>
              </tr>
              {section.rows.map((row) => (
                <tr key={row.parameter}>
                  <td>{section.label(row)}</td>
                  <td>{sig(row.mean, 4)}</td>
                  <td>{sig(row.median, 4)}</td>
                  <td className="nowrap">{sig(row.q05, 4)} – {sig(row.q95, 4)}</td>
                  <td>{sig(row.sd, 3)}</td>
                  <td className={row.r_hat > 1.01 ? 'flag' : 'muted'}>{row.r_hat.toFixed(3)}</td>
                  <td className={row.ess_bulk < 400 || row.ess_tail < 400 ? 'flag' : 'muted'}>
                    {Math.round(row.ess_bulk)} / {Math.round(row.ess_tail)}
                  </td>
                </tr>
              ))}
            </tbody>
          ))}
      </table>
      <p className="muted small">
        Intervals are equal-tailed: 5% of the posterior lies below, 5% above. For
        reliability the mean is the probability of surviving given everything
        observed; for skewed quantities such as life, read the median.
        {dropped.length > 0 && <> Not shown because every draw was infinite: {dropped.join(', ')}.</>}
      </p>
    </section>
  )
}

// ---------------------------------------------------------------------------
// diagnostics

type DiagnosticView = 'Traces' | 'Marginals' | 'Joint'

function DiagnosticsView({
  diagnostics,
  posterior,
}: {
  diagnostics: Diagnostic[]
  posterior: Posterior | null
}) {
  const [mode, setMode] = useState<DiagnosticView>('Traces')
  return (
    <div className="stack">
      {diagnostics.map((diagnostic, index) => (
        <div
          key={index}
          className={`banner banner--${
            diagnostic.severity === 'ok' ? 'ok' : diagnostic.severity === 'note' ? 'note' : 'warn'
          }`}
        >
          <strong>{diagnostic.message}</strong>
          {diagnostic.advice && <div className="muted">{diagnostic.advice}</div>}
        </div>
      ))}
      <div className="viz-toolbar">
        <span className="muted small">How the sampler explored the posterior.</span>
        <div className="segmented" role="group" aria-label="Diagnostic view">
          {(['Traces', 'Marginals', 'Joint'] as const).map((name) => (
            <button
              key={name}
              className={`segmented__option${mode === name ? ' segmented__option--on' : ''}`}
              aria-pressed={mode === name}
              onClick={() => setMode(name)}
            >
              {name}
            </button>
          ))}
        </div>
      </div>
      {!posterior && <p className="muted">Loading the draws…</p>}
      {posterior && mode === 'Traces' && <TracesView posterior={posterior} />}
      {posterior && mode === 'Marginals' && <DistributionsView posterior={posterior} />}
      {posterior && mode === 'Joint' && <JointView posterior={posterior} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// distributions

function DistributionsView({ posterior }: { posterior: Posterior }) {
  const columns = [...posterior.parameters, ...(posterior.derived ?? [])]

  return (
    <div className="grid2">
      {columns.map((name) => (
        <MarginalCard
          key={name}
          label={posterior.labels?.[name] ?? name}
          values={posterior.columns[name] ?? []}
        />
      ))}
    </div>
  )
}

const BINS = 48

/** Ratio of largest to smallest draw past which a linear axis is useless. */
const LOG_RATIO = 1e4

/** How far past the central 99% a tail must run before the axis clips it. */
const CLIP_RATIO = 20

function MarginalCard({ label, values }: { label: string; values: number[] }) {
  const stats = useMemo(() => {
    const finite = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
    const min = finite[0]
    const max = finite[finite.length - 1]
    const lo = quantile(finite, 0.005)
    const hi = quantile(finite, 0.995)

    // A derived quantity such as mean life routinely spans many orders of
    // magnitude, and drawn on a linear axis the whole posterior collapses into
    // one bar at the left edge.  A log axis shows the actual shape and drops
    // nothing.  Where the spread is merely long-tailed rather than
    // multiplicative, the axis is windowed instead and the caption says so --
    // silently cropping a tail would misrepresent it.
    const log = min > 0 && max / min > LOG_RATIO
    const clipped = !log && hi > lo && max - min > CLIP_RATIO * (hi - lo)

    return {
      median: quantile(finite, 0.5),
      q05: quantile(finite, 0.05),
      q95: quantile(finite, 0.95),
      min,
      lo,
      hi,
      max,
      log,
      clipped,
    }
  }, [values])

  // Plotly bins in the data's own units even when the axis is logarithmic, so
  // a log *axis* alone would still produce one bar and 47 empty decades.  The
  // draws are taken into log space here and the axis is labelled in powers of
  // ten to put them back.
  const data = useMemo(
    () => [
      {
        type: 'histogram',
        x: stats.log
          ? values.filter((v) => Number.isFinite(v) && v > 0).map(Math.log10)
          : values,
        ...(stats.clipped
          ? { xbins: { start: stats.lo, end: stats.hi, size: (stats.hi - stats.lo) / BINS } }
          : { nbinsx: BINS }),
        histnorm: 'probability density',
        // A hairline of surface colour between bars, rather than a stroke
        // drawn around each one.
        marker: { color: BLUE, line: { width: 1, color: '#ffffff' } },
        hovertemplate: stats.log
          ? '10<sup>%{x:.2f}</sup><br>density per decade %{y:.3g}<extra></extra>'
          : '%{x}<br>density %{y:.3g}<extra></extra>',
      },
    ],
    [values, stats.log, stats.clipped, stats.lo, stats.hi],
  )

  const decades = useMemo(() => {
    if (!stats.log) return null
    const first = Math.floor(Math.log10(stats.min))
    const last = Math.ceil(Math.log10(stats.max))
    const stride = Math.max(1, Math.ceil((last - first) / 7))
    const values: number[] = []
    for (let power = first; power <= last; power += stride) values.push(power)
    return { tickvals: values, ticktext: values.map((p) => `10<sup>${p}</sup>`) }
  }, [stats.log, stats.min, stats.max])

  const layout = useMemo(() => {
    // Plotly places shapes and annotations in the axis's own coordinates, and
    // a log axis counts in powers of ten.
    const at = stats.log ? Math.log10(stats.median) : stats.median
    const marked = Number.isFinite(at)
    return {
      margin: { l: 52, r: 16, t: 10, b: 44 },
      xaxis: {
        title: stats.log ? `${label} (log scale)` : label,
        ...(stats.log && decades ? decades : {}),
        ...(stats.clipped ? { range: [stats.lo, stats.hi] } : {}),
      },
      yaxis: {
        title: stats.log ? 'density per decade' : 'density',
        rangemode: 'tozero',
      },
      shapes: marked
        ? [
            {
              type: 'line',
              x0: at,
              x1: at,
              yref: 'paper',
              y0: 0,
              y1: 1,
              line: { color: VIOLET, width: 1.5 },
            },
          ]
        : [],
      annotations: marked
        ? [
            {
              x: at,
              yref: 'paper',
              y: 1,
              text: 'median',
              showarrow: false,
              yanchor: 'bottom',
              font: { size: 11, color: VIOLET },
            },
          ]
        : [],
    }
  }, [label, stats.median, stats.log, stats.clipped, stats.lo, stats.hi, decades])

  return (
    <section className="card">
      <h3>{label}</h3>
      <Plot
        data={data}
        layout={layout}
        height={260}
        description={`Posterior distribution of ${label}`}
      />
      <p className="muted small">
        median {fmt(stats.median)} · 90% interval {fmt(stats.q05)} to{' '}
        {fmt(stats.q95)}
        {stats.clipped && (
          <> · axis clipped: the tail runs out to {fmt(stats.max)}</>
        )}
      </p>
    </section>
  )
}

// ---------------------------------------------------------------------------
// traces

function TracesView({ posterior }: { posterior: Posterior }) {
  return (
    <div className="stack">
      {posterior.parameters.map((name) => (
        <TraceCard key={name} posterior={posterior} name={name} />
      ))}
    </div>
  )
}

function TraceCard({ posterior, name }: { posterior: Posterior; name: string }) {
  const data = useMemo(() => traceSeries(posterior, name), [posterior, name])
  const layout = useMemo(
    () => ({
      margin: { l: 58, r: 16, t: 10, b: 44 },
      xaxis: { title: 'draw' },
      yaxis: { title: name },
      showlegend: true,
      hovermode: 'closest',
    }),
    [name],
  )

  return (
    <section className="card">
      <h3>{name}</h3>
      <Plot
        data={data}
        layout={layout}
        height={240}
        description={`Sampler trace for ${name}, one line per chain`}
      />
      <p className="muted small">
        Chains should overlap and look like noise around a stable level. A chain
        drifting apart from the others means they have not agreed.
      </p>
    </section>
  )
}

function traceSeries(posterior: Posterior, name: string) {
  const values = posterior.columns[name] ?? []
  const chains = posterior.chain
  // Thinner than the 2px house line: four of these overlap on purpose, and at
  // full weight the overlap turns into a solid block instead of a texture.
  const line = { width: 1.2 }

  if (chains.length !== values.length) {
    return [
      {
        type: 'scattergl',
        mode: 'lines',
        y: values,
        name: 'all draws',
        line: { ...line, color: SERIES[0] },
      },
    ]
  }

  const grouped = new Map<number, { x: number[]; y: number[] }>()
  values.forEach((value, index) => {
    const chain = chains[index]
    if (!grouped.has(chain)) grouped.set(chain, { x: [], y: [] })
    const series = grouped.get(chain)!
    series.x.push(series.x.length)
    series.y.push(value)
  })
  return [...grouped.entries()].map(([chain, series], index) => ({
    type: 'scattergl',
    mode: 'lines',
    x: series.x,
    y: series.y,
    name: `chain ${chain}`,
    line: { ...line, color: SERIES[index % SERIES.length] },
  }))
}

// ---------------------------------------------------------------------------
// joint

type JointMode = '3d' | '2d'

function JointView({ posterior }: { posterior: Posterior }) {
  const [mode, setMode] = useState<JointMode>('3d')

  const pairs = useMemo(() => {
    const out: [string, string][] = []
    const names = posterior.parameters
    for (let i = 0; i < names.length; i += 1) {
      for (let j = i + 1; j < names.length; j += 1) out.push([names[i], names[j]])
    }
    return out
  }, [posterior])

  if (pairs.length === 0) {
    return (
      <section className="card">
        <p className="muted">
          This model has a single parameter, so there is no joint distribution
          to show. See Distributions instead.
        </p>
      </section>
    )
  }

  return (
    <div className="stack">
      <div className="viz-toolbar">
        <span className="muted small">
          Posterior density over each pair of parameters.
        </span>
        <div className="segmented" role="group" aria-label="How to draw the joint density">
          <button
            className={`segmented__option${mode === '3d' ? ' segmented__option--on' : ''}`}
            onClick={() => setMode('3d')}
            aria-pressed={mode === '3d'}
          >
            3D surface
          </button>
          <button
            className={`segmented__option${mode === '2d' ? ' segmented__option--on' : ''}`}
            onClick={() => setMode('2d')}
            aria-pressed={mode === '2d'}
          >
            2D contour
          </button>
        </div>
      </div>

      <div className="grid2">
        {pairs.map(([x, y]) => (
          <JointCard key={`${x}:${y}`} posterior={posterior} x={x} y={y} mode={mode} />
        ))}
      </div>
    </div>
  )
}

function JointCard({
  posterior,
  x,
  y,
  mode,
}: {
  posterior: Posterior
  x: string
  y: string
  mode: JointMode
}) {
  const xs = posterior.columns[x] ?? []
  const ys = posterior.columns[y] ?? []
  const grid = useMemo(() => density2d(xs, ys), [xs, ys])

  const hover = `${x} %{x:.4g}<br>${y} %{y:.4g}<br>density %{z:.3g}<extra></extra>`

  const data = useMemo(() => {
    if (!grid) return []
    if (mode === '3d') {
      return [
        {
          type: 'surface',
          x: grid.x,
          y: grid.y,
          z: grid.z,
          colorscale: DENSITY_SCALE,
          showscale: false,
          // The shadow on the floor is the 2D contour of the same grid, so the
          // shape can be read off without rotating anything.
          contours: {
            z: { show: true, usecolormap: true, project: { z: true }, width: 1 },
          },
          lighting: { ambient: 0.78, diffuse: 0.5, specular: 0.06, roughness: 0.9 },
          hovertemplate: hover,
        },
      ]
    }
    // Draws outside the grid's range would sit on bare surface with no
    // contour under them, so the two views agree on what is being shown.
    const inside: [number, number][] = []
    const [xLo, xHi] = [grid.x[0], grid.x[grid.x.length - 1]]
    const [yLo, yHi] = [grid.y[0], grid.y[grid.y.length - 1]]
    for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
      if (xs[i] >= xLo && xs[i] <= xHi && ys[i] >= yLo && ys[i] <= yHi) {
        inside.push([xs[i], ys[i]])
      }
    }
    const dots = thin(inside, 1500)

    return [
      {
        type: 'contour',
        x: grid.x,
        y: grid.y,
        z: grid.z,
        colorscale: DENSITY_SCALE_CLEAR,
        showscale: false,
        ncontours: 12,
        contours: { coloring: 'fill' },
        line: { width: 0.6, color: 'rgba(255, 255, 255, 0.55)' },
        hovertemplate: hover,
      },
      {
        type: 'scattergl',
        mode: 'markers',
        x: dots.map((d) => d[0]),
        y: dots.map((d) => d[1]),
        marker: { size: 2.5, color: 'rgba(13, 54, 107, 0.4)' },
        hoverinfo: 'skip',
        name: 'draws',
      },
    ]
  }, [grid, mode, hover, xs, ys])

  const layout = useMemo(() => {
    if (mode === '3d') {
      return {
        // The scene needs room for the tick labels that hang off the bottom
        // corners of the cube; at zero margin they are cut by the card.
        margin: { l: 8, r: 8, t: 8, b: 24 },
        scene: {
          xaxis: { ...SCENE_AXIS, title: x },
          yaxis: { ...SCENE_AXIS, title: y },
          zaxis: { ...SCENE_AXIS, title: 'density' },
          // Pulled in and widened so the surface fills the card rather than
          // floating in the middle of it.
          camera: { eye: { x: 1.5, y: -1.6, z: 0.78 } },
          aspectmode: 'manual',
          aspectratio: { x: 1.3, y: 1.3, z: 0.8 },
        },
      }
    }
    return {
      margin: { l: 58, r: 16, t: 10, b: 46 },
      xaxis: { title: x },
      yaxis: { title: y },
      hovermode: 'closest',
    }
  }, [mode, x, y])

  return (
    <section className="card">
      <h3>
        {x} vs {y}
      </h3>
      {grid ? (
        <Plot
          data={data}
          layout={layout}
          height={mode === '3d' ? 440 : 340}
          description={`Joint posterior density of ${x} and ${y}`}
        />
      ) : (
        <div className="placeholder">Not enough draws to estimate a density.</div>
      )}
      <p className="muted small">
        A ridge running diagonally means the data cannot tell these two apart —
        only their combination is pinned down.
        {mode === '3d' && ' Drag to rotate; the shadow underneath is the same density seen from above.'}
      </p>
    </section>
  )
}

// ---------------------------------------------------------------------------
// reliability

interface ReliabilityProps {
  curve: ReliabilityOverTime | undefined
  /** Reliability at each mission, marked on the curve. */
  marks: SummaryRow[]
  empirical?: { steps: { t: number; s: number }[]; censored: { t: number; s: number }[] }
}

function ReliabilityView({ curve, marks, empirical }: ReliabilityProps) {
  const data = useMemo(() => {
    if (!curve) return []
    const traces = reliabilityTraces(curve)
    if (empirical && empirical.steps.length > 1) {
      // The data's own curve, so the eye can check the fit: a model that
      // misses the steps is the wrong model, however narrow its band.
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: empirical.steps.map((p) => p.t),
        y: empirical.steps.map((p) => p.s),
        line: { color: '#2b2b2b', width: 1.4, shape: 'hv' },
        name: 'data (Kaplan–Meier)',
        hovertemplate: '%{x}: %{y:.3f}<extra>data</extra>',
      })
      if (empirical.censored.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          x: empirical.censored.map((p) => p.t),
          y: empirical.censored.map((p) => p.s),
          marker: { symbol: 'line-ns-open', size: 9, color: '#2b2b2b', line: { width: 1.4 } },
          name: 'censored',
          hovertemplate: '%{x}<extra>censored</extra>',
        })
      }
    }
    const points = marks
      .map((row) => ({ at: missionOf(row).at, row }))
      .filter((m): m is { at: number; row: SummaryRow } => m.at !== null && m.at > 0)
    if (points.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'markers+text',
        x: points.map((p) => p.at),
        y: points.map((p) => p.row.mean),
        error_y: {
          type: 'data',
          symmetric: false,
          array: points.map((p) => p.row.q95 - p.row.mean),
          arrayminus: points.map((p) => p.row.mean - p.row.q05),
          color: CRITICAL,
          thickness: 1.5,
          width: 5,
        },
        text: points.map((p) => sig(p.row.mean)),
        textposition: 'middle right',
        textfont: { color: CRITICAL, size: 12 },
        marker: { color: CRITICAL, size: 8 },
        name: 'at the mission',
        hovertemplate: '%{x}: %{y:.4f}<extra>at the mission</extra>',
      })
    }
    return traces
  }, [curve, marks, empirical])

  const axisLabel = curve
    ? curve.kind === 'time'
      ? `time${curve.unit ? ` (${curve.unit})` : ''}`
      : 'demands survived'
    : ''

  // The model's curve sets the window; the data's steps may run further and
  // are cut there rather than stretching the axis past the curve.
  const right = curve?.points.length ? curve.points[curve.points.length - 1].x : null
  // A curve that never drops below ~0.5 (rare failures, short missions) is a
  // flat line on a 0-1 axis; zoom in, and keep the axis title honest.
  const floor = useMemo(() => {
    if (!curve?.points.length) return 0
    const lowest = Math.min(...curve.points.map((p) => p.q05).filter(Number.isFinite))
    if (!(lowest > 0.5)) return 0
    return Math.max(0, Math.floor((lowest - 0.15 * (1 - lowest)) * 100) / 100)
  }, [curve])
  const layout = useMemo(
    () => ({
      margin: { l: 58, r: 20, t: 10, b: 48 },
      xaxis: { title: axisLabel, ...(right ? { range: [0, right] } : { rangemode: 'tozero' }) },
      yaxis: {
        title: floor > 0 ? `reliability R (axis from ${floor})` : 'reliability R',
        range: [floor, 1 + 0.02 * (1 - floor)],
      },
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: -0.2 },
      hovermode: 'x unified',
    }),
    [axisLabel, right, floor],
  )

  if (!curve || curve.points.length === 0) {
    return (
      <section className="card">
        <h3>Reliability</h3>
        <p className="muted">
          There is no reliability curve for this run: it had neither a mission
          nor any data to set a time scale by. Set a mission time on the Model
          tab and run again.
        </p>
      </section>
    )
  }

  return (
    <section className="card">
      <h3>Reliability over {curve.kind === 'time' ? 'time' : 'demands'}</h3>
      <Plot
        data={data}
        layout={layout}
        height={400}
        description="Posterior reliability against time, with a 90% credible band"
      />
      <p className="muted small">
        The line is the posterior mean, the probability of surviving that long
        given everything observed; the band holds 90% of the posterior at each
        point, so it widens where the data stop saying much. Red marks the
        mission.{empirical && ' The black steps are the data alone (Kaplan–Meier, ticks for censored units): the fitted curve should run through them.'}
      </p>
    </section>
  )
}

function reliabilityTraces(curve: ReliabilityOverTime): Record<string, unknown>[] {
  const x = curve.points.map((p) => p.x)
  const reversed = [...x].reverse()

  const traces: Record<string, unknown>[] = [
    {
      type: 'scatter',
      x: [...x, ...reversed],
      y: [
        ...curve.points.map((p) => p.q95),
        ...[...curve.points].reverse().map((p) => p.q05),
      ],
      fill: 'toself',
      fillcolor: BLUE_WASH,
      line: { color: 'transparent' },
      hoverinfo: 'skip',
      name: '90% interval',
      showlegend: true,
    },
    {
      type: 'scatter',
      mode: 'lines',
      x,
      y: curve.points.map((p) => p.median),
      line: { color: VIOLET, width: 1.5, dash: 'dash' },
      name: 'median',
      hovertemplate: '%{y:.4f}<extra>median</extra>',
    },
    {
      type: 'scatter',
      mode: 'lines',
      x,
      y: curve.points.map((p) => p.mean),
      line: { color: BLUE, width: 2.5 },
      name: 'mean',
      hovertemplate: '%{y:.4f}<extra>mean</extra>',
    },
  ]

  // The mission is a threshold rather than a series, so it wears the reserved
  // red and arrives as its own trace -- which is what makes it switchable from
  // the legend.
  const missions = curve.mission.filter((m) => Number.isFinite(m) && m > 0)
  if (missions.length > 0) {
    const mx: (number | null)[] = []
    const my: (number | null)[] = []
    for (const m of missions) {
      mx.push(m, m, null)
      my.push(0, 1, null)
    }
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: mx,
      y: my,
      line: { color: CRITICAL, width: 2, dash: 'dot' },
      name: missions.length > 1 ? 'mission times' : 'mission',
      hoverinfo: 'skip',
    })
  }
  return traces
}

function fmtElapsed(elapsed: Record<string, number>): string {
  const parts = Object.entries(elapsed)
    .filter(([, seconds]) => seconds > 0.01)
    .map(([stage, seconds]) => `${stage} ${seconds.toFixed(1)}s`)
  return parts.join(' · ')
}
