import { useState } from 'react'

import type {
  LikelihoodMeta,
  Meta,
  ModelSpec,
  ParamMeta,
  ResolvedParam,
  SpecValidation,
} from '../api/types'
import { NumberField } from '../components/NumberField'
import { PriorCurve } from '../components/PriorCurve'
import { compatibleFamilies, defaultHyper, formatBound } from './defaults'

interface Props {
  meta: Meta
  spec: ModelSpec
  likelihood: LikelihoodMeta
  validation: SpecValidation | null
  onChangeLikelihood: (id: string) => void
  onChangeSpec: (spec: ModelSpec) => void
}

export function ModelPanel({
  meta,
  spec,
  likelihood,
  validation,
  onChangeLikelihood,
  onChangeSpec,
}: Props) {
  const [showSource, setShowSource] = useState(false)

  const patchParam = (name: string, patch: Partial<ModelSpec['params'][string]>) => {
    onChangeSpec({
      ...spec,
      params: { ...spec.params, [name]: { ...spec.params[name], ...patch } },
    })
  }

  const resolvedFor = (name: string): ResolvedParam | undefined =>
    validation?.parameters.find((p) => p.name === name)

  return (
    <div className="stack">
      <section className="card">
        <h2>Likelihood</h2>
        <p className="muted">
          What kind of observation each row of your data represents.
        </p>
        <select
          className="control control--wide"
          value={likelihood.id}
          onChange={(e) => onChangeLikelihood(e.target.value)}
        >
          {meta.likelihoods.map((l) => (
            <option key={l.id} value={l.id}>
              {l.label}
            </option>
          ))}
        </select>
        <p className="description">{likelihood.description}</p>

        <div className="row row--wrap">
          {likelihood.mission === 'time' ? (
            <label className="field">
              <span>Mission time(s)</span>
              <input
                className="control"
                value={(spec.mission_times ?? []).join(', ')}
                placeholder="e.g. 100, 500, 1000"
                onChange={(e) =>
                  onChangeSpec({
                    ...spec,
                    mission_times: e.target.value
                      .split(',')
                      .map((v) => Number(v.trim()))
                      .filter((v) => Number.isFinite(v)),
                  })
                }
              />
              <small>
                Reliability is reported at each of these. Give several to get a
                curve.
              </small>
            </label>
          ) : (
            <label className="field">
              <span>Mission demands</span>
              <NumberField
                className="control"
                integer
                min={0}
                value={spec.mission_demands ?? 1}
                onChange={(value) => onChangeSpec({ ...spec, mission_demands: value })}
              />
              <small>How many demands the unit must survive.</small>
            </label>
          )}

          <label className="field">
            <span>Time unit</span>
            <input
              className="control control--narrow"
              value={spec.time_unit ?? 'h'}
              onChange={(e) => onChangeSpec({ ...spec, time_unit: e.target.value })}
            />
            <small>
              Printed on every plot and export. It matters: weighting by
              relevance is not invariant to the unit.
            </small>
          </label>

          <label className="field">
            <span>Curve runs to</span>
            <NumberField
              className="control control--narrow"
              nullable
              placeholder="auto"
              value={spec.curve_horizon ?? null}
              onChange={(v) => onChangeSpec({ ...spec, curve_horizon: v })}
            />
            <small>
              Where the reliability curve stops, in{' '}
              {likelihood.mission === 'time' ? (spec.time_unit ?? 'h') : 'demands'}
              . Leave empty for 1.3 × the mission, which shows where the
              estimate is heading without implying the extrapolation is data.
            </small>
          </label>

          <label className="field field--check">
            <input
              type="checkbox"
              checked={spec.prior_only ?? false}
              onChange={(e) => onChangeSpec({ ...spec, prior_only: e.target.checked })}
            />
            <span>Ignore the data (sample the prior only)</span>
            <small>
              Fits the prior alone, so you can see what it implies before the
              data has any say.
            </small>
          </label>
        </div>
      </section>

      {validation && !validation.ok && (
        <div className="banner banner--error">
          <strong>This model cannot be fitted yet.</strong>
          <ul>
            {validation.errors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {likelihood.parameters.map((param) => (
        <ParameterCard
          key={param.name}
          meta={meta}
          param={param}
          spec={spec}
          resolved={resolvedFor(param.name)}
          onPatch={(patch) => patchParam(param.name, patch)}
        />
      ))}

      {validation?.stan_source && (
        <section className="card">
          <button className="link" onClick={() => setShowSource((v) => !v)}>
            {showSource ? 'Hide' : 'Show'} the generated Stan program
          </button>
          {showSource && (
            <pre className="source">{validation.stan_source}</pre>
          )}
        </section>
      )}
    </div>
  )
}

interface ParamProps {
  meta: Meta
  param: ParamMeta
  spec: ModelSpec
  resolved: ResolvedParam | undefined
  onPatch: (patch: Partial<ModelSpec['params'][string]>) => void
}

function ParameterCard({ meta, param, spec, resolved, onPatch }: ParamProps) {
  const current = spec.params[param.name]
  const families = compatibleFamilies(meta.priors, param)
  const family = meta.priors.find((p) => p.id === current.prior.family)

  const preview = resolved?.preview
  const mass = preview?.retained_mass

  return (
    <section className="card">
      <div className="card__head">
        <h2>
          {param.label} <span className="pill">{param.role}</span>
        </h2>
      </div>
      <p className="description">{param.description}</p>

      <div className="param">
        <div className="param__controls">
          <label className="field">
            <span>Prior</span>
            <select
              className="control"
              value={current.prior.family}
              onChange={(e) => {
                const next = meta.priors.find((p) => p.id === e.target.value)!
                onPatch({
                  prior: { family: next.id, hyper: defaultHyper(next) },
                })
              }}
            >
              {families.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>

          {family?.hyperparameters.map((hyper) => (
            <label className="field" key={hyper.name}>
              <span>{hyper.label}</span>
              <NumberField
                className="control control--narrow"
                value={current.prior.hyper[hyper.name] ?? hyper.default}
                onChange={(value) =>
                  onPatch({
                    prior: {
                      ...current.prior,
                      hyper: {
                        ...current.prior.hyper,
                        [hyper.name]: value ?? hyper.default,
                      },
                    },
                  })
                }
              />
            </label>
          ))}

          <label className="field">
            <span>Lower bound</span>
            <NumberField
              className="control control--narrow"
              nullable
              placeholder="none"
              value={current.lower}
              onChange={(value) => onPatch({ lower: value })}
            />
          </label>
          <label className="field">
            <span>Upper bound</span>
            <NumberField
              className="control control--narrow"
              nullable
              placeholder="none"
              value={current.upper}
              onChange={(value) => onPatch({ upper: value })}
            />
          </label>

          {param.presets.length > 0 && (
            <div className="field">
              <span>Presets</span>
              <div className="row">
                {param.presets.map((preset) => (
                  <button
                    key={preset.label}
                    className="chip"
                    onClick={() =>
                      onPatch({
                        lower: preset.lower ?? current.lower,
                        upper: preset.upper ?? current.upper,
                      })
                    }
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="param__preview">
          {preview && preview.x.length > 0 ? (
            <>
              <PriorCurve
                x={preview.x}
                pdf={preview.pdf}
                q05={preview.q05}
                q95={preview.q95}
              />
              <dl className="stats">
                <div>
                  <dt>median</dt>
                  <dd>{fmt(preview.median)}</dd>
                </div>
                <div>
                  <dt>90% interval</dt>
                  <dd>
                    {fmt(preview.q05)} – {fmt(preview.q95)}
                  </dd>
                </div>
              </dl>
            </>
          ) : (
            <div className="placeholder">No preview available</div>
          )}

          {resolved && (
            <p className="muted small">
              In force:{' '}
              <strong>
                {formatBound(resolved.effective_lower, '−∞')} to{' '}
                {formatBound(resolved.effective_upper, '+∞')}
              </strong>
              {mass !== null && mass !== undefined && (
                <> · keeps {(mass * 100).toFixed(1)}% of the prior</>
              )}
            </p>
          )}
        </div>
      </div>

      {/*
        The bounds silently narrow the prior, so what it becomes has to be
        said out loud -- this is the failure the legacy models had, where a
        lognormal on a [0,1] parameter quietly lost half its mass.
      */}
      {preview?.warning && (
        <div className="banner banner--warn">{preview.warning}</div>
      )}
      {family?.notes && <p className="muted small">{family.notes}</p>}
    </section>
  )
}

function fmt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(2)
  return String(Number(value.toPrecision(4)))
}
