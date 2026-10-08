import { useEffect, useState } from 'react'

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
import { decimal, type Lang, useI18n } from '../i18n'
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
  const { t, lang } = useI18n()
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
        <h2>{t('model.likelihood')}</h2>
        <p className="muted">{t('model.likelihoodIntro')}</p>
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
              <span>{t('model.missionTimes')}</span>
              <MissionTimesField
                lang={lang}
                value={spec.mission_times ?? []}
                placeholder={t('model.missionTimesPlaceholder')}
                onChange={(times) => onChangeSpec({ ...spec, mission_times: times })}
              />
              <small>{t('model.missionTimesHelp')}</small>
            </label>
          ) : (
            <label className="field">
              <span>{t('model.missionDemands')}</span>
              <NumberField
                className="control"
                integer
                min={0}
                value={spec.mission_demands ?? 1}
                onChange={(value) => onChangeSpec({ ...spec, mission_demands: value })}
              />
              <small>{t('model.missionDemandsHelp')}</small>
            </label>
          )}

          <label className="field">
            <span>{t('model.timeUnit')}</span>
            <input
              className="control control--narrow"
              value={spec.time_unit ?? 'h'}
              onChange={(e) => onChangeSpec({ ...spec, time_unit: e.target.value })}
            />
            <small>{t('model.timeUnitHelp')}</small>
          </label>

          <label className="field">
            <span>{t('model.curveTo')}</span>
            <NumberField
              className="control control--narrow"
              nullable
              placeholder={t('model.auto')}
              value={spec.curve_horizon ?? null}
              onChange={(v) => onChangeSpec({ ...spec, curve_horizon: v })}
            />
            <small>
              {t('model.curveToHelp', {
                unit: likelihood.mission === 'time' ? (spec.time_unit ?? 'h') : t('model.demands'),
              })}
            </small>
          </label>

          <label className="field field--check">
            <input
              type="checkbox"
              checked={spec.prior_only ?? false}
              onChange={(e) => onChangeSpec({ ...spec, prior_only: e.target.checked })}
            />
            <span>{t('model.priorOnly')}</span>
            <small>{t('model.priorOnlyHelp')}</small>
          </label>
        </div>
      </section>

      {validation && !validation.ok && (
        <div className="banner banner--error">
          <strong>{t('model.cannotFit')}</strong>
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
            {t(showSource ? 'model.hideSource' : 'model.showSource')}
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
  const { t } = useI18n()
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
            <span>{t('model.prior')}</span>
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
            <span>{t('model.lower')}</span>
            <NumberField
              className="control control--narrow"
              nullable
              placeholder={t('model.none')}
              value={current.lower}
              onChange={(value) => onPatch({ lower: value })}
            />
          </label>
          <label className="field">
            <span>{t('model.upper')}</span>
            <NumberField
              className="control control--narrow"
              nullable
              placeholder={t('model.none')}
              value={current.upper}
              onChange={(value) => onPatch({ upper: value })}
            />
          </label>

          {param.presets.length > 0 && (
            <div className="field">
              <span>{t('model.presets')}</span>
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
                  <dt>{t('model.median')}</dt>
                  <dd>{fmt(preview.median)}</dd>
                </div>
                <div>
                  <dt>{t('model.interval90')}</dt>
                  <dd>
                    {fmt(preview.q05)} – {fmt(preview.q95)}
                  </dd>
                </div>
              </dl>
            </>
          ) : (
            <div className="placeholder">{t('model.noPreview')}</div>
          )}

          {resolved && (
            <p className="muted small">
              {t('model.inForce')}{' '}
              <strong>
                {formatBound(resolved.effective_lower, '−∞')} {t('model.to')}{' '}
                {formatBound(resolved.effective_upper, '+∞')}
              </strong>
              {mass !== null && mass !== undefined && (
                <> · {t('model.keeps', { pct: decimal((mass * 100).toFixed(1)) })}</>
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
        <div className="banner banner--warn">
          {/* The truncation warning is the common one and is rebuilt here, in
              the reader's language; any other comes from the server as is. */}
          {mass !== null && mass !== undefined && mass < 0.99
            ? t('model.massWarning', {
                pct: decimal((mass * 100).toFixed(1)),
                family: family?.label ?? '',
              })
            : preview.warning}
        </div>
      )}
      {family?.notes && <p className="muted small">{family.notes}</p>}
    </section>
  )
}

function fmt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return decimal(value.toExponential(2))
  return decimal(String(Number(value.toPrecision(4))))
}

/**
 * Mission times as typed text.
 *
 * Kept as a buffer and committed when it parses, as NumberField does: parsing
 * on every keystroke would turn "100," into "100, 0" before the next digit
 * arrives.  English separates with commas; Portuguese, whose decimal mark is
 * the comma, separates with semicolons.
 */
function MissionTimesField({
  lang,
  value,
  placeholder,
  onChange,
}: {
  lang: Lang
  value: number[]
  placeholder: string
  onChange: (times: number[]) => void
}) {
  const show = (times: number[]) =>
    times.map((v) => decimal(String(v))).join(lang === 'pt' ? '; ' : ', ')
  const [buffer, setBuffer] = useState(() => show(value))

  const parse = (text: string): number[] | null => {
    const parts = text
      .split(lang === 'pt' ? ';' : /[,;]/)
      .map((part) => part.trim())
      .filter((part) => part !== '')
    const numbers = parts.map((part) => Number(lang === 'pt' ? part.replace(',', '.') : part))
    return numbers.every((n) => Number.isFinite(n) && n >= 0) ? numbers : null
  }

  // Follow outside changes (an example loaded, the language switched), but
  // leave a half-typed buffer alone while it still means the same numbers.
  useEffect(() => {
    const parsed = parse(buffer)
    if (!parsed || parsed.join() !== value.join()) setBuffer(show(value))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])
  useEffect(() => {
    setBuffer(show(value))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang])

  return (
    <input
      className="control"
      value={buffer}
      placeholder={placeholder}
      onChange={(e) => {
        setBuffer(e.target.value)
        const parsed = parse(e.target.value)
        if (parsed) onChange(parsed)
      }}
      onBlur={() => setBuffer(show(value))}
    />
  )
}
