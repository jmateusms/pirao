import type { SamplerConfig } from '../api/types'
import { NumberField } from '../components/NumberField'
import { type Key, numberLocale, useI18n } from '../i18n'

interface Props {
  /** Plain-language explanations, served by the backend so there is one copy. */
  help: Record<string, string>
  sampler: SamplerConfig
  onChange: (sampler: SamplerConfig) => void
}

const FIELDS: {
  key: keyof SamplerConfig
  min?: number
  max?: number
  integer?: boolean
  optional?: boolean
}[] = [
  { key: 'chains', min: 1, max: 16, integer: true },
  { key: 'iter_warmup', min: 1, integer: true },
  { key: 'iter_sampling', min: 1, integer: true },
  { key: 'adapt_delta', min: 0.5, max: 0.999 },
  { key: 'max_treedepth', min: 1, max: 20, integer: true },
  { key: 'seed', integer: true, optional: true },
]

export function SamplerPanel({ help, sampler, onChange }: Props) {
  const { t } = useI18n()
  return (
    <div className="stack">
      <section className="card">
        <h2>{t('sampler.title')}</h2>
        <p className="muted">{t('sampler.intro')}</p>

        <div className="settings">
          {FIELDS.map((field) => (
            <label className="setting" key={field.key}>
              <span className="setting__label">{t(`sampler.${field.key}` as Key)}</span>
              <NumberField
                className="control control--narrow"
                integer={field.integer}
                min={field.min}
                max={field.max}
                nullable={field.optional}
                placeholder={field.optional ? t('sampler.random') : undefined}
                value={sampler[field.key] as number | null}
                onChange={(value) =>
                  onChange({ ...sampler, [field.key]: value } as SamplerConfig)
                }
              />
              <small className="setting__help">{help[field.key] ?? ''}</small>
            </label>
          ))}
        </div>

        <p className="muted small">
          {t('sampler.total', {
            chains: sampler.chains,
            draws: sampler.iter_sampling.toLocaleString(numberLocale()),
            total: (sampler.chains * sampler.iter_sampling).toLocaleString(numberLocale()),
          })}
        </p>
      </section>
    </div>
  )
}
