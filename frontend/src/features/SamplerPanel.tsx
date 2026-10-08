import type { SamplerConfig } from '../api/types'
import { NumberField } from '../components/NumberField'

interface Props {
  /** Plain-language explanations, served by the backend so there is one copy. */
  help: Record<string, string>
  sampler: SamplerConfig
  onChange: (sampler: SamplerConfig) => void
}

const FIELDS: {
  key: keyof SamplerConfig
  label: string
  min?: number
  max?: number
  integer?: boolean
  optional?: boolean
}[] = [
  { key: 'chains', label: 'Chains', min: 1, max: 16, integer: true },
  { key: 'iter_warmup', label: 'Warmup draws', min: 1, integer: true },
  { key: 'iter_sampling', label: 'Kept draws', min: 1, integer: true },
  { key: 'adapt_delta', label: 'Target acceptance', min: 0.5, max: 0.999 },
  { key: 'max_treedepth', label: 'Max tree depth', min: 1, max: 20, integer: true },
  { key: 'seed', label: 'Random seed', integer: true, optional: true },
]

export function SamplerPanel({ help, sampler, onChange }: Props) {
  return (
    <div className="stack">
      <section className="card">
        <h2>Sampling</h2>
        <p className="muted">
          The defaults suit these models. Change them if the run reports a
          problem — each setting below says which problem it addresses.
        </p>

        <div className="settings">
          {FIELDS.map((field) => (
            <label className="setting" key={field.key}>
              <span className="setting__label">{field.label}</span>
              <NumberField
                className="control control--narrow"
                integer={field.integer}
                min={field.min}
                max={field.max}
                nullable={field.optional}
                placeholder={field.optional ? 'random' : undefined}
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
          {sampler.chains} chains × {sampler.iter_sampling} draws ={' '}
          {sampler.chains * sampler.iter_sampling} draws in total. Sampling
          these models takes a moment; the wait you will notice is the one-off
          compilation of a model structure you have not used before.
        </p>
      </section>
    </div>
  )
}
