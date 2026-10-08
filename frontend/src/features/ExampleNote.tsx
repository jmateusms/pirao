import type { Example } from '../api/types'
import { useI18n } from '../i18n'

interface Props {
  example: Example
  onClose: () => void
}

/** What the loaded example is, where its data come from, and what to look for. */
export function ExampleNote({ example, onClose }: Props) {
  const { t, lang } = useI18n()
  const pt = lang === 'pt'
  const illustrative = example.data_kind === 'illustrative'
  return (
    <div className="banner banner--note example-note">
      <div className="example-note__head">
        <strong>{pt ? example.title_pt : example.title}</strong>
        <span className={`pill${illustrative ? ' pill--warn' : ''}`}>
          {t(illustrative ? 'example.illustrative' : 'example.public')}
        </span>
        <button className="link example-note__close" onClick={onClose}>
          {t('example.close')}
        </button>
      </div>
      <p>{pt ? example.note_pt : example.note}</p>
      <p className="muted small">
        {t('example.source')} {pt ? example.source_pt : example.source}
      </p>
    </div>
  )
}
