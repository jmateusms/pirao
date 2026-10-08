import type { Example } from '../api/types'

interface Props {
  example: Example
  onClose: () => void
}

/** What the loaded example is, where its data come from, and what to look for. */
export function ExampleNote({ example, onClose }: Props) {
  const illustrative = example.data_kind === 'illustrative'
  return (
    <div className="banner banner--note example-note">
      <div className="example-note__head">
        <strong>{example.title}</strong>
        <span className={`pill${illustrative ? ' pill--warn' : ''}`}>
          {illustrative ? 'illustrative data' : 'public data'}
        </span>
        <button className="link example-note__close" onClick={onClose}>
          close
        </button>
      </div>
      <p>{example.note}</p>
      <p className="muted small">Source: {example.source}</p>
    </div>
  )
}
