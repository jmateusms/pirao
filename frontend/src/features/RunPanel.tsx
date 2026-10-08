import { useEffect, useState } from 'react'

import { api } from '../api/client'
import type { RunState } from '../api/types'

interface Props {
  run: RunState | null
  canRun: boolean
  onStart: () => void
  onFinished: (run: RunState) => void
}

/**
 * Stages in the order they happen.  Compilation is listed as its own step
 * because it is the one that actually takes time -- sampling these models is a
 * matter of milliseconds, so a bar labelled "sampling" would sit at zero
 * through the entire wait and then jump to done.
 */
const STAGES: { id: string; label: string }[] = [
  { id: 'resolve', label: 'Checking the model' },
  { id: 'render', label: 'Assembling the Stan program' },
  { id: 'compile', label: 'Compiling' },
  { id: 'sample', label: 'Sampling' },
  { id: 'summarize', label: 'Summarising' },
  { id: 'done', label: 'Finished' },
]

export function RunPanel({ run, canRun, onStart, onFinished }: Props) {
  const [live, setLive] = useState<RunState | null>(run)

  useEffect(() => setLive(run), [run])

  useEffect(() => {
    if (!run?.id || run.status === 'done' || run.status === 'failed') return

    const stop = api.streamRun(run.id, (message) => {
      if (message.kind === 'progress') {
        setLive((prev) =>
          prev
            ? {
                ...prev,
                stage: String(message.stage),
                message: String(message.message),
                fraction: (message.fraction as number | null) ?? null,
              }
            : prev,
        )
      } else if (message.kind === 'end' || message.kind === 'done' || message.kind === 'failed') {
        api
          .getRun(run.id)
          .then((final) => {
            setLive(final)
            onFinished(final)
          })
          .catch(() => undefined)
      }
    })
    return stop
  }, [run?.id, run?.status, onFinished])

  if (!live) {
    return (
      <div className="stack">
        <section className="card">
          <h2>Run</h2>
          <p className="muted">
            {canRun
              ? 'Everything checks out. Start when you are ready.'
              : 'Fix the problems flagged on the Model and Data tabs first.'}
          </p>
          <button className="primary" disabled={!canRun} onClick={onStart}>
            Run MCMC
          </button>
        </section>
      </div>
    )
  }

  const activeIndex = STAGES.findIndex((s) => s.id === live.stage)
  const finished = live.status === 'done'
  const failed = live.status === 'failed'
  const cancelled = live.status === 'cancelled'

  return (
    <div className="stack">
      <section className="card">
        <div className="card__head">
          <h2>Run {live.id && <span className="pill">{live.id}</span>}</h2>
          {live.status === 'running' && (
            <button className="chip" onClick={() => api.cancelRun(live.id)}>
              Cancel
            </button>
          )}
        </div>

        <ol className="stages">
          {STAGES.map((stage, index) => {
            const state =
              failed && index === activeIndex
                ? 'failed'
                : finished || index < activeIndex
                  ? 'done'
                  : index === activeIndex
                    ? 'active'
                    : 'pending'
            return (
              <li key={stage.id} className={`stage stage--${state}`}>
                <span className="stage__mark" />
                <span>{stage.label}</span>
              </li>
            )
          })}
        </ol>

        <p className={failed ? 'muted' : ''}>{live.message}</p>

        {live.status === 'running' && (
          <div className="progress">
            <div
              className={`progress__bar${
                live.fraction === null ? ' progress__bar--indeterminate' : ''
              }`}
              style={
                live.fraction !== null
                  ? { width: `${Math.round(live.fraction * 100)}%` }
                  : undefined
              }
            />
          </div>
        )}

        {failed && (
          <div className="banner banner--error">
            <strong>The run did not finish.</strong>
            <p>{live.error}</p>
            {live.error_details.length > 0 && (
              <ul className="issues">
                {live.error_details.map((detail, i) => (
                  <li key={i}>
                    {detail.row !== null && <strong>Row {detail.row} </strong>}
                    {detail.column && <>· {detail.column} </>}
                    {detail.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {cancelled && (
          <div className="banner banner--warn">This run was cancelled.</div>
        )}

        {live.warnings.map((warning) => (
          <div className="banner banner--warn" key={warning}>
            {warning}
          </div>
        ))}

        {(failed || cancelled) && (
          <div className="row">
            <button className="primary" disabled={!canRun} onClick={onStart}>
              Run again
            </button>
          </div>
        )}
      </section>
    </div>
  )
}
