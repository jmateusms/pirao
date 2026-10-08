import { useEffect, useState } from 'react'

import { api } from '../api/client'
import type { RunState } from '../api/types'
import { type Key, useI18n } from '../i18n'

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
const STAGES = ['resolve', 'render', 'compile', 'sample', 'summarize', 'done'] as const

export function RunPanel({ run, canRun, onStart, onFinished }: Props) {
  const { t } = useI18n()
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
          <h2>{t('run.title')}</h2>
          <p className="muted">{t(canRun ? 'run.ready' : 'run.fixFirst')}</p>
          <button className="primary" disabled={!canRun} onClick={onStart}>
            {t('app.run')}
          </button>
        </section>
      </div>
    )
  }

  const activeIndex = STAGES.findIndex((s) => s === live.stage)
  const finished = live.status === 'done'
  const failed = live.status === 'failed'
  const cancelled = live.status === 'cancelled'

  return (
    <div className="stack">
      <section className="card">
        <div className="card__head">
          <h2>
            {t('run.title')} {live.id && <span className="pill">{live.id}</span>}
          </h2>
          {live.status === 'running' && (
            <button className="chip" onClick={() => api.cancelRun(live.id)}>
              {t('run.cancel')}
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
              <li key={stage} className={`stage stage--${state}`}>
                <span className="stage__mark" />
                <span>{t(`stage.${stage}` as Key)}</span>
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
            <strong>{t('run.didNotFinish')}</strong>
            <p>{live.error}</p>
            {live.error_details.length > 0 && (
              <ul className="issues">
                {live.error_details.map((detail, i) => (
                  <li key={i}>
                    {detail.row !== null && <strong>{t('run.row', { n: detail.row })} </strong>}
                    {detail.column && <>· {detail.column} </>}
                    {detail.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {cancelled && (
          <div className="banner banner--warn">{t('run.cancelled')}</div>
        )}

        {live.warnings.map((warning) => (
          <div className="banner banner--warn" key={warning}>
            {warning}
          </div>
        ))}

        {(failed || cancelled) && (
          <div className="row">
            <button className="primary" disabled={!canRun} onClick={onStart}>
              {t('run.again')}
            </button>
          </div>
        )}
      </section>
    </div>
  )
}
