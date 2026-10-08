import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { api } from './api/client'
import type {
  CellError,
  Example,
  ExampleSummary,
  Meta,
  ModelSpec,
  RunState,
  Row,
  SamplerConfig,
  SpecValidation,
} from './api/types'
import { DataPanel } from './features/DataPanel'
import { ModelPanel } from './features/ModelPanel'
import { ResultsPanel } from './features/ResultsPanel'
import { RunPanel } from './features/RunPanel'
import { SamplerPanel } from './features/SamplerPanel'
import { defaultSpecFor } from './features/defaults'
import { ExampleNote } from './features/ExampleNote'
import { type Key, type Lang, useI18n } from './i18n'

/** The inputs live in the side panel; the run and its results fill the rest. */
const TABS = ['Model', 'Data', 'Sampler'] as const
type Tab = (typeof TABS)[number]

const SIDE_KEY = 'pirao.sideWidth'
const SIDE_MIN = 340
const SIDE_DEFAULT = 470

function readSideWidth(): number {
  try {
    const stored = Number(window.localStorage.getItem(SIDE_KEY))
    if (Number.isFinite(stored) && stored >= SIDE_MIN) return stored
  } catch {
    /* storage can be unavailable; the default is fine */
  }
  return SIDE_DEFAULT
}

function clampSide(width: number): number {
  return Math.round(Math.max(SIDE_MIN, Math.min(width, window.innerWidth - 380)))
}

const DEFAULT_SAMPLER: SamplerConfig = {
  chains: 4,
  iter_warmup: 1000,
  iter_sampling: 1000,
  seed: null,
  adapt_delta: 0.8,
  max_treedepth: 10,
}

export function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [fatal, setFatal] = useState<string | null>(null)
  const [toolchain, setToolchain] = useState<{
    ok: boolean
    version?: string
    error?: string
    hint?: string
  } | null>(null)

  const { t, lang, setLang } = useI18n()
  const [tab, setTab] = useState<Tab>('Model')
  const [spec, setSpec] = useState<ModelSpec | null>(null)
  const [rows, setRows] = useState<Row[]>([])
  const [sampler, setSampler] = useState<SamplerConfig>(DEFAULT_SAMPLER)

  const [validation, setValidation] = useState<SpecValidation | null>(null)
  const [dataErrors, setDataErrors] = useState<CellError[]>([])
  const [dataWarnings, setDataWarnings] = useState<string[]>([])
  const [run, setRun] = useState<RunState | null>(null)
  // What the run was fitted with.  The panels on the left may be edited while
  // the results are on screen, and the results must keep describing the run.
  const [runSpec, setRunSpec] = useState<ModelSpec | null>(null)
  const [runValidation, setRunValidation] = useState<SpecValidation | null>(null)

  const [sideWidth, setSideWidth] = useState(readSideWidth)
  const dragging = useRef(false)
  const storeSideWidth = useCallback((width: number) => {
    try {
      window.localStorage.setItem(SIDE_KEY, String(width))
    } catch {
      /* not worth failing over */
    }
  }, [])

  const [examples, setExamples] = useState<ExampleSummary[]>([])
  const [example, setExample] = useState<Example | null>(null)
  // Bumped on every example load, as the panels' key: they hold local input
  // buffers that must start over from the example's values.
  const [loadSeq, setLoadSeq] = useState(0)

  const loadExample = useCallback(async (id: string) => {
    if (!id) {
      setExample(null)
      return
    }
    try {
      const loaded = await api.example(id)
      setSpec(loaded.spec)
      setRows(loaded.rows)
      setSampler({ ...DEFAULT_SAMPLER, ...loaded.sampler })
      setRun(null)
      setExample(loaded)
      setLoadSeq((n) => n + 1)
      setTab('Model')
      const url = new URL(window.location.href)
      url.searchParams.set('example', id)
      window.history.replaceState(null, '', url)
    } catch (e) {
      setFatal(t('app.exampleFailed', { id, error: String(e) }))
    }
  }, [t])

  const clearExample = useCallback(() => {
    setExample(null)
    const url = new URL(window.location.href)
    url.searchParams.delete('example')
    window.history.replaceState(null, '', url)
  }, [])

  // The registry's labels and help come back in the chosen language.  The
  // spec is only seeded the first time -- and only if an example has not
  // already filled it in while the catalogue was loading.
  useEffect(() => {
    api
      .meta(lang)
      .then((m) => {
        setMeta(m)
        setSpec((current) => current ?? defaultSpecFor(m, m.likelihoods[0].id))
      })
      .catch((e) => setFatal(String(e)))
  }, [lang])

  useEffect(() => {
    api.health().then(setToolchain).catch(() => setToolchain(null))
    api
      .examples()
      .then((list) => {
        setExamples(list)
        // `?example=<id>` opens straight into one, as faultree's GUI does --
        // handy for a link on a slide.
        const wanted = new URLSearchParams(window.location.search).get('example')
        if (wanted && list.some((e) => e.id === wanted)) void loadExample(wanted)
      })
      .catch(() => setExamples([]))
  }, [loadExample])

  // Re-resolve whenever the model changes.  The backend owns bound
  // intersection and prior truncation, so the UI asks rather than reimplements
  // -- there is exactly one place those rules live.
  useEffect(() => {
    if (!spec) return
    let cancelled = false
    const timer = setTimeout(() => {
      api
        .validateSpec(spec)
        .then((v) => !cancelled && setValidation(v))
        .catch(() => !cancelled && setValidation(null))
    }, 200)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [spec])

  const revalidateData = useCallback(
    async (nextRows: Row[]) => {
      if (!spec) return
      try {
        const result = await api.validateData(spec, nextRows)
        setDataErrors(result.ok ? [] : result.errors)
        setDataWarnings(result.warnings ?? [])
      } catch {
        setDataErrors([])
      }
    },
    [spec],
  )

  useEffect(() => {
    void revalidateData(rows)
  }, [rows, revalidateData])

  const likelihood = useMemo(
    () => meta?.likelihoods.find((l) => l.id === spec?.likelihood) ?? null,
    [meta, spec],
  )

  const onChangeLikelihood = useCallback(
    (id: string) => {
      if (!meta) return
      // Parameters, priors and required columns all change together, so the
      // spec is rebuilt rather than patched.  Data is kept when the columns
      // still apply, since re-entering a table is expensive.
      const next = defaultSpecFor(meta, id)
      const previous = meta.likelihoods.find((l) => l.id === spec?.likelihood)
      const target = meta.likelihoods.find((l) => l.id === id)
      const compatible =
        previous &&
        target &&
        previous.required_columns.join() === target.required_columns.join()
      if (!compatible) setRows([])
      setSpec(next)
      clearExample()
    },
    [meta, spec, clearExample],
  )

  const startRun = useCallback(async () => {
    if (!spec) return
    setRunSpec(spec)
    setRunValidation(validation)
    try {
      const started = await api.createRun(spec, rows, sampler)
      setRun(started)
    } catch (e) {
      setRun({
        id: '',
        status: 'failed',
        stage: 'resolve',
        message: t('app.couldNotStart'),
        fraction: null,
        error: e instanceof Error ? e.message : String(e),
        error_details: [],
        warnings: [],
        created_utc: '',
      })
    }
  }, [spec, rows, sampler, validation, t])

  if (fatal) {
    return (
      <div className="app app--message">
        <div className="banner banner--error">
          <strong>{t('app.unreachable')}</strong> {fatal}
          <div className="muted">
            {t('app.startWith')} <code>pirao gui</code>.
          </div>
        </div>
      </div>
    )
  }

  if (!meta || !spec || !likelihood) {
    return <div className="app app--message app--loading">{t('app.loading')}</div>
  }

  const specOk = validation?.ok ?? false
  const dataOk = dataErrors.length === 0
  const canRun = specOk && dataOk

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <svg className="brand__mark" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 5 C 8 5, 10 6, 13 12 S 18 19, 21 19" />
            <path className="brand__band" d="M3 3.5 C 9 3.5, 11 6, 14 11.5 S 19 17.5, 21 17.5" />
          </svg>
          <div>
            <div className="brand__name">pirão</div>
            <div className="brand__sub" title={t('app.sub')}>
              {lang === 'en' ? (
                <>
                  <b>P</b>robabilistic <b>I</b>nference for <b>R</b>eliability{' '}
                  <b>A</b>nalysis from <b>O</b>bserved data
                </>
              ) : (
                t('app.sub')
              )}
            </div>
          </div>
        </div>
        {examples.length > 0 && (
          <select
            className="control topbar__examples"
            aria-label={t('app.examplesAria')}
            value={example?.id ?? ''}
            onChange={(e) => {
              if (e.target.value) void loadExample(e.target.value)
              else clearExample()
            }}
          >
            <option value="">{t('app.examples')}</option>
            {examples.map((item) => (
              <option key={item.id} value={item.id}>
                {lang === 'pt' ? item.title_pt : item.title}
              </option>
            ))}
          </select>
        )}
        <div className="topbar__spacer" />
        {toolchain?.version && (
          <span className="topbar__version">pirão {toolchain.version}</span>
        )}
        <div className="seg" role="group" aria-label={t('app.lang')}>
          {(['pt', 'en'] as Lang[]).map((code) => (
            <button
              key={code}
              className={`seg__option${lang === code ? ' seg__option--on' : ''}`}
              aria-pressed={lang === code}
              onClick={() => setLang(code)}
            >
              {code.toUpperCase()}
            </button>
          ))}
        </div>
        <button className="primary" disabled={!canRun} onClick={startRun}>
          {t('app.run')}
        </button>
      </header>

      <div className="layout" style={{ gridTemplateColumns: `${sideWidth}px 6px minmax(0, 1fr)` }}>
        <aside className="side" aria-label={t('app.sideAria')}>
          <nav className="tabs side__tabs" role="tablist">
            {TABS.map((name) => {
              const problems =
                (name === 'Model' && !specOk) || (name === 'Data' && !dataOk)
              return (
                <button
                  key={name}
                  role="tab"
                  aria-selected={tab === name}
                  className={`tab${tab === name ? ' tab--active' : ''}${
                    problems ? ' tab--problem' : ''
                  }`}
                  onClick={() => setTab(name)}
                >
                  {t(`tab.${name}` as Key)}
                  {name === 'Data' && rows.length > 0 && (
                    <span className="tab__count">{rows.length}</span>
                  )}
                  {problems && <span className="tab__dot" aria-label={t('app.hasProblems')} />}
                </button>
              )
            })}
          </nav>

          <div className="side__body" key={loadSeq}>
            {tab === 'Model' && (
              <ModelPanel
                meta={meta}
                spec={spec}
                likelihood={likelihood}
                validation={validation}
                onChangeLikelihood={onChangeLikelihood}
                onChangeSpec={setSpec}
              />
            )}
            {tab === 'Data' && (
              <DataPanel
                likelihood={likelihood}
                rows={rows}
                errors={dataErrors}
                warnings={dataWarnings}
                onChangeRows={setRows}
              />
            )}
            {tab === 'Sampler' && (
              <SamplerPanel
                help={meta.sampler_help}
                sampler={sampler}
                onChange={setSampler}
              />
            )}
          </div>
        </aside>

        <div
          className="splitter"
          role="separator"
          aria-orientation="vertical"
          aria-label={t('app.resizeSide')}
          aria-valuenow={sideWidth}
          tabIndex={0}
          onPointerDown={(e) => {
            dragging.current = true
            e.currentTarget.setPointerCapture(e.pointerId)
          }}
          onPointerMove={(e) => {
            if (dragging.current) setSideWidth(clampSide(e.clientX))
          }}
          onPointerUp={(e) => {
            dragging.current = false
            e.currentTarget.releasePointerCapture(e.pointerId)
            storeSideWidth(sideWidth)
          }}
          onDoubleClick={() => {
            setSideWidth(SIDE_DEFAULT)
            storeSideWidth(SIDE_DEFAULT)
          }}
          onKeyDown={(e) => {
            const step = e.key === 'ArrowLeft' ? -24 : e.key === 'ArrowRight' ? 24 : 0
            if (!step) return
            e.preventDefault()
            const next = clampSide(sideWidth + step)
            setSideWidth(next)
            storeSideWidth(next)
          }}
        />

        <main className="main">
          {toolchain && !toolchain.ok && (
            <div className="banner banner--error">
              <strong>{t('app.noStan')}</strong> {toolchain.error}
              {toolchain.hint && <div className="muted">{toolchain.hint}</div>}
            </div>
          )}

          {example && <ExampleNote example={example} onClose={clearExample} />}

          {run && run.status === 'done' && run.summary ? (
            <ResultsPanel
              run={run}
              spec={runSpec ?? spec}
              validation={runValidation}
              priorLabels={Object.fromEntries(meta.priors.map((p) => [p.id, p.label]))}
            />
          ) : run ? (
            <RunPanel run={run} canRun={canRun} onStart={startRun} onFinished={setRun} />
          ) : (
            <section className="card ready">
              <h2>{t('ready.title')}</h2>
              <p className="muted">{t('ready.body')}</p>
              {!canRun && (
                <p className="muted small">
                  {t('ready.fixFirst', { tab: t(!specOk ? 'tab.Model' : 'tab.Data') })}
                </p>
              )}
              <button className="primary" disabled={!canRun} onClick={startRun}>
                {t('app.run')}
              </button>
            </section>
          )}
        </main>
      </div>
    </>
  )
}
