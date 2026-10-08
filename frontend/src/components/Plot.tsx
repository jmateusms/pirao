/**
 * Minimal Plotly wrapper.
 *
 * Plotly is pulled in with a dynamic import so it lands in its own chunk: it
 * is several megabytes and is not needed until a run has finished, so making
 * the first paint wait for it would be a poor trade.
 *
 * The figure is updated with `Plotly.react` and torn down only when the
 * component unmounts.  That distinction matters for the 3D views: `purge`
 * followed by a fresh draw would throw away the camera, so every re-render
 * would spin a rotated surface back to its starting angle.
 */

import { useEffect, useRef, useState } from 'react'

import { BASE_LAYOUT, CONFIG } from './viz'

interface Props {
  data: unknown[]
  layout?: Record<string, unknown>
  config?: Record<string, unknown>
  height?: number
  /** Announced to screen readers in place of the canvas. */
  description?: string
}

interface PlotlyModule {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => void
  purge: (el: HTMLElement) => void
  Plots: { resize: (el: HTMLElement) => void }
}

let plotlyPromise: Promise<PlotlyModule> | null = null

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-dist-min').then(
      (module) => ((module as unknown as { default?: PlotlyModule }).default ??
        module) as unknown as PlotlyModule,
    )
  }
  return plotlyPromise
}

/**
 * Fold a chart's layout into the shared theme.
 *
 * `xaxis` and `yaxis` are merged a level deeper than the rest: a caller that
 * only wants to name an axis would otherwise replace the whole object and take
 * the hairline grid, the tick style and the fonts down with it.
 */
function merge(height: number, layout?: Record<string, unknown>) {
  const base = BASE_LAYOUT as Record<string, Record<string, unknown>>
  const given = (layout ?? {}) as Record<string, Record<string, unknown>>
  return {
    ...BASE_LAYOUT,
    height,
    ...layout,
    xaxis: { ...base.xaxis, ...given.xaxis },
    yaxis: { ...base.yaxis, ...given.yaxis },
  }
}

export function Plot({ data, layout, config, height = 320, description }: Props) {
  const node = useRef<HTMLDivElement>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let disposed = false
    const element = node.current
    if (!element) return

    loadPlotly()
      .then((Plotly) => {
        if (disposed) return
        Plotly.react(element, data, merge(height, layout), { ...CONFIG, ...config })
      })
      .catch(() => {
        if (!disposed) setFailed(true)
      })

    return () => {
      disposed = true
    }
  }, [data, layout, config, height])

  // `responsive` only follows the window; the panes can also be resized by
  // dragging the splitter, so watch the element itself.
  useEffect(() => {
    const element = node.current
    if (!element || typeof ResizeObserver === 'undefined') return
    let frame = 0
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        if (element.querySelector('.main-svg')) {
          loadPlotly().then((Plotly) => Plotly.Plots.resize(element)).catch(() => undefined)
        }
      })
    })
    observer.observe(element)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
    }
  }, [])

  // Teardown belongs to the component's life, not to each redraw.
  useEffect(() => {
    const element = node.current
    return () => {
      if (element) loadPlotly().then((Plotly) => Plotly.purge(element)).catch(() => undefined)
    }
  }, [])

  if (failed) {
    return <div className="placeholder">The plotting library failed to load.</div>
  }
  return (
    <div
      ref={node}
      className="plot"
      style={{ minHeight: height }}
      role="img"
      aria-label={description}
    />
  )
}
