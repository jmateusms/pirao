/**
 * The inline density preview beside each parameter's controls.
 *
 * The curve shown is the *truncated* density the backend computed, not the
 * family the user picked -- those differ whenever a bound cuts into the
 * support, and the difference is the whole point of showing it.
 *
 * It redraws whenever the backend returns a new preview, which is at most
 * every 200ms while a hyperparameter is being tuned (App debounces the
 * validation call), so a real chart is affordable here and buys a readable
 * axis and a hover readout that the hand-drawn SVG never had.
 */

import { useMemo } from 'react'

import { Plot } from './Plot'
import { BLUE, BLUE_WASH } from './viz'

interface Props {
  x: number[]
  pdf: number[]
  /** Marked on the curve, so the summary numbers have a place on the picture. */
  q05?: number | null
  q95?: number | null
  height?: number
}

export function PriorCurve({ x, pdf, q05, q95, height = 108 }: Props) {
  const data = useMemo(() => {
    const y = pdf.map((value) => (Number.isFinite(value) ? value : 0))
    const traces: unknown[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x,
        y,
        fill: 'tozeroy',
        fillcolor: BLUE_WASH,
        line: { color: BLUE, width: 2, shape: 'spline', smoothing: 0.4 },
        hovertemplate: '%{x:.4g}<br>density %{y:.3g}<extra></extra>',
      },
    ]

    // The 90% interval, drawn as the slice of the curve it actually covers
    // rather than as two bare rules.
    if (q05 != null && q95 != null && Number.isFinite(q05) && Number.isFinite(q95)) {
      const inside = x
        .map((value, index) => ({ value, density: y[index] }))
        .filter((point) => point.value >= q05 && point.value <= q95)
      if (inside.length > 1) {
        traces.unshift({
          type: 'scatter',
          mode: 'lines',
          x: inside.map((p) => p.value),
          y: inside.map((p) => p.density),
          fill: 'tozeroy',
          fillcolor: BLUE_WASH,
          line: { color: 'transparent' },
          hoverinfo: 'skip',
        })
      }
    }
    return traces
  }, [x, pdf, q05, q95])

  const layout = useMemo(
    () => ({
      margin: { l: 8, r: 8, t: 6, b: 24 },
      xaxis: { showgrid: false, ticks: 'outside', ticklen: 3, nticks: 5 },
      yaxis: { visible: false, rangemode: 'tozero' },
      hovermode: 'x',
    }),
    [],
  )

  const config = useMemo(() => ({ displayModeBar: false }), [])

  if (x.length < 2) return <div className="placeholder">No preview</div>

  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      height={height}
      description="Prior density, after the bounds have been applied"
    />
  )
}
