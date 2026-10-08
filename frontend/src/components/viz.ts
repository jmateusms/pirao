/**
 * Chart tokens and the shared Plotly theme.
 *
 * One module owns every colour and every layout default the charts use, so a
 * plot cannot quietly drift into its own look.  The palette is not a matter of
 * taste: the categorical slots were run through a colour-vision validator and
 * kept only in an order that clears it.
 *
 *   slots 1-4 (blue, orange, aqua, violet), all pairs, on white:
 *     CVD separation      worst pair dE 9.2   (target >= 8)
 *     normal vision       worst pair dE 16.3  (floor 15)
 *   slots 1-8, adjacent pairs (lines):
 *     CVD separation      worst pair dE 9.2
 *     normal vision       worst pair dE 20.8
 *
 * Reordering these is therefore a measurement, not a preference.  Chains never
 * exceed four in the default sampler settings, which is why the four
 * all-pairs-safe hues come first -- overlaid trace lines are the one place
 * where any two series can end up adjacent on screen.
 */

/** Categorical identity: chains, and any other "which one is this" encoding. */
export const SERIES = [
  '#2a78d6', // blue
  '#eb6834', // orange
  '#1baf7a', // aqua
  '#4a3aa7', // violet
  '#e87ba4', // magenta
  '#008300', // green
  '#eda100', // yellow
  '#e34948', // red
] as const

export const BLUE = SERIES[0]
export const VIOLET = SERIES[3]

/** Reserved for the mission marker: a threshold, not a series. */
export const CRITICAL = '#d03b3b'

/** The blue series hue as a wash, for interval bands and histogram fills. */
export const BLUE_WASH = 'rgba(42, 120, 214, 0.16)'

/**
 * Sequential magnitude, one hue, light to dark.  Used for posterior density,
 * where the lightest step means "nothing here" and is meant to recede into the
 * surface.
 */
export const DENSITY_SCALE: [number, string][] = [
  [0, '#f6f9fe'],
  [0.15, '#cde2fb'],
  [0.3, '#9ec5f4'],
  [0.45, '#6da7ec'],
  [0.6, '#3987e5'],
  [0.75, '#256abf'],
  [0.9, '#184f95'],
  [1, '#0d366b'],
]

/**
 * The same ramp with the empty end made transparent.
 *
 * A filled contour paints its lowest band across the whole plotting rectangle,
 * which reads as a box drawn around the data rather than as "nothing here".
 * Letting the surface show through says the same thing without the box.
 */
export const DENSITY_SCALE_CLEAR: [number, string][] = [
  [0, 'rgba(246, 249, 254, 0)'],
  [0.15, '#cde2fb'],
  [0.3, '#9ec5f4'],
  [0.45, '#6da7ec'],
  [0.6, '#3987e5'],
  [0.75, '#256abf'],
  [0.9, '#184f95'],
  [1, '#0d366b'],
]

const INK = '#2b2b2b'
const MUTED = '#5e5e5a'
const LINE = '#d8d4cc'
const SURFACE = '#ffffff'

const FONT = {
  family: 'system-ui, -apple-system, "Segoe UI", sans-serif',
  size: 12,
  color: MUTED,
}

/** Axis chrome: solid hairlines one step off the surface, never dashed. */
const AXIS = {
  gridcolor: LINE,
  gridwidth: 1,
  zeroline: false,
  linecolor: LINE,
  ticks: 'outside' as const,
  ticklen: 4,
  tickcolor: LINE,
  automargin: true,
  title: { font: { ...FONT, color: MUTED, size: 12 } },
}

export const BASE_LAYOUT: Record<string, unknown> = {
  margin: { l: 58, r: 18, t: 14, b: 48 },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: FONT,
  showlegend: false,
  hoverlabel: {
    bgcolor: SURFACE,
    bordercolor: LINE,
    font: { ...FONT, color: INK },
  },
  xaxis: AXIS,
  yaxis: AXIS,
  legend: {
    orientation: 'h',
    yanchor: 'bottom',
    y: 1.02,
    x: 0,
    font: { ...FONT, color: INK },
  },
}

/** The same chrome for a 3D scene, where Plotly names everything differently. */
export const SCENE_AXIS = {
  gridcolor: LINE,
  zerolinecolor: LINE,
  showbackground: true,
  backgroundcolor: 'rgba(0, 0, 0, 0.015)',
  tickfont: FONT,
  titlefont: { ...FONT, color: MUTED },
}

export const CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: [
    'select2d',
    'lasso2d',
    'autoScale2d',
    'toggleSpikelines',
  ],
}

/** Format a number for a label or a table cell: readable before precise. */
export function fmt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(2)
  return String(Number(value.toPrecision(4)))
}
