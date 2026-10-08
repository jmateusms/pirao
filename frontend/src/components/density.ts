/**
 * A smoothed two-dimensional density from posterior draws.
 *
 * Computed here rather than left to Plotly's `histogram2dcontour` for one
 * reason: the same grid then feeds both the 3D surface and the 2D contour, so
 * switching between them cannot change what is being shown -- only how it is
 * drawn.
 *
 * Binning followed by a Gaussian blur is a kernel density estimate evaluated
 * on a lattice.  It is the cheap version, and it is the right one here: the
 * grid is 64 x 64 and the input is a few thousand draws, so the difference
 * from an exact KDE is far below the width of a plotted contour.
 */

export interface Density2D {
  x: number[]
  y: number[]
  /** `z[row][col]`, row indexing `y` and column indexing `x`. */
  z: number[][]
}

const BINS = 64
/** Blur width, in bins.  Wide enough to hide the lattice, narrow enough to
 *  leave a genuine ridge looking like a ridge. */
const SIGMA = 1.15

export function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN
  const position = (sorted.length - 1) * p
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return sorted[lower]
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
}

/** Trim the extreme tails so a handful of draws cannot flatten the picture. */
function robustRange(values: number[]): [number, number] {
  const sorted = [...values].sort((a, b) => a - b)
  let lo = quantile(sorted, 0.002)
  let hi = quantile(sorted, 0.998)
  if (!(hi > lo)) {
    const centre = sorted[Math.floor(sorted.length / 2)] ?? 0
    lo = centre - 0.5
    hi = centre + 0.5
  }
  const pad = (hi - lo) * 0.04
  return [lo - pad, hi + pad]
}

function gaussianKernel(sigma: number): number[] {
  const radius = Math.max(1, Math.ceil(sigma * 3))
  const kernel: number[] = []
  let total = 0
  for (let offset = -radius; offset <= radius; offset += 1) {
    const weight = Math.exp(-(offset * offset) / (2 * sigma * sigma))
    kernel.push(weight)
    total += weight
  }
  return kernel.map((weight) => weight / total)
}

/** Blur along one axis; called twice, because a Gaussian is separable. */
function blurRows(grid: number[][], kernel: number[]): number[][] {
  const radius = (kernel.length - 1) / 2
  return grid.map((row) =>
    row.map((_, index) => {
      let sum = 0
      for (let k = -radius; k <= radius; k += 1) {
        // Clamp at the edges: mass that falls outside stays at the boundary
        // rather than wrapping round to the opposite corner.
        const source = Math.min(row.length - 1, Math.max(0, index + k))
        sum += row[source] * kernel[k + radius]
      }
      return sum
    }),
  )
}

function transpose(grid: number[][]): number[][] {
  return grid[0].map((_, col) => grid.map((row) => row[col]))
}

export function density2d(xs: number[], ys: number[]): Density2D | null {
  const n = Math.min(xs.length, ys.length)
  if (n < 8) return null

  const [xLo, xHi] = robustRange(xs)
  const [yLo, yHi] = robustRange(ys)
  const dx = (xHi - xLo) / BINS
  const dy = (yHi - yLo) / BINS
  if (!(dx > 0) || !(dy > 0)) return null

  const counts: number[][] = Array.from({ length: BINS }, () =>
    new Array<number>(BINS).fill(0),
  )
  let kept = 0
  for (let i = 0; i < n; i += 1) {
    const col = Math.floor((xs[i] - xLo) / dx)
    const row = Math.floor((ys[i] - yLo) / dy)
    if (col < 0 || col >= BINS || row < 0 || row >= BINS) continue
    counts[row][col] += 1
    kept += 1
  }
  if (kept === 0) return null

  const kernel = gaussianKernel(SIGMA)
  let z = blurRows(counts, kernel)
  z = transpose(blurRows(transpose(z), kernel))

  // Counts per cell become a density per unit area, so the height means the
  // same thing whatever the parameter's scale happens to be.
  const scale = 1 / (kept * dx * dy)
  z = z.map((row) => row.map((value) => value * scale))

  return {
    x: Array.from({ length: BINS }, (_, i) => xLo + (i + 0.5) * dx),
    y: Array.from({ length: BINS }, (_, i) => yLo + (i + 0.5) * dy),
    z,
  }
}

/** Evenly spread subsample, for the scatter overlay on the 2D view. */
export function thin<T>(values: T[], limit: number): T[] {
  if (values.length <= limit) return values
  const step = values.length / limit
  return Array.from({ length: limit }, (_, i) => values[Math.floor(i * step)])
}
