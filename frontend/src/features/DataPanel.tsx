import {
  DataEditor,
  GridCellKind,
  type EditableGridCell,
  type GridCell,
  type GridColumn,
  type Item,
} from '@glideapps/glide-data-grid'
import '@glideapps/glide-data-grid/dist/index.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { api } from '../api/client'
import type { CellError, LikelihoodMeta, Row } from '../api/types'

interface Props {
  likelihood: LikelihoodMeta
  rows: Row[]
  errors: CellError[]
  warnings: string[]
  onChangeRows: (rows: Row[]) => void
}

/** Columns are derived from the likelihood, so only relevant ones are shown. */
function columnsFor(likelihood: LikelihoodMeta): string[] {
  return ['device', ...likelihood.required_columns, 'relevance']
}

const INTEGER_COLUMNS = new Set(['n', 'failure', 'failures'])

/** Narrowest a column may become when the widths are shared out. */
const MIN_COLUMN_WIDTH = 96
/** The numbered gutter down the left, which is not one of our columns. */
const ROW_MARKER_WIDTH = 44

// An empty table used to render a single row on an otherwise empty page.
// Twelve gives a real sense of "a table" without a mostly-empty grid
// dominating the screen.
const MIN_VISIBLE_ROWS = 12
const ROW_HEIGHT = 34
const HEADER_HEIGHT = 60
const DEFAULT_GRID_HEIGHT = HEADER_HEIGHT + MIN_VISIBLE_ROWS * ROW_HEIGHT
const MIN_GRID_HEIGHT = HEADER_HEIGHT + 3 * ROW_HEIGHT
const MAX_GRID_HEIGHT = 900

// Module-level rather than component state, so a resize survives switching
// to another tab and back -- DataPanel unmounts on every tab change, and
// "sticks for the session" means the drag has to outlive that.
let sessionGridHeight = DEFAULT_GRID_HEIGHT

export function DataPanel({
  likelihood,
  rows,
  errors,
  warnings,
  onChangeRows,
}: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const names = useMemo(() => columnsFor(likelihood), [likelihood])

  const [gridHeight, setGridHeight] = useState(sessionGridHeight)
  const dragStart = useRef<{ y: number; height: number } | null>(null)

  const resize = (next: number) => {
    const clamped = Math.min(MAX_GRID_HEIGHT, Math.max(MIN_GRID_HEIGHT, next))
    sessionGridHeight = clamped
    setGridHeight(clamped)
  }

  /** Row/column pairs the backend rejected, for per-cell highlighting. */
  const badCells = useMemo(() => {
    const set = new Set<string>()
    for (const error of errors) {
      if (error.row !== null && error.column) set.add(`${error.row - 1}:${error.column}`)
    }
    return set
  }, [errors])

  const tableErrors = useMemo(() => errors.filter((e) => e.row === null), [errors])

  // The grid draws to a canvas, so the columns have to be given pixel widths;
  // measuring the card and sharing them out is what stops a four-column table
  // huddling against the left edge of an otherwise empty card.
  const wrap = useRef<HTMLDivElement>(null)
  const [available, setAvailable] = useState(0)

  useEffect(() => {
    const element = wrap.current
    if (!element) return
    setAvailable(element.getBoundingClientRect().width)
    const observer = new ResizeObserver(([entry]) =>
      setAvailable(entry.contentRect.width),
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const columns: GridColumn[] = useMemo(() => {
    const weight = (name: string) => (name === 'device' ? 1.5 : 1)
    const total = names.reduce((sum, name) => sum + weight(name), 0)
    const usable = Math.max(0, available - ROW_MARKER_WIDTH)
    return names.map((name) => ({
      id: name,
      title: name,
      width: Math.max(MIN_COLUMN_WIDTH, Math.floor((usable * weight(name)) / total)),
    }))
  }, [names, available])

  const getCell = useCallback(
    ([col, row]: Item): GridCell => {
      const name = names[col]
      const value = rows[row]?.[name]
      const invalid = badCells.has(`${row}:${name}`)
      const theme = invalid
        ? { bgCell: '#fdeaea', textDark: '#a11' }
        : undefined

      if (name === 'device') {
        return {
          kind: GridCellKind.Text,
          data: value == null ? '' : String(value),
          displayData: value == null ? '' : String(value),
          allowOverlay: true,
          themeOverride: theme,
        }
      }
      return {
        kind: GridCellKind.Number,
        data: value == null || value === '' ? undefined : Number(value),
        displayData: value == null || value === '' ? '' : String(value),
        allowOverlay: true,
        themeOverride: theme,
      }
    },
    [names, rows, badCells],
  )

  const onCellEdited = useCallback(
    ([col, row]: Item, newValue: EditableGridCell) => {
      const name = names[col]
      const next = rows.map((r) => ({ ...r }))
      while (next.length <= row) next.push(emptyRow(names))

      if (newValue.kind === GridCellKind.Text) {
        next[row][name] = newValue.data
      } else if (newValue.kind === GridCellKind.Number) {
        const raw = newValue.data
        next[row][name] =
          raw === undefined || Number.isNaN(raw)
            ? null
            : INTEGER_COLUMNS.has(name)
              ? Math.round(raw)
              : raw
      }
      onChangeRows(next)
    },
    [names, rows, onChangeRows],
  )

  // Pasting a block from Excel is the main way a real table gets in here, so
  // the grid must accept more rows than it currently shows.
  const onPaste = useCallback(
    (target: Item, values: readonly (readonly string[])[]) => {
      const [startCol, startRow] = target
      const next = rows.map((r) => ({ ...r }))
      values.forEach((line, lineIndex) => {
        const rowIndex = startRow + lineIndex
        while (next.length <= rowIndex) next.push(emptyRow(names))
        line.forEach((cell, cellIndex) => {
          const name = names[startCol + cellIndex]
          if (!name) return
          const text = cell.trim()
          if (name === 'device') {
            next[rowIndex][name] = text
          } else {
            const parsed = Number(text.replace(',', '.'))
            next[rowIndex][name] = text === '' || Number.isNaN(parsed) ? null : parsed
          }
        })
      })
      onChangeRows(next)
      return false // we applied it ourselves
    },
    [names, rows, onChangeRows],
  )

  const upload = async (file: File) => {
    setUploadError(null)
    try {
      const parsed = await api.parseFile(file)
      onChangeRows(parsed.rows.map((r) => pick(r, names)))
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="stack">
      <section className="card">
        <div className="card__head">
          <h2>Data</h2>
          <div className="row">
            <a
              className="chip"
              href={api.templateUrl(likelihood.id, 'csv')}
              download
            >
              Template (CSV)
            </a>
            <a
              className="chip"
              href={api.templateUrl(likelihood.id, 'xlsx')}
              download
            >
              Template (Excel)
            </a>
            <button className="chip" onClick={() => fileInput.current?.click()}>
              Upload…
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,.xlsx,.xlsm,.xltx"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) void upload(file)
                e.target.value = ''
              }}
            />
          </div>
        </div>
        <p className="muted">
          One row per unit. <code>failure = 1</code> means the unit failed;{' '}
          <code>0</code> means it survived and the row is censored.{' '}
          <code>relevance</code> is optional and defaults to 1. You can paste a
          block straight from Excel.
        </p>

        {uploadError && <div className="banner banner--error">{uploadError}</div>}
        {tableErrors.map((error) => (
          <div className="banner banner--error" key={error.message}>
            {error.message}
          </div>
        ))}

        <div className="grid-wrap" ref={wrap}>
          <DataEditor
            columns={columns}
            rows={Math.max(rows.length, MIN_VISIBLE_ROWS)}
            getCellContent={getCell}
            onCellEdited={onCellEdited}
            onPaste={onPaste}
            getCellsForSelection={true}
            rowMarkers="number"
            smoothScrollY
            trailingRowOptions={{ hint: 'New row…', sticky: true }}
            onRowAppended={() => {
              onChangeRows([...rows, emptyRow(names)])
            }}
            height={gridHeight}
          />
          <div
            className="grid-resize"
            role="separator"
            aria-orientation="horizontal"
            aria-label="Resize the table"
            tabIndex={0}
            style={{ touchAction: 'none' }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              dragStart.current = { y: e.clientY, height: gridHeight }
            }}
            onPointerMove={(e) => {
              if (!dragStart.current) return
              resize(dragStart.current.height + (e.clientY - dragStart.current.y))
            }}
            onPointerUp={(e) => {
              if (dragStart.current) e.currentTarget.releasePointerCapture(e.pointerId)
              dragStart.current = null
            }}
            onPointerCancel={(e) => {
              if (dragStart.current) e.currentTarget.releasePointerCapture(e.pointerId)
              dragStart.current = null
            }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowUp') {
                e.preventDefault()
                resize(gridHeight - ROW_HEIGHT)
              } else if (e.key === 'ArrowDown') {
                e.preventDefault()
                resize(gridHeight + ROW_HEIGHT)
              }
            }}
          />
        </div>

        <div className="row row--between">
          <span className="muted small">
            {rows.length} row{rows.length === 1 ? '' : 's'}
          </span>
          <div className="row">
            <button
              className="chip"
              onClick={() => onChangeRows([...rows, emptyRow(names)])}
            >
              Add row
            </button>
            <button
              className="chip"
              disabled={rows.length === 0}
              onClick={() => onChangeRows(rows.slice(0, -1))}
            >
              Remove last
            </button>
            <button
              className="chip"
              disabled={rows.length === 0}
              onClick={() => onChangeRows([])}
            >
              Clear
            </button>
          </div>
        </div>
      </section>

      {errors.filter((e) => e.row !== null).length > 0 && (
        <section className="card card--error">
          <h3>Problems to fix</h3>
          <ul className="issues">
            {errors
              .filter((e) => e.row !== null)
              .map((e, i) => (
                <li key={i}>
                  <strong>Row {e.row}</strong>
                  {e.column ? ` · ${e.column}` : ''} — {e.message}
                </li>
              ))}
          </ul>
        </section>
      )}

      {warnings.map((warning) => (
        <div className="banner banner--warn" key={warning}>
          {warning}
        </div>
      ))}
    </div>
  )
}

function emptyRow(names: string[]): Row {
  const row: Row = {}
  for (const name of names) row[name] = name === 'relevance' ? 1 : null
  return row
}

function pick(row: Row, names: string[]): Row {
  const out: Row = {}
  for (const name of names) out[name] = row[name] ?? null
  return out
}
