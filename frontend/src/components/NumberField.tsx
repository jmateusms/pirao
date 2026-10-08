import { useEffect, useState } from 'react'

interface Props {
  value: number | null
  onChange: (value: number | null) => void
  /** Empty commits `null` instead of being left uncommitted. */
  nullable?: boolean
  /** Rounds on commit; does not restrict which characters can be typed. */
  integer?: boolean
  /** Applied on blur only -- clamping while typing would make "1"
   *  unreachable when min is 10. */
  min?: number
  max?: number
  placeholder?: string
  className?: string
  id?: string
  disabled?: boolean
  ariaLabel?: string
}

// A "complete" number: no trailing "." or "e", no bare "-" or ".". Anything
// else ("", "-", ".", "1.", "1e", "-1e-") is a mid-typing state rather than
// a mistake, and is left on screen instead of being forced into a number.
const COMPLETE_NUMBER = /^-?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$/

function format(value: number | null): string {
  return value === null || value === undefined ? '' : String(value)
}

/**
 * What `buffer` would commit to, or `undefined` if it isn't there yet.
 * `undefined` covers both plain garbage ("1e-") and, for a non-nullable
 * field, an empty box -- the state right after clearing a leading 0 to type
 * something else, which must not be read back as "0".
 */
function parse(buffer: string, nullable: boolean): number | null | undefined {
  if (buffer === '') return nullable ? null : undefined
  const normalized = buffer.replace(',', '.')
  return COMPLETE_NUMBER.test(normalized) ? Number(normalized) : undefined
}

export function NumberField({
  value,
  onChange,
  nullable = false,
  integer = false,
  min,
  max,
  placeholder,
  className,
  id,
  disabled,
  ariaLabel,
}: Props) {
  const [buffer, setBuffer] = useState(() => format(value))

  // Re-sync only when `value` itself changes and disagrees with what's on
  // screen -- an external change such as switching prior family or clicking
  // a preset. Checking this on every keystroke (rather than just when the
  // prop changes) would fight the user: a partial, uncommitted buffer -- or
  // an intentionally empty, non-nullable one -- has to survive the
  // re-renders that follow our own onChange.
  useEffect(() => {
    const parsed = parse(buffer, nullable)
    if (parsed !== undefined && parsed !== value) setBuffer(format(value))
  }, [value])

  const commit = (raw: string) => {
    const parsed = parse(raw, nullable)
    if (parsed === undefined) return
    onChange(parsed === null ? null : integer ? Math.round(parsed) : parsed)
  }

  return (
    <input
      id={id}
      type="text"
      inputMode="decimal"
      className={className}
      placeholder={placeholder}
      disabled={disabled}
      aria-label={ariaLabel}
      value={buffer}
      onChange={(e) => {
        setBuffer(e.target.value)
        commit(e.target.value)
      }}
      onBlur={() => {
        const parsed = parse(buffer, nullable)
        if (parsed === undefined) {
          setBuffer(format(value)) // nothing usable was typed -- revert
          return
        }
        if (parsed === null) return // nullable + empty, already committed
        let next = integer ? Math.round(parsed) : parsed
        if (min !== undefined) next = Math.max(min, next)
        if (max !== undefined) next = Math.min(max, next)
        setBuffer(String(next))
        onChange(next)
      }}
    />
  )
}
