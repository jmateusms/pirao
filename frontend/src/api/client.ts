/** Thin fetch wrapper over the backend. */

import type {
  DataValidation,
  Example,
  ExampleSummary,
  Meta,
  ModelSpec,
  Posterior,
  RunState,
  Row,
  SamplerConfig,
  SpecValidation,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
    ...init,
  })
  if (!response.ok) {
    let detail: unknown
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      detail = body.detail ?? body
      if (typeof detail === 'string') message = detail
    } catch {
      /* a non-JSON error body is not worth failing over */
    }
    throw new ApiError(message, response.status, detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  meta: (lang: 'pt' | 'en' = 'en') => request<Meta>(`/meta?lang=${lang}`),

  health: () =>
    request<{ ok: boolean; version?: string; error?: string; hint?: string }>(
      '/health',
    ),

  examples: () => request<ExampleSummary[]>('/examples'),

  example: (id: string) => request<Example>(`/examples/${encodeURIComponent(id)}`),

  validateSpec: (spec: ModelSpec) =>
    request<SpecValidation>('/specs/validate', {
      method: 'POST',
      body: JSON.stringify(spec),
    }),

  validateData: (spec: ModelSpec, rows: Row[]) =>
    request<DataValidation>('/data/validate', {
      method: 'POST',
      body: JSON.stringify({ spec, rows }),
    }),

  async parseFile(file: File): Promise<{ rows: Row[]; ignored_columns: string[] }> {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch(`${BASE}/data/parse`, {
      method: 'POST',
      body: form,
    })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new ApiError(body.detail ?? 'That file could not be read.', response.status)
    }
    return response.json()
  },

  createRun: (spec: ModelSpec, rows: Row[], sampler: SamplerConfig) =>
    request<RunState>('/runs', {
      method: 'POST',
      body: JSON.stringify({ spec, rows, sampler }),
    }),

  getRun: (id: string) => request<RunState>(`/runs/${id}`),

  cancelRun: (id: string) =>
    request<{ cancelled: boolean }>(`/runs/${id}`, { method: 'DELETE' }),

  posterior: (id: string) => request<Posterior>(`/runs/${id}/posterior`),

  async source(id: string): Promise<string> {
    const response = await fetch(`${BASE}/runs/${id}/source`)
    if (!response.ok) throw new ApiError('No generated source yet.', response.status)
    return response.text()
  },

  templateUrl: (likelihood: string, format: 'csv' | 'xlsx') =>
    `${BASE}/templates/${likelihood}.${format}`,

  drawsUrl: (id: string, format: 'csv' | 'parquet') =>
    `${BASE}/runs/${id}/draws?format=${format}`,

  bundleUrl: (id: string) => `${BASE}/runs/${id}/bundle`,

  /**
   * Subscribe to a run's stage changes.
   *
   * Compilation is the stage that actually takes time -- sampling these models
   * is a matter of milliseconds -- so the stream exists mainly to show that
   * something is happening while the C++ toolchain works.
   */
  streamRun(
    id: string,
    onMessage: (message: Record<string, unknown>) => void,
  ): () => void {
    const source = new EventSource(`${BASE}/runs/${id}/events`)
    source.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        onMessage(message)
        if (message.kind === 'end') source.close()
      } catch {
        /* ignore a malformed frame rather than tearing down the stream */
      }
    }
    source.onerror = () => source.close()
    return () => source.close()
  },
}
