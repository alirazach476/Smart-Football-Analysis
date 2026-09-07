import type {
  AnalyticsBundle,
  JobDetail,
  JobListResponse,
  UploadResponse,
  VisualItem,
} from '../types/api'

const API = '/api/v1'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || res.statusText || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; models_available: boolean }>('/health'),

  listMatches: (params?: { status?: string; limit?: number }) => {
    const q = new URLSearchParams()
    if (params?.status) q.set('status', params.status)
    if (params?.limit) q.set('limit', String(params.limit))
    const qs = q.toString()
    return request<JobListResponse>(`/matches${qs ? `?${qs}` : ''}`)
  },

  getMatch: (id: string) => request<JobDetail>(`/matches/${id}`),

  uploadMatch: (file: File, options?: Record<string, unknown>) => {
    const form = new FormData()
    form.append('file', file)
    if (options) form.append('options', JSON.stringify(options))
    return request<UploadResponse>('/matches', { method: 'POST', body: form })
  },

  deleteMatch: (id: string) =>
    request<{ message: string }>(`/matches/${id}`, { method: 'DELETE' }),

  getAnalytics: (id: string, type: string) =>
    request<{ data: Record<string, unknown> }>(`/matches/${id}/analytics/${type}`),

  loadAllAnalytics: async (id: string): Promise<AnalyticsBundle> => {
    const types = ['possession', 'passes', 'shots', 'goals', 'speed', 'heatmaps'] as const
    const bundle: AnalyticsBundle = {}
    await Promise.all(
      types.map(async (t) => {
        try {
          const res = await api.getAnalytics(id, t)
          bundle[t] = res.data
        } catch {
          /* optional analytics */
        }
      }),
    )
    return bundle
  },

  listVisuals: (id: string) =>
    request<{ items: VisualItem[] }>(`/matches/${id}/visuals`),

  videoUrl: (id: string, kind: 'birdseye' | 'highlights' = 'birdseye') =>
    `${API}/matches/${id}/video?kind=${kind}`,

  visualUrl: (id: string, category: string, name: string) =>
    `${API}/matches/${id}/visuals/file?category=${category}&name=${encodeURIComponent(name)}`,

  reportUrl: (id: string, format: 'html' | 'json' = 'html') =>
    `${API}/matches/${id}/report?format=${format}`,
}
