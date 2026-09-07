import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { JobDetail } from '../types/api'

export function useJobPolling(jobId: string | null, intervalMs = 2000) {
  const [job, setJob] = useState<JobDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)

  const refresh = useCallback(async () => {
    if (!jobId) return
    try {
      setLoading(true)
      const data = await api.getMatch(jobId)
      setJob(data)
      setError(null)
      return data
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load match')
      return null
    } finally {
      setLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    if (!jobId) {
      setJob(null)
      return
    }
    refresh()
    timer.current = setInterval(async () => {
      const data = await api.getMatch(jobId)
      setJob(data)
      if (data.status === 'completed' || data.status === 'failed') {
        if (timer.current) clearInterval(timer.current)
      }
    }, intervalMs)
    return () => {
      if (timer.current) clearInterval(timer.current)
    }
  }, [jobId, intervalMs, refresh])

  return { job, error, loading, refresh }
}
