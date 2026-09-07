import { motion } from 'framer-motion'
import { Download, Loader2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import { GoalTimeline } from '../components/analytics/GoalTimeline'
import { HeatmapPanel } from '../components/analytics/HeatmapPanel'
import { MatchStatus } from '../components/analytics/MatchStatus'
import { PassNetworkViz } from '../components/analytics/PassNetworkViz'
import { PlayerStatsTable } from '../components/analytics/PlayerStatsTable'
import { PossessionChart } from '../components/analytics/PossessionChart'
import { ProcessedVideo } from '../components/analytics/ProcessedVideo'
import { ShotMapViz } from '../components/analytics/ShotMapViz'
import { SpeedChart } from '../components/analytics/SpeedChart'
import { VisualGallery } from '../components/analytics/VisualGallery'
import { useJobPolling } from '../hooks/useJobPolling'
import type {
  AnalyticsBundle,
  GoalEvent,
  PassEvent,
  PlayerPossession,
  PlayerSpeed,
  ShotEvent,
} from '../types/api'

export function MatchDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { job, error } = useJobPolling(id ?? null)
  const [analytics, setAnalytics] = useState<AnalyticsBundle | null>(null)
  const [loadingAnalytics, setLoadingAnalytics] = useState(false)

  useEffect(() => {
    if (!id || job?.status !== 'completed') return
    setLoadingAnalytics(true)
    api
      .loadAllAnalytics(id)
      .then(setAnalytics)
      .finally(() => setLoadingAnalytics(false))
  }, [id, job?.status])

  if (!id) return <p className="p-8 text-wc-coral">Invalid match ID</p>
  if (error) return <p className="p-8 text-wc-coral">{error}</p>
  if (!job) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-10 w-10 text-wc-gold animate-spin" />
      </div>
    )
  }

  const possession = analytics?.possession as {
    summary?: {
      team_a_pct?: number
      team_b_pct?: number
      loose_pct?: number
      contested_pct?: number
      total_possession_sec?: number
    }
    players?: PlayerPossession[]
  } | undefined

  const speedData = analytics?.speed as { players?: PlayerSpeed[] } | undefined
  const passesData = analytics?.passes as { passes?: PassEvent[] } | undefined
  const shotsData = analytics?.shots as { shots?: ShotEvent[] } | undefined
  const goalsData = analytics?.goals as {
    goals?: GoalEvent[]
    final_score?: Record<string, number>
  } | undefined

  const isReady = job.status === 'completed'

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
      <MatchStatus job={job} />

      {isReady && (
        <div className="mb-6 flex flex-wrap gap-3">
          <a
            href={api.reportUrl(id, 'html')}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-lg bg-wc-panel border border-wc-border px-4 py-2 text-sm hover:border-wc-gold transition"
          >
            <Download className="h-4 w-4" />
            Download Report
          </a>
        </div>
      )}

      {!isReady && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card-glass rounded-2xl p-12 text-center"
        >
          <Loader2 className="mx-auto h-12 w-12 text-wc-gold animate-spin mb-4" />
          <p className="text-lg font-medium">Analysis in progress…</p>
          <p className="text-wc-muted text-sm mt-2">
            YOLOv8 is processing your match. This page updates automatically.
          </p>
        </motion.div>
      )}

      {isReady && loadingAnalytics && (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 text-wc-gold animate-spin" />
        </div>
      )}

      {isReady && !loadingAnalytics && (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="lg:col-span-2">
            <ProcessedVideo jobId={id} />
          </div>

          <PossessionChart
            teamA={possession?.summary?.team_a_pct ?? job.summary?.possession?.team_a_pct ?? 50}
            teamB={possession?.summary?.team_b_pct ?? job.summary?.possession?.team_b_pct ?? 50}
            loose={possession?.summary?.loose_pct ?? job.summary?.possession?.loose_pct}
            contested={possession?.summary?.contested_pct}
            totalPossessionSec={possession?.summary?.total_possession_sec}
          />

          <GoalTimeline
            goals={goalsData?.goals ?? []}
            finalScore={goalsData?.final_score ?? job.summary?.goals?.final_score}
          />

          <div className="lg:col-span-2">
            <PlayerStatsTable
              possessionPlayers={possession?.players}
              speedPlayers={speedData?.players}
            />
          </div>

          <SpeedChart players={speedData?.players ?? []} />

          <ShotMapViz shots={shotsData?.shots ?? []} />

          <PassNetworkViz passes={passesData?.passes ?? []} team="Team A" />
          <PassNetworkViz passes={passesData?.passes ?? []} team="Team B" />

          <div className="lg:col-span-2">
            <HeatmapPanel jobId={id} />
          </div>

          <VisualGallery jobId={id} category="pass_network" title="Pass Network (PNG)" />
          <VisualGallery jobId={id} category="shot_map" title="Shot Map (PNG)" />
        </div>
      )}
    </div>
  )
}
