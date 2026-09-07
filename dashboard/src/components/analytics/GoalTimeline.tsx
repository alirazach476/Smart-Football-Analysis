import { motion } from 'framer-motion'
import { Goal } from 'lucide-react'
import { Card } from '../ui/Card'
import type { GoalEvent } from '../../types/api'

interface Props {
  goals: GoalEvent[]
  finalScore?: Record<string, number>
}

export function GoalTimeline({ goals, finalScore }: Props) {
  const sorted = [...goals].sort((a, b) => a.timestamp_sec - b.timestamp_sec)

  return (
    <Card title="Goal Timeline" subtitle="Chronological match events">
      {finalScore && Object.keys(finalScore).length > 0 && (
        <div className="mb-6 flex items-center justify-center gap-6 rounded-xl bg-wc-deep/80 py-4">
          {Object.entries(finalScore).map(([team, score], i) => (
            <div key={team} className="text-center">
              <p className="text-xs text-wc-muted uppercase tracking-wider">{team}</p>
              <p className="text-4xl font-bold gold-gradient tabular-nums">{score}</p>
              {i === 0 && <span className="text-wc-muted text-2xl mx-4">—</span>}
            </div>
          ))}
        </div>
      )}
      {sorted.length === 0 ? (
        <p className="text-wc-muted text-center py-8">No goals detected</p>
      ) : (
        <div className="relative pl-6 border-l-2 border-wc-gold/40 space-y-6">
          {sorted.map((g, i) => (
            <motion.div
              key={g.goal_id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="relative"
            >
              <span className="absolute -left-[1.6rem] top-1 flex h-5 w-5 items-center justify-center rounded-full bg-wc-gold text-black">
                <Goal className="h-3 w-3" />
              </span>
              <div className="rounded-xl bg-wc-deep/60 p-4 border border-wc-border/50">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-semibold text-wc-green-light">{g.scoring_team}</span>
                  <span className="text-sm font-mono text-wc-gold">
                    {Math.floor(g.timestamp_sec / 60)}'
                    {String(Math.floor(g.timestamp_sec % 60)).padStart(2, '0')}
                  </span>
                </div>
                <p className="text-sm text-wc-muted mt-1">
                  Scorer: #{g.scorer_track_id ?? '?'} · Score {g.scoreline} ·{' '}
                  {g.goal_line} goal
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </Card>
  )
}
