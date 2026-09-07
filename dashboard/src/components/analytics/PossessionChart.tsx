import { motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { Card } from '../ui/Card'

interface Props {
  teamA: number
  teamB: number
  loose?: number
  contested?: number
  totalPossessionSec?: number
}

const COLORS = ['#7ed321', '#2196f3', '#2a2a2a', '#d4af37']

export function PossessionChart({
  teamA,
  teamB,
  loose = 0,
  contested = 0,
  totalPossessionSec,
}: Props) {
  // teamA/teamB are % of assigned time; loose/contested are % of total time.
  // Convert team shares to total-time percentages for a coherent pie chart.
  const assignedShare = Math.max(0, 100 - loose - contested)
  const teamATotal = (teamA * assignedShare) / 100
  const teamBTotal = (teamB * assignedShare) / 100

  const data = [
    { name: 'Team A', value: teamATotal },
    { name: 'Team B', value: teamBTotal },
    ...(loose > 0 ? [{ name: 'Loose', value: loose }] : []),
    ...(contested > 0 ? [{ name: 'Contested', value: contested }] : []),
  ].filter((d) => d.value > 0.1)

  const lowConfidence =
    (totalPossessionSec !== undefined && totalPossessionSec < 5) || loose > 60

  return (
    <Card title="Possession" subtitle="Ball control share">
      {lowConfidence && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-wc-gold/40 bg-wc-gold/10 px-3 py-2 text-xs text-wc-gold">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <span>
            Low ball-tracking confidence on this clip ({totalPossessionSec?.toFixed(1) ?? '?'}s
            assigned). Re-upload after the latest fixes, or use a clearer wide-angle video.
          </span>
        </div>
      )}
      <div className="flex flex-col sm:flex-row items-center gap-6">
        <div className="h-48 w-48 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data.length ? data : [{ name: 'No data', value: 100 }]}
                cx="50%"
                cy="50%"
                innerRadius={52}
                outerRadius={72}
                paddingAngle={3}
                dataKey="value"
              >
                {(data.length ? data : [{ name: 'No data', value: 100 }]).map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip
                formatter={(v) => [`${Number(v).toFixed(1)}%`, '']}
                contentStyle={{
                  background: '#141414',
                  border: '1px solid #2a2a2a',
                  borderRadius: 8,
                  color: '#fff',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="flex-1 space-y-3 w-full">
          {data.map((d, i) => (
            <motion.div
              key={d.name}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center gap-2">
                  <span
                    className="h-3 w-3 rounded-full"
                    style={{ background: COLORS[i % COLORS.length] }}
                  />
                  {d.name}
                </span>
                <span className="font-semibold tabular-nums">{d.value.toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full bg-wc-deep overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, d.value)}%` }}
                  transition={{ duration: 0.8, delay: i * 0.1 }}
                  className="h-full rounded-full"
                  style={{ background: COLORS[i % COLORS.length] }}
                />
              </div>
            </motion.div>
          ))}
          {assignedShare > 0 && (
            <p className="text-xs text-wc-muted pt-1">
              Team split (when possessed): Team A {teamA.toFixed(0)}% · Team B{' '}
              {teamB.toFixed(0)}%
            </p>
          )}
        </div>
      </div>
    </Card>
  )
}
