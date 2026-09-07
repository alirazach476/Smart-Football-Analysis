import { motion } from 'framer-motion'
import { Card } from '../ui/Card'
import type { PlayerPossession, PlayerSpeed } from '../../types/api'

interface Props {
  possessionPlayers?: PlayerPossession[]
  speedPlayers?: PlayerSpeed[]
}

export function PlayerStatsTable({ possessionPlayers = [], speedPlayers = [] }: Props) {
  const speedMap = new Map(speedPlayers.map((p) => [p.track_id, p]))
  const rows = possessionPlayers.length
    ? possessionPlayers
    : speedPlayers.map((s) => ({
        track_id: s.track_id,
        class_name: s.class_name,
        team: s.team,
        possession_time_sec: 0,
        possession_pct: 0,
        touches: 0,
      }))

  return (
    <Card title="Player Statistics" subtitle="Possession, touches & distance covered">
      <div className="overflow-x-auto -mx-2">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-wc-muted border-b border-wc-border">
              <th className="pb-3 px-2 font-medium">#</th>
              <th className="pb-3 px-2 font-medium">Team</th>
              <th className="pb-3 px-2 font-medium text-right">Poss %</th>
              <th className="pb-3 px-2 font-medium text-right">Touches</th>
              <th className="pb-3 px-2 font-medium text-right">Distance</th>
              <th className="pb-3 px-2 font-medium text-right">Max Speed</th>
              <th className="pb-3 px-2 font-medium text-right">Sprints</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 20).map((p, i) => {
              const spd = speedMap.get(p.track_id)
              return (
                <motion.tr
                  key={p.track_id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.03 }}
                  className="border-b border-wc-border/50 hover:bg-wc-deep/50"
                >
                  <td className="py-2.5 px-2 font-mono text-wc-gold">{p.track_id}</td>
                  <td className="py-2.5 px-2">
                    <span
                      className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${
                        p.team === 'Team A'
                          ? 'bg-wc-green/20 text-wc-green-light'
                          : p.team === 'Team B'
                            ? 'bg-wc-cyan/20 text-wc-cyan'
                            : 'bg-wc-border text-wc-muted'
                      }`}
                    >
                      {p.team || '—'}
                    </span>
                  </td>
                  <td className="py-2.5 px-2 text-right tabular-nums">
                    {p.possession_pct?.toFixed(1) ?? '—'}%
                  </td>
                  <td className="py-2.5 px-2 text-right tabular-nums">{p.touches ?? '—'}</td>
                  <td className="py-2.5 px-2 text-right tabular-nums">
                    {spd ? `${spd.total_distance_m.toFixed(0)} m` : '—'}
                  </td>
                  <td className="py-2.5 px-2 text-right tabular-nums text-wc-gold">
                    {spd ? `${spd.max_speed_kmh.toFixed(1)} km/h` : '—'}
                  </td>
                  <td className="py-2.5 px-2 text-right tabular-nums">{spd?.sprint_count ?? '—'}</td>
                </motion.tr>
              )
            })}
          </tbody>
        </table>
        {rows.length === 0 && (
          <p className="text-center text-wc-muted py-8">No player data available</p>
        )}
      </div>
    </Card>
  )
}
