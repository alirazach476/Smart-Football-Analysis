import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card } from '../ui/Card'
import type { PlayerSpeed } from '../../types/api'

interface Props {
  players: PlayerSpeed[]
}

export function SpeedChart({ players }: Props) {
  const data = [...players]
    .sort((a, b) => b.max_speed_kmh - a.max_speed_kmh)
    .slice(0, 12)
    .map((p) => ({
      name: `#${p.track_id}`,
      avg: Math.round(p.avg_speed_kmh * 10) / 10,
      max: Math.round(p.max_speed_kmh * 10) / 10,
      team: p.team,
    }))

  return (
    <Card title="Speed Analysis" subtitle="Average vs maximum speed (km/h)">
      {data.length === 0 ? (
        <p className="text-wc-muted text-center py-12">No speed data</p>
      ) : (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
              <XAxis dataKey="name" tick={{ fill: '#a0a0a0', fontSize: 11 }} />
              <YAxis tick={{ fill: '#a0a0a0', fontSize: 11 }} unit=" km/h" />
              <Tooltip
                contentStyle={{
                  background: '#141414',
                  border: '1px solid #2a2a2a',
                  borderRadius: 8,
                  color: '#fff',
                }}
              />
              <Bar dataKey="avg" name="Avg" fill="#2196f3" radius={[4, 4, 0, 0]} />
              <Bar dataKey="max" name="Max" fill="#d4af37" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}
