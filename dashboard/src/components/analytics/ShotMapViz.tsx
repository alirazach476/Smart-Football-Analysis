import { Card } from '../ui/Card'
import type { ShotEvent } from '../../types/api'

const PITCH_W = 12000
const PITCH_H = 7000

interface Props {
  shots: ShotEvent[]
}

export function ShotMapViz({ shots }: Props) {
  const toSvg = (x: number, y: number) => ({
    cx: 4 + (x / PITCH_W) * 92,
    cy: 4 + (y / PITCH_H) * 50,
  })

  return (
    <Card title="Shot Map" subtitle="Shot locations on the pitch">
      {shots.length === 0 ? (
        <p className="text-wc-muted text-center py-12">No shots detected</p>
      ) : (
        <div className="relative aspect-[12/7] w-full rounded-xl overflow-hidden bg-[#1a4d2e] border border-wc-border">
          <svg viewBox="0 0 100 58" className="w-full h-full">
            <rect x="2" y="2" width="96" height="54" fill="none" stroke="#fff" strokeWidth="0.3" opacity="0.5" />
            <line x1="50" y1="2" x2="50" y2="56" stroke="#fff" strokeWidth="0.2" opacity="0.4" />
            {/* Goals */}
            <rect x="0" y="22" width="2" height="14" fill="#d4af37" opacity="0.6" />
            <rect x="98" y="22" width="2" height="14" fill="#d4af37" opacity="0.6" />

            {shots.map((s) => {
              const { cx, cy } = toSvg(s.location_x_cm, s.location_y_cm)
              const color =
                s.outcome === 'on_target'
                  ? '#7ed321'
                  : s.outcome === 'blocked'
                    ? '#d4af37'
                    : '#ff5722'
              const r = 1.2 + Math.min(s.speed_kmh / 40, 1.5)
              return (
                <circle
                  key={s.shot_id}
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill={color}
                  fillOpacity={s.outcome === 'off_target' ? 0 : 0.85}
                  stroke={color}
                  strokeWidth={s.outcome === 'off_target' ? 0.4 : 0}
                />
              )
            })}
          </svg>
          <div className="absolute bottom-2 left-2 flex gap-3 text-xs text-white/80">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#7ed321]" /> On target
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full border border-[#ff5722]" /> Off target
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-sm bg-[#d4af37]" /> Blocked
            </span>
          </div>
        </div>
      )}
    </Card>
  )
}
