import { useMemo } from 'react'
import { Card } from '../ui/Card'
import type { PassEvent } from '../../types/api'

const PITCH_W = 12000
const PITCH_H = 7000

interface Props {
  passes: PassEvent[]
  team?: string
}

export function PassNetworkViz({ passes, team }: Props) {
  const filtered = useMemo(
    () => (team ? passes.filter((p) => p.team === team) : passes),
    [passes, team],
  )

  const { nodes, edges, attempts, positions } = useMemo(() => {
    const pos: Record<number, { x: number; y: number; count: number }> = {}
    const edgeMap: Record<string, { count: number; success: number }> = {}
    const attempts: Array<{
      from: number
      endX: number
      endY: number
    }> = []

    for (const p of filtered) {
      if (p.to_track_id == null) {
        if (p.start_x_cm != null && p.start_y_cm != null) {
          const n = pos[p.from_track_id] ?? { x: 0, y: 0, count: 0 }
          n.x += p.start_x_cm
          n.y += p.start_y_cm
          n.count++
          pos[p.from_track_id] = n
        }
        if (
          p.start_x_cm != null &&
          p.start_y_cm != null &&
          p.end_x_cm != null &&
          p.end_y_cm != null
        ) {
          attempts.push({
            from: p.from_track_id,
            endX: p.end_x_cm,
            endY: p.end_y_cm,
          })
        }
        continue
      }
      const key = `${p.from_track_id}-${p.to_track_id}`
      const e = edgeMap[key] ?? { count: 0, success: 0 }
      e.count++
      if (p.successful) e.success++
      edgeMap[key] = e

      if (p.start_x_cm != null && p.start_y_cm != null) {
        const n = pos[p.from_track_id] ?? { x: 0, y: 0, count: 0 }
        n.x += p.start_x_cm
        n.y += p.start_y_cm
        n.count++
        pos[p.from_track_id] = n
      }
      if (p.end_x_cm != null && p.end_y_cm != null) {
        const n = pos[p.to_track_id] ?? { x: 0, y: 0, count: 0 }
        n.x += p.end_x_cm
        n.y += p.end_y_cm
        n.count++
        pos[p.to_track_id] = n
      }
    }

    for (const id of Object.keys(pos)) {
      const n = pos[Number(id)]
      n.x /= n.count
      n.y /= n.count
    }

    return {
      nodes: Object.keys(pos).map(Number),
      edges: Object.entries(edgeMap).map(([k, v]) => {
        const [from, to] = k.split('-').map(Number)
        return { from, to, ...v }
      }),
      attempts,
      positions: pos,
    }
  }, [filtered])

  const toSvg = (x: number, y: number) => ({
    sx: (x / PITCH_W) * 100,
    sy: (y / PITCH_H) * 100,
  })

  return (
    <Card title="Pass Network" subtitle={team || 'All teams'}>
      {filtered.length === 0 ? (
        <p className="text-wc-muted text-center py-12">No pass data</p>
      ) : (
        <div className="relative aspect-[12/7] w-full rounded-xl overflow-hidden bg-[#1a4d2e] border border-wc-border">
          <svg viewBox="0 0 100 58.33" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
            {/* Pitch lines */}
            <rect x="2" y="2" width="96" height="54.33" fill="none" stroke="#fff" strokeWidth="0.3" opacity="0.5" />
            <line x1="50" y1="2" x2="50" y2="56.33" stroke="#fff" strokeWidth="0.2" opacity="0.4" />
            <circle cx="50" cy="29.17" r="5" fill="none" stroke="#fff" strokeWidth="0.2" opacity="0.4" />

            {attempts.map((a, i) => {
              const p1 = positions[a.from]
              if (!p1) return null
              const from = toSvg(p1.x, p1.y)
              const to = toSvg(a.endX, a.endY)
              return (
                <line
                  key={`att-${i}`}
                  x1={from.sx}
                  y1={from.sy}
                  x2={to.sx}
                  y2={to.sy}
                  stroke="#888"
                  strokeWidth="0.25"
                  strokeDasharray="1 1"
                  opacity={0.55}
                />
              )
            })}

            {edges.map((e, i) => {
              const p1 = positions[e.from]
              const p2 = positions[e.to]
              if (!p1 || !p2) return null
              const a = toSvg(p1.x, p1.y)
              const b = toSvg(p2.x, p2.y)
              const rate = e.success / e.count
              const color = rate > 0.7 ? '#7ed321' : rate > 0.4 ? '#d4af37' : '#ff5722'
              return (
                <line
                  key={i}
                  x1={a.sx}
                  y1={a.sy}
                  x2={b.sx}
                  y2={b.sy}
                  stroke={color}
                  strokeWidth={0.3 + e.count * 0.15}
                  opacity={0.7}
                  markerEnd="url(#arrow)"
                />
              )
            })}

            <defs>
              <marker id="arrow" markerWidth="4" markerHeight="4" refX="3" refY="2" orient="auto">
                <polygon points="0 0, 4 2, 0 4" fill="#d4af37" />
              </marker>
            </defs>

            {nodes.map((id) => {
              const p = positions[id]
              if (!p) return null
              const { sx, sy } = toSvg(p.x, p.y)
              return (
                <g key={id}>
                  <circle cx={sx} cy={sy} r="1.8" fill="#000" stroke="#fff" strokeWidth="0.3" />
                  <text x={sx} y={sy + 0.5} textAnchor="middle" fontSize="1.5" fill="#fff">
                    {id}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>
      )}
    </Card>
  )
}
