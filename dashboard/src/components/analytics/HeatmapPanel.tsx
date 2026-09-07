import { useEffect, useState } from 'react'
import { Card } from '../ui/Card'
import { api } from '../../api/client'
import type { VisualItem } from '../../types/api'

interface Props {
  jobId: string
}

export function HeatmapPanel({ jobId }: Props) {
  const [visuals, setVisuals] = useState<VisualItem[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api
      .listVisuals(jobId)
      .then((res) => {
        const heatmaps = res.items.filter((v) => v.category === 'heatmap')
        setVisuals(heatmaps)
        if (heatmaps[0]) setSelected(heatmaps[0].name)
      })
      .catch(() => setVisuals([]))
      .finally(() => setLoading(false))
  }, [jobId])

  const current = visuals.find((v) => v.name === selected)

  return (
    <Card title="Heatmaps" subtitle="Player & team positional density">
      {loading ? (
        <p className="text-wc-muted text-center py-12">Loading heatmaps…</p>
      ) : visuals.length === 0 ? (
        <p className="text-wc-muted text-center py-12">
          No heatmaps — enable <code className="text-wc-gold">enable_heatmaps</code> at upload
        </p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2 mb-4">
            {visuals.map((v) => (
              <button
                key={v.id}
                type="button"
                onClick={() => setSelected(v.name)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  selected === v.name
                    ? 'bg-wc-gold text-black'
                    : 'bg-wc-deep text-wc-muted hover:text-white'
                }`}
              >
                {v.name.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
          {current && (
            <img
              src={api.visualUrl(jobId, 'heatmap', current.filename.replace('.png', ''))}
              alt={current.name}
              className="w-full rounded-xl border border-wc-border"
            />
          )}
        </>
      )}
    </Card>
  )
}
