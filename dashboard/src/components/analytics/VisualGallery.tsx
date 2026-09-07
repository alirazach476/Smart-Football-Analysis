import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import { Card } from '../ui/Card'
import type { VisualItem } from '../../types/api'

interface Props {
  jobId: string
  category: 'pass_network' | 'shot_map'
  title: string
}

export function VisualGallery({ jobId, category, title }: Props) {
  const [items, setItems] = useState<VisualItem[]>([])
  const [selected, setSelected] = useState<string | null>(null)

  useEffect(() => {
    api.listVisuals(jobId).then((res) => {
      const filtered = res.items.filter((v) => v.category === category)
      setItems(filtered)
      if (filtered[0]) setSelected(filtered[0].name)
    })
  }, [jobId, category])

  if (items.length === 0) return null

  const current = items.find((v) => v.name === selected)

  return (
    <Card title={title} subtitle="Server-rendered visualization">
      <div className="flex flex-wrap gap-2 mb-4">
        {items.map((v) => (
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
          src={api.visualUrl(jobId, category, current.filename.replace('.png', ''))}
          alt={current.name}
          className="w-full rounded-xl border border-wc-border"
        />
      )}
    </Card>
  )
}
