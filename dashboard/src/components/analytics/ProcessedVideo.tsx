import { useState } from 'react'
import { Play, Sparkles } from 'lucide-react'
import { Card } from '../ui/Card'
import { api } from '../../api/client'

interface Props {
  jobId: string
}

export function ProcessedVideo({ jobId }: Props) {
  const [kind, setKind] = useState<'birdseye' | 'highlights'>('birdseye')

  return (
    <Card title="Processed Video" subtitle="Bird's-eye tactical view & highlights">
      <div className="flex gap-2 mb-4">
        <button
          type="button"
          onClick={() => setKind('birdseye')}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
            kind === 'birdseye'
              ? 'bg-wc-gold text-black'
              : 'bg-wc-deep text-wc-muted hover:text-white'
          }`}
        >
          <Play className="h-4 w-4" />
          Bird's-Eye
        </button>
        <button
          type="button"
          onClick={() => setKind('highlights')}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
            kind === 'highlights'
              ? 'bg-wc-orange text-black'
              : 'bg-wc-deep text-wc-muted hover:text-white'
          }`}
        >
          <Sparkles className="h-4 w-4" />
          Highlights
        </button>
      </div>
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black border border-wc-border">
        <video
          key={`${jobId}-${kind}`}
          src={api.videoUrl(jobId, kind)}
          controls
          className="h-full w-full object-contain"
          preload="metadata"
        >
          Your browser does not support video playback.
        </video>
      </div>
    </Card>
  )
}
