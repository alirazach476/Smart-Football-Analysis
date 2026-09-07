import { motion } from 'framer-motion'
import { Upload, Film, Loader2 } from 'lucide-react'
import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'

export function MatchUpload() {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  const upload = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('video/')) {
        setError('Please select a video file.')
        return
      }
      setUploading(true)
      setError(null)
      try {
        const res = await api.uploadMatch(file, { enable_heatmaps: true })
        navigate(`/match/${res.id}`)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Upload failed')
      } finally {
        setUploading(false)
      }
    },
    [navigate],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) upload(file)
    },
    [upload],
  )

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`relative rounded-2xl border-2 border-dashed p-10 text-center transition-all wc26-stripe ${
        dragging
          ? 'border-wc-gold bg-wc-gold/10'
          : 'border-wc-border bg-wc-panel/60 hover:border-wc-gold/40'
      }`}
    >
      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-wc-gold/15 ring-2 ring-wc-gold/30">
        {uploading ? (
          <Loader2 className="h-8 w-8 text-wc-gold animate-spin" />
        ) : (
          <Upload className="h-8 w-8 text-wc-gold" />
        )}
      </div>
      <h2 className="text-xl font-semibold mb-2">Upload Match Video</h2>
      <p className="text-wc-muted text-sm mb-6 max-w-md mx-auto">
        Drop your football match footage here for AI-powered tactical analysis —
        possession, passes, shots, goals, heatmaps and more.
      </p>
      <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-gradient-to-r from-wc-gold to-wc-orange px-6 py-3 font-bold text-black shadow-lg shadow-wc-gold/20 transition hover:brightness-110 hover:shadow-wc-gold/40">
        <Film className="h-5 w-5" />
        {uploading ? 'Uploading…' : 'Select Video'}
        <input
          type="file"
          accept="video/*"
          className="hidden"
          disabled={uploading}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) upload(f)
          }}
        />
      </label>
      {error && <p className="mt-4 text-sm text-wc-coral">{error}</p>}
    </motion.div>
  )
}
