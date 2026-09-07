import { motion } from 'framer-motion'
import { AlertCircle, CheckCircle2, Clock, Loader2 } from 'lucide-react'
import type { JobDetail } from '../../types/api'

interface Props {
  job: JobDetail
}

export function MatchStatus({ job }: Props) {
  const icons = {
    queued: <Clock className="h-5 w-5 text-wc-muted" />,
    processing: <Loader2 className="h-5 w-5 text-wc-gold animate-spin" />,
    completed: <CheckCircle2 className="h-5 w-5 text-wc-green-light" />,
    failed: <AlertCircle className="h-5 w-5 text-wc-coral" />,
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="card-glass wc26-stripe rounded-2xl p-5 mb-6"
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs text-wc-muted uppercase tracking-wider">Match</p>
          <h2 className="text-xl font-semibold truncate max-w-md">{job.original_filename}</h2>
        </div>
        <div className="flex items-center gap-2 capitalize">
          {icons[job.status]}
          <span className="font-medium">{job.status}</span>
        </div>
      </div>
      {(job.status === 'processing' || job.status === 'queued') && (
        <div className="mt-4">
          <div className="flex justify-between text-xs text-wc-muted mb-1">
            <span>Processing</span>
            <span>{Math.round(job.progress * 100)}%</span>
          </div>
          <div className="h-2 rounded-full bg-wc-deep overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-wc-gold via-wc-orange to-wc-red"
              initial={{ width: 0 }}
              animate={{ width: `${job.progress * 100}%` }}
            />
          </div>
        </div>
      )}
      {job.status === 'failed' && job.error && (
        <p className="mt-3 text-sm text-wc-coral bg-wc-coral/10 rounded-lg p-3">{job.error}</p>
      )}
    </motion.div>
  )
}
