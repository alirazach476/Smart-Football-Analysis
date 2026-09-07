import { motion } from 'framer-motion'
import { ChevronRight, Film } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { MatchUpload } from '../components/upload/MatchUpload'
import type { JobSummary } from '../types/api'

export function HomePage() {
  const [matches, setMatches] = useState<JobSummary[]>([])

  useEffect(() => {
    api.listMatches({ limit: 10 }).then((r) => setMatches(r.items)).catch(() => {})
  }, [])

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10 text-center"
      >
        <motion.img
          src="/logo.png"
          alt="FIFA World Cup 2026"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="mx-auto h-24 w-24 sm:h-32 sm:w-32 object-contain mb-4 drop-shadow-2xl"
        />
        <p className="text-xs sm:text-sm font-bold tracking-[0.3em] text-wc-gold uppercase mb-2">
          USA · México · Canada 2026
        </p>
        <h2 className="text-3xl sm:text-5xl font-bold mb-3">
          <span className="gold-gradient">Smart Football Analytics</span>
        </h2>
        <p className="text-wc-muted max-w-2xl mx-auto text-sm sm:text-base">
          Upload match footage for AI-powered tactical insights — possession, passing networks,
          shot maps, player speed, goals and automated highlights.
        </p>
      </motion.div>

      <MatchUpload />

      {matches.length > 0 && (
        <section className="mt-12">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Film className="h-5 w-5 text-wc-gold" />
            Recent Matches
          </h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {matches.map((m, i) => (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <Link
                  to={`/match/${m.id}`}
                  className="card-glass wc26-stripe flex items-center justify-between rounded-xl p-4 hover:border-wc-gold/40 transition group"
                >
                  <div className="min-w-0">
                    <p className="font-medium truncate">{m.original_filename}</p>
                    <p className="text-xs text-wc-muted capitalize mt-1">{m.status}</p>
                  </div>
                  <ChevronRight className="h-5 w-5 text-wc-muted group-hover:text-wc-gold shrink-0 transition-colors" />
                </Link>
              </motion.div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
