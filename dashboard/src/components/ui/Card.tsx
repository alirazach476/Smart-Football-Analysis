import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface CardProps {
  title?: string
  subtitle?: string
  children: ReactNode
  className?: string
  delay?: number
}

export function Card({ title, subtitle, children, className = '', delay = 0 }: CardProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className={`card-glass wc26-stripe rounded-2xl p-5 ${className}`}
    >
      {(title || subtitle) && (
        <header className="mb-4 pt-1">
          {title && <h3 className="text-lg font-semibold text-white">{title}</h3>}
          {subtitle && <p className="text-sm text-wc-muted mt-0.5">{subtitle}</p>}
        </header>
      )}
      {children}
    </motion.section>
  )
}
