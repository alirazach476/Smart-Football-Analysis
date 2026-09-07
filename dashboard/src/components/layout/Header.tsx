import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-wc-border/80 bg-black/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-2.5 sm:px-6">
        <Link to="/" className="flex items-center gap-3 group">
          <motion.img
            src="/logo.png"
            alt="FIFA World Cup 2026"
            whileHover={{ scale: 1.04 }}
            className="h-12 w-12 sm:h-14 sm:w-14 object-contain drop-shadow-lg"
          />
          <div>
            <p className="text-[10px] sm:text-xs font-bold tracking-[0.25em] text-wc-gold uppercase">
              FIFA World Cup
            </p>
            <h1 className="text-base sm:text-lg font-bold leading-tight text-white">
              2026 <span className="gold-gradient">Analytics</span>
            </h1>
          </div>
        </Link>
        <nav className="hidden sm:flex items-center gap-6 text-sm text-wc-muted">
          <Link to="/" className="hover:text-wc-gold transition-colors font-medium">
            Dashboard
          </Link>
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="hover:text-wc-gold transition-colors font-medium"
          >
            API
          </a>
        </nav>
      </div>
      {/* WC26 rainbow accent bar */}
      <div
        className="h-0.5 w-full"
        style={{
          background:
            'linear-gradient(90deg, #0066ff, #7ed321, #d4af37, #ff6b00, #e53935, #7b2fbe)',
        }}
      />
    </header>
  )
}
