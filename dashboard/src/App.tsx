import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Header } from './components/layout/Header'
import { HomePage } from './pages/HomePage'
import { MatchDetailPage } from './pages/MatchDetailPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="pitch-pattern min-h-screen">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/match/:id" element={<MatchDetailPage />} />
          </Routes>
        </main>
        <footer className="border-t border-wc-border/50 py-6 text-center text-xs text-wc-muted">
          <img src="/logo.png" alt="" className="inline-block h-6 w-6 opacity-60 mr-2 align-middle" />
          Smart Football Match Analysis · FIFA World Cup 2026
        </footer>
      </div>
    </BrowserRouter>
  )
}
