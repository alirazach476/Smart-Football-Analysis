# FIFA World Cup 2026 · Match Analytics Dashboard

Modern React analytics dashboard for the Smart Football Match Analysis System.

## Stack

- **React 19** + TypeScript + Vite
- **Tailwind CSS 4** — FIFA WC 2026 theme (navy, pitch green, trophy gold)
- **Framer Motion** — page transitions & micro-interactions
- **Recharts** — possession pie, speed bar charts
- **React Router** — home + match detail views

## Features

| Feature | Component |
|---------|-----------|
| Upload match videos | `MatchUpload` |
| Processed video player | `ProcessedVideo` (bird's-eye + highlights) |
| Player statistics | `PlayerStatsTable` |
| Possession % | `PossessionChart` |
| Heatmaps | `HeatmapPanel` |
| Pass network | `PassNetworkViz` + PNG gallery |
| Shot map | `ShotMapViz` + PNG gallery |
| Speed charts | `SpeedChart` |
| Goal timeline | `GoalTimeline` |
| Live job status | `MatchStatus` with polling |

## Development

**Terminal 1 — API backend:**
```bash
cd ..
python run_api.py --reload
```

**Terminal 2 — Dashboard:**
```bash
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to `http://127.0.0.1:8000`.

## Production build

```bash
npm run build
# Serve dist/ with any static host, or configure your reverse proxy
# to proxy /api to the FastAPI backend.
```

Set `VITE_API_URL` if the API is on a different origin (optional future env).

## Design

FIFA World Cup 2026 inspired palette:
- Deep navy backgrounds (`#0a1628`)
- Pitch green accents (`#00a651`)
- Trophy gold highlights (`#f4d03f`)
- Coral for Team B (`#e85d4c`)
- Glass-morphism cards with pitch-pattern hero background
