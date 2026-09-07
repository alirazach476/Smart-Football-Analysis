"""
Heatmap exporter: PNG images, JSON data, and interactive HTML dashboard.

Writes all rendered heatmaps to an output directory and produces a single
self-contained `heatmaps.html` file with embedded data for interactive
exploration (layer switching, hover values, player selection).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2

from .heatmap_collector import HeatmapData
from .heatmap_renderer import HeatmapRenderer


class HeatmapExporter:
    """Saves heatmap PNGs, JSON, and an interactive HTML visualisation."""

    def __init__(
        self,
        output_dir: str = "output/heatmaps",
        json_path: str = "output/heatmap_data.json",
        html_path: str = "output/heatmaps.html",
        pitch_scale: float = 0.06,
    ) -> None:
        self.output_dir = output_dir
        self.json_path = json_path
        self.html_path = html_path
        self.renderer = HeatmapRenderer(pitch_scale=pitch_scale)

    def export(self, data: HeatmapData, source: str = "") -> dict:
        """
        Render all heatmaps, save PNGs + JSON + interactive HTML.

        Returns a summary dict with paths and counts.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        _ensure_parent(self.json_path)
        _ensure_parent(self.html_path)

        images = self.renderer.render_all(data)
        png_paths: dict[str, str] = {}

        for name, img in images.items():
            path = os.path.join(self.output_dir, f"{name}.png")
            cv2.imwrite(path, img)
            png_paths[name] = path

        payload = data.to_dict()
        payload["source"] = source
        payload["png_files"] = {k: os.path.basename(v) for k, v in png_paths.items()}

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self._write_html(payload, png_paths)

        return {
            "png_count": len(png_paths),
            "output_dir": self.output_dir,
            "json_path": self.json_path,
            "html_path": self.html_path,
            "frames_processed": data.frames_processed,
            "players": len(data.players),
        }

    def _write_html(self, payload: dict, png_paths: dict[str, str]) -> None:
        """Generate a self-contained interactive HTML dashboard."""
        # Relative paths from html file to png dir for <img> tags.
        html_dir = Path(self.html_path).parent
        out_dir = Path(self.output_dir)
        try:
            rel_dir = os.path.relpath(out_dir, html_dir)
        except ValueError:
            rel_dir = str(out_dir)

        img_srcs = {
            name: f"{rel_dir}/{os.path.basename(path)}".replace("\\", "/")
            for name, path in png_paths.items()
        }

        # Embed numeric grids for canvas hover (compact JSON).
        grid_json = json.dumps(
            {
                "players": {
                    k: v["grid"] for k, v in payload.get("players", {}).items()
                },
                "teams": payload.get("teams", {}),
                "ball": payload.get("ball", []),
                "pressure": payload.get("pressure", []),
                "zone_control": payload.get("zone_control", []),
                "meta": {
                    "grid_cols": payload["grid"]["cols"],
                    "grid_rows": payload["grid"]["rows"],
                    "zone_cols": payload["zones"]["cols"],
                    "zone_rows": payload["zones"]["rows"],
                    "pitch_length_cm": payload["pitch_length_cm"],
                    "pitch_width_cm": payload["pitch_width_cm"],
                    "team_names": payload.get("team_names", ["Team A", "Team B"]),
                    "frames": payload.get("frames_processed", 0),
                },
            }
        )

        layers = []
        for name in sorted(img_srcs.keys()):
            layer_type = "player" if name.startswith("player_") else name.split("_")[0]
            layers.append({"id": name, "type": layer_type, "src": img_srcs[name]})

        layers_json = json.dumps(layers)
        player_ids = json.dumps(list(payload.get("players", {}).keys()))

        html = _HTML_TEMPLATE.format(
            layers_json=layers_json,
            grid_json=grid_json,
            player_ids=player_ids,
            source=payload.get("source", ""),
            frames=payload.get("frames_processed", 0),
        )

        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(html)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Football Heatmaps</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1419; color: #e7e9ea; }}
  header {{ padding: 16px 24px; border-bottom: 1px solid #2f3336; }}
  header h1 {{ font-size: 1.25rem; font-weight: 600; }}
  header p {{ color: #71767b; font-size: 0.85rem; margin-top: 4px; }}
  .layout {{ display: grid; grid-template-columns: 260px 1fr; min-height: calc(100vh - 70px); }}
  aside {{ padding: 16px; border-right: 1px solid #2f3336; overflow-y: auto; }}
  aside h2 {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #71767b; margin: 16px 0 8px; }}
  aside h2:first-child {{ margin-top: 0; }}
  .btn {{ display: block; width: 100%; text-align: left; padding: 10px 12px; margin-bottom: 4px;
          background: transparent; border: 1px solid transparent; border-radius: 8px;
          color: #e7e9ea; cursor: pointer; font-size: 0.9rem; }}
  .btn:hover {{ background: #1a1f26; }}
  .btn.active {{ background: #1d3a2a; border-color: #00ba7c; color: #00ba7c; }}
  main {{ padding: 16px; display: flex; flex-direction: column; align-items: center; }}
  .viewer {{ position: relative; max-width: 100%; }}
  .viewer img {{ max-width: 100%; border-radius: 8px; border: 1px solid #2f3336; }}
  #tooltip {{ position: fixed; display: none; background: rgba(0,0,0,0.85); color: #fff;
              padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; pointer-events: none;
              z-index: 100; max-width: 220px; }}
  .stats {{ margin-top: 12px; padding: 12px; background: #1a1f26; border-radius: 8px;
            font-size: 0.85rem; width: 100%; max-width: 800px; }}
  .stats span {{ color: #00ba7c; font-weight: 600; }}
  #playerSelect {{ width: 100%; padding: 8px; background: #1a1f26; color: #e7e9ea;
                  border: 1px solid #2f3336; border-radius: 8px; margin-bottom: 8px; }}
  .legend {{ display: flex; gap: 16px; margin-top: 8px; font-size: 0.8rem; color: #71767b; }}
  .legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
<header>
  <h1>Football Heatmap Dashboard</h1>
  <p>Source: {source} &middot; {frames} frames with homography</p>
</header>
<div class="layout">
  <aside>
    <h2>Heatmap layers</h2>
    <div id="layerButtons"></div>
    <h2>Players</h2>
    <select id="playerSelect"><option value="">-- select player --</option></select>
    <h2>About</h2>
    <p style="font-size:0.8rem;color:#71767b;line-height:1.5">
      Hover over the pitch to inspect cell values. Switch layers to compare
      player, team, ball, zone-control and pressure maps.
    </p>
    <div class="legend" id="legend"></div>
  </aside>
  <main>
    <div class="viewer">
      <img id="heatmapImg" src="" alt="heatmap">
    </div>
    <div class="stats" id="stats">Select a layer to view the heatmap.</div>
  </main>
</div>
<div id="tooltip"></div>
<script>
const LAYERS = {layers_json};
const GRID = {grid_json};
const PLAYER_IDS = {player_ids};

let currentLayer = LAYERS.length ? LAYERS[0].id : '';
let currentGrid = null;

const img = document.getElementById('heatmapImg');
const tooltip = document.getElementById('tooltip');
const stats = document.getElementById('stats');
const layerButtons = document.getElementById('layerButtons');
const playerSelect = document.getElementById('playerSelect');
const legend = document.getElementById('legend');

// Build layer buttons.
const groups = {{ tactical: [], team: [], player: [], ball: [], pressure: [], zone: [] }};
LAYERS.forEach(l => {{
  if (l.id.startsWith('player_')) groups.player.push(l);
  else if (l.id.startsWith('team_')) groups.team.push(l);
  else if (l.id === 'ball') groups.ball.push(l);
  else if (l.id === 'pressure') groups.pressure.push(l);
  else if (l.id === 'zone_control') groups.zone.push(l);
  else groups.tactical.push(l);
}});

function addButtons(items, label) {{
  if (!items.length) return;
  const h = document.createElement('h2');
  h.textContent = label;
  layerButtons.appendChild(h);
  items.forEach(l => {{
    const b = document.createElement('button');
    b.className = 'btn' + (l.id === currentLayer ? ' active' : '');
    b.textContent = l.id.replace(/_/g, ' ');
    b.onclick = () => selectLayer(l.id);
    b.dataset.id = l.id;
    layerButtons.appendChild(b);
  }});
}}
addButtons(groups.zone, 'Zone control');
addButtons(groups.pressure, 'Pressure');
addButtons(groups.ball, 'Ball');
addButtons(groups.team, 'Teams');
addButtons(groups.player, 'Players');

// Player dropdown.
PLAYER_IDS.forEach(id => {{
  const o = document.createElement('option');
  o.value = 'player_' + id;
  o.textContent = 'Player #' + id;
  playerSelect.appendChild(o);
}});
playerSelect.onchange = () => {{ if (playerSelect.value) selectLayer(playerSelect.value); }};

function gridForLayer(id) {{
  if (id.startsWith('player_')) {{
    const pid = id.replace('player_', '').split('_')[0];
    return GRID.players[pid] || null;
  }}
  if (id.startsWith('team_')) {{
    const t = id.replace('team_', '').replace('_', ' ');
    const names = GRID.meta.team_names;
  for (const n of names) {{
      if (id.includes(n.replace(/ /g, '_'))) return GRID.teams[n];
    }}
  }}
  if (id === 'ball') return GRID.ball;
  if (id === 'pressure') return GRID.pressure;
  if (id === 'zone_control') return GRID.zone_control;
  return null;
}}

function selectLayer(id) {{
  currentLayer = id;
  const layer = LAYERS.find(l => l.id === id);
  if (!layer) return;
  img.src = layer.src;
  currentGrid = gridForLayer(id);
  document.querySelectorAll('.btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.id === id);
  }});
  updateStats();
  updateLegend(id);
}}

function updateStats() {{
  if (!currentGrid) {{ stats.innerHTML = 'No grid data for this layer.'; return; }}
  const flat = currentGrid.flat();
  const max = Math.max(...flat);
  const sum = flat.reduce((a,b) => a+b, 0);
  const nonzero = flat.filter(v => v > 0).length;
  stats.innerHTML = `Layer: <span>${{currentLayer.replace(/_/g,' ')}}</span> &middot; `
    + `Max cell: <span>${{max.toFixed(1)}}</span> &middot; `
    + `Total: <span>${{sum.toFixed(0)}}</span> &middot; `
    + `Active cells: <span>${{nonzero}}</span>`;
}}

function updateLegend(id) {{
  if (id === 'zone_control') {{
    legend.innerHTML = '<span><i style="background:#4488ff"></i>Team A</span>'
      + '<span><i style="background:#ff4444"></i>Team B</span>'
      + '<span><i style="background:#286428"></i>Contested</span>';
  }} else {{
    legend.innerHTML = '<span><i style="background:linear-gradient(90deg,#0000ff,#00ff00,#ffff00,#ff0000)"></i>Low &rarr; High density</span>';
  }}
}}

img.addEventListener('mousemove', e => {{
  if (!currentGrid) return;
  const rect = img.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  const isZone = currentLayer === 'zone_control';
  const cols = isZone ? GRID.meta.zone_cols : GRID.meta.grid_cols;
  const rows = isZone ? GRID.meta.zone_rows : GRID.meta.grid_rows;
  const col = Math.min(Math.floor(x * cols), cols - 1);
  const row = Math.min(Math.floor(y * rows), rows - 1);
  const val = currentGrid[row] ? currentGrid[row][col] : 0;
  const xM = ((col + 0.5) / cols * GRID.meta.pitch_length_cm / 100).toFixed(1);
  const yM = ((row + 0.5) / rows * GRID.meta.pitch_width_cm / 100).toFixed(1);
  let txt = `Pos: ${{xM}}m x ${{yM}}m<br>Value: ${{val.toFixed(2)}}`;
  if (isZone) txt += val > 0 ? ' (Team A)' : val < 0 ? ' (Team B)' : ' (contested)';
  tooltip.innerHTML = txt;
  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY + 14) + 'px';
}});
img.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

if (LAYERS.length) selectLayer(LAYERS[0].id);
</script>
</body>
</html>
"""
