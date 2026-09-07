"""
Heatmap PNG renderer.

Turns accumulated density grids into colourised images overlaid on a top-down
pitch. Uses OpenCV colormaps + alpha blending so the pitch markings stay visible.
"""

from __future__ import annotations

import cv2
import numpy as np

from .heatmap_collector import GRID_COLS, GRID_ROWS, ZONE_COLS, ZONE_ROWS, HeatmapData
from .pitch import PitchView


class HeatmapRenderer:
    """Renders heatmap grids to PNG-ready BGR images."""

    def __init__(self, pitch_scale: float = 0.06) -> None:
        self.pitch = PitchView(scale=pitch_scale)
        self._out_w = self.pitch.pitch_w_px
        self._out_h = self.pitch.pitch_h_px

    def render_all(self, data: HeatmapData) -> dict[str, np.ndarray]:
        """Return a dict of heatmap_name -> BGR image (with pitch background)."""
        images: dict[str, np.ndarray] = {}

        for tid, grid in data.players.items():
            meta = data.player_meta.get(tid, {})
            label = f"player_{tid}"
            if meta.get("team"):
                label += f"_{meta['team'].replace(' ', '')}"
            images[label] = self._render_density(
                grid, colormap=cv2.COLORMAP_JET, title=f"Player #{tid}"
            )

        for team_name, grid in data.teams.items():
            safe = team_name.replace(" ", "_")
            cmap = (
                cv2.COLORMAP_COOL
                if team_name == data.team_names[0]
                else cv2.COLORMAP_HOT
            )
            images[f"team_{safe}"] = self._render_density(
                grid, colormap=cmap, title=f"{team_name} heatmap"
            )

        images["ball"] = self._render_density(
            data.ball, colormap=cv2.COLORMAP_HOT, title="Ball heatmap"
        )
        images["pressure"] = self._render_density(
            data.pressure, colormap=cv2.COLORMAP_INFERNO, title="Pressure map"
        )
        images["zone_control"] = self._render_zone_control(data)

        return images

    def _render_density(
        self,
        grid: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        title: str = "",
        blur_ksize: int = 15,
    ) -> np.ndarray:
        """Blur, normalise, colourise and blend a density grid onto the pitch."""
        base = self.pitch.blank()
        # Crop to pitch area (exclude padding).
        pitch_area = base[
            self.pitch.padding : self.pitch.padding + self._out_h,
            self.pitch.padding : self.pitch.padding + self._out_w,
        ].copy()

        if grid.max() <= 0:
            return base

        # Gaussian smooth for a classic heatmap look.
        smooth = cv2.GaussianBlur(grid.astype(np.float32), (blur_ksize, blur_ksize), 0)
        norm = smooth / smooth.max()
        heat_small = (norm * 255).astype(np.uint8)
        heat = cv2.resize(
            heat_small, (self._out_w, self._out_h), interpolation=cv2.INTER_LINEAR
        )
        colored = cv2.applyColorMap(heat, colormap)

        # Alpha blend: transparent where density is zero.
        alpha = (norm * 0.72).astype(np.float32)
        alpha = cv2.resize(alpha, (self._out_w, self._out_h), interpolation=cv2.INTER_LINEAR)
        alpha_3 = np.stack([alpha, alpha, alpha], axis=2)

        blended = (colored * alpha_3 + pitch_area * (1 - alpha_3)).astype(np.uint8)
        base[
            self.pitch.padding : self.pitch.padding + self._out_h,
            self.pitch.padding : self.pitch.padding + self._out_w,
        ] = blended

        if title:
            cv2.putText(
                base, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
        return base

    def _render_zone_control(self, data: HeatmapData) -> np.ndarray:
        """Render zone-control as Team A (blue) vs Team B (red)."""
        base = self.pitch.blank()
        pitch_area = base[
            self.pitch.padding : self.pitch.padding + self._out_h,
            self.pitch.padding : self.pitch.padding + self._out_w,
        ].copy()

        zc = data.zone_control
        if np.abs(zc).max() <= 0:
            return base

        # Map [-1, 1] to BGR: Team B red, neutral green, Team A blue.
        zone_img = np.zeros((ZONE_ROWS, ZONE_COLS, 3), dtype=np.uint8)
        for r in range(ZONE_ROWS):
            for c in range(ZONE_COLS):
                v = zc[r, c]
                if v > 0.05:
                    zone_img[r, c] = (255, int(80 * (1 - v)), int(80 * (1 - v)))  # blue
                elif v < -0.05:
                    zone_img[r, c] = (int(80 * (1 + v)), int(80 * (1 + v)), 255)  # red
                else:
                    zone_img[r, c] = (40, 100, 40)  # contested

        zone_big = cv2.resize(
            zone_img, (self._out_w, self._out_h), interpolation=cv2.INTER_NEAREST
        )
        # Draw zone grid lines.
        for c in range(1, ZONE_COLS):
            x = int(c / ZONE_COLS * self._out_w)
            cv2.line(zone_big, (x, 0), (x, self._out_h), (255, 255, 255), 1)
        for r in range(1, ZONE_ROWS):
            y = int(r / ZONE_ROWS * self._out_h)
            cv2.line(zone_big, (0, y), (self._out_w, y), (255, 255, 255), 1)

        blended = cv2.addWeighted(pitch_area, 0.35, zone_big, 0.65, 0)
        base[
            self.pitch.padding : self.pitch.padding + self._out_h,
            self.pitch.padding : self.pitch.padding + self._out_w,
        ] = blended
        cv2.putText(
            base, "Zone control (blue=Team A, red=Team B)", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return base
