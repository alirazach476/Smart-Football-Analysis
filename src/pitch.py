"""
Top-down football pitch renderer (bird's-eye canvas).

Draws a clean 2D pitch to scale from the real-world dimensions in
`field_config`, and provides the cm -> pixel mapping used to place transformed
players and the ball onto the minimap.

The pitch coordinate system matches the homography target plane:
  x in [0, PITCH_LENGTH] cm  (goal-to-goal)
  y in [0, PITCH_WIDTH]  cm  (touchline-to-touchline)
"""

from __future__ import annotations

import cv2
import numpy as np

from . import field_config as fc

# Pitch surface colour (BGR) and line colour.
PITCH_COLOR = (40, 120, 40)   # green
LINE_COLOR = (255, 255, 255)  # white


class PitchView:
    """A reusable top-down pitch canvas with cm<->pixel conversion."""

    def __init__(self, scale: float = 0.06, padding: int = 40) -> None:
        # scale: pixels per centimetre. 0.06 -> 12000cm pitch = 720px long.
        self.scale = scale
        self.padding = padding

        self.pitch_w_px = int(fc.PITCH_LENGTH * scale)   # image width  (x)
        self.pitch_h_px = int(fc.PITCH_WIDTH * scale)    # image height (y)
        self.canvas_w = self.pitch_w_px + 2 * padding
        self.canvas_h = self.pitch_h_px + 2 * padding

        # Pre-render the empty pitch once; each frame copies it.
        self._base = self._render_base()

    # ------------------------------------------------------------------ #
    def cm_to_px(self, x_cm: float, y_cm: float) -> tuple[int, int]:
        """Map a real-world pitch coordinate (cm) to a canvas pixel."""
        px = int(self.padding + x_cm * self.scale)
        py = int(self.padding + y_cm * self.scale)
        return px, py

    def blank(self) -> np.ndarray:
        """A fresh copy of the rendered empty pitch."""
        return self._base.copy()

    # ------------------------------------------------------------------ #
    def _render_base(self) -> np.ndarray:
        """Draw the static pitch markings once."""
        canvas = np.full(
            (self.canvas_h, self.canvas_w, 3), PITCH_COLOR, dtype=np.uint8
        )
        t = 2  # line thickness

        L, W = fc.PITCH_LENGTH, fc.PITCH_WIDTH
        pb, pbl = fc.PENALTY_BOX_WIDTH, fc.PENALTY_BOX_LENGTH
        gb, gbl = fc.GOAL_BOX_WIDTH, fc.GOAL_BOX_LENGTH
        r = fc.CENTRE_CIRCLE_RADIUS
        psd = fc.PENALTY_SPOT_DISTANCE

        def line(p1, p2):
            cv2.line(canvas, self.cm_to_px(*p1), self.cm_to_px(*p2), LINE_COLOR, t, cv2.LINE_AA)

        def rect(p1, p2):
            cv2.rectangle(canvas, self.cm_to_px(*p1), self.cm_to_px(*p2), LINE_COLOR, t)

        # Outer boundary (sidelines + goal lines).
        rect((0, 0), (L, W))

        # Halfway line.
        line((L / 2, 0), (L / 2, W))

        # Centre circle + centre spot.
        cv2.ellipse(
            canvas, self.cm_to_px(L / 2, W / 2),
            (int(r * self.scale), int(r * self.scale)), 0, 0, 360, LINE_COLOR, t, cv2.LINE_AA,
        )
        cv2.circle(canvas, self.cm_to_px(L / 2, W / 2), 3, LINE_COLOR, -1, cv2.LINE_AA)

        # Left / right penalty boxes.
        rect((0, (W - pb) / 2), (pbl, (W + pb) / 2))
        rect((L - pbl, (W - pb) / 2), (L, (W + pb) / 2))

        # Left / right goal boxes.
        rect((0, (W - gb) / 2), (gbl, (W + gb) / 2))
        rect((L - gbl, (W - gb) / 2), (L, (W + gb) / 2))

        # Penalty spots.
        cv2.circle(canvas, self.cm_to_px(psd, W / 2), 3, LINE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, self.cm_to_px(L - psd, W / 2), 3, LINE_COLOR, -1, cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------ #
    def draw_player(
        self,
        canvas: np.ndarray,
        x_cm: float,
        y_cm: float,
        color: tuple[int, int, int],
        label: str | None = None,
    ) -> None:
        """Draw a player marker at a pitch coordinate (cm)."""
        # Skip anything transformed well outside the pitch (bad homography).
        if not self._on_extended_pitch(x_cm, y_cm):
            return
        px, py = self.cm_to_px(x_cm, y_cm)
        cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)
        if label:
            cv2.putText(
                canvas, label, (px + 7, py - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
            )

    def draw_trajectory(
        self,
        canvas: np.ndarray,
        trail: list[tuple[float, float]],
        color: tuple[int, int, int] = (0, 200, 255),
    ) -> None:
        """Draw the ball's recent path as a fading poly-line (cm coords)."""
        pts = [
            self.cm_to_px(x, y)
            for (x, y) in trail
            if self._on_extended_pitch(x, y)
        ]
        n = len(pts)
        if n < 2:
            return
        for i in range(1, n):
            # Older segments thinner/dimmer, newest brightest.
            frac = i / n
            thickness = 1 + int(round(2 * frac))
            c = tuple(int(ch * (0.35 + 0.65 * frac)) for ch in color)
            cv2.line(canvas, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)

    def draw_ball(self, canvas: np.ndarray, x_cm: float, y_cm: float) -> None:
        """Draw the ball marker at a pitch coordinate (cm)."""
        if not self._on_extended_pitch(x_cm, y_cm):
            return
        px, py = self.cm_to_px(x_cm, y_cm)
        cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

    def _on_extended_pitch(self, x_cm: float, y_cm: float) -> bool:
        """Allow a margin outside the pitch, reject wildly off points."""
        margin = 500  # cm
        return (
            -margin <= x_cm <= fc.PITCH_LENGTH + margin
            and -margin <= y_cm <= fc.PITCH_WIDTH + margin
        )
