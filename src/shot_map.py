"""
Shot map visualization on a top-down pitch canvas.

Plots each shot at its pitch location with a marker encoding outcome:

  * on target  -> green filled circle
  * off target -> hollow red circle
  * blocked    -> orange square

Marker size scales with shot speed. A thin trajectory line shows the projected
path toward the goal.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import field_config as fc
from .shot_detector import ShotEvent, ShotOutcome
from .pitch import PitchView

_ON_TARGET = (80, 200, 80)    # green (BGR)
_OFF_TARGET = (80, 80, 220)   # red
_BLOCKED = (40, 160, 240)     # orange


def render_shot_map(
    shots: list[ShotEvent],
    team: str | None = None,
    pitch_scale: float = 0.06,
    title: str | None = None,
) -> np.ndarray:
    """Draw a shot map for one team (or all teams if team is None)."""
    pitch = PitchView(scale=pitch_scale)
    canvas = pitch.blank()

    filtered = [s for s in shots if team is None or s.team == team]

    max_speed = max((s.speed_kmh for s in filtered), default=1.0) or 1.0

    for s in filtered:
        px, py = pitch.cm_to_px(s.location_x_cm, s.location_y_cm)
        color = _outcome_color(s.outcome)
        radius = int(5 + 8 * min(s.speed_kmh / max_speed, 1.0))

        # Projected trajectory to the goal line.
        if s.projected_goal_y_cm is not None:
            goal_x = 0.0 if s.target_goal == "left" else float(fc.PITCH_LENGTH)
            gy = min(max(s.projected_goal_y_cm, 0), fc.PITCH_WIDTH)
            gp = pitch.cm_to_px(goal_x, gy)
            cv2.line(canvas, (px, py), gp, color, 1, cv2.LINE_AA)

        if s.outcome == ShotOutcome.ON_TARGET.value:
            cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)
        elif s.outcome == ShotOutcome.OFF_TARGET.value:
            cv2.circle(canvas, (px, py), radius, color, 2, cv2.LINE_AA)
        else:  # blocked
            cv2.rectangle(
                canvas, (px - radius, py - radius), (px + radius, py + radius),
                color, -1, cv2.LINE_AA,
            )

    if title:
        cv2.putText(
            canvas, title, (pitch.padding, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    _draw_legend(canvas, pitch.padding, pitch.canvas_h - 62)
    return canvas


def _outcome_color(outcome: str) -> tuple[int, int, int]:
    if outcome == ShotOutcome.ON_TARGET.value:
        return _ON_TARGET
    if outcome == ShotOutcome.BLOCKED.value:
        return _BLOCKED
    return _OFF_TARGET


def _draw_legend(canvas: np.ndarray, x: int, y: int) -> None:
    items = [
        ("on target", _ON_TARGET),
        ("off target", _OFF_TARGET),
        ("blocked", _BLOCKED),
    ]
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 18
        cv2.circle(canvas, (x + 6, yy - 4), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (x + 18, yy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )
