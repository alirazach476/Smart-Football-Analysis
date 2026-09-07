"""
Pass network visualization on a top-down pitch canvas.

Renders player nodes at their average pitch positions and directed edges
between passer/receiver pairs. Edge thickness scales with pass count; colour
encodes success rate (green = high completion, red = low).
"""

from __future__ import annotations

import cv2
import numpy as np

from .pass_detector import PassEvent
from .pitch import PitchView


def render_pass_network(
    passes: list[PassEvent],
    player_positions: dict[int, tuple[float, float]],
    team: str | None = None,
    pitch_scale: float = 0.06,
    title: str | None = None,
) -> np.ndarray:
    """
    Draw a pass network for one team (or all teams if team is None).

    Returns a BGR image of the pitch with nodes and edges.
    """
    pitch = PitchView(scale=pitch_scale)
    canvas = pitch.blank()

    filtered = [p for p in passes if team is None or p.team == team]
    if not filtered:
        if title:
            cv2.putText(
                canvas, title, (pitch.padding, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return canvas

    # Aggregate edges passer -> receiver.
    edges: dict[tuple[int, int], dict] = {}
    for p in filtered:
        if p.to_track_id is None:
            continue
        key = (p.from_track_id, p.to_track_id)
        e = edges.setdefault(key, {"count": 0, "success": 0})
        e["count"] += 1
        if p.successful:
            e["success"] += 1

    # Draw edges first (under nodes).
    max_count = max(e["count"] for e in edges.values()) if edges else 1
    for (src, dst), meta in edges.items():
        if src not in player_positions or dst not in player_positions:
            continue
        p1 = pitch.cm_to_px(*player_positions[src])
        p2 = pitch.cm_to_px(*player_positions[dst])
        rate = meta["success"] / meta["count"]
        color = _success_color(rate)
        thickness = max(1, int(1 + 4 * meta["count"] / max_count))
        cv2.arrowedLine(canvas, p1, p2, color, thickness, cv2.LINE_AA, tipLength=0.15)

    # Unsuccessful / incomplete passes: dashed line to end coordinate.
    for p in filtered:
        if p.to_track_id is not None or p.successful:
            continue
        if p.from_track_id not in player_positions:
            continue
        p1 = pitch.cm_to_px(*player_positions[p.from_track_id])
        p2 = pitch.cm_to_px(p.end_x_cm, p.end_y_cm)
        cv2.line(canvas, p1, p2, (140, 140, 140), 1, cv2.LINE_AA)

    # Draw nodes for players involved in this network.
    involved: set[int] = set()
    for p in filtered:
        involved.add(p.from_track_id)
        if p.to_track_id is not None:
            involved.add(p.to_track_id)

    for tid in involved:
        if tid not in player_positions:
            continue
        px, py = pitch.cm_to_px(*player_positions[tid])
        cv2.circle(canvas, (px, py), 14, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 14, (255, 255, 255), 2, cv2.LINE_AA)
        label = str(tid)
        cv2.putText(
            canvas, label, (px - 8, py + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    if title:
        cv2.putText(
            canvas, title, (pitch.padding, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    # Legend.
    _draw_legend(canvas, pitch.padding, pitch.canvas_h - 50)
    return canvas


def render_pass_map(
    passes: list[PassEvent],
    team: str | None = None,
    pitch_scale: float = 0.06,
    title: str | None = None,
) -> np.ndarray:
    """
    Draw individual pass trajectories on the pitch (successful vs failed).
    """
    pitch = PitchView(scale=pitch_scale)
    canvas = pitch.blank()

    filtered = [p for p in passes if team is None or p.team == team]
    for p in filtered:
        if p.end_x_cm is None or p.end_y_cm is None:
            continue
        p1 = pitch.cm_to_px(p.start_x_cm, p.start_y_cm)
        p2 = pitch.cm_to_px(p.end_x_cm, p.end_y_cm)
        color = (80, 200, 80) if p.successful else (80, 80, 220)
        cv2.arrowedLine(canvas, p1, p2, color, 1, cv2.LINE_AA, tipLength=0.12)

    if title:
        cv2.putText(
            canvas, title, (pitch.padding, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return canvas


def _success_color(rate: float) -> tuple[int, int, int]:
    """BGR colour from red (0%) to green (100%)."""
    r = int(255 * (1.0 - rate))
    g = int(255 * rate)
    return (0, g, r)


def _draw_legend(canvas: np.ndarray, x: int, y: int) -> None:
    cv2.putText(
        canvas, "Edge width = pass count", (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "Edge colour = completion rate", (x, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )
