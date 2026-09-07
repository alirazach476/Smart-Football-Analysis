"""
Frame annotation utilities.

Draws bounding boxes, class labels with confidence, and an FPS overlay
onto video frames. Pure drawing logic only - no model or I/O here.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig, color_for_track_id
from .detector import Detection
from .ball_tracker import BallState


class Annotator:
    """Draws detections and an FPS counter onto frames."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw all bounding boxes and labels onto a copy-free frame."""
        for det in detections:
            color = self._box_color(det)

            # Bounding box.
            cv2.rectangle(
                frame,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                self.config.box_thickness,
            )

            self._draw_label(frame, self._label_for(det), det.x1, det.y1, color)

        return frame

    def _label_for(self, det: Detection) -> str:
        """Build the label text shown above a box."""
        # Players with a known team: show the team name prominently.
        if det.team is not None:
            tid = f" #{det.track_id}" if det.track_id is not None else ""
            return f"{det.team}{tid} {det.confidence:.2f}"
        # Tracked non-players (referee / goalkeeper / unassigned player).
        if det.track_id is not None:
            return f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
        # Plain detection mode.
        return f"{det.class_name} {det.confidence:.2f}"

    def _box_color(self, det: Detection) -> tuple[int, int, int]:
        """
        Choose a box colour, in priority order:

        1. Team colour  - players assigned to a team are drawn in that team's
           representative jersey colour (referees/goalkeepers are unaffected and
           keep their own class colour, so they stay visually separate).
        2. Per-track ID colour - in tracking mode when `id_colors` is on.
        3. Per-class colour - the default detection scheme.
        """
        if det.team_color is not None:
            return det.team_color
        if self.config.enable_tracking and self.config.id_colors and det.track_id is not None:
            return color_for_track_id(det.track_id)
        return self.config.color_for(det.class_name)

    def draw_ball_tracking(
        self,
        frame: np.ndarray,
        ball_state: BallState | None,
        trajectory: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Draw the ball trajectory trail, current position, and speed readout.

        Detected positions are drawn solid yellow; Kalman-predicted (coasted)
        positions are drawn as a hollow circle so you can see when the filter is
        bridging a gap.
        """
        if not trajectory and ball_state is None:
            return frame

        ball_color = self.config.color_for("ball")  # yellow

        # --- Trajectory polyline (fading older segments) -------------------
        if len(trajectory) >= 2:
            for i in range(1, len(trajectory)):
                # Fade alpha from dim (old) to bright (recent).
                alpha = i / len(trajectory)
                thickness = max(1, int(1 + alpha * 2))
                shade = int(80 + alpha * 175)
                color = (0, shade, shade)  # yellow-ish gradient in BGR
                cv2.line(
                    frame,
                    trajectory[i - 1],
                    trajectory[i],
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        if ball_state is None:
            return frame

        cx, cy = int(ball_state.x), int(ball_state.y)

        # --- Current ball marker -----------------------------------------
        if ball_state.detected:
            cv2.circle(frame, (cx, cy), 8, ball_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Coasting: hollow ring shows this is a Kalman prediction.
            cv2.circle(frame, (cx, cy), 8, ball_color, 2, cv2.LINE_AA)

        # --- Coordinates + speed label -----------------------------------
        conf_str = (
            f" conf:{ball_state.confidence:.2f}" if ball_state.confidence else ""
        )
        mode = "DET" if ball_state.detected else "PRED"
        label = (
            f"ball ({mode}) ({cx},{cy}) "
            f"{ball_state.speed:.0f}px/s{conf_str}"
        )
        self._draw_label(frame, label, cx + 12, cy - 12, ball_color)

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw the current FPS in the top-left corner."""
        text = f"FPS: {fps:.1f}"
        # Draw a subtle black outline first for readability, then white text.
        cv2.putText(
            frame, text, (10, 30), self.font, 0.8, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, (10, 30), self.font, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a filled label background with text above the box."""
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, self.config.font_scale, self.config.font_thickness
        )

        # Position the label just above the box; clamp to the top edge.
        top = max(y - text_h - baseline - 2, 0)

        # Filled rectangle behind the text so it is always legible.
        cv2.rectangle(
            frame,
            (x, top),
            (x + text_w, top + text_h + baseline + 2),
            color,
            thickness=-1,
        )

        # Black text on the coloured background.
        cv2.putText(
            frame,
            label,
            (x, top + text_h + 1),
            self.font,
            self.config.font_scale,
            (0, 0, 0),
            self.config.font_thickness,
            cv2.LINE_AA,
        )
