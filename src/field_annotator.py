"""
Field visualization.

Draws the detected pitch onto a frame: the pitch lines (sidelines, goal lines,
penalty boxes, goal boxes, halfway line) coloured by feature group, the centre
circle, individual keypoints, and an optional legend. Pure drawing only.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import field_config as fc
from .field_detector import FieldDetection, FieldKeypoint


class FieldAnnotator:
    """Renders a `FieldDetection` onto frames."""

    def __init__(
        self,
        draw_labels: bool = True,
        draw_points: bool = True,
        draw_lines: bool = True,
        line_thickness: int = 2,
        point_radius: int = 5,
        font_scale: float = 0.4,
    ) -> None:
        self.draw_labels = draw_labels
        self.draw_points = draw_points
        self.draw_lines = draw_lines
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(self, frame: np.ndarray, detection: FieldDetection) -> np.ndarray:
        """Draw all pitch elements onto the frame in-place and return it."""
        kp_by_index = {k.index: k for k in detection.keypoints}

        if self.draw_lines:
            self._draw_edges(frame, kp_by_index)
            self._draw_centre_circle(frame, kp_by_index)

        if self.draw_points:
            self._draw_keypoints(frame, detection.visible_keypoints)

        return frame

    # ------------------------------------------------------------------ #
    def _draw_edges(
        self, frame: np.ndarray, kp_by_index: dict[int, FieldKeypoint]
    ) -> None:
        """Draw each pitch line only when both its endpoints are visible."""
        for edge in fc.EDGES:
            a = kp_by_index.get(edge.a)
            b = kp_by_index.get(edge.b)
            if a is None or b is None or not a.visible or not b.visible:
                continue
            color = fc.color_for_group(edge.group)
            cv2.line(
                frame,
                (int(a.x), int(a.y)),
                (int(b.x), int(b.y)),
                color,
                self.line_thickness,
                cv2.LINE_AA,
            )

    def _draw_centre_circle(
        self, frame: np.ndarray, kp_by_index: dict[int, FieldKeypoint]
    ) -> None:
        """
        Draw the centre circle from its visible extreme keypoints.

        With >= 3 visible circle points we fit a rotated ellipse (robust under
        perspective) via minAreaRect; with 2 we approximate a circle.
        """
        pts = [
            (int(kp_by_index[i].x), int(kp_by_index[i].y))
            for i in fc.CENTRE_CIRCLE_INDICES
            if i in kp_by_index and kp_by_index[i].visible
        ]
        if len(pts) < 2:
            return

        color = fc.color_for_group(fc.GROUP_CENTRE_CIRCLE)
        arr = np.array(pts, dtype=np.float32)

        if len(pts) >= 3:
            # Rotated rect bounding the circle points -> inscribed ellipse.
            (cx, cy), (w, h), angle = cv2.minAreaRect(arr)
            axes = (max(int(w / 2), 1), max(int(h / 2), 1))
            cv2.ellipse(
                frame, (int(cx), int(cy)), axes, angle, 0, 360, color,
                self.line_thickness, cv2.LINE_AA,
            )
        else:
            # Two points: use their span as the diameter of an approximate circle.
            (x1, y1), (x2, y2) = pts
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = int(np.hypot(x2 - x1, y2 - y1) / 2)
            if radius > 0:
                cv2.circle(
                    frame, (cx, cy), radius, color, self.line_thickness, cv2.LINE_AA
                )

    def _draw_keypoints(
        self, frame: np.ndarray, keypoints: list[FieldKeypoint]
    ) -> None:
        """Draw each visible keypoint as a filled dot, optionally labelled."""
        for kp in keypoints:
            color = fc.color_for_group(kp.group)
            center = (int(kp.x), int(kp.y))
            cv2.circle(frame, center, self.point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, center, self.point_radius, (0, 0, 0), 1, cv2.LINE_AA)

            if self.draw_labels:
                # Compact index label keeps the frame readable; full names live
                # in the exported JSON.
                cv2.putText(
                    frame,
                    str(kp.index),
                    (center[0] + 6, center[1] - 6),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def draw_summary(self, frame: np.ndarray, detection: FieldDetection) -> np.ndarray:
        """Draw a small top-left readout of how many keypoints were found."""
        n_visible = len(detection.visible_keypoints)
        text = f"pitch {detection.pitch_confidence:.2f} | keypoints {n_visible}/{fc.NUM_KEYPOINTS}"
        cv2.putText(frame, text, (10, 30), self.font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 30), self.font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame
