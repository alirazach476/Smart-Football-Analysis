"""
Homography estimation between the video frame and the real pitch plane.

Given the detected field keypoints (pixel coordinates) and their known
real-world positions on the pitch template (centimetres, from `field_config`),
this estimates the 3x3 homography `H` that maps

    frame pixels  --H-->  pitch centimetres

which is exactly what we need to project player/ball ground positions onto a
top-down bird's-eye view.

Robustness features for live video:
  * RANSAC in `cv2.findHomography` rejects mismatched keypoints.
  * A "last good" matrix is reused on frames with too few keypoints (e.g. the
    camera zoomed into a corner), so the minimap does not flicker/empty out.
  * Optional exponential smoothing of `H` reduces jitter between frames.
"""

from __future__ import annotations

import numpy as np
import cv2

from . import field_config as fc
from .field_detector import FieldDetection

# Template vertices as a (32, 2) array in centimetres.
_TEMPLATE = np.array(fc.TEMPLATE_VERTICES, dtype=np.float32)


class HomographyEstimator:
    """Estimates and applies the frame->pitch homography."""

    def __init__(
        self,
        min_keypoints: int = 4,
        ransac_thresh: float = 10.0,
        smoothing: float = 0.5,
        max_stale_frames: int = 30,
    ) -> None:
        # Need >= 4 correspondences for a homography; more is better.
        self.min_keypoints = max(min_keypoints, 4)
        self.ransac_thresh = ransac_thresh
        # smoothing in [0, 1): 0 = no smoothing, higher = smoother/laggier.
        self.smoothing = smoothing
        # Give up on the stale matrix after this many frames without a fresh fit.
        self.max_stale_frames = max_stale_frames

        self._H: np.ndarray | None = None       # current (smoothed) matrix
        self._last_good_H: np.ndarray | None = None
        self._stale = 0

    # ------------------------------------------------------------------ #
    def update(self, detection: FieldDetection) -> np.ndarray | None:
        """
        Recompute the homography from the frame's visible keypoints.

        Returns the current usable matrix (fresh, smoothed, or last-good),
        or None if none is available yet.
        """
        src, dst = self._correspondences(detection)

        if len(src) >= self.min_keypoints:
            H, _mask = cv2.findHomography(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh
            )
            if H is not None:
                self._H = self._smooth(self._H, H)
                self._last_good_H = self._H
                self._stale = 0
                return self._H

        # Not enough points (or fit failed): coast on the last good matrix.
        self._stale += 1
        if self._last_good_H is not None and self._stale <= self.max_stale_frames:
            return self._last_good_H
        return None

    def transform_points(
        self, points: np.ndarray, H: np.ndarray | None = None
    ) -> np.ndarray | None:
        """
        Map an array of frame points (N, 2) to pitch cm coordinates (N, 2).

        Uses the provided `H`, or the current internal matrix if omitted.
        """
        matrix = H if H is not None else self._H
        if matrix is None or points is None or len(points) == 0:
            return None
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, matrix)
        return out.reshape(-1, 2)

    def transform_point(
        self, x: float, y: float, H: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        """Convenience: transform a single (x, y) frame point to pitch cm."""
        result = self.transform_points(np.array([[x, y]], dtype=np.float32), H)
        if result is None:
            return None
        return float(result[0][0]), float(result[0][1])

    @property
    def matrix(self) -> np.ndarray | None:
        return self._H

    @property
    def is_ready(self) -> bool:
        return self._H is not None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _correspondences(
        detection: FieldDetection,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build matched (frame_px, template_cm) arrays from visible keypoints."""
        src_list: list[tuple[float, float]] = []
        dst_list: list[tuple[float, float]] = []
        for kp in detection.visible_keypoints:
            src_list.append((kp.x, kp.y))
            dst_list.append(tuple(_TEMPLATE[kp.index]))
        src = np.array(src_list, dtype=np.float32)
        dst = np.array(dst_list, dtype=np.float32)
        return src, dst

    def _smooth(
        self, prev: np.ndarray | None, new: np.ndarray
    ) -> np.ndarray:
        """Exponentially smooth the homography for temporal stability."""
        if prev is None or self.smoothing <= 0.0:
            return new
        # Normalise both by their [2,2] element so they are directly comparable,
        # then blend.
        a = prev / prev[2, 2]
        b = new / new[2, 2]
        blended = self.smoothing * a + (1.0 - self.smoothing) * b
        return blended / blended[2, 2]
