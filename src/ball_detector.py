"""
Dedicated ball detection + tracking using a trained YOLOv8 ball model.

The custom `football-players-detection.pt` model often misses the ball because
it is tiny and fast. A model trained specifically to detect the football
(``football-ball-detection.pt``) is far more reliable for the ball alone, so
this module runs that dedicated model as the authoritative ball detector.

Design notes:
  * If a COCO-pretrained model is used instead, pass ``classes=[32]`` to select
    the "sports ball" class. For a single-class custom ball model, leave
    ``classes=None`` and the most confident detection is taken.
  * If the weights can't be loaded (missing file / offline), the detector
    disables itself gracefully and the pipeline falls back to the custom
    players model's own ball detections.
  * A light nearest-neighbour + exponential smoother keeps the tracked ball
    position continuous across frames and rejects sudden teleports.
"""

from __future__ import annotations

import math

import numpy as np


class BallDetector:
    """Trained-YOLOv8 ball detector with a simple single-object tracker."""

    def __init__(
        self,
        model_path: str = "football-ball-detection/football-ball-detection.pt",
        conf_threshold: float = 0.15,
        device: str | None = None,
        imgsz: int = 1280,
        smoothing: float = 0.65,
        max_jump_px: float = 280.0,
        max_coast_frames: int = 25,
        classes: list[int] | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.smoothing = smoothing
        self.max_jump_px = max_jump_px
        self.max_coast_frames = max_coast_frames
        # Optional class filter (e.g. [32] for COCO "sports ball"). For a
        # single-class custom ball model this stays None.
        self.classes = classes

        self._available = False
        self._model = None
        self._device = device

        # Tracker state (pixel coordinates of the ball centre).
        self._last: tuple[float, float] | None = None
        self._coast = 0

        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
            self._available = True
            print(f"[info] ball model loaded: {model_path}")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully if unavailable
            print(f"[warn] ball detector unavailable ({model_path}): {exc}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        """
        Return the ball centre (cx, cy) in pixel coordinates, or None.

        Runs the trained detector, picks the most confident ball box, applies a
        jump filter + exponential smoothing, and coasts on the last known
        position for a few frames when the ball is briefly missed.
        """
        if not self._available or self._model is None:
            return None

        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self._device,
            classes=self.classes,
            verbose=False,
        )
        result = results[0]
        best: tuple[float, float] | None = None
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            # Highest-confidence ball.
            i = int(np.argmax(confs))
            x1, y1, x2, y2 = xyxy[i]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Reject implausible teleports from the last known position.
            if self._last is not None:
                jump = math.hypot(cx - self._last[0], cy - self._last[1])
                if jump > self.max_jump_px and self._coast < self.max_coast_frames:
                    # Treat as a spurious detection; coast instead.
                    best = None
                else:
                    best = (cx, cy)
            else:
                best = (cx, cy)

        if best is not None:
            self._last = self._smooth(self._last, best)
            self._coast = 0
            return self._last

        # Missed this frame: coast on the last position briefly.
        if self._last is not None and self._coast < self.max_coast_frames:
            self._coast += 1
            return self._last

        self._last = None
        return None

    def reset(self) -> None:
        self._last = None
        self._coast = 0

    # ------------------------------------------------------------------ #
    def _smooth(
        self, prev: tuple[float, float] | None, cur: tuple[float, float]
    ) -> tuple[float, float]:
        if prev is None or self.smoothing <= 0.0:
            return cur
        a = self.smoothing
        return (a * prev[0] + (1 - a) * cur[0], a * prev[1] + (1 - a) * cur[1])
