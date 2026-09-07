"""
Advanced football (ball) tracker using a Kalman filter.

YOLOv8 supplies noisy, intermittent ball detections. This module keeps a
continuous estimate by:

    predict (Kalman)  ->  associate nearest detection  ->  correct / coast

The state is a constant-velocity model [x, y, vx, vy], which handles fast
ball movement well. When the ball is temporarily invisible (occlusion, motion
blur, out of frame) the filter keeps predicting for up to `max_coast_frames`,
bridging short gaps without losing the trajectory.

Outputs per frame: coordinates, velocity, speed, and whether the position came
from a real detection or a Kalman prediction. The full history is exported to
`ball_trajectory.json`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection


@dataclass
class BallState:
    """Ball position and motion at a single frame."""

    frame: int
    timestamp_sec: float
    x: float
    y: float
    vx: float          # pixels per second
    vy: float          # pixels per second
    speed: float       # pixels per second (magnitude)
    detected: bool     # True = YOLO measurement, False = Kalman coast
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "vx": round(self.vx, 2),
            "vy": round(self.vy, 2),
            "speed_px_per_sec": round(self.speed, 2),
            "detected": self.detected,
            "confidence": round(self.confidence, 4) if self.confidence else None,
        }


class BallTracker:
    """
    Kalman-filter ball tracker fed by YOLOv8 ball detections.

    Tuned for football: high process noise (fast kicks), permissive association
  gate (motion blur drops confidence but position is still roughly right),
    and a generous coast window for brief disappearances.
    """

    # Kalman state layout: [x, y, vx, vy]
    _STATE_DIM = 4
    _MEAS_DIM = 2

    def __init__(
        self,
        fps: float = 25.0,
        ball_conf_threshold: float = 0.10,
        gate_radius: float = 120.0,
        max_coast_frames: int = 20,
        trajectory_length: int = 60,
        process_noise: float = 1e-1,
        measurement_noise: float = 5.0,
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 25.0
        self.ball_conf_threshold = ball_conf_threshold
        self.gate_radius = gate_radius
        self.max_coast_frames = max_coast_frames
        self.trajectory_length = trajectory_length

        self._kf = self._build_kalman(process_noise, measurement_noise)
        self._initialized = False
        self._missed = 0
        self._history: list[BallState] = []
        self._current: BallState | None = None

    # ------------------------------------------------------------------ #
    # Kalman setup
    # ------------------------------------------------------------------ #
    def _build_kalman(
        self, process_noise: float, measurement_noise: float
    ) -> cv2.KalmanFilter:
        """Create a constant-velocity Kalman filter."""
        kf = cv2.KalmanFilter(self._STATE_DIM, self._MEAS_DIM)
        kf.transitionMatrix = np.array(
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )

        # Process noise: higher values let the filter track abrupt direction /
        # speed changes (fast passes, deflections).
        kf.processNoiseCov = np.eye(self._STATE_DIM, dtype=np.float32) * process_noise

        # Measurement noise: ball boxes are small and jittery, especially
        # under motion blur, so we don't over-trust any single detection.
        kf.measurementNoiseCov = (
            np.eye(self._MEAS_DIM, dtype=np.float32) * measurement_noise
        )

        # Initial state uncertainty.
        kf.errorCovPost = np.eye(self._STATE_DIM, dtype=np.float32) * 500.0
        return kf

    # ------------------------------------------------------------------ #
    # Per-frame update
    # ------------------------------------------------------------------ #
    def update(
        self, frame_index: int, detections: list[Detection]
    ) -> BallState | None:
        """
        Ingest YOLO detections for one frame and return the ball state.

        Returns None only before the first ball has ever been seen.
        """
        timestamp = frame_index * self.dt
        ball_dets = self._ball_detections(detections)

        if self._initialized:
            # --- Predict -------------------------------------------------
            predicted = self._kf.predict()
            pred_x, pred_y = float(predicted[0, 0]), float(predicted[1, 0])

            # --- Associate -----------------------------------------------
            match = self._associate(ball_dets, pred_x, pred_y)

            if match is not None:
                cx, cy, conf = match
                self._kf.correct(np.array([[cx], [cy]], dtype=np.float32))
                self._missed = 0
                detected = True
                confidence = conf
            else:
                self._missed += 1
                detected = False
                confidence = None
                if self._missed > self.max_coast_frames:
                    # Ball has been gone too long; reset and wait for a fresh
                    # high-confidence sighting.
                    self._initialized = False
                    self._current = None
                    return None

            state = self._kf.statePost
            x, y, vx, vy = (
                float(state[0, 0]),
                float(state[1, 0]),
                float(state[2, 0]),
                float(state[3, 0]),
            )
        else:
            # --- Initialise on first usable detection --------------------
            if not ball_dets:
                return None
            best = max(ball_dets, key=lambda d: d.confidence)
            cx, cy = best.center
            self._kf.statePost = np.array(
                [[cx], [cy], [0.0], [0.0]], dtype=np.float32
            )
            self._initialized = True
            self._missed = 0
            x, y, vx, vy = float(cx), float(cy), 0.0, 0.0
            detected = True
            confidence = best.confidence

        speed = math.hypot(vx, vy)
        ball_state = BallState(
            frame=frame_index,
            timestamp_sec=timestamp,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            speed=speed,
            detected=detected,
            confidence=confidence,
        )
        self._current = ball_state
        self._history.append(ball_state)
        return ball_state

    def _ball_detections(self, detections: list[Detection]) -> list[Detection]:
        """Keep ball class detections above the (low) ball confidence floor."""
        return [
            d
            for d in detections
            if d.class_name == "ball" and d.confidence >= self.ball_conf_threshold
        ]

    def _associate(
        self,
        ball_dets: list[Detection],
        pred_x: float,
        pred_y: float,
    ) -> tuple[float, float, float] | None:
        """
        Pick the detection nearest the Kalman prediction within the gate.

        The gate is what makes blur-tolerant tracking work: a smeared ball may
        have low confidence but its centre is still close to where the filter
        expects it, so we associate on *position* not confidence alone.
        """
        if not ball_dets:
            return None

        best: tuple[float, float, float] | None = None
        best_dist = self.gate_radius

        for det in ball_dets:
            cx, cy = det.center
            dist = math.hypot(cx - pred_x, cy - pred_y)
            if dist < best_dist:
                best_dist = dist
                best = (float(cx), float(cy), det.confidence)

        return best

    # ------------------------------------------------------------------ #
    # Accessors + export
    # ------------------------------------------------------------------ #
    @property
    def current(self) -> BallState | None:
        return self._current

    @property
    def is_active(self) -> bool:
        return self._initialized and self._current is not None

    def trajectory_points(self) -> list[tuple[int, int]]:
        """Recent (x, y) integer points for drawing the trail."""
        recent = self._history[-self.trajectory_length :]
        return [(int(s.x), int(s.y)) for s in recent]

    def summary(self) -> dict:
        """Aggregate stats for the run summary / JSON header."""
        speeds = [s.speed for s in self._history]
        detected_count = sum(1 for s in self._history if s.detected)
        return {
            "total_points": len(self._history),
            "detected_points": detected_count,
            "predicted_points": len(self._history) - detected_count,
            "max_speed_px_per_sec": round(max(speeds), 2) if speeds else 0.0,
            "avg_speed_px_per_sec": round(sum(speeds) / len(speeds), 2)
            if speeds
            else 0.0,
        }

    def save_json(self, path: str) -> None:
        """Write the full ball trajectory and summary statistics."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

        payload = {
            "fps": self.fps,
            **self.summary(),
            "trajectory": [s.to_dict() for s in self._history],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
